package phonmatchnet

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"
)

// bundled holds the default models. See ASSETS.md for provenance.
//
//go:embed models/*/config.json models/*/*.onnx
var bundled embed.FS

const (
	hiddenSize    = 256
	embeddingSize = 96
)

// Config formats written by tools/export.
const (
	detectorFormat = "naptrap-phonmatchnet-onnx-v1"
	googleFormat   = "naptrap-google-speech-embedding-onnx-v1"
	g2pFormat      = "naptrap-g2p-en-decoder-onnx-v1"
)

type detectorConfig struct {
	Format     string             `json:"format"`
	Thresholds map[string]float64 `json:"thresholds"`
	Requires   map[string]struct {
		SHA256 map[string]string `json:"sha256"`
	} `json:"requires"`
	SHA256 map[string]string `json:"sha256"`
}

type googleConfig struct {
	Format        string            `json:"format"`
	EmbeddingSize int               `json:"embedding_size"`
	SHA256        map[string]string `json:"sha256"`
}

type g2pConfig struct {
	Format          string            `json:"format"`
	HiddenSize      int               `json:"hidden_size"`
	Graphemes       []string          `json:"graphemes"`
	Phonemes        int               `json:"phonemes"`
	StartToken      int64             `json:"start_token"`
	EndToken        int64             `json:"end_token"`
	MaxDecoderSteps int               `json:"max_decoder_steps"`
	SHA256          map[string]string `json:"sha256"`
}

// models holds the verified graph bytes and parsed configuration.
type models struct {
	detector, google, g2pEncoder, g2pDecoder []byte
	g2p                                      g2pConfig
	thresholds                               map[float64]float64
}

// loadModels reads the embedded models, or dir when set, and checks every
// graph against the SHA-256 in its config and the companion pins in the
// PhonMatchNet config.
func loadModels(dir string) (models, error) {
	var root fs.FS = os.DirFS(dir)
	if dir == "" {
		var err error
		if root, err = fs.Sub(bundled, "models"); err != nil {
			return models{}, err
		}
	}
	var m models
	var detector detectorConfig
	var google googleConfig
	for _, c := range []struct {
		file   string
		value  any
		format *string
		want   string
	}{
		{"phonmatchnet/config.json", &detector, &detector.Format, detectorFormat},
		{"google/config.json", &google, &google.Format, googleFormat},
		{"g2p/config.json", &m.g2p, &m.g2p.Format, g2pFormat},
	} {
		data, err := fs.ReadFile(root, c.file)
		if err != nil {
			return models{}, fmt.Errorf("load model config (see ASSETS.md): %w", err)
		}
		if err = json.Unmarshal(data, c.value); err != nil {
			return models{}, fmt.Errorf("%s: %w", c.file, err)
		}
		if *c.format != c.want {
			return models{}, fmt.Errorf("%s: format %q, expected %q", c.file, *c.format, c.want)
		}
	}
	g := m.g2p
	if g.HiddenSize != hiddenSize || len(g.Graphemes) < 4 || g.Phonemes < 4 || g.MaxDecoderSteps < 1 ||
		g.StartToken < 0 || g.StartToken >= int64(g.Phonemes) || g.EndToken < 0 || g.EndToken >= int64(g.Phonemes) {
		return models{}, fmt.Errorf("g2p/config.json: unsupported configuration")
	}
	if google.EmbeddingSize != embeddingSize {
		return models{}, fmt.Errorf("google/config.json: embedding size %d, expected %d", google.EmbeddingSize, embeddingSize)
	}
	m.thresholds = make(map[float64]float64, len(detector.Thresholds))
	for key, threshold := range detector.Thresholds {
		far, err := strconv.ParseFloat(key, 64)
		if err != nil || !(far > 0 && far < 1) || math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
			return models{}, fmt.Errorf("phonmatchnet/config.json: invalid threshold %q: %v", key, threshold)
		}
		m.thresholds[far] = threshold
	}
	if len(m.thresholds) == 0 {
		return models{}, fmt.Errorf("phonmatchnet/config.json: no thresholds")
	}
	for _, f := range []struct {
		dir, file string
		sums      map[string]string
		requires  string
		data      *[]byte
	}{
		{"phonmatchnet", "model.onnx", detector.SHA256, "", &m.detector},
		{"google", "model.onnx", google.SHA256, "google", &m.google},
		{"g2p", "encoder.onnx", g.SHA256, "g2p", &m.g2pEncoder},
		{"g2p", "decoder.onnx", g.SHA256, "g2p", &m.g2pDecoder},
	} {
		name := path.Join(f.dir, f.file)
		data, err := fs.ReadFile(root, name)
		if err != nil {
			return models{}, fmt.Errorf("load model (see ASSETS.md): %w", err)
		}
		sum := sha256.Sum256(data)
		digest := hex.EncodeToString(sum[:])
		if f.sums[f.file] != digest {
			return models{}, fmt.Errorf("%s does not match the SHA-256 in %s/config.json", name, f.dir)
		}
		if f.requires != "" && detector.Requires[f.requires].SHA256[f.file] != digest {
			return models{}, fmt.Errorf("%s is not the %s graph this PhonMatchNet model was trained with", name, f.requires)
		}
		*f.data = data
	}
	return m, nil
}

// TensorInfo describes one graph input or output. Dynamic dimensions are -1.
type TensorInfo struct {
	Name  string  `json:"name"`
	Shape []int64 `json:"shape"`
	Type  string  `json:"type"`
}

// GraphInfo describes one ONNX graph's inputs and outputs.
type GraphInfo struct {
	Inputs  []TensorInfo `json:"inputs"`
	Outputs []TensorInfo `json:"outputs"`
}

// ModelInfo describes the four loaded graphs.
type ModelInfo struct {
	Google     GraphInfo `json:"google"`
	Detector   GraphInfo `json:"detector"`
	G2PEncoder GraphInfo `json:"g2p_encoder"`
	G2PDecoder GraphInfo `json:"g2p_decoder"`
}

type tensorSpec struct {
	name  string
	shape []int64
	dtype ort.TensorElementDataType
}

type graphSpec struct {
	name            string
	inputs, outputs []tensorSpec
}

func (s graphSpec) inputNames() []string  { return names(s.inputs) }
func (s graphSpec) outputNames() []string { return names(s.outputs) }

func names(specs []tensorSpec) []string {
	result := make([]string, len(specs))
	for i, s := range specs {
		result[i] = s.name
	}
	return result
}

const (
	float32Type = ort.TensorElementDataTypeFloat
	int64Type   = ort.TensorElementDataTypeInt64
)

var (
	googleSpec = graphSpec{"google speech embedding",
		[]tensorSpec{{"waveform", []int64{-1}, float32Type}},
		[]tensorSpec{{"embeddings", []int64{-1, embeddingSize}, float32Type}}}
	detectorSpec = graphSpec{"phonmatchnet",
		[]tensorSpec{
			{"waveform", []int64{-1}, float32Type},
			{"google", []int64{-1, embeddingSize}, float32Type},
			{"text", []int64{-1, hiddenSize}, float32Type},
		},
		[]tensorSpec{{"logit", []int64{1}, float32Type}}}
	g2pEncoderSpec = graphSpec{"g2p encoder",
		[]tensorSpec{{"graphemes", []int64{-1}, int64Type}},
		[]tensorSpec{
			{"hidden", []int64{1, hiddenSize}, float32Type},
			{"separator", []int64{1, hiddenSize}, float32Type},
		}}
	g2pDecoderSpec = graphSpec{"g2p decoder",
		[]tensorSpec{
			{"token", []int64{1}, int64Type},
			{"hidden", []int64{1, hiddenSize}, float32Type},
		},
		[]tensorSpec{
			{"next_hidden", []int64{1, hiddenSize}, float32Type},
			{"logits", []int64{1, -1}, float32Type},
		}}
)

// inspectGraph checks a graph's tensors against spec, in order.
func inspectGraph(data []byte, spec graphSpec) (GraphInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(data)
	if err != nil {
		return GraphInfo{}, fmt.Errorf("inspect %s graph: %w", spec.name, err)
	}
	var info GraphInfo
	for _, group := range []struct {
		actual []ort.InputOutputInfo
		want   []tensorSpec
		into   *[]TensorInfo
	}{{inputs, spec.inputs, &info.Inputs}, {outputs, spec.outputs, &info.Outputs}} {
		if len(group.actual) != len(group.want) {
			return GraphInfo{}, fmt.Errorf("incompatible %s graph: %d tensors, expected %d", spec.name, len(group.actual), len(group.want))
		}
		for i, item := range group.actual {
			want := group.want[i]
			if item.Name != want.name || item.OrtValueType != ort.ONNXTypeTensor || item.DataType != want.dtype || !sameShape(item.Dimensions, want.shape) {
				return GraphInfo{}, fmt.Errorf("incompatible %s tensor: %s", spec.name, &item)
			}
			*group.into = append(*group.into, TensorInfo{item.Name, append([]int64(nil), item.Dimensions...), item.DataType.String()})
		}
	}
	return info, nil
}

// sameShape treats -1 in want as any dimension; the element counts of such
// outputs are checked after each run.
func sameShape(actual ort.Shape, want []int64) bool {
	if len(actual) != len(want) {
		return false
	}
	for i, dim := range actual {
		if want[i] != -1 && dim != want[i] {
			return false
		}
	}
	return true
}
