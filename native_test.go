package phonmatchnet

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/androiddrew/go-phonmatchnet/internal/nativetest"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// Tolerances against naptrap's PyTorch outputs. tools/export measures the
// ONNX graphs at about 1e-5 of PyTorch on CPU; these leave room for other CPUs.
const (
	featureTolerance   = 1e-4
	embeddingTolerance = 1e-3
	logitTolerance     = 1e-3
)

func testLibrary(t *testing.T) string {
	t.Helper()
	library := os.Getenv("PHONMATCHNET_TEST_ORT")
	if library == "" {
		t.Skip("set PHONMATCHNET_TEST_ORT to an ONNX Runtime library for native tests")
	}
	return library
}

func newTestEngine(t *testing.T, config Config) *Engine {
	t.Helper()
	engine, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	})
	return engine
}

func maxDiff(t *testing.T, got []float32, want [][]float32) float64 {
	t.Helper()
	if len(want) == 0 || len(got) != len(want)*len(want[0]) {
		t.Fatalf("got %d values, want %d rows", len(got), len(want))
	}
	worst := 0.0
	for i, row := range want {
		for j, value := range row {
			worst = math.Max(worst, math.Abs(float64(got[i*len(row)+j]-value)))
		}
	}
	return worst
}

func TestReferenceParity(t *testing.T) {
	engine := newTestEngine(t, Config{ORTLibrary: testLibrary(t)})
	ref := loadReference(t)
	ctx := context.Background()

	for far, threshold := range engine.Thresholds() {
		if want := ref.Thresholds[strconv.FormatFloat(far, 'g', -1, 64)]; want != threshold {
			t.Errorf("threshold for FAR %g: %v, want %v", far, threshold, want)
		}
	}
	var worstFeature, worstEmbedding, worstLogit float64
	for _, k := range ref.Keywords {
		keyword, err := engine.Keyword(ctx, k.Keyword)
		if err != nil {
			t.Fatalf("%q: %v", k.Keyword, err)
		}
		if keyword.Text() != k.Normalized || keyword.Positions() != len(k.Features) {
			t.Fatalf("%q: %q with %d positions, want %q with %d", k.Keyword, keyword.Text(), keyword.Positions(), k.Normalized, len(k.Features))
		}
		diff := maxDiff(t, keyword.features, k.Features)
		if diff > featureTolerance {
			t.Errorf("%q: text features differ by %g", k.Keyword, diff)
		}
		worstFeature = math.Max(worstFeature, diff)
	}
	for _, r := range ref.Embeddings {
		samples := readWAV(t, r.Clip)
		waveform, err := ort.NewTensor(ort.NewShape(int64(len(samples))), samples)
		if err != nil {
			t.Fatal(err)
		}
		outputs, err := run(engine.google, []ort.Value{waveform}, 1)
		if err != nil {
			t.Fatal(err)
		}
		diff := maxDiff(t, outputs[0].(*ort.Tensor[float32]).GetData(), r.Embeddings)
		if diff > embeddingTolerance {
			t.Errorf("%s: embeddings differ by %g", r.Clip, diff)
		}
		worstEmbedding = math.Max(worstEmbedding, diff)
		_ = outputs[0].Destroy()
		_ = waveform.Destroy()
	}
	keywords := map[string]Keyword{}
	for _, d := range ref.Detections {
		keyword, ok := keywords[d.Keyword]
		if !ok {
			var err error
			if keyword, err = engine.Keyword(ctx, d.Keyword); err != nil {
				t.Fatal(err)
			}
			keywords[d.Keyword] = keyword
		}
		result, err := engine.Detect(ctx, Request{Samples: readWAV(t, d.Clip), SampleRate: SampleRate, Keyword: keyword})
		if err != nil {
			t.Fatalf("%s / %q: %v", d.Clip, d.Keyword, err)
		}
		worstLogit = math.Max(worstLogit, math.Abs(result.Logit-d.Logit))
		if math.Abs(result.Logit-d.Logit) > logitTolerance || result.Accepted != d.Accepted {
			t.Errorf("%s / %q: logit %.6f accepted %v, naptrap %.6f accepted %v",
				d.Clip, d.Keyword, result.Logit, result.Accepted, d.Logit, d.Accepted)
		}
		if result.TargetFAR != DefaultFAR || result.Keyword != keyword.Text() {
			t.Errorf("%s / %q: result %+v", d.Clip, d.Keyword, result)
		}
	}
	t.Logf("max |go - naptrap|: text features %.3g, embeddings %.3g, logits %.3g over %d keywords, %d clips, %d detections",
		worstFeature, worstEmbedding, worstLogit, len(ref.Keywords), len(ref.Embeddings), len(ref.Detections))
}

func TestRequestValidation(t *testing.T) {
	engine := newTestEngine(t, Config{ORTLibrary: testLibrary(t), Threads: 2})
	ctx := context.Background()
	keyword, err := engine.Keyword(ctx, "hey computer")
	if err != nil {
		t.Fatal(err)
	}
	clip := readWAV(t, "said-mister.wav")
	nan := append([]float32(nil), clip...)
	nan[10] = float32(math.NaN())
	loud := append([]float32(nil), clip...)
	loud[10] = 1.5
	for _, c := range []struct {
		name    string
		request Request
		want    error
	}{
		{"rate", Request{Samples: clip, SampleRate: 8000, Keyword: keyword}, ErrInvalidAudio},
		{"short", Request{Samples: clip[:MinSamples-1], SampleRate: SampleRate, Keyword: keyword}, ErrInvalidAudio},
		{"long", Request{Samples: make([]float32, MaxSamples+1), SampleRate: SampleRate, Keyword: keyword}, ErrInvalidAudio},
		{"nan", Request{Samples: nan, SampleRate: SampleRate, Keyword: keyword}, ErrInvalidAudio},
		{"loud", Request{Samples: loud, SampleRate: SampleRate, Keyword: keyword}, ErrInvalidAudio},
		{"zero keyword", Request{Samples: clip, SampleRate: SampleRate}, ErrInvalidKeyword},
		{"far", Request{Samples: clip, SampleRate: SampleRate, Keyword: keyword, FAR: 0.05}, ErrUnknownFAR},
	} {
		if _, err := engine.Detect(ctx, c.request); !errors.Is(err, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.want)
		}
	}
	for _, text := range []string{"route 66", "!!!", strings.Repeat("supercalifragilistic ", 12)} {
		if _, err := engine.Keyword(ctx, text); !errors.Is(err, ErrInvalidKeyword) {
			t.Errorf("Keyword(%q): %v", text, err)
		}
	}
	for _, length := range []int{MinSamples, MaxSamples} {
		samples := make([]float32, length)
		copy(samples, clip)
		if _, err := engine.Detect(ctx, Request{Samples: samples, SampleRate: SampleRate, Keyword: keyword, FAR: 0.1}); err != nil {
			t.Errorf("%d samples: %v", length, err)
		}
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := engine.Detect(canceled, Request{Samples: clip, SampleRate: SampleRate, Keyword: keyword}); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled Detect: %v", err)
	}
	if _, err := engine.Keyword(canceled, "hey computer"); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled Keyword: %v", err)
	}
}

func TestLifecycle(t *testing.T) {
	library := testLibrary(t)
	ctx := context.Background()
	clip := readWAV(t, "said-mister.wav")
	// A graph that matches its hashes but is not ONNX: the sessions loaded
	// before it must be released and the environment kept.
	bad := copyModels(t)
	writeModel(t, bad, "g2p/decoder.onnx", []byte("not ONNX"))
	if _, err := New(Config{ORTLibrary: library, ModelDir: bad}); err == nil || !strings.Contains(err.Error(), "g2p decoder") {
		t.Fatalf("malformed graph: %v", err)
	}
	if !ort.IsInitialized() {
		t.Fatal("failed New destroyed the environment")
	}
	for range 3 {
		engine, err := New(Config{ORTLibrary: library})
		if err != nil {
			t.Fatal(err)
		}
		second, err := New(Config{ORTLibrary: library})
		if err != nil {
			t.Fatal(err)
		}
		if err := second.Close(); err != nil {
			t.Fatal(err)
		}
		keyword, err := engine.Keyword(ctx, "said mister")
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				result, err := engine.Detect(ctx, Request{Samples: clip, SampleRate: SampleRate, Keyword: keyword})
				if err != nil || !result.Accepted {
					t.Errorf("concurrent Detect: %+v, %v", result, err)
				}
			})
		}
		wg.Wait()
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := engine.Detect(ctx, Request{Samples: clip, SampleRate: SampleRate, Keyword: keyword}); !errors.Is(err, ErrClosed) {
			t.Fatalf("Detect after Close: %v", err)
		}
		if _, err := engine.Keyword(ctx, "said mister"); !errors.Is(err, ErrClosed) {
			t.Fatalf("Keyword after Close: %v", err)
		}
	}
}

func TestModelInfo(t *testing.T) {
	engine := newTestEngine(t, Config{ORTLibrary: testLibrary(t)})
	info := engine.ModelInfo()
	if len(info.Detector.Inputs) != 3 || info.G2PDecoder.Outputs[1].Shape[1] != 74 {
		t.Fatalf("model info: %+v", info)
	}
	if init := engine.Initialization(); init.RuntimeVersion == "" {
		t.Fatalf("initialization: %+v", init)
	}
}

func TestCallerOwnedEnvironment(t *testing.T) {
	library := testLibrary(t)
	if !nativetest.Subprocess(t) {
		return
	}
	if _, err := New(Config{}); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("uninitialized caller-owned environment: %v", err)
	}
	if err := ortenv.Init(library); err != nil {
		t.Fatal(err)
	}
	engine, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Keyword(context.Background(), "hey computer"); err != nil {
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("Close destroyed a caller-owned environment")
	}
}

// ModelDir problems are reported before ONNX Runtime is touched.
func TestModelDirValidation(t *testing.T) {
	if _, err := New(Config{ModelDir: t.TempDir()}); err == nil || !strings.Contains(err.Error(), "ASSETS.md") {
		t.Errorf("empty directory: %v", err)
	}
	if _, err := New(Config{Threads: -1}); err == nil {
		t.Error("negative threads accepted")
	}
	tampered := copyModels(t)
	if err := os.WriteFile(filepath.Join(tampered, "google/model.onnx"), []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{ModelDir: tampered}); err == nil || !strings.Contains(err.Error(), "SHA-256") {
		t.Errorf("tampered graph: %v", err)
	}
	// A self-consistent google/ directory that is not the one PhonMatchNet pins.
	swapped := copyModels(t)
	updateConfig(t, swapped, "google/config.json", func(config map[string]any) {
		config["sha256"].(map[string]any)["model.onnx"] = writeFile(t, swapped, "google/model.onnx", []byte("other"))
	})
	if _, err := New(Config{ModelDir: swapped}); err == nil || !strings.Contains(err.Error(), "trained with") {
		t.Errorf("swapped companion: %v", err)
	}
	wrongFormat := copyModels(t)
	updateConfig(t, wrongFormat, "g2p/config.json", func(config map[string]any) { config["format"] = "other" })
	if _, err := New(Config{ModelDir: wrongFormat}); err == nil || !strings.Contains(err.Error(), "format") {
		t.Errorf("wrong format: %v", err)
	}
}

// copyModels copies the embedded models into a temporary ModelDir.
func copyModels(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	err := fs.WalkDir(bundled, "models", func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, err := bundled.ReadFile(name)
		if err != nil {
			return err
		}
		target := filepath.Join(dir, strings.TrimPrefix(name, "models/"))
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o600)
	})
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func writeFile(t *testing.T, dir, name string, data []byte) string {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// writeModel replaces a graph and updates every hash that pins it.
func writeModel(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	digest := writeFile(t, dir, name, data)
	component, file, _ := strings.Cut(name, "/")
	updateConfig(t, dir, component+"/config.json", func(config map[string]any) {
		config["sha256"].(map[string]any)[file] = digest
	})
	updateConfig(t, dir, "phonmatchnet/config.json", func(config map[string]any) {
		if pin, ok := config["requires"].(map[string]any)[component].(map[string]any); ok {
			pin["sha256"].(map[string]any)[file] = digest
		}
	})
}

func updateConfig(t *testing.T, dir, name string, update func(map[string]any)) {
	t.Helper()
	path := filepath.Join(dir, name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	update(config)
	if data, err = json.Marshal(config); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkDetect(b *testing.B) {
	library := os.Getenv("PHONMATCHNET_TEST_ORT")
	if library == "" {
		b.Skip("set PHONMATCHNET_TEST_ORT to an ONNX Runtime library for native benchmarks")
	}
	engine, err := New(Config{ORTLibrary: library})
	if err != nil {
		b.Fatal(err)
	}
	defer engine.Close()
	ctx := context.Background()
	keyword, err := engine.Keyword(ctx, "hey computer")
	if err != nil {
		b.Fatal(err)
	}
	clip := make([]float32, 3*SampleRate/2)
	b.Run("Keyword", func(b *testing.B) {
		for b.Loop() {
			if _, err := engine.Keyword(ctx, "hey computer"); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("Detect1.5s", func(b *testing.B) {
		for b.Loop() {
			if _, err := engine.Detect(ctx, Request{Samples: clip, SampleRate: SampleRate, Keyword: keyword}); err != nil {
				b.Fatal(err)
			}
		}
	})
}
