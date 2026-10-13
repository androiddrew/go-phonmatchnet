package phonmatchnet

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// reference is testdata/reference.json, recorded from naptrap's PyTorch
// models by tools/export/fixtures.py.
type reference struct {
	Thresholds    map[string]float64 `json:"thresholds"`
	Normalization []struct {
		Input      string `json:"input"`
		Normalized string `json:"normalized"`
		Error      string `json:"error"`
	} `json:"normalization"`
	Keywords []struct {
		Keyword    string      `json:"keyword"`
		Normalized string      `json:"normalized"`
		Features   [][]float32 `json:"features"`
	} `json:"keywords"`
	Embeddings []struct {
		Clip       string      `json:"clip"`
		Embeddings [][]float32 `json:"embeddings"`
	} `json:"embeddings"`
	Detections []struct {
		Clip     string  `json:"clip"`
		Keyword  string  `json:"keyword"`
		Logit    float64 `json:"logit"`
		Score    float64 `json:"score"`
		Accepted bool    `json:"accepted"`
	} `json:"detections"`
}

func loadReference(t *testing.T) reference {
	t.Helper()
	data, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var r reference
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// readWAV decodes a mono 16-bit PCM WAV exactly like naptrap.load_wav.
func readWAV(t *testing.T, name string) []float32 {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	samples, err := decodeWAV(data)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return samples
}

func decodeWAV(data []byte) ([]float32, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("not a WAV file")
	}
	var format struct {
		Tag, Channels uint16
		Rate          uint32
		_             uint32
		_             uint16
		Bits          uint16
	}
	seenFormat := false
	for chunk := data[12:]; len(chunk) >= 8; {
		id, size := string(chunk[:4]), int(binary.LittleEndian.Uint32(chunk[4:8]))
		if size > len(chunk)-8 {
			return nil, errors.New("truncated chunk")
		}
		body := chunk[8 : 8+size]
		switch id {
		case "fmt ":
			if err := binary.Read(bytes.NewReader(body), binary.LittleEndian, &format); err != nil {
				return nil, err
			}
			seenFormat = true
		case "data":
			if !seenFormat || format.Tag != 1 || format.Channels != 1 || format.Rate != SampleRate || format.Bits != 16 {
				return nil, errors.New("expected mono 16-bit PCM at 16 kHz")
			}
			samples := make([]float32, size/2)
			for i := range samples {
				samples[i] = float32(int16(binary.LittleEndian.Uint16(body[2*i:]))) / 32768
			}
			return samples, nil
		}
		chunk = chunk[8+size+size%2:]
	}
	return nil, errors.New("no data chunk")
}
