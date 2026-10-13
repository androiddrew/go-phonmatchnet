// Package phonmatchnet detects text-defined wake words in 16 kHz mono PCM with
// naptrap's PhonMatchNet model and ONNX Runtime.
package phonmatchnet

import (
	"errors"

	"github.com/androiddrew/go-phonmatchnet/internal/ortruntime"
)

// SampleRate is the required input rate in Hz. Input is mono.
const SampleRate = 16000

// Clip length bounds: one clip of 0.5 to 2 seconds.
const (
	MinSamples = SampleRate / 2
	MaxSamples = 2 * SampleRate
)

// MaxKeywordPositions bounds a keyword's text features: decoder steps plus
// one separator between words.
const MaxKeywordPositions = 128

// DefaultFAR selects the strictest calibrated operating point when Request.FAR
// is zero.
const DefaultFAR = 0.01

var (
	// ErrNotInitialized is returned by New when ORTLibrary is empty and the
	// caller has not initialized the ONNX Runtime environment.
	ErrNotInitialized = ortruntime.ErrNotInitialized
	// ErrClosed is returned by requests on a closed engine.
	ErrClosed = errors.New("phonmatchnet engine is closed")
	// ErrInvalidAudio reports a wrong sample rate, a clip outside 0.5-2 seconds,
	// or a sample that is nonfinite or outside [-1, 1].
	ErrInvalidAudio = errors.New("invalid audio")
	// ErrInvalidKeyword reports keyword text with no English letters, with
	// digits, or longer than MaxKeywordPositions features. It is also returned
	// for a zero Keyword.
	ErrInvalidKeyword = errors.New("invalid keyword")
	// ErrUnknownFAR is returned when Request.FAR is not a calibrated operating
	// point; see Engine.Thresholds.
	ErrUnknownFAR = errors.New("FAR is not a calibrated operating point")
)

// Config selects the models and configures the sessions.
type Config struct {
	// ORTLibrary is an ONNX Runtime library path or OS-loader name. When set, the
	// engine initializes the environment through ortenv, which retains it until
	// process exit. When empty, the caller owns the environment and must
	// initialize it before New and destroy it only after Close.
	ORTLibrary string
	// ModelDir overrides the embedded models with a directory holding
	// phonmatchnet/, google/ and g2p/ laid out like the embedded models/ (the
	// three Hugging Face -onnx downloads). Usually empty.
	ModelDir string
	// Threads sets intra-op threads per session. Zero selects 1.
	Threads int
}

// Request scores one clip against a keyword. Samples must remain unmodified
// until Detect returns. No gain, clipping, trimming, channel conversion or
// resampling is performed.
type Request struct {
	Samples    []float32 `json:"-"`
	SampleRate int       `json:"sample_rate"`
	// Keyword comes from Engine.Keyword. Reuse it across clips.
	Keyword Keyword `json:"-"`
	// FAR selects the operating point by its target false-accept rate. Zero
	// selects DefaultFAR.
	FAR float64 `json:"far"`
}

// Result is one detection. Score is sigmoid(Logit), the keyword-match
// probability. Accepted reports Score >= Threshold.
type Result struct {
	Keyword   string  `json:"keyword"` // normalized keyword text
	Score     float64 `json:"score"`
	Logit     float64 `json:"logit"`
	Threshold float64 `json:"threshold"`
	TargetFAR float64 `json:"target_far"`
	Accepted  bool    `json:"accepted"`
	Timings   Timings `json:"timings"`
}

// Timings is a diagnostic breakdown of one Detect call. Each stage includes
// tensor creation, the synchronous Run and output validation.
type Timings struct {
	EmbeddingSeconds float64 `json:"embedding_seconds"` // Google speech embedding graph
	DetectorSeconds  float64 `json:"detector_seconds"`  // PhonMatchNet graph
}
