package phonmatchnet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/androiddrew/go-phonmatchnet/internal/ortruntime"
	ort "github.com/yalue/onnxruntime_go"
)

// Engine detects keywords. Requests and Close are serialized.
type Engine struct {
	mu                           sync.Mutex
	google, detector             *ort.DynamicAdvancedSession
	g2pEncoder, g2pDecoder       *ort.DynamicAdvancedSession
	g2p                          g2pConfig
	graphemes                    map[rune]int64
	unknownGrapheme, endGrapheme int64
	thresholds                   map[float64]float64
	info                         ModelInfo
	initialization               Initialization
	closed                       bool
}

// Initialization is a diagnostic breakdown of New's loading time.
// ModelSeconds includes reading, hashing and inspecting the graphs as well as
// creating the sessions. RuntimeVersion is the loaded native ONNX Runtime
// version.
type Initialization struct {
	RuntimeVersion string  `json:"runtime_version"`
	RuntimeSeconds float64 `json:"runtime_initialization_seconds"`
	ModelSeconds   float64 `json:"model_load_seconds"`
}

// Initialization returns how long New spent in each loading stage.
func (e *Engine) Initialization() Initialization { return e.initialization }

// ModelInfo returns the validated tensor metadata of the four graphs.
func (e *Engine) ModelInfo() ModelInfo { return e.info }

// New loads and validates the models before accepting requests. Close is
// mandatory. Concurrent calls on one engine are serialized, including Close.
func New(c Config) (_ *Engine, err error) {
	if c.Threads < 0 {
		return nil, errors.New("threads must be nonnegative")
	}
	m, err := loadModels(c.ModelDir)
	if err != nil {
		return nil, err
	}
	e := &Engine{g2p: m.g2p, thresholds: m.thresholds, graphemes: make(map[rune]int64)}
	for i, g := range m.g2p.Graphemes {
		switch g {
		case "<unk>":
			e.unknownGrapheme = int64(i)
		case "</s>":
			e.endGrapheme = int64(i)
		default:
			if r := []rune(g); len(r) == 1 {
				e.graphemes[r[0]] = int64(i)
			}
		}
	}
	start := time.Now()
	if err = ortruntime.Init(c.ORTLibrary); err != nil {
		return nil, err
	}
	e.initialization.RuntimeVersion = ort.GetVersion()
	e.initialization.RuntimeSeconds = time.Since(start).Seconds()
	start = time.Now()
	defer func() { e.initialization.ModelSeconds = time.Since(start).Seconds() }()
	options, err := sessionOptions(c.Threads)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, options.Destroy())
		if err != nil {
			err = errors.Join(err, e.destroySessions())
		}
	}()
	for _, g := range []struct {
		data    []byte
		spec    graphSpec
		info    *GraphInfo
		session **ort.DynamicAdvancedSession
	}{
		{m.google, googleSpec, &e.info.Google, &e.google},
		{m.detector, detectorSpec, &e.info.Detector, &e.detector},
		{m.g2pEncoder, g2pEncoderSpec, &e.info.G2PEncoder, &e.g2pEncoder},
		{m.g2pDecoder, g2pDecoderSpec, &e.info.G2PDecoder, &e.g2pDecoder},
	} {
		if *g.info, err = inspectGraph(g.data, g.spec); err != nil {
			return nil, err
		}
		*g.session, err = ort.NewDynamicAdvancedSessionWithONNXData(g.data, g.spec.inputNames(), g.spec.outputNames(), options)
		if err != nil {
			return nil, fmt.Errorf("load %s session: %w", g.spec.name, err)
		}
	}
	return e, nil
}

func sessionOptions(threads int) (_ *ort.SessionOptions, err error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if threads == 0 {
		threads = 1
	}
	for _, set := range []func() error{
		func() error { return o.SetIntraOpNumThreads(threads) },
		func() error { return o.SetInterOpNumThreads(1) },
		func() error { return o.SetExecutionMode(ort.ExecutionModeSequential) },
		func() error { return o.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll) },
	} {
		if err = set(); err != nil {
			return nil, errors.Join(err, o.Destroy())
		}
	}
	return o, nil
}

func (e *Engine) destroySessions() error {
	var errs []error
	for _, s := range []**ort.DynamicAdvancedSession{&e.google, &e.detector, &e.g2pEncoder, &e.g2pDecoder} {
		if *s != nil {
			errs = append(errs, (*s).Destroy())
			*s = nil
		}
	}
	return errors.Join(errs...)
}

// Close destroys the sessions. It is idempotent. The ONNX Runtime environment
// stays initialized for other engines and later calls to New.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	return e.destroySessions()
}

// Thresholds returns the calibrated operating points: target false-accept
// rate to score threshold. Each was fitted on held-out development speakers;
// the achieved rate on other audio differs, so applications should validate
// a threshold on their own positive and negative clips and compare it with
// Result.Score.
func (e *Engine) Thresholds() map[float64]float64 {
	thresholds := make(map[float64]float64, len(e.thresholds))
	for far, threshold := range e.thresholds {
		thresholds[far] = threshold
	}
	return thresholds
}

// Detect scores one clip against a keyword.
func (e *Engine) Detect(ctx context.Context, r Request) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if err := validateAudio(r.Samples, r.SampleRate); err != nil {
		return Result{}, err
	}
	positions := r.Keyword.Positions()
	if positions < 1 || positions > MaxKeywordPositions || len(r.Keyword.features)%hiddenSize != 0 {
		return Result{}, fmt.Errorf("%w: create the Keyword with Engine.Keyword", ErrInvalidKeyword)
	}
	far := r.FAR
	if far == 0 {
		far = DefaultFAR
	}
	threshold, ok := e.thresholds[far]
	if !ok {
		return Result{}, fmt.Errorf("%w: %g; choose one of %v", ErrUnknownFAR, far, e.fars())
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Result{}, ErrClosed
	}
	result := Result{Keyword: r.Keyword.text, Threshold: threshold, TargetFAR: far}
	start := time.Now()
	samples := len(r.Samples)
	waveform, err := ort.NewTensor(ort.NewShape(int64(samples)), r.Samples)
	if err != nil {
		return Result{}, err
	}
	defer func() { _ = waveform.Destroy() }()
	outputs, err := run(e.google, []ort.Value{waveform}, 1)
	if err != nil {
		return Result{}, fmt.Errorf("google speech embedding: %w", err)
	}
	embeddings := outputs[0]
	defer func() { _ = embeddings.Destroy() }()
	if err = checkOutput(embeddings, int64(samples/1280), embeddingSize); err != nil {
		return Result{}, fmt.Errorf("google speech embedding: %w", err)
	}
	result.Timings.EmbeddingSeconds = time.Since(start).Seconds()
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	start = time.Now()
	text, err := ort.NewTensor(ort.NewShape(int64(positions), hiddenSize), r.Keyword.features)
	if err != nil {
		return Result{}, err
	}
	defer func() { _ = text.Destroy() }()
	outputs, err = run(e.detector, []ort.Value{waveform, embeddings, text}, 1)
	if err != nil {
		return Result{}, fmt.Errorf("phonmatchnet: %w", err)
	}
	logit, err := floats(outputs[0], 1)
	_ = outputs[0].Destroy()
	if err != nil {
		return Result{}, fmt.Errorf("phonmatchnet: %w", err)
	}
	result.Timings.DetectorSeconds = time.Since(start).Seconds()
	result.Logit = float64(logit[0])
	result.Score = 1 / (1 + math.Exp(-result.Logit))
	result.Accepted = result.Score >= threshold
	return result, nil
}

func (e *Engine) fars() []float64 {
	fars := make([]float64, 0, len(e.thresholds))
	for far := range e.thresholds {
		fars = append(fars, far)
	}
	sort.Float64s(fars)
	return fars
}

func validateAudio(samples []float32, rate int) error {
	if rate != SampleRate {
		return fmt.Errorf("%w: sample rate %d; resample to mono %d Hz", ErrInvalidAudio, rate, SampleRate)
	}
	if len(samples) < MinSamples || len(samples) > MaxSamples {
		return fmt.Errorf("%w: %d samples; clips must be 0.5-2 seconds (%d-%d samples)",
			ErrInvalidAudio, len(samples), MinSamples, MaxSamples)
	}
	for _, s := range samples {
		if !(s >= -1 && s <= 1) {
			return fmt.Errorf("%w: samples must be finite and within [-1, 1]", ErrInvalidAudio)
		}
	}
	return nil
}

// checkOutput verifies a float32 output's [rows, cols] shape and finiteness.
func checkOutput(value ort.Value, rows, cols int64) error {
	tensor, ok := value.(*ort.Tensor[float32])
	if !ok {
		return errors.New("output must be a float32 tensor")
	}
	if shape := tensor.GetShape(); len(shape) != 2 || shape[0] != rows || shape[1] != cols {
		return fmt.Errorf("output shape %v, expected [%d %d]", shape, rows, cols)
	}
	if !finite(tensor.GetData()) {
		return errors.New("output contains nonfinite values")
	}
	return nil
}

func finite(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

// run executes a session with runtime-allocated outputs. On error every
// allocated output is destroyed.
func run(session *ort.DynamicAdvancedSession, inputs []ort.Value, count int) ([]ort.Value, error) {
	outputs := make([]ort.Value, count)
	if err := session.Run(inputs, outputs); err != nil {
		for _, output := range outputs {
			if output != nil {
				_ = output.Destroy()
			}
		}
		return nil, err
	}
	return outputs, nil
}
