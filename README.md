# go-phonmatchnet

`github.com/androiddrew/go-phonmatchnet` (package `phonmatchnet`) detects
**text-defined wake words** with ONNX Runtime. Type a keyword, give it a short
clip of 16 kHz mono audio, and it returns the probability that the keyword was
spoken. New wake words need no recordings or training.

It runs [naptrap](https://github.com/androiddrew/naptrap)'s independently
trained English [PhonMatchNet](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en)
(Lee & Cho, Interspeech 2023) and matches naptrap's PyTorch scores to about
1e-5. All models are embedded (7.8 MB), so the only external requirement is an
installed ONNX Runtime. Audio decoding, resampling, microphone windowing and
any CLI or serving layer belong to the application.

```go
import (
    "context"
    "errors"
    phonmatchnet "github.com/androiddrew/go-phonmatchnet"
)

func detect(ctx context.Context, samples []float32) (result phonmatchnet.Result, err error) {
    engine, err := phonmatchnet.New(phonmatchnet.Config{
        ORTLibrary: "libonnxruntime.so", // OS loader name or explicit installed-library path
    })
    if err != nil { return result, err }
    defer func() { err = errors.Join(err, engine.Close()) }()

    keyword, err := engine.Keyword(ctx, "hey computer") // reuse across clips
    if err != nil { return result, err }
    return engine.Detect(ctx, phonmatchnet.Request{
        Samples: samples, SampleRate: phonmatchnet.SampleRate, Keyword: keyword,
    })
}
```

`result.Score` is the keyword-match probability (0–1) and `result.Accepted`
reports `Score >= Threshold`. Reuse one engine across requests.

## API and guarantees

- `Keyword` normalizes the text exactly as naptrap does (accents and
  punctuation removed, lowercase, apostrophes inside words kept) and computes
  its g2p decoder features. Numbers must be spelled out ("route sixty six").
  `NormalizeKeyword` exposes the normalization alone. A keyword may produce at
  most `MaxKeywordPositions` (128) feature rows.
- `Detect` scores **one clip of 0.5–2 seconds** (`MinSamples`–`MaxSamples`)
  of finite samples in [-1, 1] at `SampleRate`; otherwise `ErrInvalidAudio`.
  Samples are not rescaled, clipped, trimmed, mixed or resampled, and must not
  be mutated during the call.
- `Result` also carries the raw `Logit`, the operating point used and stage
  `Timings`. `ModelInfo` returns the validated tensor metadata of the four
  graphs; `Initialization` reports loading times and the native runtime version.
- Errors: `ErrInvalidAudio`, `ErrInvalidKeyword`, `ErrUnknownFAR`, `ErrClosed`,
  `ErrNotInitialized`, plus context cancellation.

## Thresholds

`Request.FAR` selects a calibrated operating point by its target per-clip
false-accept rate; `Engine.Thresholds` lists them:

| FAR | Score threshold |
|---|---|
| `0.01` (default) | 0.7149 |
| `0.1` | 0.0026 |

These were fitted on held-out LibriSpeech development speakers. **The targets
do not hold on other audio:** on phonetically confusable phrases the 1%
threshold accepted 31% of negatives. In the test fixtures, Kokoro-synthesized
"hey commuter" scores up to 0.99 for the keyword "hey computer". For a real
product, record positive and negative clips in your setting and choose your
own threshold on `Result.Score`. See the
[model card](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en) for
the evaluation.

## Listening to a microphone

The model scores single clips; it is not a streaming listener. To watch live
audio, window it yourself, e.g. 1.5 s windows every 0.25 s, compute the
keyword once, and debounce repeated accepts. Detection of a 1.5 s window takes
about 5.5 ms and a two-word keyword about 0.7 ms on one thread of a Ryzen 5
5600X (`go test -bench .`).

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by every model in it.

- **Set `ORTLibrary`** (simplest): the engine initializes the environment through
  [`github.com/androiddrew/ortenv`](https://github.com/androiddrew/ortenv) on first
  use. The environment stays loaded until process exit, so closing the last engine
  never unloads the runtime. Every component using ortenv in the process must use
  the same library selector, and nothing may destroy the environment ortenv owns.
- **Leave `ORTLibrary` empty** when your application already manages ONNX Runtime
  itself (directly through `onnxruntime_go` or through ortenv). Initialize the
  environment before `New` and destroy it only after `Close`; otherwise `New`
  returns `ErrNotInitialized`.

`Config.Threads` sets intra-op threads per session (default 1). With many models
in one process, keep it small to avoid oversubscribing the CPU.

## Runtime and behavior

Close every engine. Each engine serializes its requests and Close. Native Run
cannot be interrupted mid-call; cancellation is checked between graph runs.
Inference runs on CPU; the models are small enough that GPU execution is not
offered. Requires Go 1.25 or newer with CGO. Tested: Linux/amd64, Go 1.25.0
and 1.26.4, ORT 1.22.0 and 1.22.2. This is a known tested configuration, not
an exact native-version restriction.

Limits inherited from the model: English only, trained on read audiobook
speech (expect lower accuracy with noise, far-field microphones, accents or
children); no speaker verification, so anyone saying the keyword triggers it;
similar-sounding phrases are often confused; not for security or
safety-critical use.

## Development and tools

```bash
go test ./...
go vet ./...
```

For native tests, set `PHONMATCHNET_TEST_ORT` to an ONNX Runtime library path
or loader name, then run `go test -race ./...`. The tests compare keyword
features, speech embeddings and 80 clip/keyword scores with naptrap's PyTorch
outputs in `testdata/reference.json`. Native tests skip when the variable is
unset. Tests that need ONNX Runtime uninitialized run in a fresh subprocess,
because ortenv keeps the environment loaded until process exit.

[`tools/export`](tools/export) regenerates the embedded models from naptrap's
pinned PyTorch releases, verifies them against naptrap, and assembles the
Hugging Face repositories. `tools/export/fixtures.py` rebuilds the test
fixtures (Python, not needed to use the library).

See [ASSETS.md](ASSETS.md), [THIRD_PARTY.md](THIRD_PARTY.md), LICENSE and NOTICE.

## Citation

If the PhonMatchNet model is useful in your research, please cite:

```bibtex
@inproceedings{lee23phonmatchnet,
  title     = {PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords},
  author    = {Lee, Yong-Hyeok and Cho, Namhyun},
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {3964--3968},
  year      = {2023}
}
```

go-phonmatchnet is not affiliated with or endorsed by NCSOFT, the PhonMatchNet
authors, Google or the g2p_en authors.
