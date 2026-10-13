# Embedded models

go-phonmatchnet ships all four ONNX graphs inside the module and embeds them in
your binary (about 7.8 MB), so `New` needs no file paths.

| Files | Size | Source | License |
|---|---:|---|---|
| `models/phonmatchnet/model.onnx`, `config.json` | 2.3 MB | [naptrap-phonmatchnet-en](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en) v1, with its 40-bin log-mel frontend | Apache-2.0 |
| `models/google/model.onnx`, `config.json` | 2.2 MB | [google-speech-embedding-pytorch](https://huggingface.co/androiddrew/google-speech-embedding-pytorch) v1, with its 32-bin log-mel frontend | Apache-2.0 |
| `models/g2p/encoder.onnx`, `decoder.onnx`, `config.json` | 3.3 MB | [g2p-en-decoder-pytorch](https://huggingface.co/androiddrew/g2p-en-decoder-pytorch) v1 | Apache-2.0 |

The same files are published on Hugging Face, one repository per model, each
with its model card and conversion tools:

- [androiddrew/naptrap-phonmatchnet-en-onnx](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en-onnx)
- [androiddrew/google-speech-embedding-onnx](https://huggingface.co/androiddrew/google-speech-embedding-onnx)
- [androiddrew/g2p-en-decoder-onnx](https://huggingface.co/androiddrew/g2p-en-decoder-onnx)

Every file can be regenerated from its pinned PyTorch source with
[`tools/export`](tools/export), which also checks that the result matches the
committed bytes and naptrap's outputs.

## The graphs

| Graph | Inputs | Outputs |
|---|---|---|
| `google/model.onnx` | `waveform` float32 `[N]` | `embeddings` `[floor(N/1280), 96]` |
| `phonmatchnet/model.onnx` | `waveform` `[N]`, `google` `[G, 96]`, `text` `[P, 256]` | `logit` `[1]` |
| `g2p/encoder.onnx` | `graphemes` int64 `[L]` | `hidden` `[1, 256]`, `separator` `[1, 256]` |
| `g2p/decoder.onnx` | `token` int64 `[1]`, `hidden` `[1, 256]` | `next_hidden` `[1, 256]`, `logits` `[1, 74]` |

The spectrogram frontends are part of the graphs, so callers pass raw samples.
Each FFT is expressed as a strided convolution with a fixed DFT basis, which
reproduces naptrap's `torch.stft`/`torch.fft` features to about 1e-5. The g2p
greedy decode runs in Go: one decoder step per call, at most 20 per word.

## Overriding the embedded models

`Config.ModelDir` replaces all of them. It must contain `phonmatchnet/`,
`google/` and `g2p/` laid out as above, which is what downloading the three
Hugging Face repositories into those directory names gives you:

```bash
hf download androiddrew/naptrap-phonmatchnet-en-onnx --revision v1 --local-dir models/phonmatchnet
hf download androiddrew/google-speech-embedding-onnx --revision v1 --local-dir models/google
hf download androiddrew/g2p-en-decoder-onnx --revision v1 --local-dir models/g2p
```

`New` checks every graph against the SHA-256 in its `config.json`, and the
two encoders against the pins in `phonmatchnet/config.json`, so a mismatched
set fails before any session is created. Tensor names, types and shapes are
validated when each graph loads.

## What is not included

**ONNX Runtime.** Install it yourself; select it with `ORTLibrary` or own the
environment in your application (see the README). The current Go binding
requests C API 22; native 1.22.0 and 1.22.2 are tested baselines, not
exact-version restrictions.
