# Export tooling

`export.py` rebuilds every model go-phonmatchnet embeds (`models/`) from
naptrap's PyTorch releases on Hugging Face, pinned by commit and SHA-256:

| Source | Commit | Output |
|---|---|---|
| `androiddrew/naptrap-phonmatchnet-en` v1 | `e8f6c45bff04d7368cbe14979392b7d52dc88ba8` | `phonmatchnet/model.onnx`, `config.json` |
| `androiddrew/google-speech-embedding-pytorch` v1 | `ad7a3dd93b02f83797a040857425d6343d98680a` | `google/model.onnx`, `config.json` |
| `androiddrew/g2p-en-decoder-pytorch` v1 | `fc55559c25a12fce6e0cf9062729e7f2a9fc1ddc` | `g2p/encoder.onnx`, `decoder.onnx`, `config.json` |

The PyTorch modules come from naptrap 0.1.0, so the graphs are traced from the
same code that defines the reference outputs. The trained values are copied
unchanged. Two things are re-expressed for ONNX:

- **Spectrogram frontends.** `torch.stft` and `torch.fft.rfft` do not export
  reliably, so each FFT is a strided `Conv` with a fixed cos/sin DFT basis
  that has the original window folded in. Reflect and zero padding, power or
  magnitude, the mel matrix and the log are ordinary graph operators. The
  graphs take raw samples.
- **The g2p greedy decode.** The encoder is an ONNX `GRU`; the decoder is one
  step, which the caller runs until the end token. Data-dependent loops stay
  out of the graph.

After exporting, the script runs every graph with ONNX Runtime against
naptrap's PyTorch modules (17 clips of speech and noise from 8,000 to
32,000 samples long, eight keywords) and fails outside 1e-3 on embeddings and
logits or 1e-5 on text features. The measured maxima are about 4e-5, 1e-5 and 5.5e-6.

```bash
cd tools/export
uv run --frozen python export.py --output out --check
```

`--check` compares each output with the committed file and fails on any
difference. With the locked environment in `uv.lock` (Python 3.12, torch
2.6.0+cpu, onnx 1.17.0, onnxruntime 1.22.1, naptrap 0.1.0) on Linux/amd64,
all seven outputs reproduce byte for byte. Copy `out/` over `models/` only
when intentionally changing the embedded models.

## Hugging Face repositories

```bash
uv run --frozen python hf.py --output hf-out
```

assembles `hf-out/naptrap-phonmatchnet-en-onnx`, `google-speech-embedding-onnx`
and `g2p-en-decoder-onnx` from the committed `models/`, the cards and NOTICE
files in `hf/`, the Apache-2.0 LICENSE and this directory as `conversion/`.

## Test fixtures

`fixtures.py` builds `testdata/`:

```bash
uv run --frozen python fixtures.py synth --server http://host:8880   # Kokoro clips
uv run --frozen python fixtures.py reference                         # naptrap outputs
```

`synth` requests each phrase from an OpenAI-compatible Kokoro speech server
(`kokoro-v1.0`; voices `af_heart`, `am_michael`, `bf_emma`), converts it to
16 kHz 16-bit mono with ffmpeg and adds 0.2 s of silence on each side. TTS
output is not guaranteed to be bit-stable, so the clips are committed.
`reference` scores the committed clips with naptrap's PyTorch models and
writes `testdata/reference.json`.
