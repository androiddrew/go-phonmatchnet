---
license: apache-2.0
library_name: onnx
language:
  - en
pipeline_tag: audio-classification
tags:
  - keyword-spotting
  - wake-word
  - user-defined-keyword
  - open-vocabulary
  - phonmatchnet
  - onnx
datasets:
  - openslr/librispeech_asr
base_model: androiddrew/naptrap-phonmatchnet-en
---

# PhonMatchNet (independently trained, English), ONNX

An ONNX conversion of
[`androiddrew/naptrap-phonmatchnet-en`](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en),
a **text-defined keyword spotter**: give it a short audio clip and a keyword
typed as text, and it returns the probability that the keyword was spoken.
The architecture follows **PhonMatchNet** (Lee & Cho, Interspeech 2023).

This is the model used by the Go library
[go-phonmatchnet](https://github.com/androiddrew/go-phonmatchnet), which embeds
it. Training, evaluation and limitations are documented on the
[PyTorch model card](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en);
the weights here are the same values.

> **Independent work.** These weights were trained independently by Drew
> Bednar using the BSD-3-Clause PyTorch training code that NCSOFT published
> for PhonMatchNet, with modifications. They are **not the paper authors'
> weights**, and this release is **not affiliated with or endorsed by NCSOFT
> or the PhonMatchNet authors**. NCSOFT's license notice is in
> `licenses/PhonMatchNet-BSD-3-Clause.txt`.

## Files

- `model.onnx`: opset 17, float32. The 40-bin log-mel frontend is inside the
  graph, so it takes raw samples.
- `config.json`: calibrated thresholds, input contracts, SHA-256 of
  `model.onnx`, and the pinned SHA-256 of the two companion graphs it was
  trained with.
- `conversion/`: the export script, its locked environment and README.

| Input | Shape | Source |
|---|---|---|
| `waveform` | float32 `[N]`, mono 16 kHz in [-1, 1], 8,000 ≤ N ≤ 32,000 | your clip (0.5–2 s) |
| `google` | float32 `[floor(N / 1280), 96]` | [google-speech-embedding-onnx](https://huggingface.co/androiddrew/google-speech-embedding-onnx) on the same clip |
| `text` | float32 `[P, 256]`, 1 ≤ P ≤ 128 | [g2p-en-decoder-onnx](https://huggingface.co/androiddrew/g2p-en-decoder-onnx) on the keyword |

Output `logit` `[1]`; the keyword-match score is `sigmoid(logit)`.

## Usage

Go, with the models embedded:

```go
engine, err := phonmatchnet.New(phonmatchnet.Config{ORTLibrary: "libonnxruntime.so"})
keyword, err := engine.Keyword(ctx, "hey computer")
result, err := engine.Detect(ctx, phonmatchnet.Request{
    Samples: samples, SampleRate: phonmatchnet.SampleRate, Keyword: keyword,
})
```

Python, with `onnxruntime`, after downloading the three `-onnx` repositories
into `phonmatchnet/`, `google/` and `g2p/`; `text` comes from the g2p
repository's keyword loop (see its card):

```python
import numpy as np
import onnxruntime as ort

google = ort.InferenceSession("google/model.onnx")
detector = ort.InferenceSession("phonmatchnet/model.onnx")

(embeddings,) = google.run(None, {"waveform": clip})       # clip: float32 [N]
(logit,) = detector.run(None, {"waveform": clip, "google": embeddings, "text": text})
score = 1 / (1 + np.exp(-logit[0]))
```

## Thresholds

| Target FAR | Score threshold |
|---|---|
| 0.01 | 0.7149 |
| 0.1 | 0.0026 |

Fitted on held-out LibriSpeech development speakers. **They do not transfer
reliably to other audio**: on phonetically confusable phrases the 1% threshold
accepted 31% of negatives. Choose your own threshold on recordings from your
deployment.

## Conversion

`conversion/export.py` loads the PyTorch release at commit
`e8f6c45bff04d7368cbe14979392b7d52dc88ba8` with naptrap 0.1.0 and traces it to
ONNX. The trained values are unchanged; the computation was re-expressed:

- the centered STFT (`torch.stft`, reflect padding) is a strided convolution
  with a fixed DFT basis, the window folded in;
- the mel matrix and window buffers are stored as graph constants;
- the inputs are one unbatched clip.

The script fails unless the graph matches naptrap within 1e-3 on logits over
17 clips of speech and noise from 0.5 to 2 s and eight keywords; the measured
maximum difference is about 1e-5. With the locked environment it reproduces `model.onnx` byte for byte.

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). Training code © NCSOFT Corporation,
BSD-3-Clause (`licenses/PhonMatchNet-BSD-3-Clause.txt`). Training data:
LibriSpeech, CC BY 4.0.

## Citation

```bibtex
@inproceedings{lee23phonmatchnet,
  title     = {PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords},
  author    = {Lee, Yong-Hyeok and Cho, Namhyun},
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {3964--3968},
  year      = {2023},
  eprint    = {2308.16511},
  archivePrefix = {arXiv}
}
```
