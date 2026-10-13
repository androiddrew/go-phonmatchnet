---
license: apache-2.0
library_name: onnx
pipeline_tag: feature-extraction
tags:
  - audio
  - speech
  - keyword-spotting
  - wake-word
  - speech-embedding
  - onnx
base_model: androiddrew/google-speech-embedding-pytorch
---

# Google speech_embedding/1, ONNX

An ONNX conversion of Google's
[`speech_embedding/1`](https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1)
(Lin et al., 2020), made from the PyTorch port
[`androiddrew/google-speech-embedding-pytorch`](https://huggingface.co/androiddrew/google-speech-embedding-pytorch).
It maps 16 kHz audio to one **96-dimensional embedding every 80 ms**. Those
embeddings describe speech sounds; the model does not produce transcripts.

It is the frozen audio feature extractor of
[`androiddrew/naptrap-phonmatchnet-en-onnx`](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en-onnx)
and is embedded in [go-phonmatchnet](https://github.com/androiddrew/go-phonmatchnet).

> This conversion is not affiliated with or endorsed by Google. The original
> model is published by Google under the Apache License 2.0. This conversion
> keeps that license; see `LICENSE` and `NOTICE`.

## Files

- `model.onnx`: opset 17, float32. Input `waveform` float32 `[N]`; output
  `embeddings` `[floor(N / 1280), 96]`. The padding and log-mel frontend are
  inside the graph.
- `config.json`: the preprocessing contract and SHA-256 of `model.onnx`.
- `conversion/`: the export script, its locked environment and README.

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
(embeddings,) = session.run(None, {"waveform": clip})   # clip: mono 16 kHz float32 [N]
```

## Preprocessing contract (inside the graph)

| Step | Value |
|---|---|
| Input | mono 16 kHz float32 in [-1, 1]; the keyword model uses 0.5–2 s clips |
| Padding | 11,120 leading zero samples, no trimming |
| STFT | 400-sample Hann window, hop 160, 512-point FFT, no centering, magnitude |
| Mel | 32 bins, natural `log(mel + 0.001)` |
| Output | `floor(samples / 1280)` frames × 96 |

The keyword models were trained on features computed this way, one clip at a
time. Slicing features computed over a longer recording gives different values.

## Conversion and accuracy

`conversion/export.py` loads the PyTorch port at commit
`ad7a3dd93b02f83797a040857425d6343d98680a` with naptrap 0.1.0 and traces it.
All trained parameters, BatchNorm statistics, the window and the mel matrix
are unchanged. The FFT (`torch.fft.rfft` of 400-sample frames padded to 512)
is a strided convolution with a fixed DFT basis, the window folded in.

The script fails unless the graph matches the PyTorch port within 1e-3 over
17 clips of speech and noise from 8,000 to 32,000 samples long. The measured
maximum difference is 4e-5, smaller than the PyTorch port's own difference
from the TensorFlow original (7.2e-5 on real speech). With the locked environment it reproduces
`model.onnx` byte for byte.

## Training data and limitations

Google reports training the original on roughly 200 million two-second YouTube
clips covering about 5,000 keywords (Lin et al., 2020). No clip-level
inventory is available. This repository adds no training. Behaviour on
non-English speech, far-field audio or unusual microphones is unknown.

## Citation

```bibtex
@article{lin2020training,
  title   = {Training Keyword Spotters with Limited and Synthesized Speech Data},
  author  = {Lin, James and Kilgour, Kevin and Roblek, Dominik and Sharifi, Matthew},
  journal = {arXiv preprint arXiv:2002.01322},
  year    = {2020}
}
```
