# Attribution and asset terms

Original code is Apache-2.0 (LICENSE/NOTICE). Adapted behavior and embedded
models retain upstream attribution; retained notices are under
`third_party/licenses/`.

| Component | Source/version | Terms |
|---|---|---|
| Keyword normalization and g2p decoder-state loop | naptrap 0.1.0, https://github.com/androiddrew/naptrap | Apache-2.0 |
| PhonMatchNet model (embedded as `models/phonmatchnet/*`) | `androiddrew/naptrap-phonmatchnet-en` `e8f6c45…`; modified (ONNX conversion, mel frontend in-graph) | Apache-2.0; trained with NCSOFT's PhonMatchNet code, BSD-3-Clause, `PhonMatchNet-BSD-3-Clause.txt` |
| PhonMatchNet training data | LibriSpeech train-clean-100/360, https://www.openslr.org/12 | CC BY 4.0 |
| Google speech embedding (embedded as `models/google/*`) | `androiddrew/google-speech-embedding-pytorch` `ad7a3dd…`, from Google `speech_embedding/1`; modified (ONNX conversion) | Apache-2.0 |
| g2p_en decoder (embedded as `models/g2p/*`) | `androiddrew/g2p-en-decoder-pytorch` `fc55559…`, from g2p_en 2.1.0 `checkpoint20.npz`; modified (split into encoder/decoder-step ONNX graphs) | Apache-2.0 |
| golang.org/x/text | v0.41.0 | BSD-3-Clause, `golang-x-text.txt` |
| onnxruntime_go | v1.22.0 | External Go dependency, MIT |
| ortenv | github.com/androiddrew/ortenv v0.2.0 | MIT; Copyright (c) 2026 Drew Bednar |

Test fixtures (`testdata/`, not part of the library):

| Files | Source | Terms |
|---|---|---|
| `testdata/said-mister.wav` | LibriSpeech utterance 6188-78885-0016 (train-clean-360), samples 3040-11680, via `androiddrew/naptrap-phonmatchnet-en` | CC BY 4.0 |
| `testdata/clips/*.wav` | Synthesized with Kokoro-82M v1.0 (voices `af_heart`, `am_michael`, `bf_emma`), resampled to 16 kHz | Kokoro-82M is Apache-2.0 |
| `testdata/reference.json` | naptrap 0.1.0 outputs for the clips above | Apache-2.0 |
