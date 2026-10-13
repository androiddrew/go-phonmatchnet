# Changelog

## [Unreleased]

### Added

- Text-defined keyword detection with naptrap's English PhonMatchNet through ONNX Runtime.
- Embedded ONNX graphs (PhonMatchNet, Google speech embedding, g2p_en encoder and decoder step)
  with SHA-256 and companion-pin checks; `ModelDir` optionally overrides them.
- Spectrogram frontends inside the graphs; the g2p greedy decode runs in Go.
- naptrap-compatible keyword normalization, calibrated FAR operating points, raw logits and scores.
- Process-lifetime ORT coordination through ortenv, with caller-owned mode.
- Validated tensor metadata, stage timings, cancellation checks and explicit errors.
- `tools/export` to regenerate the models and Hugging Face repositories from pinned PyTorch
  sources, and parity fixtures recorded from naptrap 0.1.0.

### Known limits

- One 0.5–2 s clip per call; no streaming listener, voice activity detection or endpointing.
- English only; calibrated thresholds do not transfer reliably to other audio.
- Linux/amd64 is the tested platform; other architectures require their own native validation.
