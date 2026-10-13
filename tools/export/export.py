"""Convert naptrap's PyTorch PhonMatchNet models to the ONNX graphs go-phonmatchnet embeds.

Sources are naptrap's three Hugging Face repositories at their v1 commits,
pinned by commit and SHA-256. The PyTorch modules come from naptrap 0.1.0, so
every graph is traced from the same code that produced the reference outputs.

Outputs, in the repository's models/ layout:

  phonmatchnet/model.onnx   waveform [N], google [G,96], text [P,256] -> logit [1]
  google/model.onnx         waveform [N] -> embeddings [floor(N/1280), 96]
  g2p/encoder.onnx          graphemes [L] -> hidden [1,256], separator [1,256]
  g2p/decoder.onnx          token [1], hidden [1,256] -> next_hidden [1,256], logits [1,74]
  */config.json             metadata the Go engine reads

The STFT frontends are inside the graphs. Each FFT is written as a strided
convolution with a fixed DFT basis (the window folded in), which ONNX exports
exactly; torch.stft and torch.fft do not export reliably. The g2p greedy
decode stays a loop in Go around the single-step decoder graph.

Usage: uv run --frozen python export.py --output out [--check]
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
from torch.nn import functional as F

from naptrap.models.phonmatchnet import PhonMatchNet
from naptrap.text import GRAPHEMES

REPO = Path(__file__).resolve().parents[2]
OPSET = 17
HF = "https://huggingface.co/androiddrew/{repo}/resolve/{commit}/{file}"
SOURCES = {
    "naptrap-phonmatchnet-en": ("e8f6c45bff04d7368cbe14979392b7d52dc88ba8", {
        "config.json": "33b0a0decd5b563ebc67cd248008f8fcfcca8a2e71c03b79da9677470b7a619a",
        "model.safetensors": "6cdb3f86b6ab57a00d47d3436b16c7ffd9c4b0d049cfcb98d49a478877213226",
        "examples/said-mister.wav": "770afe00e123da53a8ef37be2b637083b2b84cdaeba59ab2d060878d349b2403",
    }),
    "google-speech-embedding-pytorch": ("ad7a3dd93b02f83797a040857425d6343d98680a", {
        "config.json": "cd241bf118d3e16c74e3850ed6c75340216af00c8810835c95b8b620c6f22b6f",
        "model.safetensors": "80a10015f4eb8ff9ba1aa10b56b1d6e27ef9f9f3419ebbae56fb43e9afce37be",
    }),
    "g2p-en-decoder-pytorch": ("fc55559c25a12fce6e0cf9062729e7f2a9fc1ddc", {
        "config.json": "d1f02ff5209f688537731abdb849c5b793538abecdb5b9c19c1a0c0f25a03b11",
        "model.safetensors": "bba37553a3414f90c7ed0cd465925aa15def67f42cc59bf43b8c633e1eb7b445",
    }),
}
OUTPUTS = [
    "phonmatchnet/model.onnx",
    "phonmatchnet/config.json",
    "google/model.onnx",
    "google/config.json",
    "g2p/encoder.onnx",
    "g2p/decoder.onnx",
    "g2p/config.json",
]
# Clip lengths cover the accepted range, both frame-length rules and every
# boundary where N % 160 or N % 1280 changes the frame count.
LENGTHS = [8000, 8001, 8159, 8160, 12800, 12801, 16000, 23999, 31840, 32000]
KEYWORDS = ["hey computer", "said mister", "okay naptrap", "don't stop", "Café au lait", "a",
            "supercalifragilisticexpialidocious", "the quick brown fox jumps over the lazy dog"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(cache):
    """Download each pinned source file once and verify it."""
    roots = {}
    for repo, (commit, files) in SOURCES.items():
        root = cache / repo / commit
        for name, digest in files.items():
            path = root / name
            if not path.exists() or sha256(path) != digest:
                path.parent.mkdir(parents=True, exist_ok=True)
                url = HF.format(repo=repo, commit=commit, file=name)
                print("download", url, file=sys.stderr)
                with urllib.request.urlopen(url) as response:
                    path.write_bytes(response.read())
            if sha256(path) != digest:
                raise SystemExit(f"{repo}/{name}: SHA-256 mismatch")
        roots[repo] = root
    return roots


def dft_basis(window, n_fft, offset=0):
    """[2*(n_fft//2+1), 1, taps] cos/-sin DFT rows with the window applied.

    The window sits at ``offset`` inside the n_fft-point frame; taps after it
    multiply zeros, so they are dropped.
    """
    bins = n_fft // 2 + 1
    padded = torch.zeros(n_fft, dtype=torch.float64)
    padded[offset:offset + window.numel()] = window.double()
    n = torch.arange(n_fft, dtype=torch.float64)
    angle = 2 * math.pi * torch.arange(bins, dtype=torch.float64)[:, None] * n[None, :] / n_fft
    basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * padded
    length = offset + window.numel()
    return basis[:, :length].float().unsqueeze(1).contiguous()


class GoogleGraph(nn.Module):
    """naptrap GoogleSpeechEmbedding with its rfft frontend as a DFT convolution."""

    def __init__(self, google):
        super().__init__()
        self.google = google
        self.register_buffer("basis", dft_basis(google.window, 512))

    def forward(self, waveform):
        padded = F.pad(waveform, (11120, 0)).reshape(1, 1, -1)
        spectrum = F.conv1d(padded, self.basis, stride=160)[0]
        real, imag = spectrum[:257], spectrum[257:]
        magnitude = torch.sqrt(real * real + imag * imag).transpose(0, 1)
        log_mel = torch.log(magnitude @ self.google.mel_matrix + .001)
        return self.google.encode_log_mel(log_mel)


class PhonMatchNetGraph(nn.Module):
    """naptrap PhonMatchNetwork with its centered STFT frontend as a DFT convolution."""

    def __init__(self, network):
        super().__init__()
        self.network = network
        frontend = network.audio.frontend
        # torch.stft centers a 400-sample window inside the 512-point frame.
        self.register_buffer("basis", dft_basis(frontend.window, 512, offset=56))
        self.register_buffer("mel_matrix", frontend.mel_matrix.clone())

    def forward(self, waveform, google, text):
        samples = waveform.shape[0]
        padded = F.pad(waveform.reshape(1, 1, -1), (256, 256), mode="reflect")
        spectrum = F.conv1d(padded, self.basis, stride=160)[0]
        real, imag = spectrum[:257], spectrum[257:]
        power = (real * real + imag * imag).transpose(0, 1)
        keyword_mel = torch.log10(power @ self.mel_matrix + 1e-7)
        keyword_mel = keyword_mel[:(samples + 159) // 160].unsqueeze(0)
        return self.network(keyword_mel, google.unsqueeze(0), text.unsqueeze(0))


class G2PEncoderGraph(nn.Module):
    """Grapheme IDs of one word (ending in </s>) -> final encoder GRU state.

    The g2p_en cell has PyTorch's GRU gate layout (r, z, n with the reset gate
    applied after the hidden projection), so nn.GRU exports it as an ONNX GRU.
    The decoder's <pad> phoneme embedding is returned as the word separator.
    """

    def __init__(self, text):
        super().__init__()
        size = text.size
        self.embedding = nn.Parameter(text.enc_emb.detach().clone())
        self.gru = nn.GRU(size, size, batch_first=True)
        cell = text.encoder
        self.gru.weight_ih_l0.data.copy_(cell.w_ih)
        self.gru.weight_hh_l0.data.copy_(cell.w_hh)
        self.gru.bias_ih_l0.data.copy_(cell.b_ih)
        self.gru.bias_hh_l0.data.copy_(cell.b_hh)
        self.register_buffer("pad", text.dec_emb[0:1].detach().clone())

    def forward(self, graphemes):
        _, hidden = self.gru(self.embedding[graphemes].unsqueeze(0))
        return hidden[0], self.pad


class G2PDecoderGraph(nn.Module):
    """One greedy decoder step: previous phoneme ID and state -> new state, logits."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def forward(self, token, hidden):
        hidden = self.text.decoder(self.text.dec_emb[token], hidden)
        return hidden, hidden @ self.text.fc_w.T + self.text.fc_b


def export(module, path, args, inputs, outputs, axes):
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                          dynamic_axes=axes, opset_version=OPSET, do_constant_folding=True, dynamo=False)
    model = onnx.load(str(path))
    # A constant output (the g2p separator) is folded into an initializer and
    # loses its requested name; restore it with an Identity node.
    for output, name in zip(model.graph.output, outputs):
        if output.name != name:
            model.graph.node.append(onnx.helper.make_node("Identity", [output.name], [name]))
            output.name = name
    onnx.checker.check_model(model, full_check=True)
    # Drop the exporter's version stamp so the bytes depend only on the graph.
    model.producer_version = ""
    onnx.save(model, str(path))


def companion_config(source_config, repo, commit, fmt, files, extra):
    config = {
        "format": fmt,
        "source": {
            "repo_id": "androiddrew/" + repo,
            "commit": commit,
            "format": source_config["format"],
            "sha256": source_config["sha256"]["model.safetensors"],
        },
        "opset": OPSET,
    }
    config.update(extra)
    config["sha256"] = {name: sha256(path) for name, path in files}
    return config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(roots, output):
    pm_root = roots["naptrap-phonmatchnet-en"]
    google_root = roots["google-speech-embedding-pytorch"]
    g2p_root = roots["g2p-en-decoder-pytorch"]
    model = PhonMatchNet(pm_root, google_root, g2p_root)
    google, text = model.google_encoder, model.text_encoder
    configs = {key: json.loads((roots[repo] / "config.json").read_text()) for key, repo in [
        ("phonmatchnet", "naptrap-phonmatchnet-en"), ("google", "google-speech-embedding-pytorch"),
        ("g2p", "g2p-en-decoder-pytorch")]}

    wave = torch.zeros(16000)
    export(GoogleGraph(google), output / "google/model.onnx", (wave,), ["waveform"], ["embeddings"],
           {"waveform": {0: "samples"}, "embeddings": {0: "frames"}})
    export(PhonMatchNetGraph(model._model), output / "phonmatchnet/model.onnx",
           (wave, torch.zeros(12, 96), torch.zeros(9, 256)), ["waveform", "google", "text"], ["logit"],
           {"waveform": {0: "samples"}, "google": {0: "google_frames"}, "text": {0: "positions"}})
    export(G2PEncoderGraph(text), output / "g2p/encoder.onnx", (torch.tensor([5, 6, 2]),),
           ["graphemes"], ["hidden", "separator"], {"graphemes": {0: "length"}})
    export(G2PDecoderGraph(text), output / "g2p/decoder.onnx", (torch.tensor([2]), torch.zeros(1, 256)),
           ["token", "hidden"], ["next_hidden", "logits"], {})

    source = configs["phonmatchnet"]
    commit = SOURCES["naptrap-phonmatchnet-en"][0]
    write_json(output / "phonmatchnet/config.json", companion_config(
        source, "naptrap-phonmatchnet-en", commit, "naptrap-phonmatchnet-onnx-v1",
        [("model.onnx", output / "phonmatchnet/model.onnx")], {
            "architecture": source["architecture"],
            "task": source["task"],
            "language": source["language"],
            "thresholds": source["thresholds"],
            "threshold_note": source["threshold_note"],
            "inputs": {
                "waveform": "[N] mono 16 kHz float32 in [-1, 1], 8000 <= N <= 32000",
                "google": "[floor(N / 1280), 96] output of google/model.onnx for the same waveform",
                "text": "[P, 256] g2p decoder features, 1 <= P <= 128",
            },
            "outputs": {"logit": "[1] keyword logit; score = sigmoid(logit)"},
            "frontend": source["frontend"],
            # The companion graphs this model was trained with, pinned like the
            # PyTorch release pins its encoders.
            "requires": {
                "google": {"repo_id": "androiddrew/google-speech-embedding-onnx", "revision": "v1",
                           "sha256": {"model.onnx": sha256(output / "google/model.onnx")}},
                "g2p": {"repo_id": "androiddrew/g2p-en-decoder-onnx", "revision": "v1",
                        "sha256": {"encoder.onnx": sha256(output / "g2p/encoder.onnx"),
                                   "decoder.onnx": sha256(output / "g2p/decoder.onnx")}},
            },
        }))
    source = configs["google"]
    write_json(output / "google/config.json", companion_config(
        source, "google-speech-embedding-pytorch", SOURCES["google-speech-embedding-pytorch"][0],
        "naptrap-google-speech-embedding-onnx-v1", [("model.onnx", output / "google/model.onnx")], {
            "embedding_size": source["embedding_size"],
            "inputs": {"waveform": "[N] mono 16 kHz float32, N >= 1280"},
            "outputs": {"embeddings": "[floor(N / 1280), 96]"},
            "preprocessing": source["preprocessing"],
        }))
    source = configs["g2p"]
    write_json(output / "g2p/config.json", companion_config(
        source, "g2p-en-decoder-pytorch", SOURCES["g2p-en-decoder-pytorch"][0],
        "naptrap-g2p-en-decoder-onnx-v1", [("encoder.onnx", output / "g2p/encoder.onnx"),
                                           ("decoder.onnx", output / "g2p/decoder.onnx")], {
            "hidden_size": source["hidden_size"],
            "graphemes": GRAPHEMES,
            "phonemes": source["phonemes"],
            "start_token": 2,
            "end_token": 3,
            "max_decoder_steps": source["max_decoder_steps"],
            "inputs": {
                "encoder": "graphemes [L] int64: a word's grapheme IDs followed by </s> (2)",
                "decoder": "token [1] int64 (start with 2), hidden [1,256] float32",
            },
            "outputs": {
                "encoder": "hidden [1,256]; separator [1,256], the row placed between words",
                "decoder": "next_hidden [1,256], one feature row; logits [1,74], argmax is the next token",
            },
        }))
    return model


class OnnxText:
    """The Go engine's keyword algorithm, run with ONNX Runtime."""

    def __init__(self, output):
        self.encoder = onnxruntime.InferenceSession(str(output / "g2p/encoder.onnx"))
        self.decoder = onnxruntime.InferenceSession(str(output / "g2p/decoder.onnx"))
        self.index = {g: i for i, g in enumerate(GRAPHEMES)}

    def __call__(self, normalized):
        rows = []
        for i, word in enumerate(normalized.split()):
            ids = np.array([self.index.get(c, 1) for c in word] + [2], dtype=np.int64)
            hidden, separator = self.encoder.run(None, {"graphemes": ids})
            if i:
                rows.append(separator)
            token = 2
            for _ in range(20):
                hidden, logits = self.decoder.run(None, {"token": np.array([token], dtype=np.int64),
                                                         "hidden": hidden})
                token = int(logits.argmax())
                if token == 3:
                    break
                rows.append(hidden)
        return np.concatenate(rows)


def verify(model, output, wav):
    """Compare every graph with naptrap's PyTorch modules; fail outside tolerance."""
    google_session = onnxruntime.InferenceSession(str(output / "google/model.onnx"))
    pm_session = onnxruntime.InferenceSession(str(output / "phonmatchnet/model.onnx"))
    text = OnnxText(output)
    generator = torch.Generator().manual_seed(7)
    clips = [("said-mister", wav)]
    for n in LENGTHS:
        noise = torch.randn(n, generator=generator) * 0.1
        clips.append((f"noise-{n}", noise.clamp(-1, 1)))
        if n >= wav.numel():
            clips.append((f"said-mister-pad-{n}", F.pad(wav, (0, n - wav.numel()))))
    worst = {"google": 0.0, "text": 0.0, "logit": 0.0}
    keywords = {}
    for keyword in KEYWORDS:
        expected = model.keyword(keyword)
        actual = text(expected.text)
        if actual.shape != tuple(expected.features.shape):
            raise SystemExit(f"text positions differ for {keyword!r}: {actual.shape} vs {tuple(expected.features.shape)}")
        worst["text"] = max(worst["text"], float(np.abs(actual - expected.features.numpy()).max()))
        keywords[keyword] = (expected, actual)
    for name, clip in clips:
        clip = clip.float().contiguous()
        g_ref = model.google_encoder(clip)
        (g_onnx,) = google_session.run(None, {"waveform": clip.numpy()})
        worst["google"] = max(worst["google"], float(np.abs(g_onnx - g_ref.numpy()).max()))
        mel_ref = model._frontend(clip)[0]
        for keyword, (expected, features) in keywords.items():
            ref = model.predict({"keyword_mel": mel_ref, "google": g_ref, "text": expected.features})
            (logit,) = pm_session.run(None, {"waveform": clip.numpy(), "google": g_onnx, "text": features})
            worst["logit"] = max(worst["logit"], abs(float(logit[0]) - ref.details["keyword_logit"]))
    limits = {"google": 1e-3, "text": 1e-5, "logit": 1e-3}
    for key, value in worst.items():
        print(f"max |onnx - torch| {key}: {value:.3g} (limit {limits[key]:g})")
    failed = [key for key, value in worst.items() if value > limits[key]]
    if failed:
        raise SystemExit("ONNX graphs differ from naptrap: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / ".cache")
    parser.add_argument("--check", action="store_true", help="compare outputs with the committed models/")
    args = parser.parse_args()
    torch.manual_seed(0)
    roots = fetch(args.cache)
    if args.output.exists():
        shutil.rmtree(args.output)
    model = build(roots, args.output)
    from naptrap import load_wav
    verify(model, args.output, load_wav(roots["naptrap-phonmatchnet-en"] / "examples/said-mister.wav"))
    if args.check:
        changed = [name for name in OUTPUTS
                   if not (REPO / "models" / name).exists()
                   or sha256(args.output / name) != sha256(REPO / "models" / name)]
        if changed:
            raise SystemExit("differs from models/: " + ", ".join(changed))
        print(f"all {len(OUTPUTS)} outputs match models/")


if __name__ == "__main__":
    main()
