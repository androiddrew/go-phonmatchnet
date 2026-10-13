"""Build go-phonmatchnet's test fixtures.

  synth      Synthesize the Kokoro test clips into testdata/clips/ through an
             OpenAI-compatible speech server, resampled to 16 kHz with ffmpeg.
             TTS output is not guaranteed to be bit-stable, so the clips are
             committed and only regenerated deliberately.
  reference  Score the committed clips with naptrap's PyTorch models and write
             testdata/reference.json, the values the Go tests compare against.

Usage:
  uv run --frozen python fixtures.py synth --server http://host:8880
  uv run --frozen python fixtures.py reference
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import naptrap
import torch
from naptrap import load_wav
from naptrap.models.phonmatchnet import PhonMatchNet
from naptrap.text import normalize_keyword

from export import REPO, SOURCES, fetch

TESTDATA = REPO / "testdata"
VOICES = ["af_heart", "am_michael", "bf_emma"]
PHRASES = ["hey computer", "said mister", "okay naptrap", "hey commuter", "turn on the lights"]
# Silence added before and after each synthesized phrase.
MARGIN = 3200
# Keywords whose full text features are recorded.
FEATURE_KEYWORDS = ["hey computer", "said mister", "okay naptrap", "don't stop", "Café au lait", "a",
                    "supercalifragilisticexpialidocious", "the quick brown fox jumps over the lazy dog"]
NORMALIZATION = ["Hey, Computer!", "  okay   NAPTRAP  ", "CAFÉ olé", "rock’n’roll", "don't", "naïve-ish",
                 "x2", "route 66", "!!!", "", "日本語"]
# Embeddings are recorded for these clips; the Go tests compare them directly.
EMBEDDING_CLIPS = ["said-mister.wav", "clips/af_heart-hey-computer.wav"]


def slug(text):
    return text.replace(" ", "-")


def synth(server):
    out = TESTDATA / "clips"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for voice in VOICES:
            for phrase in PHRASES:
                request = urllib.request.Request(
                    server.rstrip("/") + "/v1/audio/speech",
                    data=json.dumps({"model": "kokoro-v1.0", "input": phrase, "voice": voice,
                                     "response_format": "wav"}).encode(),
                    headers={"Content-Type": "application/json"})
                raw = Path(tmp) / "raw.wav"
                with urllib.request.urlopen(request) as response:
                    raw.write_bytes(response.read())
                target = out / f"{voice}-{slug(phrase)}.wav"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", "16000",
                                "-sample_fmt", "s16", "-af", f"adelay={MARGIN * 1000 // 16000}ms,apad=pad_len={MARGIN}",
                                "-bitexact", str(target)], check=True)
                with wave.open(str(target)) as clip:
                    frames = clip.getnframes()
                if not naptrap.features.MIN_SAMPLES <= frames <= naptrap.features.MAX_SAMPLES:
                    raise SystemExit(f"{target.name}: {frames} samples is outside 0.5-2 s")
                print(target.relative_to(REPO), frames)


def rounded(tensor):
    return [[float(f"{v:.7g}") for v in row] for row in tensor.tolist()]


def reference():
    roots = fetch(Path(__file__).resolve().parent / ".cache")
    pm = roots["naptrap-phonmatchnet-en"]
    shutil.copyfile(pm / "examples/said-mister.wav", TESTDATA / "said-mister.wav")
    model = PhonMatchNet(pm, roots["google-speech-embedding-pytorch"], roots["g2p-en-decoder-pytorch"])
    clips = ["said-mister.wav"] + sorted(str(p.relative_to(TESTDATA)) for p in (TESTDATA / "clips").glob("*.wav"))
    keywords = PHRASES
    normalization = []
    for text in NORMALIZATION:
        try:
            normalization.append({"input": text, "normalized": normalize_keyword(text)})
        except ValueError as error:
            normalization.append({"input": text, "error": str(error)})
    features = []
    for text in FEATURE_KEYWORDS:
        keyword = model.keyword(text)
        features.append({"keyword": text, "normalized": keyword.text, "features": rounded(keyword.features)})
    embeddings, detections = [], []
    for name in clips:
        clip = load_wav(TESTDATA / name)
        if name in EMBEDDING_CLIPS:
            embeddings.append({"clip": name, "embeddings": rounded(model.google_encoder(clip))})
        for text in keywords:
            prediction = model.detect(clip, text)
            detections.append({"clip": name, "keyword": text, "logit": prediction.details["keyword_logit"],
                               "score": prediction.score, "accepted": prediction.accepted})
    result = {
        "generator": {"naptrap": naptrap.__version__, "torch": torch.__version__,
                      "models": {repo: commit for repo, (commit, _) in SOURCES.items()}},
        "thresholds": model.config["thresholds"],
        "normalization": normalization,
        "keywords": features,
        "embeddings": embeddings,
        "detections": detections,
    }
    (TESTDATA / "reference.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(detections)} detections, {len(features)} keywords, {len(embeddings)} embeddings")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synth").add_argument("--server", required=True)
    commands.add_parser("reference")
    args = parser.parse_args()
    if args.command == "synth":
        synth(args.server)
    else:
        reference()


if __name__ == "__main__":
    main()
