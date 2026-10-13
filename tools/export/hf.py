"""Assemble the three Hugging Face -onnx repositories from the committed models/.

Each repository gets its graphs and config.json, its model card and NOTICE
from hf/<repo>/, the Apache-2.0 LICENSE, and conversion/ with this export
tool. Run export.py --check first so models/ is known to be reproducible.

Usage: uv run --frozen python hf.py --output hf-out
"""

import argparse
import shutil
from pathlib import Path

from export import REPO

HERE = Path(__file__).resolve().parent
REPOS = {
    "naptrap-phonmatchnet-en-onnx": ("phonmatchnet", ["model.onnx", "config.json"]),
    "google-speech-embedding-onnx": ("google", ["model.onnx", "config.json"]),
    "g2p-en-decoder-onnx": ("g2p", ["encoder.onnx", "decoder.onnx", "config.json"]),
}
CONVERSION = ["export.py", "pyproject.toml", "uv.lock", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for repo, (component, files) in REPOS.items():
        target = args.output / repo
        if target.exists():
            shutil.rmtree(target)
        (target / "conversion").mkdir(parents=True)
        for name in files:
            shutil.copyfile(REPO / "models" / component / name, target / name)
        for name in ["README.md", "NOTICE"]:
            shutil.copyfile(HERE / "hf" / repo / name, target / name)
        shutil.copyfile(REPO / "LICENSE", target / "LICENSE")
        for name in CONVERSION:
            shutil.copyfile(HERE / name, target / "conversion" / name)
        if component == "phonmatchnet":
            (target / "licenses").mkdir()
            shutil.copyfile(REPO / "third_party/licenses/PhonMatchNet-BSD-3-Clause.txt",
                            target / "licenses/PhonMatchNet-BSD-3-Clause.txt")
        print(target)


if __name__ == "__main__":
    main()
