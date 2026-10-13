---
license: apache-2.0
library_name: onnx
language:
  - en
pipeline_tag: feature-extraction
tags:
  - grapheme-to-phoneme
  - g2p
  - keyword-spotting
  - wake-word
  - onnx
base_model: androiddrew/g2p-en-decoder-pytorch
---

# g2p_en decoder: keyword text features, ONNX

The [g2p_en](https://github.com/Kyubyong/g2p) grapheme-to-phoneme model
(Kyubyong Park & Jongseok Kim, Apache-2.0, release 2.1.0) as two ONNX graphs,
converted from
[`androiddrew/g2p-en-decoder-pytorch`](https://huggingface.co/androiddrew/g2p-en-decoder-pytorch).
The values are unchanged.

It is **not** used to output phoneme strings. The keyword model
[`androiddrew/naptrap-phonmatchnet-en-onnx`](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en-onnx)
was trained on the decoder's **hidden states**. It is embedded in
[go-phonmatchnet](https://github.com/androiddrew/go-phonmatchnet).

> Not affiliated with or endorsed by the g2p_en authors. See `LICENSE` and `NOTICE`.

## Files

| Graph | Inputs | Outputs |
|---|---|---|
| `encoder.onnx` | `graphemes` int64 `[L]`: a word's grapheme IDs, then `</s>` (2) | `hidden` `[1, 256]`; `separator` `[1, 256]`, the row placed between words |
| `decoder.onnx` | `token` int64 `[1]`, `hidden` `[1, 256]` | `next_hidden` `[1, 256]`; `logits` `[1, 74]` |

`config.json` lists the grapheme vocabulary, start/end phoneme IDs, the
20-step limit and the SHA-256 of both graphs. `conversion/` holds the export
script, its locked environment and README.

## Keyword features

1. Normalize the keyword: NFKD, drop combining marks, lowercase, and keep
   words matching `[a-z]+(?:'[a-z]+)*`. Numbers must be spelled out.
2. Encode each word: grapheme IDs from `config.json` (`<unk>` for `'`), then `</s>`.
3. Decode greedily from token 2 (`<s>`) for up to 20 steps; each step's
   `next_hidden` is one feature row, until the argmax token is 3 (`</s>`),
   whose state is not kept.
4. Put the `separator` row between words.

```python
import numpy as np
import onnxruntime as ort

encoder = ort.InferenceSession("encoder.onnx")
decoder = ort.InferenceSession("decoder.onnx")
graphemes = {g: i for i, g in enumerate(config["graphemes"])}

def features(normalized):
    rows = []
    for i, word in enumerate(normalized.split()):
        ids = np.array([graphemes.get(c, 1) for c in word] + [2], dtype=np.int64)
        hidden, separator = encoder.run(None, {"graphemes": ids})
        if i:
            rows.append(separator)
        token = 2
        for _ in range(20):
            hidden, logits = decoder.run(None, {"token": np.array([token]), "hidden": hidden})
            token = int(logits.argmax())
            if token == 3:
                break
            rows.append(hidden)
    return np.concatenate(rows)   # float32 [positions, 256]
```

## Conversion and accuracy

`conversion/export.py` loads the PyTorch release at commit
`fc55559c25a12fce6e0cf9062729e7f2a9fc1ddc` with naptrap 0.1.0. The grapheme
GRU has PyTorch's gate layout, so the encoder is an ONNX `GRU` operator; the
decoder is one GRU-cell step plus the output projection. Over eight keywords,
including apostrophes, accents, one letter, a 34-letter word and a nine-word
sentence, every feature length matched naptrap and the maximum difference was
5.5e-6. With the locked environment it reproduces both graphs byte for byte.

## Limitations

The model was trained on CMUdict pronunciations of American English words.
Unusual names, brand names and non-English words get its best-guess
pronunciation, which may not match how people say them. The dictionary-lookup,
homograph and number-expansion steps of the g2p_en package are not used.
