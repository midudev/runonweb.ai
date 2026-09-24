"""Fold a Kev LoRA adapter into its Qwen3.5 base (fp32 math, fp32 output) so the ONNX builder sees plain weights.

usage: python merge.py <base_dir> <kev_dir> <out_dir>
"""
import json, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, kev, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((kev / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]

ad = safe_open(kev / "adapter_model.safetensors", "pt")
pairs = {}
for k in ad.keys():
    stem, part = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
    pairs.setdefault(stem, {})[part.split(".")[0]] = ad.get_tensor(k).float()

tensors = {}
for fn in sorted(base.glob("*.safetensors")):
    f = safe_open(fn, "pt")
    for k in f.keys():
        if k.startswith("mtp.") or k.startswith("model.visual."):
            continue  # text backbone only: no MTP head, no vision tower
        tensors[k] = f.get_tensor(k).float()

merged = 0
for stem, ab in pairs.items():
    key = f"model.language_model.{stem}.weight"
    w = tensors[key]
    delta = (ab["B"] @ ab["A"]) * scale
    assert delta.shape == w.shape, (key, delta.shape, w.shape)
    tensors[key] = w + delta
    merged += 1
print(f"merged {merged} LoRA pairs (scale {scale})")

save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
conf = json.loads((base / "config.json").read_text())
conf["text_config"]["dtype"] = "float32"
conf["text_config"]["mtp_num_hidden_layers"] = 0
(out / "config.json").write_text(json.dumps(conf, indent=2))
for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"]:
    if (base / name).exists():
        shutil.copy(base / name, out / name)
