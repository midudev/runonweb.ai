"""Assemble the browser bundle: ONNX backbone + pointer head (head.bin) + kev.json + tokenizer.

usage: python assemble.py <kev_dir> <onnx_dir> <base_dir> <out_dir> <name> <kev_revision>
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

kev, onnx_dir, base, out, name, kev_rev = sys.argv[1:7]
kev, onnx_dir, base, out = map(Path, (kev, onnx_dir, base, out))
(out / "onnx").mkdir(parents=True, exist_ok=True)

meta = torch.load(kev / "head.pt", map_location="cpu", weights_only=False)
h = meta["head"]
parts = [h["q.weight"], h["q.bias"], h["k.weight"], h["k.bias"]]
blob = np.concatenate([p.float().numpy().ravel() for p in parts]).astype("<f4")
(out / "head.bin").write_bytes(blob.tobytes())

tok = AutoTokenizer.from_pretrained(base)
special = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
ids = [tok.convert_tokens_to_ids(t) for t in special]
cfg_text = json.loads((base / "config.json").read_text())["text_config"]
layer_types = cfg_text["layer_types"]

config = {
    "name": name,
    "format": "kev-onnx/1",
    "source": {"checkpoint": f"jaredpalmer/{name}", "revision": kev_rev, "base": meta["base"], "base_revision": meta["base_revision"]},
    "temperature": meta["temperature"],
    "hidden_size": cfg_text["hidden_size"],
    "head_dim": meta["head_dim"],
    "head_file": "head.bin",
    "head_layout": ["q.weight", "q.bias", "k.weight", "k.bias"],
    "onnx": {"model": "onnx/model.onnx", "data": "onnx/model.onnx.data"},
    "tokens": dict(zip(["state", "question", "option", "option_end", "decide"], ids)),
    "max_state_tokens": 8192,
    "max_row_tokens": 8192,
    "cache": {
        "num_hidden_layers": cfg_text["num_hidden_layers"],
        "layer_types": layer_types,
        "num_key_value_heads": cfg_text["num_key_value_heads"],
        "attention_head_dim": cfg_text["head_dim"],
        "conv_dim": 2 * cfg_text["linear_num_key_heads"] * cfg_text["linear_key_head_dim"] + cfg_text["linear_num_value_heads"] * cfg_text["linear_value_head_dim"],
        "conv_kernel": cfg_text["linear_conv_kernel_dim"] - 1,
        "linear_heads": cfg_text["linear_num_value_heads"],
        "linear_key_dim": cfg_text["linear_key_head_dim"],
        "linear_value_dim": cfg_text["linear_value_head_dim"],
    },
}
(out / "kev.json").write_text(json.dumps(config, indent=2) + "\n")

shutil.copy(onnx_dir / "model.onnx", out / "onnx" / "model.onnx")
shutil.copy(onnx_dir / "model.onnx.data", out / "onnx" / "model.onnx.data")
for f in ["tokenizer.json", "tokenizer_config.json", "LICENSE"]:
    shutil.copy(base / f, out / f)
card = (Path(__file__).parent / "MODEL_CARD.md").read_text()
(out / "README.md").write_text(card.replace("{name}", name).replace("{base}", meta["base"]).replace("{revision}", meta["base_revision"]).replace("{kev_revision}", kev_rev))
print(json.dumps(config["tokens"]), config["temperature"], blob.nbytes)
