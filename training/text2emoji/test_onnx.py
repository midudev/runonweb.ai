"""Greedy decode with onnxruntime only (no torch) to verify the exported files."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
suffix = sys.argv[2] if len(sys.argv) > 2 else "_quantized"
out = ROOT / "out" / name
tok = Tokenizer.from_file(str(out / "tokenizer.json"))
cfg = json.loads((out / "config.json").read_text())
PAD, EOS = cfg["pad_token_id"], cfg["eos_token_id"]
enc = ort.InferenceSession(str(out / "onnx" / f"encoder_model{suffix}.onnx"))
dec = ort.InferenceSession(str(out / "onnx" / f"decoder_model_merged{suffix}.onnx"))
dec_inputs = [i.name for i in dec.get_inputs()]
n_layers, n_heads, d_kv = cfg["num_decoder_layers"], cfg["num_heads"], cfg["d_kv"]


def translate(text: str, max_new=20) -> str:
    ids = np.array([tok.encode(text).ids + [EOS]], dtype=np.int64)
    att = np.ones_like(ids)
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": att})[0]
    past = {}
    for l in range(n_layers):
        for kind in ("decoder", "encoder"):
            for kv in ("key", "value"):
                past[f"past_key_values.{l}.{kind}.{kv}"] = np.zeros((1, n_heads, 0, d_kv), dtype=np.float32)
    cur = np.array([[PAD]], dtype=np.int64)
    result = []
    for step in range(max_new):
        feed = {"input_ids": cur, "encoder_attention_mask": att, "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([step > 0])}
        feed.update({k: v for k, v in past.items() if k in dec_inputs})
        outs = dec.run(None, feed)
        names = [o.name for o in dec.get_outputs()]
        logits = outs[0]
        logits[0, -1, result] = -1e9
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        result.append(nxt)
        cur = np.array([[nxt]], dtype=np.int64)
        for n, v in zip(names, outs):
            # encoder (cross-attention) cache is only produced on step 0; keep it afterwards
            if n.startswith("present.") and (step == 0 or ".decoder." in n):
                past[n.replace("present.", "past_key_values.")] = v
    return tok.decode(result).replace(" ", "")


tests = ["I love pizza and my dog", "Happy birthday! Let's party tonight",
         "It is raining and I forgot my umbrella", "Going to the gym then a coffee with friends",
         "I am so tired of work, need a vacation on the beach", "Learning to code in JavaScript is fun"]
t0 = time.time()
for t in tests:
    print(f"{t!r:60s} -> {translate(t)}")
print(f"{(time.time()-t0)/len(tests)*1000:.0f} ms/sentence")
