"""Accuracy and drift of ONNX exports on labelled Kev suite records (state pass + cached branch rows, as in the browser).

usage: KEV_REPO=/path/to/kev python parity.py <kev_dir> <tokenizer_dir> <n_per_suite> <model.onnx> [<model.onnx> ...]
The first model is the reference for drift: build an fp32 export (`-p fp32` in export.sh), which matches Kev's
PyTorch fp32 path to 2e-5.
"""
import json, os, random, sys, time
import numpy as np
import onnxruntime as ort
import torch
KEV = os.environ["KEV_REPO"]
sys.path.insert(0, KEV)
from kev.api import SystemOneRequest, to_record, question_keys
from kev.model import encode, rows_of, ContextOverflow
from transformers import AutoTokenizer

kev_dir, tok_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
models = sys.argv[4:]
SUITES = [f"{KEV}/evals/{s}/development.jsonl" for s in ("v7/decision-v7", "v4/transfer-v4", "external/scienthoon-v1", "external/semif-v1")]
tok = AutoTokenizer.from_pretrained(tok_dir)
head = torch.load(f"{kev_dir}/head.pt", map_location="cpu", weights_only=False)
Wq, bq = head["head"]["q.weight"].numpy(), head["head"]["q.bias"].numpy()
Wk, bk = head["head"]["k.weight"].numpy(), head["head"]["k.bias"].numpy()
T = head["temperature"]

items = []
rng = random.Random(0)
for path in SUITES:
    recs = [json.loads(l) for l in open(path, encoding="utf-8").read().split("\n") if l.strip()]
    rng.shuffle(recs)
    taken = 0
    for r in recs:
        if taken >= n: break
        qs = {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")} for k, q in r["questions"].items()}
        req = SystemOneRequest(state=r["state"], questions=qs)
        rec, meta = to_record(req)
        try:
            enc = encode(tok, rec, max_state=384, max_branch=1024, strict=True)
        except ContextOverflow:
            continue
        labels = []
        for (qid, q), m in zip(r["questions"].items(), meta):
            lab = q["label"]
            key = ("true" if lab else "false") if q["type"] == "noul" else str(lab)
            labels.append(m["keys"].index(key))
        S, _, rows = rows_of(enc)
        items.append({"suite": path.split("/")[-2], "S": S, "rows": rows, "labels": labels})
        taken += 1
print(f"{len(items)} records, {sum(len(i['rows']) for i in items)} questions")


def runner(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    past_names = {n.split(".", 1)[1] if n.startswith("past.") else n.removeprefix("past_key_values."): n for n in inputs if n.startswith("past")}
    kv = next(i for n, i in inputs.items() if n.endswith(".key"))
    fdt = np.float16 if kv.type == "tensor(float16)" else np.float32
    outs = [o.name for o in sess.get_outputs()]

    def empty():
        feed = {}
        for name, i in inputs.items():
            if not name.startswith("past"): continue
            shape = [1 if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape]
            if name.endswith(".key") or name.endswith(".value"): shape[-1] = 256
            feed[name] = np.zeros(shape, dtype=fdt)
        return feed

    def run(ids, start, past):
        feed = dict(past)
        feed["input_ids"] = np.array([ids], dtype=np.int64)
        feed["attention_mask"] = np.ones((1, start + len(ids)), dtype=np.int64)
        pos = np.arange(start, start + len(ids), dtype=np.int64)[None]
        if "position_ids" in inputs: feed["position_ids"] = np.stack([pos, pos, pos])
        res = dict(zip(outs, sess.run(None, feed)))
        present = {past_names[k.split(".", 1)[1]]: v for k, v in res.items() if k.startswith("present")}
        return res["hidden_states"][0].astype(np.float32), present

    def probs(item):
        _, past = run(item["S"], 0, empty())
        out = []
        for row in item["rows"]:
            h, _ = run(row["ids"], len(item["S"]), past)
            q = Wq @ h[row["decide"]] + bq
            k = np.stack([Wk @ h[o] + bk for o in row["opts"]])
            z = (k @ q) / 16 / T
            z = np.exp(z - z.max()); out.append(z / z.sum())
        return out
    return probs


ref = None
for path in models:
    probs = runner(path)
    t0 = time.time()
    allp = [probs(it) for it in items]
    dt = time.time() - t0
    flat = [(p, lab, it["suite"]) for it, ps in zip(items, allp) for p, lab in zip(ps, it["labels"])]
    acc = np.mean([int(np.argmax(p) == lab) for p, lab, _ in flat])
    by = {}
    for p, lab, s in flat: by.setdefault(s, []).append(int(np.argmax(p) == lab))
    brier = np.mean([np.sum((p - np.eye(len(p))[lab]) ** 2) for p, lab, _ in flat])
    line = f"{path:40s} acc {acc:.3f} brier {brier:.3f} " + " ".join(f"{s}={np.mean(v):.3f}" for s, v in by.items()) + f"  {dt:.0f}s"
    if ref is None:
        ref = [p for p, _, _ in flat]
    else:
        d = [float(np.abs(a - b).max()) for a, b in zip(ref, [p for p, _, _ in flat])]
        flips = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(ref, [p for p, _, _ in flat]))
        conf = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(ref, [p for p, _, _ in flat]) if a.max() - np.sort(a)[-2] > 0.2)
        line += f"  drift max {max(d):.3f} mean {np.mean(d):.4f} flips {flips}/{len(d)} (margin>0.2: {conf})"
    print(line, flush=True)
