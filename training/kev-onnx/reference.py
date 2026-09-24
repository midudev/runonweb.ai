"""Reference probabilities from Kev's own PyTorch fp32 path (the path the model cards report), plus the encoded token
rows, so the browser port can be checked token for token and probability for probability.

usage: KEV_REPO=/path/to/kev python reference.py <kev_dir> <requests.json> <out.json>
"""
import json, os, sys
import torch
sys.path.insert(0, os.environ["KEV_REPO"])
from kev.checkpoint import Checkpoint, LoadOptions
from kev.api import SystemOneRequest, to_record, to_answers
from kev.model import rows_of

run, reqs_path, out_path = sys.argv[1:4]
ck = Checkpoint(run)
tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, backend="torch"))
print("temperature", model.head.temperature)
out = []
for r in json.load(open(reqs_path)):
    req = SystemOneRequest(**r)
    rec, meta = to_record(req)
    enc = model.encode(tok, rec, max_state=8192, max_branch=16384)
    S, _, rows = rows_of(enc)
    probs = [p.tolist() for p in model.probs(enc)]
    out.append({"record": rec, "state_ids": S, "rows": rows, "probs": probs, "answers": to_answers(probs, meta)})
    print(json.dumps(out[-1]["answers"]))
json.dump({"temperature": model.head.temperature, "results": out}, open(out_path, "w"), ensure_ascii=False)
