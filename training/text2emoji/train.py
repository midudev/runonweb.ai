"""Train a tiny T5 (from scratch) that maps English text -> emoji sequence."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--name", default="tiny")
p.add_argument("--d_model", type=int, default=128)
p.add_argument("--d_ff", type=int, default=512)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max_src", type=int, default=64)
p.add_argument("--max_tgt", type=int, default=20)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()

torch.manual_seed(0)
random.seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")
tok.model_input_names = ["input_ids", "attention_mask"]
PAD, EOS = tok.pad_token_id, tok.eos_token_id


def load(name):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{name}.jsonl")]
    return rows[: args.limit] if args.limit else rows


class DS(Dataset):
    def __init__(self, rows):
        src = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
        tgt = tok([r["emoji"] for r in rows], add_special_tokens=False)["input_ids"]
        self.src = [s[: args.max_src - 1] + [EOS] for s in src]
        self.tgt = [t[: args.max_tgt - 1] + [EOS] for t in tgt]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


def collate(batch):
    src, tgt = zip(*batch)
    ls, lt = max(map(len, src)), max(map(len, tgt))
    ids = torch.full((len(src), ls), PAD)
    att = torch.zeros((len(src), ls), dtype=torch.long)
    lab = torch.full((len(src), lt), -100)
    for i, (s, t) in enumerate(zip(src, tgt)):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        lab[i, : len(t)] = torch.tensor(t)
    return ids, att, lab


train_rows, val_rows = load("train"), load("val")
train_dl = DataLoader(DS(train_rows), batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=True)
val_dl = DataLoader(DS(val_rows), batch_size=256, collate_fn=collate)

cfg = T5Config(
    vocab_size=len(tok),
    d_model=args.d_model,
    d_kv=args.d_model // args.heads,
    d_ff=args.d_ff,
    num_layers=args.layers,
    num_decoder_layers=args.layers,
    num_heads=args.heads,
    dropout_rate=0.1,
    feed_forward_proj="relu",
    tie_word_embeddings=True,
    pad_token_id=PAD,
    eos_token_id=EOS,
    decoder_start_token_id=PAD,
)
model = T5ForConditionalGeneration(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params={n_params/1e6:.2f}M device={device} train={len(train_rows)} steps/epoch={len(train_dl)}")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
total = int(len(train_dl) * args.epochs)
warm = min(1000, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
)


def emoji_f1(pred: str, gold: str) -> float:
    a, b = set(pred.split()), set(gold.split())
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    pr, rc = inter / len(a), inter / len(b)
    return 2 * pr * rc / (pr + rc)


@torch.no_grad()
def evaluate(n_gen=500):
    model.eval()
    losses = []
    for ids, att, lab in val_dl:
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        losses.append(out.loss.item())
    rows = val_rows[:n_gen]
    f1 = 0.0
    for i in range(0, len(rows), 64):
        chunk = rows[i : i + 64]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True, truncation=True, max_length=args.max_src)
        gen = model.generate(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), max_new_tokens=args.max_tgt, num_beams=1)
        preds = tok.batch_decode(gen, skip_special_tokens=True)
        f1 += sum(emoji_f1(p, r["emoji"]) for p, r in zip(preds, chunk))
    model.train()
    return sum(losses) / len(losses), f1 / len(rows), preds[:4], [r["emoji"] for r in chunk[:4]], [r["text"] for r in chunk[:4]]


out_dir = ROOT / "models" / args.name
out_dir.mkdir(parents=True, exist_ok=True)
step, t0, best = 0, time.time(), 1e9
model.train()
done = False
while not done:
    for ids, att, lab in train_dl:
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{total} loss {out.loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == total:
            vl, f1, preds, golds, texts = evaluate()
            print(f"== step {step} val_loss {vl:.3f} emoji_f1 {f1:.3f}", flush=True)
            for t, g, pr in zip(texts, golds, preds):
                print(f"   {t[:70]!r}\n      gold {g}\n      pred {pr}", flush=True)
            if vl < best:
                best = vl
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
                (out_dir / "generation_config.json").write_text(json.dumps({
                    "decoder_start_token_id": PAD, "eos_token_id": EOS, "pad_token_id": PAD,
                    "max_new_tokens": args.max_tgt, "num_beams": 1, "no_repeat_ngram_size": 1}, indent=2))
                print(f"   saved -> {out_dir}", flush=True)
        if step >= total:
            done = True
            break
print(f"done. best val_loss {best:.3f} params {n_params/1e6:.2f}M")
