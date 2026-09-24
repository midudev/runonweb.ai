"""Download Text2Emoji, clean it and train a small shared tokenizer.

Output:
  data/train.jsonl, data/val.jsonl   {"text": ..., "emoji": "🍕 ❤️ ..."} (emojis space-separated)
  tokenizer/tokenizer.json (+ tokenizer_config.json, special_tokens_map.json)
"""
import json
import random
import re
import unicodedata
from pathlib import Path

import regex  # supports \X grapheme clusters
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TOK = ROOT / "tokenizer"
VOCAB_SIZE = 8192
MAX_EMOJI = 12
random.seed(0)

EMOJI_RE = regex.compile(r"\X")


def is_emoji_cluster(g: str) -> bool:
    cp = g[0]
    cat = unicodedata.category(cp)
    if cat.startswith("So") or cat == "Sk":
        return True
    o = ord(cp)
    return 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x2300 <= o <= 0x23FF or 0x2B00 <= o <= 0x2BFF


def split_emojis(s: str) -> list[str]:
    out = []
    for g in EMOJI_RE.findall(s):
        g = g.strip()
        if not g:
            continue
        # drop stray variation selectors / keycap combos that are not emoji
        if is_emoji_cluster(g):
            out.append(g)
    return out


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    DATA.mkdir(exist_ok=True)
    TOK.mkdir(exist_ok=True)
    ds = load_dataset("KomeijiForce/Text2Emoji", split="train")
    rows = []
    seen = set()
    for r in ds:
        text = clean_text(str(r["text"] or ""))
        emojis = split_emojis(str(r["emoji"] or ""))
        if not text or not emojis or len(text) > 300:
            continue
        # dedupe consecutive repeats, cap length
        dedup = []
        for e in emojis:
            if not dedup or dedup[-1] != e:
                dedup.append(e)
        emojis = dedup[:MAX_EMOJI]
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "emoji": " ".join(emojis)})
    random.shuffle(rows)
    n_val = 5000
    val, train = rows[:n_val], rows[n_val:]
    for name, part in (("train", train), ("val", val)):
        with open(DATA / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} val={len(val)}")

    # ---- tokenizer: BPE, metaspace (T5-style), lowercase, shared for text + emoji ----
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "</s>", "<unk>"],
        min_frequency=3,
        show_progress=False,
    )

    def corpus():
        for r in train:
            yield r["text"]
            yield r["emoji"]

    tok.train_from_iterator(corpus(), trainer=trainer, length=2 * len(train))
    tok.save(str(TOK / "tokenizer.json"))
    (TOK / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 128,
        "pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
    }, indent=2))
    (TOK / "special_tokens_map.json").write_text(json.dumps({
        "pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}, indent=2))
    print("vocab", tok.get_vocab_size())
    for s in ["I love pizza and my dog!", "🍕 ❤️ 🐶", train[0]["text"], train[0]["emoji"]]:
        enc = tok.encode(s)
        print(repr(s), "->", enc.tokens, "unk" if 2 in enc.ids else "")
    lens = sorted(len(tok.encode(r["text"]).ids) for r in train[:20000])
    print("src len p50/p95/max", lens[len(lens)//2], lens[int(len(lens)*0.95)], lens[-1])
    lens = sorted(len(tok.encode(r["emoji"]).ids) for r in train[:20000])
    print("tgt len p50/p95/max", lens[len(lens)//2], lens[int(len(lens)*0.95)], lens[-1])


if __name__ == "__main__":
    main()
