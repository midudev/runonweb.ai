# text2emoji — training recipe

Trains the tiny text → emoji model behind `runonweb/emoji` from scratch and exports it to ONNX
in the Transformers.js layout. Whole pipeline runs on a laptop (Apple M-series, MPS) in ~15 min.

| Model | Config | Params | q8 size (enc + dec) | val loss | emoji-set F1 |
|-------|--------|--------|---------------------|----------|--------------|
| `tiny` (shipped) | T5, 3+3 layers, d_model 128, d_ff 512, 4 heads, vocab 8192 | 2.43M | 1.7 MB + 2.1 MB = 3.9 MB | 3.72 | 0.38 |
| `small` | T5, 4+4 layers, d_model 192, d_ff 768, 4 heads, vocab 8192, 4 epochs | 5.71M | 3.5 MB + 4.3 MB = 7.8 MB | 3.66 | 0.37 |

Data: [KomeijiForce/Text2Emoji](https://huggingface.co/datasets/KomeijiForce/Text2Emoji) (EmojiLM paper),
503k ChatGPT-generated English sentence → emoji pairs. ~493k remain after cleaning (dedupe, cap at 12
emojis, drop non-emoji graphemes); 5k held out. The dataset card lists no license — check before
redistributing the *data*; the trained weights are MIT.

Emoji-set F1 is against a single ChatGPT reference, so it under-counts valid alternatives; look at the
samples printed during training for a better feel.

## Run

```bash
cd training/text2emoji
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch transformers datasets tokenizers "optimum[exporters]" optimum-onnx \
  onnx onnxruntime sentencepiece accelerate regex

.venv/bin/python prepare.py            # data/*.jsonl + tokenizer/ (BPE, metaspace, lowercase, 8192)
.venv/bin/python train.py --name tiny  # ~12 min on M4 Max, saves models/tiny
.venv/bin/python export.py tiny        # out/tiny: fp32 + q8 ONNX, Transformers.js layout
.venv/bin/python test_onnx.py tiny     # greedy decode with onnxruntime only
node test_node.mjs tiny q8             # same through @huggingface/transformers
```

Ship: copy `out/tiny/*.json` and `out/tiny/onnx/*_quantized.onnx` to `public/models/text2emoji-tiny/`.

Bigger variant for comparison: `train.py --name small --d_model 192 --d_ff 768 --layers 4 --epochs 4`.
2× the size for a 0.06 drop in val loss and slightly more literal outputs ("gym" → 🏋️‍♀️,
"code" → 📚✏️💡); not worth it for the demo, so `tiny` ships.

## Design notes

- **Architecture**: stock `T5ForConditionalGeneration` so Transformers.js' `text2text-generation`
  pipeline runs it with no custom JS. Tied embeddings, ReLU FF (no gated GELU: fewer params).
- **Tokenizer**: one shared BPE (T5-style metaspace, lowercased) trained on both sides. Targets are
  emoji grapheme clusters separated by spaces, so every emoji (incl. ZWJ sequences, skin tones) is a
  single `▁🍕`-style token. Output spaces are stripped in JS.
- **Decoding**: greedy + `no_repeat_ngram_size: 1` (each emoji is one token → never repeat an emoji).
- **Export**: `optimum-cli` produces the encoder and two decoders (no-past / with-past). The merged
  decoder optimum builds puts weights inside `If` subgraphs where `quantize_dynamic` cannot reach
  them, so `export.py` quantizes the two flat decoders first and merges them with
  `optimum.onnx.merge_decoders`. fp16 is skipped (T5 layer-norm `Cast` nodes break the converter;
  encoder-decoder graphs run on WASM in ONNX Runtime Web anyway).
