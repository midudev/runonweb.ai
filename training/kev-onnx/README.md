# kev-onnx: Kev in the browser

Turns a [Kev](https://github.com/jaredpalmer/kev) checkpoint (Jared Palmer, Apache-2.0) into the bundle
`runonweb/classify` loads: ONNX backbone, pointer head, config and tokenizer. Kev is a LoRA and a pointer
head on a Qwen3.5 base; nothing here retrains it.

```bash
git clone https://github.com/jaredpalmer/kev && (cd kev && uv sync && uv pip install onnx onnxruntime onnxruntime-genai)
KEV_REPO=$PWD/kev training/kev-onnx/export.sh kev-0.8b   # ~3 min on an M4 Max, needs ~8 GB RAM
```

Output: `weights/kev-0.8b-ONNX/` (gitignored), ~750 MB:

| File | |
|------|---|
| `onnx/model.onnx` + `model.onnx.data` | backbone, `input_ids` / `attention_mask` / `position_ids` [3,B,S] + cache in, `hidden_states` + cache out |
| `head.bin` | pointer head, fp32 `q.weight` [256,1024], `q.bias`, `k.weight`, `k.bias` |
| `kev.json` | temperature, delimiter token ids, context limits, cache layout |
| `tokenizer.json`, `tokenizer_config.json` | Qwen3.5 tokenizer from the pinned base revision |
| `README.md` | model card (from `MODEL_CARD.md`) |

The site's dev server reads it through `public/models/runonweb/kev-0.8b-ONNX` (a symlink to that folder,
gitignored). Production loads `runonweb/kev-0.8b-ONNX` from the Hub:
`hf upload runonweb/kev-0.8b-ONNX weights/kev-0.8b-ONNX .`

## How it maps Kev

- `merge.py` folds the LoRA into the base in fp32 (`W + 2·B·A`) and drops the vision tower and MTP head.
- The onnxruntime-genai builder exports Qwen3.5 with Gated DeltaNet as `LinearAttention` +
  `CausalConvWithState` ops and full attention as `GroupQueryAttention`. `exclude_lm_head=true` makes the
  graph return `hidden_states`.
- Kev runs Qwen3.5 in its row form: state `<|fim_prefix|> …` once, then per question
  `<|fim_middle|> instructions <|box_start|> option <|box_end|> … <|fim_suffix|>` continuing the state's cache
  (positions keep counting). The browser does the same: one state pass, cache kept on the GPU, one pass per
  question, pointer head in JS (`(k(</opt>) · q(<decide>)) / 16 / T`, softmax).
- `reference.py` dumps Kev's PyTorch fp32 probabilities and token rows for `requests.json`; the JS tokenizer
  and encoder reproduce all 16 rows exactly.

## Quantization (Kev-0.8B, 318 questions from decision-v7, transfer-v4, scienthoon, SemIf dev)

`parity.py` runs every export the way the browser does and compares against the fp32 export, which matches
PyTorch to 2e-5. The comparison below ran on revision `54f4f87` (2026-09-21); the shipped revision `9a45d25`
(round 15, 2026-09-24) with the chosen setting: fp32 0.664, shipped 0.657, mean drift 0.028, 16 answers changed,
none with a margin above 0.2.

| Export | Size | Accuracy | Mean drift | Answers changed (margin > 0.2) |
|--------|------|----------|------------|--------------------------------|
| fp32 | 3.1 GB | 0.660 | — | — |
| int8 | 1.1 GB | 0.660 | 0.006 | 1 (0) |
| **int4 + DeltaNet int8 (shipped)** | **735 MB** | 0.648 | 0.029 | 12 (1) |
| int4 + DeltaNet int8, asymmetric | 753 MB | 0.670 | 0.029 | 10 (1) |
| int4 + DeltaNet int8, block 64 | 688 MB | 0.626 | 0.032 | 20 (2) |
| int4, block 16 | 635 MB | 0.664 | 0.050 | 25 |
| int4 | 541 MB | 0.651 | 0.059 | 35 (6) |
| int4, fp16 activations | 492 MB | 0.638 | 0.061 | 41 |
| int4, fp32 embeddings | 1.4 GB | 0.629 | 0.056 | 39 |

The error comes from the DeltaNet layers, not the embeddings. Accuracy differences under ~3 points are noise
at this sample size; drift and changed answers are the signal.

## Gotchas

- ONNX Runtime Web's default entry is the JSEP build; its `MatMulNBits` rejects 8-bit weights
  (`nbits_ == 4 || nbits_ == 2 was false`). Import `onnxruntime-web/webgpu` (native WebGPU EP).
- The builder writes the model, then fails writing `genai_config.json` (`Qwen3_5Config` has no
  `eos_token_id`). The runtime doesn't need that file; `export.sh` ignores the error.
- Cache inputs are named `past_key_values.N.{key,value}` (attention layers) and `past.N.{conv,recurrent}`
  (DeltaNet layers); outputs `present.N.*`. Empty cache = zero conv/recurrent states and zero-length KV.
- Measured on an M4 Max in Chromium: ~120 ms for three questions on WebGPU after warm-up, ~4 s on WASM.
