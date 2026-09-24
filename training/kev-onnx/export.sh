#!/bin/bash
# Kev checkpoint -> browser bundle for runonweb/classify.
#
#   KEV_REPO=/path/to/kev ./export.sh [kev-0.8b] [revision]
#
# KEV_REPO is a clone of github.com/jaredpalmer/kev with its env synced (`uv sync`) plus
# `uv pip install onnx onnxruntime onnxruntime-genai`. Output: ../../weights/<name>-ONNX
set -euo pipefail
cd "$(dirname "$0")"

NAME=${1:-kev-0.8b}
# Kev checkpoints are updated in place on the Hub; pin the one that was checked with parity.py.
# kev-0.8b 9a45d25: "round 15: documents + skills in one delta" (2026-09-24).
REVISION=${2:-9a45d25eb2ab761841196625383fa1dff0e56c1e}
PY="$KEV_REPO/.venv/bin/python"
HF="$KEV_REPO/.venv/bin/hf"
WORK=${WORK:-work}
OUT=${OUT:-../../weights/$NAME-ONNX}
mkdir -p "$WORK"

"$HF" download "jaredpalmer/$NAME" --revision "$REVISION" --local-dir "$WORK/$NAME" > /dev/null
read -r BASE REV < <("$PY" -c "import torch; m = torch.load('$WORK/$NAME/head.pt', map_location='cpu', weights_only=False); print(m['base'], m['base_revision'])")
echo "base $BASE @ $REV"
"$HF" download "$BASE" --revision "$REV" --local-dir "$WORK/base-$NAME" > /dev/null

"$PY" merge.py "$WORK/base-$NAME" "$WORK/$NAME" "$WORK/merged-$NAME"

# int4 weights (RTN, block 32, symmetric) with the Gated DeltaNet projections and their MLPs in int8,
# int4 embeddings, fp32 activations, no LM head: the graph returns `hidden_states`.
"$PY" -m onnxruntime_genai.models.builder -m "$BASE" -i "$WORK/merged-$NAME" -o "$WORK/onnx-$NAME" \
  -p int4 -e webgpu -c "$WORK/hfcache" --extra_options \
  exclude_lm_head=true exclude_mtp=true shared_embeddings=false hf_token=false use_webgpu_fp32=true \
  op_types_to_quantize=MatMul/Gather matmul_mixed_precision=linear_attn:int8 \
  || test -f "$WORK/onnx-$NAME/model.onnx"   # the builder writes the model, then fails on genai_config (no eos_token_id); not needed

"$PY" assemble.py "$WORK/$NAME" "$WORK/onnx-$NAME" "$WORK/base-$NAME" "$OUT" "$NAME" "$REVISION"
echo "bundle: $OUT"
