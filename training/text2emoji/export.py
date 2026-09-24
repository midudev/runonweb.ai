"""Export a trained model to ONNX in the Transformers.js layout.

  out/<name>/
    config.json, generation_config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json
    onnx/encoder_model.onnx, onnx/decoder_model_merged.onnx           (fp32)
    onnx/encoder_model_quantized.onnx, onnx/decoder_model_merged_quantized.onnx  (q8 / uint8 weights)
  (fp16 skipped: T5 layer-norm Cast nodes break onnxconverter_common, and encoder-decoder
   models run on WASM in ONNX Runtime Web anyway)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).parent
name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
src = ROOT / "models" / name
tmp = ROOT / "onnx" / f"{name}-raw"
out = ROOT / "out" / name
onnx_dir = out / "onnx"

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(out, ignore_errors=True)
onnx_dir.mkdir(parents=True)

subprocess.run(
    [str(ROOT / ".venv/bin/optimum-cli"), "export", "onnx", "--model", str(src),
     "--task", "text2text-generation-with-past", "--opset", "17", str(tmp)],
    check=True, capture_output=True,
)


from optimum.onnx import merge_decoders

def quant(src: Path, dst: Path) -> None:
    # same recipe as the Transformers.js conversion script (q8 = dynamic uint8 weights)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8, per_channel=False, reduce_range=False)


# Encoder: flat graph, quantize directly.
shutil.copy(tmp / "encoder_model.onnx", onnx_dir / "encoder_model.onnx")
quant(onnx_dir / "encoder_model.onnx", onnx_dir / "encoder_model_quantized.onnx")

# Decoder: optimum's merged graph hides the weights inside `If` subgraphs, where quantize_dynamic
# cannot reach them. Quantize the two flat decoders first, then merge each pair ourselves.
for suffix, fn in (("", None), ("_quantized", quant)):
    pair = []
    for part in ("decoder_model", "decoder_with_past_model"):
        p = tmp / f"{part}{suffix}.onnx"
        if fn:
            fn(tmp / f"{part}.onnx", p)
        pair.append(p)
    merge_decoders(*pair, save_path=onnx_dir / f"decoder_model_merged{suffix}.onnx", strict=False)

for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy((src / f) if (src / f).exists() else (ROOT / "tokenizer" / f), out / f)
# Transformers.js reads the tokenizer class from tokenizer_config; T5Tokenizer works with our metaspace BPE
cfg = json.loads((out / "tokenizer_config.json").read_text())
cfg["tokenizer_class"] = "T5Tokenizer"
(out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
shutil.rmtree(tmp)

for f in sorted(onnx_dir.iterdir()):
    print(f"  {f.name:40s} {f.stat().st_size/1e6:6.2f} MB")
print("->", out)
