---
license: apache-2.0
base_model: jaredpalmer/{name}
library_name: onnxruntime
pipeline_tag: text-classification
tags:
  - kev
  - decision-model
  - typesafe
  - qwen3.5
  - onnx
  - webgpu
---

# {name}-ONNX

[{name}](https://huggingface.co/jaredpalmer/{name}) (revision `{kev_revision}`) by Jared Palmer, packaged for the browser by
[runonweb](https://runonweb.ai/models/classify) (`runonweb/classify`).

Kev is a Jev-style decision model: typed questions about one text (`noul` yes/no, `choice`, `score`)
answered with calibrated probabilities, no text generation. Requests follow TypeSafe's System One API.

## What changed from the original

- The rank-16 LoRA is merged into `{base}` (revision `{revision}`) in fp32.
- The backbone is exported with the onnxruntime-genai builder without the LM head and MTP layer: the graph
  returns `hidden_states` plus the recurrent/conv/KV cache, so the state is encoded once and each question
  runs as its own row on that cache (Kev's row form).
- Weights are int4 (RTN, block 32) with the Gated DeltaNet projections and their MLPs in int8; activations
  fp32. Needs `LinearAttention` / `CausalConvWithState` and 8-bit `MatMulNBits`: ONNX Runtime Web's native
  WebGPU build (`onnxruntime-web/webgpu`) or its WASM backend.
- The pointer head is `head.bin` (fp32: `q.weight`, `q.bias`, `k.weight`, `k.bias`); `kev.json` holds the
  calibration temperature, delimiter token ids and cache layout.

On 318 questions from Kev's development suites, probabilities differ from Kev's PyTorch fp32 path by 0.03 on
average (the fp32 export matches to 2e-5); 16 answers change, none with a margin above 0.2.
Recipe: `training/kev-onnx` in the runonweb repo.

## Use

```ts
import { Classifier } from 'runonweb/classify'

const classifier = new Classifier({ model: 'midudev/{name}-ONNX' })
const { answers } = await classifier.classify({
  state: 'I was charged twice. Please fix this ASAP.',
  questions: { billing: { type: 'noul', instructions: 'Is this ticket about billing?' } },
})
```

## License

Apache-2.0, like Kev and the Qwen3.5 base. Kev's training datasets have their own licenses; see the
[original model card](https://huggingface.co/jaredpalmer/{name}).
