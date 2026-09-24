// @ts-nocheck
/**
 * Bonsai Image WebGPU engine (vendored).
 *
 * Official browser demo: https://huggingface.co/spaces/webml-community/bonsai-image-webgpu
 * Extracted pipeline: https://github.com/nishlumi/bonsai-image-pipeline-lib
 * Upstream demo: https://huggingface.co/spaces/prism-ml/Bonsai-image-demo
 * Models (Apache-2.0): prism-ml/bonsai-image-*-mlx-*
 *
 * WebGPUを利用したBonsai画像生成ライブラリ。
 * 元コード: https://huggingface.co/spaces/prism-ml/Bonsai-image-demo
 * モデル: prism-ml/bonsai-image-ternary-4B-mlx-2bit
 * 
 * 使用方法:
 *   import { BonsaiImagePipeline, BonsaiImageResult } from './bonsai-image.js';
 *   
 *   if (!await BonsaiImagePipeline.isSupported()) {
 *     throw new Error('WebGPU is not supported');
 *   }
 *   
 *   const pipeline = await BonsaiImagePipeline.from_pretrained(null, {
 *     onProgress: (info) => console.log(info),
 *   });
 *   
 *   const result = await pipeline.generate({
 *     prompt: 'A beautiful landscape',
 *     width: 1024,
 *     height: 1024,
 *   });
 *   
 *   const blob = result.toBlob();
 *   await pipeline.destroy();
 * 
 * 主要クラス:
 *   BonsaiImagePipeline  - メインの公開APIクラス (from_pretrained, generate, destroy)
 *   BonsaiImageResult    - 生成結果クラス (toBlob, toDataURL, toImageBitmap)
 *   PipelineBase         - パイプライン基底クラス (内部)
 *   HttpPipeline         - HTTP経由モデル読込パイプライン (内部)
 * 
 * @license See original repository for license information
 */

var e = Object.freeze({
  float16: `float16`,
  float32: `float32`,
  uint32: `uint32`,
});
function t(t) {
  if (t === e.float16) return 2;
  if (t === e.float32 || t === e.uint32) return 4;
  throw Error(`Unsupported dtype: ${t}`);
}
function n(e) {
  if (!Array.isArray(e) || e.length === 0)
    throw Error(`shape must be a non-empty array`);
  let t = 1;
  for (let n of e) {
    if (!Number.isInteger(n) || n <= 0)
      throw Error(`invalid shape dimension: ${n}`);
    t *= n;
  }
  return t;
}
function r(e) {
  let t = Array(e.length),
    n = 1;
  for (let r = e.length - 1; r >= 0; --r) ((t[r] = n), (n *= e[r]));
  return t;
}
var i = [`shader-f16`, `subgroups`, `chromium-experimental-subgroup-matrix`];
async function a(e = {}) {
  return new s({ host: e.host ?? (await c(e)) });
}
var o = class {
    constructor({
      runtime: e,
      dtype: i,
      shape: a,
      buffer: o,
      strides: s = r(a),
    }) {
      ((this.runtime = e),
        (this.dtype = i),
        (this.shape = a),
        (this.strides = s),
        (this.buffer = o),
        (this.size = n(a)),
        (this.byteLength = this.size * t(i)),
        (this.destroyed = !1));
    }
    destroy() {
      this.destroyed ||= (this.buffer?.destroy?.(), !0);
    }
  },
  s = class {
    constructor({ host: e }) {
      ((this.host = e),
        (this.pipelineCache = new Map()),
        (this.bindGroupCache = new Map()),
        (this.maxBindGroupCacheEntries = 4096),
        (this.bufferIds = new WeakMap()),
        (this.nextBufferId = 1),
        (this.readbackPool = new Map()),
        (this.readbackPoolBytes = 0),
        (this.maxReadbackPoolBytes = 64 * 1024 * 1024),
        (this.destroyed = !1));
    }
    caps() {
      return this.host.caps();
    }
    async destroy() {
      this.destroyed ||
        ((this.destroyed = !0),
        this.clearTransientCaches(),
        this.clearReadbackPool(),
        await this.host.destroy?.());
    }
    clearTransientCaches() {
      return { bindGroups: this.clearBindGroupCache() };
    }
    clearBindGroupCache() {
      let e = this.bindGroupCache.size;
      return (this.bindGroupCache.clear(), e);
    }
    clearReadbackPool() {
      let e = 0;
      for (let t of this.readbackPool.values())
        for (let n of t) (n.destroy?.(), e++);
      return (this.readbackPool.clear(), (this.readbackPoolBytes = 0), e);
    }
    tensorFromTypedArray(e, t, r) {
      if (!p(e, r))
        throw Error(
          `Only float16/Uint16Array, float32/Float32Array and uint32/Uint32Array tensors are supported`,
        );
      let i = n(t);
      if (r.length !== i)
        throw Error(
          `tensor data length ${r.length} does not match shape element count ${i}`,
        );
      let a = this.host.createBuffer({
          label: `tensor`,
          size: r.byteLength,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          mappedAtCreation: !0,
        }),
        s = r.constructor;
      return (
        new s(a.getMappedRange()).set(r),
        a.unmap(),
        new o({ runtime: this, dtype: e, shape: t, buffer: a })
      );
    }
    allocateWeightsBuffer({
      byteLength: t,
      dtype: n,
      shape: r,
      label: i = `weights`,
    }) {
      if (!Object.values(e).includes(n)) throw Error(`Unsupported dtype: ${n}`);
      if (!Number.isInteger(t) || t < 0)
        throw Error(`byteLength must be a nonnegative integer, got ${t}`);
      let a = this.host.createBuffer({
        label: i,
        size: t,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      return new o({ runtime: this, dtype: n, shape: r, buffer: a });
    }
    writeWeightsRange(e, t, n) {
      if (!(e instanceof o))
        throw Error(`writeWeightsRange expects a BonsaiWebGpuTensor`);
      if (!Number.isInteger(t) || t < 0)
        throw Error(`byteOffset must be a nonnegative integer, got ${t}`);
      if (t + n.byteLength > e.byteLength)
        throw Error(
          `write range [${t}, ${t + n.byteLength}] exceeds tensor byteLength ${e.byteLength}`,
        );
      this.host.writeBuffer(e.buffer, t, n);
    }
    async copyBufferToBuffer({
      src: e,
      dst: t,
      srcOffset: n = 0,
      dstOffset: r = 0,
      byteLength: i,
      wait: a = !1,
    }) {
      let o = e?.buffer ?? e,
        s = t?.buffer ?? t,
        c = this.host.device.createCommandEncoder({
          label: `copyBufferToBuffer`,
        });
      (c.copyBufferToBuffer(o, n, s, r, i),
        await this.host.submit([c.finish()], { wait: a }));
    }
    empty(r, i, a = `tensor-output`) {
      if (!Object.values(e).includes(r)) throw Error(`Unsupported dtype: ${r}`);
      let s = n(i) * t(r),
        c = this.host.createBuffer({
          label: a,
          size: s,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        });
      return new o({ runtime: this, dtype: r, shape: i, buffer: c });
    }
    readTensor(t) {
      let n = this.#e(t.byteLength),
        r = this.host.device.createCommandEncoder({ label: `readTensor` });
      (r.copyBufferToBuffer(t.buffer, 0, n, 0, t.byteLength),
        this.host.device.queue.submit([r.finish()]));
      let { dtype: i, byteLength: a } = t;
      return (async () => {
        let t = await this.host.mapRead(n, 0, a);
        if ((this.#t(a, n), i === e.float32)) return new Float32Array(t);
        if (i === e.float16) return new Uint16Array(t);
        if (i === e.uint32) return new Uint32Array(t);
        throw Error(`Unsupported dtype: ${i}`);
      })();
    }
    createUniformU32(e, t) {
      let n = new Uint32Array(e),
        r = this.host.createBuffer({
          label: t,
          size: n.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      return (this.host.writeBuffer(r, 0, n), r);
    }
    async runProgram(e, t = {}) {
      let n = await this.#n(e),
        r = t.wait ?? !1,
        i = this.host.device.createCommandEncoder({
          label: `compute-dispatch`,
        }),
        a = i.beginComputePass({ label: `compute-pass` });
      (a.setPipeline(n.pipeline),
        a.setBindGroup(0, n.bindGroup),
        a.dispatchWorkgroups(n.workgroups[0], n.workgroups[1], n.workgroups[2]),
        a.end(),
        await this.host.submit([i.finish()], { wait: r }));
    }
    #e(e) {
      let t = this.readbackPool.get(e);
      return t && t.length > 0
        ? ((this.readbackPoolBytes -= e), t.pop())
        : this.host.createBuffer({
            label: `tensor-readback`,
            size: e,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
    }
    #t(e, t) {
      if (this.readbackPoolBytes + e > this.maxReadbackPoolBytes) {
        t.destroy?.();
        return;
      }
      let n = this.readbackPool.get(e);
      (n || ((n = []), this.readbackPool.set(e, n)),
        n.push(t),
        (this.readbackPoolBytes += e));
    }
    async #n(e) {
      let {
        name: t,
        source: n,
        entryPoint: r = `main`,
        cacheKey: i = t,
        bindings: a,
        workgroups: o,
      } = e;
      if (typeof n != `string` || n.length === 0)
        throw Error(`program requires WGSL source`);
      if (!Array.isArray(a) || a.length === 0)
        throw Error(`program requires bindings`);
      if (!Array.isArray(o) || o.length !== 3)
        throw Error(`program requires a 3D workgroups array`);
      let s = await this.#r({
        name: t,
        source: n,
        entryPoint: r,
        cacheKey: i,
        layoutFactory: () =>
          this.#o(
            t,
            a.map((e) => e.type),
          ),
      });
      return {
        pipeline: s,
        bindGroup: this.#i({ name: t, cacheKey: i, pipeline: s, bindings: a }),
        workgroups: o,
      };
    }
    async #r({
      name: e,
      source: t,
      entryPoint: n,
      cacheKey: r,
      layoutFactory: i,
    }) {
      let a = this.pipelineCache.get(r);
      if (a) return a;
      let o = this.host.createShaderModule(t, e),
        s = await this.host.createComputePipeline({
          label: e,
          layout: i(),
          compute: { module: o, entryPoint: n },
        });
      return (this.pipelineCache.set(r, s), s);
    }
    #i({ name: e, cacheKey: t, pipeline: n, bindings: r }) {
      let i = r.map((e, t) => {
          let n = e.tensor?.buffer ?? e.buffer,
            r = { buffer: n, offset: e.offset ?? 0 };
          return (
            e.size !== void 0 && (r.size = e.size),
            {
              binding: e.binding ?? t,
              resource: r,
              cachePart: `${e.binding ?? t}:${this.#a(n)}:${r.offset}:${r.size ?? ``}`,
            }
          );
        }),
        a = `${t}|${i.map((e) => e.cachePart).join(`|`)}`,
        o = this.bindGroupCache.get(a);
      if (
        o === void 0 &&
        ((o = this.host.device.createBindGroup({
          label: `${e}-bind-group`,
          layout: n.getBindGroupLayout(0),
          entries: i.map(({ binding: e, resource: t }) => ({
            binding: e,
            resource: t,
          })),
        })),
        this.bindGroupCache.set(a, o),
        this.bindGroupCache.size > this.maxBindGroupCacheEntries)
      ) {
        let e = this.bindGroupCache.keys().next().value;
        this.bindGroupCache.delete(e);
      }
      return o;
    }
    #a(e) {
      let t = this.bufferIds.get(e);
      return (
        t === void 0 && ((t = this.nextBufferId++), this.bufferIds.set(e, t)),
        t
      );
    }
    #o(e, t) {
      let n = this.host.device.createBindGroupLayout({
        label: `${e}-bgl`,
        entries: t.map((e, t) => ({
          binding: t,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: e },
        })),
      });
      return this.host.device.createPipelineLayout({
        label: `${e}-layout`,
        bindGroupLayouts: [n],
      });
    }
  };
async function c(e = {}) {
  let t = globalThis.navigator?.gpu;
  if (!t) throw Error(`WebGPU is not available in this browser context`);
  let n = await t.requestAdapter({
    powerPreference: e.powerPreference ?? `high-performance`,
  });
  if (!n) throw Error(`No WebGPU adapter was returned`);
  let r = await n.requestDevice({
    requiredFeatures: l(n),
    requiredLimits: e.requiredLimits ?? u(n),
    label: e.label ?? `webgpu-ml-runtime`,
  });
  return (
    r.addEventListener(`uncapturederror`, (e) => {
      console.error(`WebGPU uncaptured error:`, e.error);
    }),
    f({ adapter: n, device: r, caps: d(r, n), destroy: () => r.destroy?.() })
  );
}
function l(e) {
  let t = new Set(
    ({}.BONSAI_DISABLE_WEBGPU_FEATURES ?? ``)
      .split(`,`)
      .map((e) => e.trim())
      .filter(Boolean),
  );
  return i.filter((n) => e.features.has(n) && !t.has(n));
}
function u(e) {
  let t = {};
  return (
    e.limits.maxBufferSize &&
      (t.maxBufferSize = Number(e.limits.maxBufferSize)),
    e.limits.maxStorageBufferBindingSize &&
      (t.maxStorageBufferBindingSize = Number(
        e.limits.maxStorageBufferBindingSize,
      )),
    t
  );
}
function d(e, t = null) {
  let n = t?.info ?? {};
  return {
    adapter: {
      vendor: n.vendor ?? ``,
      architecture: n.architecture ?? ``,
      device: n.device ?? ``,
      description: n.description ?? ``,
    },
    f16: e.features.has(`shader-f16`),
    subgroups: e.features.has(`subgroups`),
    subgroupMatrix: e.features.has(`chromium-experimental-subgroup-matrix`),
  };
}
function f({ adapter: e, device: t, caps: n, destroy: r, gpu: i }) {
  return {
    gpu: i,
    adapter: e,
    device: t,
    caps: () => n,
    createShaderModule: (e, n) => t.createShaderModule({ code: e, label: n }),
    createComputePipeline: (e) => t.createComputePipelineAsync(e),
    createBuffer: (e) => t.createBuffer(e),
    writeBuffer: (e, n, r) => t.queue.writeBuffer(e, n, r),
    submit: async (e, n = {}) => {
      (t.queue.submit(e),
        n.wait !== !1 && (await t.queue.onSubmittedWorkDone()));
    },
    mapRead: async (e, t, n) => {
      await e.mapAsync(GPUMapMode.READ, t, n);
      let r = e.getMappedRange(t, n).slice(0);
      return (e.unmap(), r);
    },
    destroy: r,
  };
}
function p(t, n) {
  return (
    (t === e.float16 && n instanceof Uint16Array) ||
    (t === e.float32 && n instanceof Float32Array) ||
    (t === e.uint32 && n instanceof Uint32Array)
  );
}
var m = BigInt(1e8),
  h = new TextDecoder(`utf-8`, { fatal: !0 }),
  g = Object.freeze({
    BOOL: 8,
    F4: 4,
    F6_E2M3: 6,
    F6_E3M2: 6,
    U8: 8,
    I8: 8,
    F8_E5M2: 8,
    F8_E4M3: 8,
    F8_E8M0: 8,
    F8_E4M3FNUZ: 8,
    F8_E5M2FNUZ: 8,
    I16: 16,
    U16: 16,
    F16: 16,
    BF16: 16,
    I32: 32,
    U32: 32,
    F32: 32,
    F64: 64,
    I64: 64,
    U64: 64,
    C64: 64,
  });
function _(e) {
  let t = g[e];
  if (!t) throw new v(`Unknown dtype: ${e}`);
  return t;
}
var v = class extends Error {
  constructor(e) {
    (super(e), (this.name = `SafeTensorsError`));
  }
};
function y(e) {
  if (e.byteLength < 8)
    throw new v(`File too small: ${e.byteLength} bytes < 8 byte header prefix`);
  let t = new DataView(e.buffer, e.byteOffset, e.byteLength).getBigUint64(
    0,
    !0,
  );
  if (t > m) throw new v(`Header length ${t} exceeds maximum 100000000`);
  let n = Number(t),
    r = 8 + n;
  if (r > e.byteLength) throw new v(`Header length ${n} exceeds buffer size`);
  let i = e.subarray(8, r),
    a;
  try {
    a = h.decode(i);
  } catch {
    throw new v(`Header is not valid UTF-8`);
  }
  let o;
  try {
    o = JSON.parse(a);
  } catch (e) {
    throw new v(`Header is not valid JSON: ${e.message}`);
  }
  if (typeof o != `object` || !o || Array.isArray(o))
    throw new v(`Header must be a JSON object`);
  return { headerByteLength: n, dataStart: r, header: o };
}
function b(e, t) {
  let n = e.__metadata__;
  if (n !== void 0 && !S(n))
    throw new v(`__metadata__ must be a {string: string} map`);
  let r = [];
  for (let [t, n] of Object.entries(e))
    t !== `__metadata__` && r.push([t, x(t, n)]);
  r.sort((e, t) => e[1].dataOffsets[0] - t[1].dataOffsets[0]);
  let i = 0;
  for (let [e, t] of r) {
    let [n, r] = t.dataOffsets;
    if (n !== i)
      throw new v(`Invalid offset for tensor ${e}: expected ${i}, got ${n}`);
    let a = t.elementCount * _(t.dtype);
    if (a % 8 != 0)
      throw new v(`Tensor ${e} has subbyte size ${a} bits not divisible by 8`);
    if (r - n !== a / 8)
      throw new v(
        `Tensor ${e} byte length ${r - n} does not match shape*dtype ${a / 8}`,
      );
    i = r;
  }
  if (t !== null && i !== t)
    throw new v(`Data length mismatch: header expects ${i} bytes, got ${t}`);
  return { metadata: n ?? null, tensors: new Map(r), dataByteLength: i };
}
function x(e, t) {
  if (typeof t != `object` || !t || Array.isArray(t))
    throw new v(`Tensor ${e}: info must be an object`);
  let { dtype: n, shape: r, data_offsets: i } = t;
  if (!(n in g)) throw new v(`Tensor ${e}: unknown dtype ${n}`);
  if (!Array.isArray(r) || !r.every((e) => Number.isInteger(e) && e >= 0))
    throw new v(`Tensor ${e}: shape must be an array of nonnegative integers`);
  if (
    !Array.isArray(i) ||
    i.length !== 2 ||
    !i.every((e) => Number.isInteger(e) && e >= 0)
  )
    throw new v(
      `Tensor ${e}: data_offsets must be a 2-element array of nonnegative integers`,
    );
  let [a, o] = i;
  if (o < a) throw new v(`Tensor ${e}: data_offsets end < begin`);
  let s = 1;
  for (let t of r)
    if (((s *= t), !Number.isSafeInteger(s)))
      throw new v(`Tensor ${e}: shape product exceeds Number.MAX_SAFE_INTEGER`);
  return { dtype: n, shape: [...r], dataOffsets: [a, o], elementCount: s };
}
function S(e) {
  if (typeof e != `object` || !e || Array.isArray(e)) return !1;
  for (let t of Object.values(e)) if (typeof t != `string`) return !1;
  return !0;
}
var C = 262144,
  w = 128 << 20,
  T = 1 << 20,
  E = 4,
  D = `safetensors-cache-v1`,
  O = `chunks`,
  k = `meta`;
async function A(e, t = {}) {
  return j(e, t, P);
}
async function j(e, t, n) {
  let r = t.cacheKey ?? (typeof e == `string` ? e : e.toString()),
    i = t.cache === !1,
    a = !!t.force,
    o = i || t.source ? null : (t.chunkCache ?? ee(t.cacheName ?? D));
  if (o && !a)
    try {
      let i = await o.getMeta?.(r);
      if (
        i &&
        i.header &&
        Number.isFinite(i.size) &&
        Number.isFinite(i.dataStart)
      ) {
        let r = await n(e, {
            ...t,
            chunkCache: o,
            knownSize: i.size,
            knownAcceptsRanges: i.acceptsRanges ?? !0,
          }),
          a = i.size - i.dataStart,
          { metadata: s, tensors: c } = b(i.header, a);
        return new M({
          source: r,
          dataStart: i.dataStart,
          metadata: s,
          tensors: c,
          headerByteLength: i.dataStart - 8,
          dataLength: a,
        });
      }
    } catch (e) {
      typeof console < `u` &&
        console.warn(`safetensors meta cache read failed: ${e.message}`);
    }
  let s = await n(e, { ...t, chunkCache: o }),
    c = t.headerProbeBytes ?? C,
    l = s.size == null ? c : Math.min(c, s.size),
    u = await s.readRange(0, l);
  if (u.byteLength < 8)
    throw new v(`Probe returned ${u.byteLength} bytes; need at least 8`);
  let d = new DataView(u.buffer, u.byteOffset, u.byteLength).getBigUint64(
    0,
    !0,
  );
  if (d > BigInt(1e8))
    throw new v(`Header length ${d} exceeds maximum 100000000`);
  let f = Number(d),
    p = 8 + f,
    m;
  if (u.byteLength >= p) m = u.subarray(0, p);
  else {
    let e = await s.readRange(u.byteLength, p);
    ((m = new Uint8Array(p)), m.set(u), m.set(e, u.byteLength));
  }
  let { header: h } = y(m),
    g = s.size == null ? null : s.size - p,
    { metadata: _, tensors: x } = b(h, g);
  return (
    o &&
      !a &&
      s.size != null &&
      o
        .putMeta?.(r, {
          size: s.size,
          dataStart: p,
          header: h,
          acceptsRanges: s.acceptsRanges,
        })
        .catch?.(() => {}),
    new M({
      source: s,
      dataStart: p,
      metadata: _,
      tensors: x,
      headerByteLength: f,
      dataLength: g,
    })
  );
}
var M = class {
  #e;
  #t;
  #n;
  constructor({
    source: e,
    dataStart: t,
    metadata: n,
    tensors: r,
    headerByteLength: i,
    dataLength: a,
  }) {
    ((this.#e = e),
      (this.#t = t),
      (this.#n = r),
      (this.metadata = n),
      (this.url = e.url),
      (this.totalSize = e.size),
      (this.headerByteLength = i),
      (this.dataByteLength = a));
  }
  names() {
    return [...this.#n.keys()];
  }
  has(e) {
    return this.#n.has(e);
  }
  info(e) {
    let t = this.#r(e);
    return {
      dtype: t.dtype,
      shape: [...t.shape],
      dataOffsets: [...t.dataOffsets],
    };
  }
  byteLength(e) {
    let t = this.#r(e);
    return t.dataOffsets[1] - t.dataOffsets[0];
  }
  async tensorBytes(e, t) {
    let [n, r] = this.#r(e).dataOffsets;
    return this.#e.readRange(this.#t + n, this.#t + r, t);
  }
  async streamAll(
    e,
    {
      concurrency: t = E,
      chunkMaxBytes: n = w,
      chunkMaxGap: r = T,
      names: i = null,
      onProgress: a,
      signal: o,
    } = {},
  ) {
    let s = i == null ? null : new Set(i);
    if (s && s.size === 0) return;
    if (s) {
      for (let e of s) if (!this.#n.has(e)) throw new v(`Unknown tensor: ${e}`);
    }
    let c = [];
    for (let [e, t] of this.#n) {
      if (s && !s.has(e)) continue;
      let [n, r] = t.dataOffsets;
      r > n && c.push({ name: e, begin: n, end: r });
    }
    if ((c.sort((e, t) => e.begin - t.begin), c.length === 0)) return;
    let l = N(c, { maxBytes: n, maxGap: r }),
      u = l.reduce((e, t) => e + (t.end - t.begin), 0),
      d = 0,
      f = new Map(),
      p = (e = {}) => {
        if (!a) return;
        let t = d;
        for (let e of f.values()) t += e;
        a({ loaded: t, total: u, ...e });
      },
      m = this.#t,
      h = this.#e;
    h.writeCachedChunk?.bind(h);
    let g = 0,
      _ = h.readTensor?.bind(h) || null,
      y = h.writeTensor?.bind(h) || null,
      b = async (t) => {
        let { begin: n, end: r, tensors: i } = l[t],
          a = m + n,
          s = m + r,
          c = r - n,
          u = null,
          g = !0,
          v = _
            ? await Promise.all(i.map((e) => _(m + e.begin, m + e.end)))
            : i.map(() => null);
        for (let e of v)
          if (!e) {
            g = !1;
            break;
          }
        if (g) {
          u = new Uint8Array(c);
          for (let e = 0; e < i.length; ++e) u.set(v[e], i[e].begin - n);
        } else
          (f.set(t, 0),
            (u = await h.readRange(a, s, {
              signal: o,
              onByteProgress: (e) => {
                (f.set(t, (f.get(t) ?? 0) + e),
                  p({ fromCache: !1, range: [a, s], inFlight: !0 }));
              },
            })));
        let b = g,
          x = i.map((e) => ({
            name: e.name,
            offset: e.begin - n,
            length: e.end - e.begin,
          })),
          S = e({ begin: a, end: s, bytes: u, tensors: x }),
          C =
            !b && y
              ? Promise.all(
                  i.map((e, t) => {
                    if (v[t]) return null;
                    let r = u.subarray(e.begin - n, e.end - n);
                    return y(m + e.begin, m + e.end, r).catch(() => {});
                  }),
                ).catch(() => {})
              : Promise.resolve();
        (await Promise.all([S, C]),
          f.delete(t),
          (d += c),
          p({ fromCache: b, range: [a, s] }));
      },
      x = async () => {
        for (;;) {
          if (o?.aborted) throw o.reason ?? Error(`aborted`);
          let e = g++;
          if (e >= l.length) return;
          await b(e);
        }
      },
      S = [];
    for (let e = 0; e < Math.min(t, l.length); ++e) S.push(x());
    await Promise.all(S);
  }
  async close() {
    await this.#e.close?.();
  }
  #r(e) {
    let t = this.#n.get(e);
    if (!t) throw new v(`Unknown tensor: ${e}`);
    return t;
  }
};
function N(e, { maxBytes: t, maxGap: n }) {
  let r = [],
    i = null;
  for (let a of e) {
    if (!i) {
      i = { begin: a.begin, end: a.end, tensors: [a] };
      continue;
    }
    let e = a.begin - i.end,
      o = a.end - i.begin;
    e <= n && o <= t
      ? ((i.end = a.end), i.tensors.push(a))
      : (r.push(i), (i = { begin: a.begin, end: a.end, tensors: [a] }));
  }
  return (i && r.push(i), r);
}
async function P(e, t) {
  let n = e instanceof URL ? e.toString() : String(e);
  if (!/^https?:/i.test(n))
    throw new v(`Expected http(s) safetensors URL, got: ${n}`);
  return F(n, t);
}
async function F(e, t) {
  let n = t.fetch ?? globalThis.fetch;
  if (typeof n != `function`)
    throw new v(`No fetch implementation available; pass options.fetch`);
  let r = !!(t.requireRangeRequests ?? t.requireRanges ?? !1),
    i,
    a;
  if (t.knownSize != null)
    ((i = t.knownSize), (a = t.knownAcceptsRanges !== !1));
  else {
    let r = await n(e, { method: `HEAD`, signal: t.signal });
    if (!r.ok) throw new v(`HEAD ${e} failed: ${r.status} ${r.statusText}`);
    let o = r.headers.get(`content-length`);
    if (
      ((a = (r.headers.get(`accept-ranges`) ?? ``)
        .toLowerCase()
        .includes(`bytes`)),
      (i = o === null ? null : Number(o)),
      o !== null && !Number.isFinite(i))
    )
      throw new v(`Invalid content-length header: ${o}`);
  }
  if (r && !a)
    throw new v(
      `Range requests are required for ${e}, but the server did not advertise Accept-Ranges: bytes`,
    );
  let o = t.cacheKey ?? e,
    s = t.cache === !1,
    c = !!t.force,
    l =
      t.chunkCache === void 0
        ? s
          ? null
          : ee(t.cacheName ?? D)
        : t.chunkCache;
  return {
    url: e,
    size: i,
    acceptsRanges: a,
    async readRange(i, o, s = {}) {
      if (i === o) return new Uint8Array();
      let c = s.signal ?? t.signal,
        l = s.onByteProgress ?? null;
      if (a) {
        let t = await n(e, {
          headers: { Range: `bytes=${i}-${o - 1}` },
          signal: c,
        });
        if (t.status !== 206 && t.status !== 200)
          throw new v(`Range ${i}-${o - 1} of ${e} failed: ${t.status}`);
        let a = await L(t, o - i, l);
        if (t.status === 200) {
          if (r)
            throw new v(
              `Range ${i}-${o - 1} of ${e} returned 200 instead of 206; refusing full-response fallback`,
            );
          return a.subarray(i, o);
        }
        if (a.byteLength !== o - i)
          throw new v(`Range ${i}-${o - 1} returned ${a.byteLength} bytes`);
        return a;
      }
      let u = await n(e, { signal: c });
      if (!u.ok) throw new v(`GET ${e} failed: ${u.status}`);
      return (await L(u, null, l)).subarray(i, o);
    },
    async readTensor(e, t) {
      if (!l || c) return null;
      try {
        return await l.get(o, e, t);
      } catch (e) {
        return (
          typeof console < `u` &&
            console.warn(`safetensors cache read failed: ${e.message}`),
          null
        );
      }
    },
    async writeTensor(e, t, n) {
      if (l)
        try {
          let r =
            n.byteOffset === 0 && n.byteLength === n.buffer.byteLength
              ? n
              : new Uint8Array(n);
          await l.put(o, e, t, r);
        } catch (e) {
          typeof console < `u` &&
            console.warn(`safetensors cache write failed: ${e.message}`);
        }
    },
  };
}
var I = new Map();
function ee(e) {
  return typeof indexedDB > `u`
    ? null
    : {
        async get(t, n, r) {
          let i = await te(e);
          return new Promise((e, a) => {
            let o = i.transaction(O, `readonly`).objectStore(O).get([t, n, r]);
            ((o.onsuccess = async () => {
              let t = o.result;
              if (!t) return e(null);
              e(new Uint8Array(await t.arrayBuffer()));
            }),
              (o.onerror = () => a(o.error)));
          });
        },
        async put(t, n, r, i) {
          let a = await te(e);
          return new Promise((e, o) => {
            let s = a
              .transaction(O, `readwrite`)
              .objectStore(O)
              .put(new Blob([i]), [t, n, r]);
            ((s.onsuccess = () => e()), (s.onerror = () => o(s.error)));
          });
        },
        async getMeta(t) {
          let n = await te(e);
          return new Promise((e, r) => {
            let i = n.transaction(k, `readonly`).objectStore(k).get(t);
            ((i.onsuccess = () => e(i.result ?? null)),
              (i.onerror = () => r(i.error)));
          });
        },
        async putMeta(t, n) {
          let r = await te(e);
          return new Promise((e, i) => {
            let a = r.transaction(k, `readwrite`).objectStore(k).put(n, t);
            ((a.onsuccess = () => e()), (a.onerror = () => i(a.error)));
          });
        },
      };
}
async function L(e, t, n) {
  if (!n || !e.body?.getReader) {
    let t = await e.arrayBuffer();
    return new Uint8Array(t);
  }
  let r = e.body.getReader();
  if (Number.isFinite(t) && t > 0) {
    let e = new Uint8Array(t),
      i = 0;
    for (;;) {
      let { done: t, value: a } = await r.read();
      if (t) break;
      (e.set(a, i), (i += a.byteLength), n(a.byteLength));
    }
    return i === t ? e : e.subarray(0, i);
  }
  let i = [],
    a = 0;
  for (;;) {
    let { done: e, value: t } = await r.read();
    if (e) break;
    (i.push(t), (a += t.byteLength), n(t.byteLength));
  }
  if (i.length === 1) return i[0];
  let o = new Uint8Array(a),
    s = 0;
  for (let e of i) (o.set(e, s), (s += e.byteLength));
  return o;
}
function te(e) {
  if (I.has(e)) return I.get(e);
  let t = new Promise((t, n) => {
    let r = indexedDB.open(e, 2);
    ((r.onupgradeneeded = (e) => {
      let t = r.result;
      (t.objectStoreNames.contains(O) || t.createObjectStore(O),
        t.objectStoreNames.contains(k) || t.createObjectStore(k));
    }),
      (r.onsuccess = () => t(r.result)),
      (r.onerror = () => n(r.error)),
      (r.onblocked = () => n(Error(`indexedDB open blocked`))));
  });
  return (I.set(e, t), t);
}
var R = class {
    constructor(e) {
      this.trie = this._build_trie(e);
    }
    _build_trie(e) {
      let t = Object.create(null);
      for (let n of e) {
        let e = t;
        for (let t = 0; t < n.length; ++t) {
          let r = n[t];
          e = e[r] ??= Object.create(null);
        }
        e.end = n;
      }
      return t;
    }
    split(e) {
      let t = [],
        n = e.length,
        r = 0,
        i = 0;
      for (; i < n; ) {
        let a = this.trie,
          o = null,
          s = i;
        for (; s < n && (a = a[e[s]]); ) (a.end && (o = a.end), ++s);
        o
          ? (i > r && t.push(e.slice(r, i)),
            t.push(o),
            (i += o.length),
            (r = i))
          : ++i;
      }
      return (r < n && t.push(e.slice(r)), t);
    }
  },
  ne = class {
    constructor(e) {
      ((this.content = e.content),
        (this.id = e.id),
        (this.single_word = e.single_word ?? !1),
        (this.lstrip = e.lstrip ?? !1),
        (this.rstrip = e.rstrip ?? !1),
        (this.special = e.special ?? !1),
        (this.normalized = e.normalized ?? !this.special));
    }
  },
  re = (() => {
    let e = [
        ...Array.from({ length: 94 }, (e, t) => t + 33),
        ...Array.from({ length: 12 }, (e, t) => t + 161),
        ...Array.from({ length: 82 }, (e, t) => t + 174),
      ],
      t = e.slice(),
      n = 0;
    for (let r = 0; r < 256; ++r)
      e.includes(r) || (e.push(r), t.push(256 + n), (n += 1));
    let r = t.map((e) => String.fromCharCode(e));
    return Object.fromEntries(e.map((e, t) => [e, r[t]]));
  })(),
  ie = ((e) => Object.fromEntries(Object.entries(e).map(([e, t]) => [t, e])))(
    re,
  ),
  ae = `.,!?…。，、।۔،`,
  oe = new Map([
    [
      `(?i:'s|'t|'re|'ve|'m|'ll|'d)`,
      `(?:'([sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD]))`,
    ],
    [
      `(?i:[sdmt]|ll|ve|re)`,
      `(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE])`,
    ],
    [`[^\\r\\n\\p{L}\\p{N}]?+`, `[^\\r\\n\\p{L}\\p{N}]?`],
    [`[^\\s\\p{L}\\p{N}]++`, `[^\\s\\p{L}\\p{N}]+`],
    [`(?>\\p{Nd}{510})`, `(?:\\p{Nd}{510})`],
    [`\\p{Nd}{3}+`, `(?:\\p{Nd}{3})+`],
    [`\\G`, ``],
    [` ?[^(\\s|[${ae}])]+`, ` ?[^\\s${ae}]+`],
  ]),
  se = `\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E`,
  ce = (e) =>
    e
      .replace(/ \./g, `.`)
      .replace(/ \?/g, `?`)
      .replace(/ \!/g, `!`)
      .replace(/ ,/g, `,`)
      .replace(/ \' /g, `'`)
      .replace(/ n't/g, `n't`)
      .replace(/ 'm/g, `'m`)
      .replace(/ 's/g, `'s`)
      .replace(/ 've/g, `'ve`)
      .replace(/ 're/g, `'re`),
  le = (e, t = !0) => {
    if (e.Regex !== void 0) {
      let t = e.Regex.replace(/\\([#&~])/g, `$1`);
      t = t
        .replace(/\\A/g, `^`)
        .replace(/\\z/g, `$`)
        .replace(/\\Z/g, `(?=\\r?\\n?$)`);
      for (let [e, n] of oe) t = t.replaceAll(e, n);
      try {
        return new RegExp(t, `gu`);
      } catch (e) {
        if (
          !(e instanceof SyntaxError) ||
          !e.message.toLowerCase().includes(`invalid property name`)
        )
          throw e;
        let n = !1,
          r = t.replace(/(\\[pP])\{([^}=]+)\}/g, (e, t, r) => {
            try {
              return (RegExp(`\\p{${r}}`, `u`), `${t}{${r}}`);
            } catch {
              return ((n = !0), `${t}{Script=${r}}`);
            }
          });
        if (!n) throw e;
        try {
          return new RegExp(r, `gu`);
        } catch {
          throw e;
        }
      }
    } else if (e.String !== void 0) {
      let n = ue(e.String);
      return new RegExp(t ? n : `(${n})`, `gu`);
    } else return (console.warn(`Unknown pattern type:`, e), null);
  },
  ue = (e) => e.replace(/[.*+?^${}()|[\]\\]/g, `\\$&`),
  de = (e, t, n) => {
    let r = [],
      i = 0;
    for (; i < e.length; ) {
      if ((r.push(e[i]), (t.get(e[i]) ?? n) !== n)) {
        ++i;
        continue;
      }
      for (; ++i < e.length && (t.get(e[i]) ?? n) === n; )
        t.get(r.at(-1)) !== n && (r[r.length - 1] += e[i]);
    }
    return r;
  },
  fe = (e) =>
    (e >= 19968 && e <= 40959) ||
    (e >= 13312 && e <= 19903) ||
    (e >= 131072 && e <= 173791) ||
    (e >= 173824 && e <= 177983) ||
    (e >= 177984 && e <= 178207) ||
    (e >= 178208 && e <= 183983) ||
    (e >= 63744 && e <= 64255) ||
    (e >= 194560 && e <= 195103),
  pe = (e) => Number.isInteger(e) || typeof e == `bigint`,
  me = (e) => {
    let t = 0;
    for (let n of e) ++t;
    return t;
  },
  he = (e) => ve(e.toLowerCase()),
  z = (...e) => Array.prototype.concat.apply([], e),
  ge = (e) => new Map(Object.entries(e)),
  _e = (e, t) => {
    let n = [],
      r = 0;
    for (let i of e.matchAll(t)) {
      let t = i[0];
      (r < i.index && n.push(e.slice(r, i.index)),
        t.length > 0 && n.push(t),
        (r = i.index + t.length));
    }
    return (r < e.length && n.push(e.slice(r)), n);
  },
  ve = (e) => e.replace(/\p{M}/gu, ``),
  B = (e, t, n = []) => {
    if (!e || Array.isArray(e) || typeof e != `object`)
      return `${t} must be a valid object`;
    for (let r of n)
      if (!(r in e)) return `${t} must contain a "${r}" property`;
    return null;
  },
  ye = (e) => e.match(/\S+/g) || [],
  V = class {
    constructor() {
      let e = function (...t) {
        return e._call(...t);
      };
      return Object.setPrototypeOf(e, new.target.prototype);
    }
  },
  H = class extends V {
    constructor(e) {
      (super(), (this.config = e));
    }
    _call(e) {
      return this.normalize(e);
    }
  },
  be = class extends H {
    tokenize_chinese_chars(e) {
      let t = [];
      for (let n = 0; n < e.length; ++n) {
        let r = e[n];
        fe(r.charCodeAt(0)) ? (t.push(` `), t.push(r), t.push(` `)) : t.push(r);
      }
      return t.join(``);
    }
    strip_accents(e) {
      return e.normalize(`NFD`).replace(/\p{Mn}/gu, ``);
    }
    is_control(e) {
      switch (e) {
        case `	`:
        case `
`:
        case `\r`:
          return !1;
        default:
          return /^\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}$/u.test(e);
      }
    }
    clean_text(e) {
      let t = [];
      for (let n of e) {
        let e = n.charCodeAt(0);
        e === 0 ||
          e === 65533 ||
          this.is_control(n) ||
          (/^\s$/.test(n) ? t.push(` `) : t.push(n));
      }
      return t.join(``);
    }
    normalize(e) {
      return (
        this.config.clean_text && (e = this.clean_text(e)),
        this.config.handle_chinese_chars &&
          (e = this.tokenize_chinese_chars(e)),
        this.config.lowercase
          ? ((e = e.toLowerCase()),
            this.config.strip_accents !== !1 && (e = this.strip_accents(e)))
          : this.config.strip_accents && (e = this.strip_accents(e)),
        e
      );
    }
  },
  xe = class extends H {
    constructor(e) {
      (super(e), (this.charsmap = e.precompiled_charsmap ?? null));
    }
    normalize(e) {
      return (
        (e = e.replace(
          /[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]/gm,
          ``,
        )),
        (e = e.replace(
          /[\u0009\u000A\u000C\u000D\u00A0\u1680\u2000-\u200F\u2028\u2029\u202F\u205F\u2581\u3000\uFEFF\uFFFD]/gm,
          ` `,
        )),
        (e = e.includes(`～`)
          ? e
              .split(`～`)
              .map((e) => e.normalize(`NFKC`))
              .join(`～`)
          : e.normalize(`NFKC`)),
        e
      );
    }
  },
  Se = class extends H {
    constructor(e) {
      (super(e), (this.normalizers = (e.normalizers ?? []).map((e) => Pe(e))));
    }
    normalize(e) {
      return this.normalizers.reduce((e, t) => (t ? t.normalize(e) : e), e);
    }
  },
  Ce = class extends H {
    normalize(e) {
      let t = le(this.config.pattern ?? {});
      return t === null ? e : e.replaceAll(t, this.config.content ?? ``);
    }
  },
  we = class extends H {
    constructor() {
      (super(...arguments), (this.form = `NFC`));
    }
    normalize(e) {
      return ((e = e.normalize(this.form)), e);
    }
  },
  Te = class extends we {
    constructor() {
      (super(...arguments), (this.form = `NFC`));
    }
  },
  Ee = class extends we {
    constructor() {
      (super(...arguments), (this.form = `NFD`));
    }
  },
  De = class extends we {
    constructor() {
      (super(...arguments), (this.form = `NFKC`));
    }
  },
  Oe = class extends we {
    constructor() {
      (super(...arguments), (this.form = `NFKD`));
    }
  },
  ke = class extends H {
    normalize(e) {
      return (
        this.config.strip_left && this.config.strip_right
          ? (e = e.trim())
          : (this.config.strip_left && (e = e.trimStart()),
            this.config.strip_right && (e = e.trimEnd())),
        e
      );
    }
  },
  Ae = class extends H {
    normalize(e) {
      return ve(e);
    }
  },
  je = class extends H {
    normalize(e) {
      return e.toLowerCase();
    }
  },
  Me = class extends H {
    normalize(e) {
      return ((e = this.config.prepend + e), e);
    }
  };
function Ne(e) {
  if (e === null) return null;
  switch (e.type) {
    case `BertNormalizer`:
      return new be(e);
    case `Precompiled`:
      return new xe(e);
    case `Sequence`:
      return new Se(e);
    case `Replace`:
      return new Ce(e);
    case `NFC`:
      return new Te(e);
    case `NFD`:
      return new Ee(e);
    case `NFKC`:
      return new De(e);
    case `NFKD`:
      return new Oe(e);
    case `Strip`:
      return new ke(e);
    case `StripAccents`:
      return new Ae(e);
    case `Lowercase`:
      return new je(e);
    case `Prepend`:
      return new Me(e);
    default:
      throw Error(`Unknown Normalizer type: ${e.type}`);
  }
}
var Pe = Ne,
  Fe = class extends V {
    pre_tokenize(e, t) {
      return (
        Array.isArray(e)
          ? e.map((e) => this.pre_tokenize_text(e, t))
          : this.pre_tokenize_text(e, t)
      ).flat();
    }
    _call(e, t) {
      return this.pre_tokenize(e, t);
    }
  },
  Ie = class extends Fe {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.add_prefix_space = this.config.add_prefix_space ?? !1),
        (this.trim_offsets = this.config.trim_offsets ?? !1),
        (this.use_regex = this.config.use_regex ?? !0),
        (this.pattern =
          /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu),
        (this.byte_encoder = re),
        (this.text_encoder = new TextEncoder()));
    }
    pre_tokenize_text(e, t) {
      return (
        this.add_prefix_space && !e.startsWith(` `) && (e = ` ` + e),
        (this.use_regex ? e.match(this.pattern) || [] : [e]).map((e) =>
          Array.from(
            this.text_encoder.encode(e),
            (e) => this.byte_encoder[e],
          ).join(``),
        )
      );
    }
  },
  Le = class extends Fe {
    pre_tokenize_text(e, t) {
      return e.match(/\w+|[^\w\s]+/g) || [];
    }
  },
  Re = class extends Fe {
    constructor(e) {
      (super(),
        (this.replacement = e.replacement ?? `▁`),
        (this.str_rep = e.str_rep || this.replacement),
        (this.prepend_scheme = e.prepend_scheme ?? `always`));
    }
    pre_tokenize_text(e, t) {
      let { section_index: n = void 0 } = t ?? {},
        r = e.replaceAll(` `, this.str_rep);
      return (
        !r.startsWith(this.replacement) &&
          (this.prepend_scheme === `always` ||
            (this.prepend_scheme === `first` && n === 0)) &&
          (r = this.str_rep + r),
        [r]
      );
    }
  },
  ze = class extends Fe {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.pattern = le(
          this.config.pattern ?? {},
          this.config.invert ?? !0,
        )));
    }
    pre_tokenize_text(e) {
      return this.pattern === null
        ? []
        : this.config.invert
          ? e.match(this.pattern) || []
          : this.config.behavior?.toLowerCase() === `removed`
            ? e.split(this.pattern).filter((e) => e)
            : _e(e, this.pattern);
    }
  },
  Be = class extends Fe {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.pattern = RegExp(`[^${se}]+|[${se}]+`, `gu`)));
    }
    pre_tokenize_text(e) {
      return e.match(this.pattern) || [];
    }
  },
  Ve = class extends Fe {
    constructor(e) {
      (super(), (this.config = e));
      let t = `[^\\d]+|\\d${this.config.individual_digits ? `` : `+`}`;
      this.pattern = new RegExp(t, `gu`);
    }
    pre_tokenize_text(e) {
      return e.match(this.pattern) || [];
    }
  },
  He = class extends Fe {
    constructor() {
      (super(), (this.pattern = RegExp(`[^\\s${se}]+|[${se}]`, `gu`)));
    }
    pre_tokenize_text(e, t) {
      return e.trim().match(this.pattern) || [];
    }
  },
  Ue = class extends Fe {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.pattern = le(this.config.pattern ?? {})),
        (this.content = this.config.content ?? ``));
    }
    pre_tokenize_text(e) {
      return this.pattern === null
        ? [e]
        : [e.replaceAll(this.pattern, this.config.content ?? ``)];
    }
  },
  We = class extends Fe {
    constructor(e) {
      (super(), (this.tokenizers = (e.pretokenizers ?? []).map((e) => Je(e))));
    }
    pre_tokenize_text(e, t) {
      return this.tokenizers.reduce(
        (e, n) => (n ? n.pre_tokenize(e, t) : e),
        [e],
      );
    }
  },
  Ge = class extends Fe {
    pre_tokenize_text(e) {
      return ye(e);
    }
  },
  Ke = class extends Fe {
    constructor(e) {
      (super(), (this.config = e), (this._length = e.length));
    }
    pre_tokenize_text(e) {
      let t = [];
      for (let n = 0; n < e.length; n += this._length)
        t.push(e.slice(n, n + this._length));
      return t;
    }
  };
function qe(e) {
  if (e === null) return null;
  switch (e.type) {
    case `BertPreTokenizer`:
      return new He();
    case `Sequence`:
      return new We(e);
    case `Whitespace`:
      return new Le();
    case `WhitespaceSplit`:
      return new Ge();
    case `Metaspace`:
      return new Re(e);
    case `ByteLevel`:
      return new Ie(e);
    case `Split`:
      return new ze(e);
    case `Punctuation`:
      return new Be(e);
    case `Digits`:
      return new Ve(e);
    case `Replace`:
      return new Ue(e);
    case `FixedLength`:
      return new Ke(e);
    default:
      throw Error(`Unknown PreTokenizer type: ${e.type}`);
  }
}
var Je = qe,
  Ye = class extends V {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.vocab = []),
        (this.tokens_to_ids = new Map()),
        (this.unk_token_id = void 0),
        (this.unk_token = void 0),
        (this.end_of_word_suffix = void 0),
        (this.fuse_unk = this.config.fuse_unk ?? !1));
    }
    _call(e) {
      let t = this.encode(e);
      return (
        this.fuse_unk && (t = de(t, this.tokens_to_ids, this.unk_token_id)),
        t
      );
    }
  },
  Xe = class extends Ye {
    constructor(e) {
      (super(e),
        (this.max_input_chars_per_word = 100),
        (this.tokens_to_ids = ge(e.vocab)),
        (this.unk_token_id = this.tokens_to_ids.get(e.unk_token)),
        (this.unk_token = e.unk_token),
        (this.max_input_chars_per_word = e.max_input_chars_per_word ?? 100),
        (this.vocab = Array(this.tokens_to_ids.size)));
      for (let [e, t] of this.tokens_to_ids) this.vocab[t] = e;
    }
    encode(e) {
      let t = [];
      for (let n of e) {
        let e = [...n];
        if (e.length > this.max_input_chars_per_word) {
          t.push(this.unk_token);
          continue;
        }
        let r = !1,
          i = 0,
          a = [];
        for (; i < e.length; ) {
          let t = e.length,
            n = null;
          for (; i < t; ) {
            let r = e.slice(i, t).join(``);
            if (
              (i > 0 && (r = this.config.continuing_subword_prefix + r),
              this.tokens_to_ids.has(r))
            ) {
              n = r;
              break;
            }
            --t;
          }
          if (n === null) {
            r = !0;
            break;
          }
          (a.push(n), (i = t));
        }
        r ? t.push(this.unk_token) : t.push(...a);
      }
      return t;
    }
  },
  Ze = class e {
    constructor(e, t) {
      ((this.is_leaf = e), (this.children = t));
    }
    static default() {
      return new e(!1, new Map());
    }
  },
  Qe = class {
    constructor() {
      this.root = Ze.default();
    }
    extend(e) {
      for (let t of e) this.push(t);
    }
    push(e) {
      let t = this.root;
      for (let n of e) {
        let e = t.children.get(n);
        (e === void 0 && ((e = Ze.default()), t.children.set(n, e)), (t = e));
      }
      t.is_leaf = !0;
    }
    *common_prefix_search(e) {
      let t = this.root;
      if (t === void 0) return;
      let n = ``;
      for (let r of e) {
        if (((n += r), (t = t.children.get(r)), t === void 0)) return;
        t.is_leaf && (yield n);
      }
    }
  },
  $e = class e {
    constructor(e, t, n, r, i) {
      ((this.token_id = e),
        (this.node_id = t),
        (this.pos = n),
        (this.length = r),
        (this.score = i),
        (this.prev = null),
        (this.backtrace_score = 0));
    }
    clone() {
      let t = new e(
        this.token_id,
        this.node_id,
        this.pos,
        this.length,
        this.score,
      );
      return (
        (t.prev = this.prev),
        (t.backtrace_score = this.backtrace_score),
        t
      );
    }
  },
  et = class {
    constructor(e, t, n) {
      ((this.chars = Array.from(e)),
        (this.len = this.chars.length),
        (this.bos_token_id = t),
        (this.eos_token_id = n),
        (this.nodes = []),
        (this.begin_nodes = Array.from({ length: this.len + 1 }, () => [])),
        (this.end_nodes = Array.from({ length: this.len + 1 }, () => [])));
      let r = new $e(this.bos_token_id ?? 0, 0, 0, 0, 0),
        i = new $e(this.eos_token_id ?? 0, 1, this.len, 0, 0);
      (this.nodes.push(r.clone()),
        this.nodes.push(i.clone()),
        this.begin_nodes[this.len].push(i),
        this.end_nodes[0].push(r));
    }
    insert(e, t, n, r) {
      let i = this.nodes.length,
        a = new $e(r, i, e, t, n);
      (this.begin_nodes[e].push(a),
        this.end_nodes[e + t].push(a),
        this.nodes.push(a));
    }
    viterbi() {
      let e = this.len,
        t = 0;
      for (; t <= e; ) {
        if (this.begin_nodes[t].length == 0) return [];
        for (let e of this.begin_nodes[t]) {
          e.prev = null;
          let n = 0,
            r = null;
          for (let i of this.end_nodes[t]) {
            let t = i.backtrace_score + e.score;
            (r === null || t > n) && ((r = i.clone()), (n = t));
          }
          if (r !== null) ((e.prev = r), (e.backtrace_score = n));
          else return [];
        }
        ++t;
      }
      let n = [],
        r = this.begin_nodes[e][0].prev;
      if (r === null) return [];
      let i = r.clone();
      for (; i.prev !== null; )
        (n.push(i.clone()), (i = i.clone().prev.clone()));
      return (n.reverse(), n);
    }
    piece(e) {
      return this.chars.slice(e.pos, e.pos + e.length).join(``);
    }
    tokens() {
      return this.viterbi().map((e) => this.piece(e));
    }
    token_ids() {
      return this.viterbi().map((e) => e.token_id);
    }
  };
function tt(e) {
  if (e.length === 0) throw Error(`Array must not be empty`);
  let t = e[0],
    n = 0;
  for (let r = 1; r < e.length; ++r) e[r] < t && ((t = e[r]), (n = r));
  return [t, n];
}
var nt = class extends Ye {
    constructor(e, t) {
      super(e);
      let n = e.vocab.length;
      ((this.vocab = Array(n)), (this.scores = Array(n)));
      for (let t = 0; t < n; ++t) [this.vocab[t], this.scores[t]] = e.vocab[t];
      ((this.unk_token_id = e.unk_id),
        (this.unk_token = this.vocab[e.unk_id]),
        (this.tokens_to_ids = new Map(this.vocab.map((e, t) => [e, t]))),
        (this.bos_token = ` `),
        (this.bos_token_id = this.tokens_to_ids.get(this.bos_token)),
        (this.eos_token = t),
        (this.eos_token_id = this.tokens_to_ids.get(this.eos_token)),
        (this.unk_token = this.vocab[this.unk_token_id]),
        (this.min_score = tt(this.scores)[0]),
        (this.unk_score = this.min_score - 10),
        (this.scores[this.unk_token_id] = this.unk_score),
        (this.trie = new Qe()),
        this.trie.extend(this.vocab),
        (this.fuse_unk = !0));
    }
    populate_nodes(e) {
      let t = e.chars,
        n = 0;
      for (; n < t.length; ) {
        let r = !1,
          i = [],
          a = t.slice(n).join(``),
          o = this.trie.common_prefix_search(a);
        for (let t of o) {
          i.push(t);
          let a = this.tokens_to_ids.get(t),
            o = this.scores[a],
            s = me(t);
          (e.insert(n, s, o, a), !r && s === 1 && (r = !0));
        }
        (r || e.insert(n, 1, this.unk_score, this.unk_token_id), (n += 1));
      }
    }
    tokenize(e) {
      let t = new et(e, this.bos_token_id, this.eos_token_id);
      return (this.populate_nodes(t), t.tokens());
    }
    encode(e) {
      let t = [];
      for (let n of e) {
        let e = this.tokenize(n);
        t.push(...e);
      }
      return t;
    }
  },
  rt = class {
    constructor(e = (e, t) => e > t, t = 1 / 0) {
      ((this._heap = []), (this._comparator = e), (this._max_size = t));
    }
    get size() {
      return this._heap.length;
    }
    is_empty() {
      return this.size === 0;
    }
    peek() {
      return this._heap[0];
    }
    push(...e) {
      return this.extend(e);
    }
    extend(e) {
      for (let t of e)
        if (this.size < this._max_size) (this._heap.push(t), this._sift_up());
        else {
          let e = this._smallest();
          this._comparator(t, this._heap[e]) &&
            ((this._heap[e] = t), this._sift_up_from(e));
        }
      return this.size;
    }
    pop() {
      let e = this.peek(),
        t = this.size - 1;
      return (
        t > 0 && this._swap(0, t),
        this._heap.pop(),
        this._sift_down(),
        e
      );
    }
    replace(e) {
      let t = this.peek();
      return ((this._heap[0] = e), this._sift_down(), t);
    }
    _parent(e) {
      return ((e + 1) >>> 1) - 1;
    }
    _left(e) {
      return (e << 1) + 1;
    }
    _right(e) {
      return (e + 1) << 1;
    }
    _greater(e, t) {
      return this._comparator(this._heap[e], this._heap[t]);
    }
    _swap(e, t) {
      let n = this._heap[e];
      ((this._heap[e] = this._heap[t]), (this._heap[t] = n));
    }
    _sift_up() {
      this._sift_up_from(this.size - 1);
    }
    _sift_up_from(e) {
      for (; e > 0 && this._greater(e, this._parent(e)); )
        (this._swap(e, this._parent(e)), (e = this._parent(e)));
    }
    _sift_down() {
      let e = 0;
      for (
        ;
        (this._left(e) < this.size && this._greater(this._left(e), e)) ||
        (this._right(e) < this.size && this._greater(this._right(e), e));
      ) {
        let t =
          this._right(e) < this.size &&
          this._greater(this._right(e), this._left(e))
            ? this._right(e)
            : this._left(e);
        (this._swap(e, t), (e = t));
      }
    }
    _smallest() {
      return 2 ** Math.floor(Math.log2(this.size)) - 1;
    }
  },
  it = class {
    constructor(e) {
      ((this.capacity = e), (this.cache = new Map()));
    }
    get(e) {
      if (!this.cache.has(e)) return;
      let t = this.cache.get(e);
      return (this.cache.delete(e), this.cache.set(e, t), t);
    }
    put(e, t) {
      (this.cache.has(e) && this.cache.delete(e),
        this.cache.set(e, t),
        this.cache.size > this.capacity &&
          this.cache.delete(this.cache.keys().next().value));
    }
    clear() {
      this.cache.clear();
    }
  },
  at = class extends Ye {
    constructor(e) {
      (super(e),
        (this.tokens_to_ids = ge(e.vocab)),
        (this.unk_token_id = this.tokens_to_ids.get(e.unk_token)),
        (this.unk_token = e.unk_token),
        (this.vocab = Array(this.tokens_to_ids.size)));
      for (let [e, t] of this.tokens_to_ids) this.vocab[t] = e;
      let t = Array.isArray(e.merges[0]);
      ((this.merges = t ? e.merges : e.merges.map((e) => e.split(` `, 2))),
        (this.bpe_ranks = new Map(
          this.merges.map((e, t) => [JSON.stringify(e), t]),
        )),
        (this.end_of_word_suffix = e.end_of_word_suffix),
        (this.continuing_subword_suffix = e.continuing_subword_suffix ?? null),
        (this.byte_fallback = this.config.byte_fallback ?? !1),
        this.byte_fallback && (this.text_encoder = new TextEncoder()),
        (this.ignore_merges = this.config.ignore_merges ?? !1),
        (this.max_length_to_cache = 256),
        (this.cache_capacity = 1e4),
        (this.cache = new it(this.cache_capacity)));
    }
    clear_cache() {
      this.cache.clear();
    }
    bpe(e) {
      if (e.length === 0) return [];
      let t = this.cache.get(e);
      if (t !== void 0) return t;
      let n = Array.from(e);
      this.end_of_word_suffix && (n[n.length - 1] += this.end_of_word_suffix);
      let r = [];
      if (n.length > 1) {
        let e = new rt((e, t) => e.score < t.score),
          t = { token: n[0], bias: 0, prev: null, next: null },
          i = t;
        for (let t = 1; t < n.length; ++t) {
          let r = { bias: t / n.length, token: n[t], prev: i, next: null };
          ((i.next = r), this.add_node(e, i), (i = r));
        }
        for (; !e.is_empty(); ) {
          let n = e.pop();
          if (n.deleted || !n.next || n.next.deleted) continue;
          if (((n.deleted = !0), (n.next.deleted = !0), n.prev)) {
            let e = { ...n.prev };
            ((n.prev.deleted = !0),
              (n.prev = e),
              e.prev ? (e.prev.next = e) : (t = e));
          }
          let r = {
            token: n.token + n.next.token,
            bias: n.bias,
            prev: n.prev,
            next: n.next.next,
          };
          (r.prev ? ((r.prev.next = r), this.add_node(e, r.prev)) : (t = r),
            r.next && ((r.next.prev = r), this.add_node(e, r)));
        }
        for (let e = t; e !== null; e = e.next) r.push(e.token);
      } else r = n;
      if (this.continuing_subword_suffix)
        for (let e = 0; e < r.length - 1; ++e)
          r[e] += this.continuing_subword_suffix;
      return (e.length < this.max_length_to_cache && this.cache.put(e, r), r);
    }
    add_node(e, t) {
      let n = this.bpe_ranks.get(JSON.stringify([t.token, t.next.token]));
      n !== void 0 && ((t.score = n + t.bias), e.push(t));
    }
    encode(e) {
      let t = [];
      for (let n of e) {
        if (this.ignore_merges && this.tokens_to_ids.has(n)) {
          t.push(n);
          continue;
        }
        let e = this.bpe(n);
        for (let n of e)
          if (this.tokens_to_ids.has(n)) t.push(n);
          else if (this.byte_fallback) {
            let e = Array.from(this.text_encoder.encode(n)).map(
              (e) => `<0x${e.toString(16).toUpperCase().padStart(2, `0`)}>`,
            );
            e.every((e) => this.tokens_to_ids.has(e))
              ? t.push(...e)
              : this.unk_token != null && t.push(this.unk_token);
          } else this.unk_token != null && t.push(this.unk_token);
      }
      return t;
    }
  },
  ot = class extends Ye {
    constructor(e, t) {
      super(e);
      let n = e.vocab;
      ((this.tokens_to_ids = ge(t.target_lang ? n[t.target_lang] : n)),
        (this.bos_token = t.bos_token),
        (this.bos_token_id = this.tokens_to_ids.get(this.bos_token)),
        (this.eos_token = t.eos_token),
        (this.eos_token_id = this.tokens_to_ids.get(this.eos_token)),
        (this.pad_token = t.pad_token),
        (this.pad_token_id = this.tokens_to_ids.get(this.pad_token)),
        (this.unk_token = t.unk_token),
        (this.unk_token_id = this.tokens_to_ids.get(this.unk_token)),
        (this.vocab = Array(this.tokens_to_ids.size)));
      for (let [e, t] of this.tokens_to_ids) this.vocab[t] = e;
    }
    encode(e) {
      return e;
    }
  };
function st(e, t) {
  switch (e.type) {
    case `WordPiece`:
      return new Xe(e);
    case `Unigram`:
      return new nt(e, t.eos_token);
    case `BPE`:
      return new at(e);
    default:
      if (e.vocab)
        return Array.isArray(e.vocab)
          ? new nt(e, t.eos_token)
          : Object.hasOwn(e, `continuing_subword_prefix`) &&
              Object.hasOwn(e, `unk_token`)
            ? Object.hasOwn(e, `merges`)
              ? new at(e)
              : new Xe(e)
            : new ot(e, {
                target_lang: t.target_lang,
                bos_token: t.bos_token,
                eos_token: t.eos_token,
                pad_token: t.pad_token,
                unk_token: t.unk_token,
              });
      throw Error(`Unknown TokenizerModel type: ${e?.type}`);
  }
}
var ct = st,
  lt = class extends V {
    constructor(e) {
      (super(), (this.config = e));
    }
    _call(e, ...t) {
      return this.post_process(e, ...t);
    }
  },
  ut = class extends lt {
    post_process(e, t = null, n = !0) {
      let r = t === null ? this.config.single : this.config.pair,
        i = [],
        a = [];
      for (let o of r)
        `SpecialToken` in o
          ? n && (i.push(o.SpecialToken.id), a.push(o.SpecialToken.type_id))
          : `Sequence` in o &&
            (o.Sequence.id === `A`
              ? ((i = z(i, e)),
                (a = z(a, Array(e.length).fill(o.Sequence.type_id))))
              : o.Sequence.id === `B` &&
                ((i = z(i, t)),
                (a = z(a, Array(t.length).fill(o.Sequence.type_id)))));
      return { tokens: i, token_type_ids: a };
    }
  },
  dt = class extends lt {
    post_process(e, t = null) {
      return { tokens: e, tokens_pair: t };
    }
  },
  ft = class extends lt {
    constructor(e) {
      (super(e), (this.sep = e.sep), (this.cls = e.cls));
    }
    post_process(e, t = null, n = !0) {
      n && (e = z([this.cls[0]], e, [this.sep[0]]));
      let r = Array(e.length).fill(0);
      if (t) {
        let i = [],
          a = n ? [this.sep[0]] : [];
        ((e = z(e, i, t, a)),
          (r = z(r, Array(t.length + i.length + a.length).fill(1))));
      }
      return { tokens: e, token_type_ids: r };
    }
  },
  pt = class extends lt {
    constructor(e) {
      (super(e), (this.sep = e.sep), (this.cls = e.cls));
    }
    post_process(e, t, n = !0) {
      n && (e = z([this.cls[0]], e, [this.sep[0]]));
      let r = Array(e.length).fill(0);
      if (t) {
        let i = n ? [this.sep[0]] : [],
          a = n ? [this.sep[0]] : [];
        ((e = z(e, i, t, a)),
          (r = z(r, Array(t.length + i.length + a.length).fill(1))));
      }
      return { tokens: e, token_type_ids: r };
    }
  },
  mt = class extends lt {
    constructor(e) {
      (super(e), (this.processors = (e.processors ?? []).map((e) => gt(e))));
    }
    post_process(e, t = null, n = !0) {
      let r = { tokens: e, tokens_pair: t };
      for (let e of this.processors)
        r = e.post_process(r.tokens, r.tokens_pair, n);
      return r;
    }
  };
function ht(e) {
  if (e === null) return null;
  switch (e.type) {
    case `TemplateProcessing`:
      return new ut(e);
    case `ByteLevel`:
      return new dt(e);
    case `BertProcessing`:
      return new ft(e);
    case `RobertaProcessing`:
      return new pt(e);
    case `Sequence`:
      return new mt(e);
    default:
      throw Error(`Unknown PostProcessor type: ${e.type}`);
  }
}
var gt = ht,
  _t = class extends V {
    constructor(e) {
      (super(),
        (this.config = e),
        (this.added_tokens = []),
        (this.end_of_word_suffix = null),
        (this.trim_offsets = `trim_offsets` in e ? e.trim_offsets : !1));
    }
    _call(e) {
      return this.decode(e);
    }
    decode(e) {
      return this.decode_chain(e).join(``);
    }
  },
  vt = class extends _t {
    constructor(e) {
      (super(e),
        (this.byte_decoder = ie),
        (this.text_decoder = new TextDecoder(`utf-8`, {
          fatal: !1,
          ignoreBOM: !0,
        })),
        (this.end_of_word_suffix = null));
    }
    convert_tokens_to_string(e) {
      let t = e.join(``),
        n = new Uint8Array([...t].map((e) => this.byte_decoder[e]));
      return this.text_decoder.decode(n);
    }
    decode_chain(e) {
      let t = [],
        n = [];
      for (let r of e)
        this.added_tokens.find((e) => e.content === r) === void 0
          ? n.push(r)
          : (n.length > 0 &&
              (t.push(this.convert_tokens_to_string(n)), (n = [])),
            t.push(r));
      return (n.length > 0 && t.push(this.convert_tokens_to_string(n)), t);
    }
  },
  yt = class extends _t {
    constructor(e) {
      (super(e), (this.cleanup = e.cleanup));
    }
    decode_chain(e) {
      return e.map((e, t) => {
        if (t !== 0) {
          let t = this.config.prefix;
          e = t && e.startsWith(t) ? e.replace(t, ``) : ` ` + e;
        }
        return (this.cleanup && (e = ce(e)), e);
      });
    }
  },
  bt = class extends _t {
    constructor(e) {
      (super(e), (this.replacement = e.replacement ?? `▁`));
    }
    decode_chain(e) {
      let t = [];
      for (let n = 0; n < e.length; ++n) {
        let r = e[n].replaceAll(this.replacement, ` `);
        (n == 0 && r.startsWith(` `) && (r = r.substring(1)), t.push(r));
      }
      return t;
    }
  },
  xt = class extends _t {
    constructor(e) {
      (super(e), (this.suffix = e.suffix ?? ``));
    }
    decode_chain(e) {
      return e.map((t, n) =>
        t.replaceAll(this.suffix, n === e.length - 1 ? `` : ` `),
      );
    }
  },
  St = class extends _t {
    constructor(e) {
      (super(e),
        (this.pad_token = e.pad_token ?? ``),
        (this.word_delimiter_token = e.word_delimiter_token ?? ``),
        (this.cleanup = e.cleanup));
    }
    convert_tokens_to_string(e) {
      if (e.length === 0) return ``;
      let t = [e[0]];
      for (let n = 1; n < e.length; ++n) e[n] !== t.at(-1) && t.push(e[n]);
      let n = t.filter((e) => e !== this.pad_token).join(``);
      return (
        this.cleanup &&
          (n = ce(n).replaceAll(this.word_delimiter_token, ` `).trim()),
        n
      );
    }
    decode_chain(e) {
      return [this.convert_tokens_to_string(e)];
    }
  },
  Ct = class extends _t {
    constructor(e) {
      (super(e), (this.decoders = (e.decoders ?? []).map((e) => kt(e))));
    }
    decode_chain(e) {
      return this.decoders.reduce((e, t) => t.decode_chain(e), e);
    }
  },
  wt = class extends _t {
    decode_chain(e) {
      let t = le(this.config.pattern),
        n = this.config.content ?? ``;
      return t === null ? e : e.map((e) => e.replaceAll(t, n));
    }
  },
  Tt = class extends _t {
    decode_chain(e) {
      return [e.join(``)];
    }
  },
  Et = class extends _t {
    constructor(e) {
      (super(e),
        (this.content = e.content ?? ``),
        (this.start = e.start ?? 0),
        (this.stop = e.stop ?? 0));
    }
    decode_chain(e) {
      return e.map((e) => {
        let t = 0;
        for (let n = 0; n < this.start && e[n] === this.content; ++n) t = n + 1;
        let n = e.length;
        for (let t = 0; t < this.stop; ++t) {
          let r = e.length - t - 1;
          if (e[r] === this.content) {
            n = r;
            continue;
          } else break;
        }
        return e.slice(t, n);
      });
    }
  },
  Dt = class extends _t {
    constructor(e) {
      (super(e), (this.text_decoder = new TextDecoder()));
    }
    decode_chain(e) {
      let t = [],
        n = [];
      for (let r of e) {
        let e = null;
        if (r.length === 6 && r.startsWith(`<0x`) && r.endsWith(`>`)) {
          let t = parseInt(r.slice(3, 5), 16);
          isNaN(t) || (e = t);
        }
        if (e !== null) n.push(e);
        else {
          if (n.length > 0) {
            let e = this.text_decoder.decode(Uint8Array.from(n));
            (t.push(e), (n = []));
          }
          t.push(r);
        }
      }
      if (n.length > 0) {
        let e = this.text_decoder.decode(Uint8Array.from(n));
        (t.push(e), (n = []));
      }
      return t;
    }
  };
function Ot(e) {
  if (e === null) return null;
  switch (e.type) {
    case `ByteLevel`:
      return new vt(e);
    case `WordPiece`:
      return new yt(e);
    case `Metaspace`:
      return new bt(e);
    case `BPEDecoder`:
      return new xt(e);
    case `CTC`:
      return new St(e);
    case `Sequence`:
      return new Ct(e);
    case `Replace`:
      return new wt(e);
    case `Fuse`:
      return new Tt(e);
    case `Strip`:
      return new Et(e);
    case `ByteFallback`:
      return new Dt(e);
    default:
      throw Error(`Unknown Decoder type: ${e.type}`);
  }
}
var kt = Ot,
  At = class {
    constructor(e, t) {
      let n = B(e, `Tokenizer`, [
        `model`,
        `decoder`,
        `post_processor`,
        `pre_tokenizer`,
        `normalizer`,
      ]);
      if (n) throw Error(n);
      let r = B(t, `Config`);
      if (r) throw Error(r);
      ((this.tokenizer = e),
        (this.config = t),
        (this.normalizer = Pe(this.tokenizer.normalizer)),
        (this.pre_tokenizer = Je(this.tokenizer.pre_tokenizer)),
        (this.model = ct(this.tokenizer.model, this.config)),
        (this.post_processor = gt(this.tokenizer.post_processor)),
        (this.decoder = kt(this.tokenizer.decoder)),
        (this.special_tokens = []),
        (this.all_special_ids = []),
        (this.added_tokens = []));
      let i = [],
        a = [];
      this.added_tokens_map = new Map();
      for (let e of this.tokenizer.added_tokens) {
        let t = new ne(e);
        if (
          (this.added_tokens.push(t),
          this.model.tokens_to_ids.set(t.content, t.id),
          (this.model.vocab[t.id] = t.content),
          t.special &&
            (this.special_tokens.push(t.content),
            this.all_special_ids.push(t.id)),
          this.added_tokens_map.set(t.content, t),
          t.normalized && this.normalizer !== null)
        ) {
          let e = this.normalizer(t.content);
          (a.push(e), this.added_tokens_map.set(e, t));
        } else i.push(t.content);
      }
      ((this.config.additional_special_tokens ?? []).forEach((e) => {
        this.special_tokens.includes(e) || this.special_tokens.push(e);
      }),
        this.decoder &&
          ((this.decoder.added_tokens = this.added_tokens),
          (this.decoder.end_of_word_suffix = this.model.end_of_word_suffix)),
        (this.splitter_unnormalized = new R(i)),
        (this.splitter_normalized = new R(a)),
        (this.remove_space = this.config.remove_space),
        (this.clean_up_tokenization_spaces =
          this.config.clean_up_tokenization_spaces ?? !0),
        (this.do_lowercase_and_remove_accent =
          this.config.do_lowercase_and_remove_accent ?? !1));
    }
    encode(
      e,
      {
        text_pair: t = null,
        add_special_tokens: n = !0,
        return_token_type_ids: r = null,
      } = {},
    ) {
      let { tokens: i, token_type_ids: a } = this.tokenize_helper(e, {
          text_pair: t,
          add_special_tokens: n,
        }),
        o = i.map(
          (e) =>
            this.added_tokens_map.get(e)?.id ??
            this.model.tokens_to_ids.get(e) ??
            this.model.unk_token_id,
        ),
        s = { ids: o, tokens: i, attention_mask: Array(o.length).fill(1) };
      return (r && a && (s.token_type_ids = a), s);
    }
    decode(e, t = {}) {
      if (!Array.isArray(e) || e.length === 0 || !pe(e[0]))
        throw Error(`token_ids must be a non-empty array of integers.`);
      let n = e.map((e) => this.model.vocab[Number(e)] ?? this.model.unk_token);
      t.skip_special_tokens &&
        (n = n.filter((e) => !this.special_tokens.includes(e)));
      let r = this.decoder ? this.decoder(n) : n.join(` `);
      return (
        this.decoder &&
          this.decoder.end_of_word_suffix &&
          ((r = r.replaceAll(this.decoder.end_of_word_suffix, ` `)),
          t.skip_special_tokens && (r = r.trim())),
        (t.clean_up_tokenization_spaces ?? this.clean_up_tokenization_spaces) &&
          (r = ce(r)),
        r
      );
    }
    tokenize(e, { text_pair: t = null, add_special_tokens: n = !1 } = {}) {
      return this.tokenize_helper(e, { text_pair: t, add_special_tokens: n })
        .tokens;
    }
    encode_text(e) {
      if (e === null) return null;
      let t = this.splitter_unnormalized.split(e);
      return (
        t.forEach((e, n) => {
          let r = this.added_tokens_map.get(e);
          r &&
            (r.lstrip && n > 0 && (t[n - 1] = t[n - 1].trimEnd()),
            r.rstrip && n < t.length - 1 && (t[n + 1] = t[n + 1].trimStart()));
        }),
        t.flatMap((e, t) => {
          if (e.length === 0) return [];
          if (this.added_tokens_map.has(e)) return [e];
          if (
            (this.remove_space === !0 && (e = e.trim().split(/\s+/).join(` `)),
            this.do_lowercase_and_remove_accent && (e = he(e)),
            this.normalizer !== null && (e = this.normalizer(e)),
            e.length === 0)
          )
            return [];
          let n = this.splitter_normalized.split(e);
          return (
            n.forEach((e, t) => {
              let r = this.added_tokens_map.get(e);
              r &&
                (r.lstrip && t > 0 && (n[t - 1] = n[t - 1].trimEnd()),
                r.rstrip &&
                  t < n.length - 1 &&
                  (n[t + 1] = n[t + 1].trimStart()));
            }),
            n.flatMap((e) => {
              if (e.length === 0) return [];
              if (this.added_tokens_map.has(e)) return [e];
              let n =
                this.pre_tokenizer === null
                  ? [e]
                  : this.pre_tokenizer(e, { section_index: t });
              return this.model(n);
            })
          );
        })
      );
    }
    tokenize_helper(e, { text_pair: t = null, add_special_tokens: n = !0 }) {
      let r = this.encode_text(e),
        i = this.encode_text(t || null);
      return this.post_processor
        ? this.post_processor(r, i, n)
        : { tokens: z(r ?? [], i ?? []) };
    }
    token_to_id(e) {
      return this.model.tokens_to_ids.get(e);
    }
    id_to_token(e) {
      return this.model.vocab[e];
    }
    get_added_tokens_decoder() {
      let e = new Map();
      for (let t of this.added_tokens) e.set(t.id, t);
      return e;
    }
    get_vocab(e = !0) {
      let t = new Map();
      for (let n = 0; n < this.model.vocab.length; ++n) {
        let r = this.model.vocab[n];
        (e || !this.added_tokens_map.has(r)) && t.set(r, n);
      }
      return t;
    }
  };
function jt(e) {
  let t = e.byteLength / 2,
    n = new Float32Array(t),
    r = new Uint32Array(n.buffer);
  if (e.byteOffset % 2 == 0) {
    let n = new Uint16Array(e.buffer, e.byteOffset, t);
    for (let e = 0; e < t; ++e) r[e] = n[e] << 16;
  } else
    for (let n = 0; n < t; ++n) {
      let t = e[n * 2];
      r[n] = ((e[n * 2 + 1] << 8) | t) << 16;
    }
  return n;
}
function Mt(e) {
  let t = e.byteLength / 2,
    n = new Uint16Array(t);
  return (Nt(e, n), n);
}
function Nt(e, t, n = 0, r = 1) {
  let i = e.byteLength / 2,
    a = zt();
  if (e.byteOffset % 2 == 0) {
    let o = new Uint16Array(e.buffer, e.byteOffset, i);
    for (let e = 0, s = n; e < i; ++e, s += r) t[s] = a[o[e]];
  } else
    for (let o = 0, s = n; o < i; ++o, s += r) {
      let n = e[o * 2];
      t[s] = a[(e[o * 2 + 1] << 8) | n];
    }
  return t;
}
function Pt(e, t, n) {
  let r = t * n;
  if (e.byteLength !== r * 2)
    throw Error(`BF16 byte length ${e.byteLength} does not match ${t}x${n}`);
  let i = new Uint16Array(r),
    a = zt();
  if (e.byteOffset % 2 == 0) {
    let o = new Uint16Array(e.buffer, e.byteOffset, r);
    for (let e = 0; e < n; ++e) {
      let r = e * t;
      for (let s = 0; s < t; ++s) i[r + s] = a[o[s * n + e]];
    }
  } else
    for (let r = 0; r < n; ++r) {
      let o = r * t;
      for (let s = 0; s < t; ++s) {
        let t = (s * n + r) * 2;
        i[o + s] = a[(e[t + 1] << 8) | e[t]];
      }
    }
  return i;
}
function Ft(e) {
  let t = e & 32768 ? -1 : 1,
    n = (e >>> 10) & 31,
    r = e & 1023;
  return n === 31
    ? r === 0
      ? t * (1 / 0)
      : NaN
    : n === 0
      ? t * 2 ** -14 * (r / 1024)
      : t * 2 ** (n - 15) * (1 + r / 1024);
}
function It(e) {
  if (Number.isNaN(e)) return 32256;
  if (e === 1 / 0) return 31744;
  if (e === -1 / 0) return 64512;
  let t = e < 0 || Object.is(e, -0) ? 32768 : 0,
    n = Math.abs(e);
  if (n === 0) return t;
  if (n >= 65504) return t | 31743;
  if (n < 5.960464477539063e-8) return t;
  if (n < 6103515625e-14) return t | Math.round(n / 5.960464477539063e-8);
  let r = Math.floor(Math.log2(n)),
    i = Math.round((n / 2 ** r - 1) * 1024);
  return i === 1024 ? t | ((r + 16) << 10) : t | ((r + 15) << 10) | i;
}
function Lt(e) {
  let t = e & 32768,
    n = (e >>> 7) & 255,
    r = e & 127;
  if (n === 255) return r === 0 ? t | 31744 : 32256;
  if (n === 0) return t;
  let i = n - 112;
  if (i >= 31) return t | 31743;
  if (i > 0) return t | (i << 10) | (r << 3);
  if (n < 103) return t;
  let a = 128 | r,
    o = 110 - n,
    s;
  return (
    (s =
      o <= 0
        ? a << -o
        : o >= 16
          ? 0
          : (a >>> o) + +((a & ((1 << o) - 1)) * 2 >= 1 << o)),
    t | Math.min(s, 1023)
  );
}
var Rt = null;
function zt() {
  if (Rt) return Rt;
  let e = new Uint16Array(65536);
  for (let t = 0; t < e.length; ++t) e[t] = Lt(t);
  return ((Rt = e), e);
}
function Bt(e, t = new Set()) {
  if (
    !(typeof e != `object` || !e) &&
    !t.has(e) &&
    (t.add(e), !(ArrayBuffer.isView(e) || e instanceof ArrayBuffer))
  ) {
    if (Wt(e)) {
      e.destroy?.();
      return;
    }
    if (Gt(e)) {
      e.destroy();
      return;
    }
    if (e instanceof Map) {
      for (let n of e.values()) Bt(n, t);
      e.clear();
      return;
    }
    if (e instanceof Set) {
      for (let n of e.values()) Bt(n, t);
      e.clear();
      return;
    }
    for (let n of Object.keys(e)) Bt(e[n], t);
  }
}
function Vt() {
  let e = [],
    t = new Set(),
    n = !1;
  return {
    track(t) {
      if (n) throw Error(`Cannot track GPU resources after scope.destroy()`);
      return (typeof t == `object` && t && e.push(t), t);
    },
    keep(e) {
      return (Ut(e, t), e);
    },
    destroy() {
      if (n) return;
      n = !0;
      let r = new Set(t);
      for (let t = e.length - 1; t >= 0; --t) Bt(e[t], r);
      ((e.length = 0), t.clear());
    },
    get size() {
      return e.length;
    },
  };
}
function Ht(e, t) {
  if (!t) return e;
  let n = new Set([`empty`, `tensorFromTypedArray`, `createUniformU32`]);
  return new Proxy(e, {
    get(e, r, i) {
      let a = Reflect.get(e, r, i);
      return typeof a == `function`
        ? n.has(r)
          ? (...n) => t.track(a.apply(e, n))
          : a.bind(e)
        : a;
    },
  });
}
function Ut(e, t) {
  if (
    !(typeof e != `object` || !e) &&
    !t.has(e) &&
    (t.add(e),
    !(ArrayBuffer.isView(e) || e instanceof ArrayBuffer) && !(Wt(e) || Gt(e)))
  ) {
    if (e instanceof Map) {
      for (let n of e.values()) Ut(n, t);
      return;
    }
    if (e instanceof Set) {
      for (let n of e.values()) Ut(n, t);
      return;
    }
    for (let n of Object.keys(e)) Ut(e[n], t);
  }
}
function Wt(e) {
  return (
    Array.isArray(e.shape) &&
    typeof e.dtype == `string` &&
    e.buffer &&
    typeof e.buffer.destroy == `function`
  );
}
function Gt(e) {
  return (
    typeof e.destroy == `function` &&
    typeof e.mapAsync == `function` &&
    typeof e.getMappedRange == `function`
  );
}
function Kt(e) {
  return Math.floor(32 / e);
}
function qt(e, t) {
  if (e % t !== 0)
    throw Error(`inFeatures (${e}) must be divisible by groupSize (${t})`);
  return Math.floor(e / t);
}
function Jt({
  scalesBytes: e,
  biasesBytes: t,
  outFeatures: n,
  inFeatures: r,
  groupSize: i,
  dtype: a = `f32`,
}) {
  let o = qt(r, i),
    s = a === `f16` ? Mt(e) : jt(e),
    c = a === `f16` ? Mt(t) : jt(t);
  if (s.length !== n * o || c.length !== n * o)
    throw Error(`scale/bias length mismatch (expected ${n * o})`);
  let l =
    a === `f16` ? new Uint16Array(n * o * 2) : new Float32Array(n * o * 2);
  for (let e = 0; e < n; ++e)
    for (let t = 0; t < o; ++t) {
      let r = (t * n + e) * 2;
      ((l[r] = s[e * o + t]), (l[r + 1] = c[e * o + t]));
    }
  if (a === `f32` || a === `f16`) return l;
  throw Error(`Unsupported scaleBias dtype: ${a}`);
}
function Yt({
  scalesBytes: e,
  biasesBytes: t,
  outFeatures: n,
  inFeatures: r,
  groupSize: i,
  dtype: a = `f32`,
}) {
  let o = qt(r, i),
    s = a === `f16` ? new Uint16Array(n * o * 2) : new Float32Array(n * o * 2);
  return (
    Xt({
      scalesBytes: e,
      biasesBytes: t,
      out: s,
      outFeatures: n,
      inFeatures: r,
      groupSize: i,
      dtype: a,
    }),
    s
  );
}
function Xt({
  scalesBytes: e,
  biasesBytes: t,
  out: n,
  outFeatures: r,
  inFeatures: i,
  groupSize: a,
  dtype: o = `f32`,
  dstElementOffset: s = 0,
}) {
  let c = qt(i, a),
    l = r * c;
  if (o === `f16`) {
    if (e.byteLength !== l * 2 || t.byteLength !== l * 2)
      throw Error(`scale/bias length mismatch (expected ${l})`);
    return (Nt(e, n, s, 2), Nt(t, n, s + 1, 2), n);
  }
  let u = jt(e),
    d = jt(t);
  if (u.length !== l || d.length !== l)
    throw Error(`scale/bias length mismatch (expected ${l})`);
  for (let e = 0; e < r * c; ++e) {
    let t = s + e * 2;
    ((n[t] = u[e]), (n[t + 1] = d[e]));
  }
  if (o === `f32`) return n;
  throw Error(`Unsupported scaleBias dtype: ${o}`);
}
function Zt() {
  let e = new Map(),
    t = new Set();
  function n(n, r) {
    if (e.has(n)) throw Error(`Duplicate handler for tensor: ${n}`);
    (t.add(n),
      e.set(n, {
        async receive(e) {
          (t.delete(n), await r(e));
        },
      }));
  }
  function r(n, r) {
    let i = {},
      a = n.length,
      o = null,
      s = null,
      c = null;
    for (let o of n) {
      if (e.has(o)) throw Error(`Duplicate handler for tensor: ${o}`);
      (t.add(o),
        e.set(o, {
          async receive(e) {
            if ((t.delete(o), (i[o] = e), --a === 0))
              try {
                (await r(i), s && s());
              } catch (e) {
                if (c) c(e);
                else throw e;
              } finally {
                for (let e of n) delete i[e];
              }
          },
        }));
    }
    return (
      (o = new Promise((e, t) => {
        ((s = e), (c = t));
      })),
      o
    );
  }
  async function i({ bytes: t, tensors: n }) {
    let r = [];
    for (let i of n) {
      let n = e.get(i.name);
      if (!n) continue;
      let a = t.subarray(i.offset, i.offset + i.length);
      r.push(n.receive(a));
    }
    r.length && (await Promise.all(r));
  }
  function a() {
    if (t.size > 0) {
      let e = [...t].slice(0, 5).join(`, `);
      throw Error(
        `Stream plan incomplete \u2014 ${t.size} tensor(s) never arrived (first: ${e})`,
      );
    }
  }
  function o() {
    return [...e.keys()];
  }
  return { tensor: n, group: r, onChunk: i, assertComplete: a, names: o };
}
function Qt(e, t, n) {
  let r = new ArrayBuffer(16),
    i = new Uint32Array(r),
    a = new Float32Array(r);
  for (let e = 0; e < 4; ++e) {
    let n = t[e];
    if (n == null) {
      i[e] = 0;
      continue;
    }
    if (`u32` in n) i[e] = n.u32 >>> 0;
    else if (`f32` in n) a[e] = n.f32;
    else throw Error(`uniform item ${e} must be {u32} or {f32}`);
  }
  return e.createUniformU32(new Uint32Array(r), n);
}
function $t(e) {
  return e === `float16` ? `f16` : `f32`;
}
function en(...e) {
  return e.includes(`float16`)
    ? `enable f16;
`
    : ``;
}
function tn(e, t) {
  return t === `float16` ? `f32(${e})` : e;
}
function nn(e, t) {
  return t === `float16` ? `f16(${e})` : e;
}
function rn(e, t) {
  return t === `float16` ? `vec4<f32>(${e})` : e;
}
function an({
  dim: e,
  eps: t,
  withWeight: n = !0,
  inputDtype: r = `float32`,
  weightDtype: i = `float32`,
  outputDtype: a = `float32`,
}) {
  let o = $t(r),
    s = $t(i),
    c = $t(a);
  return `${en(r, i, a)}struct Params { rows: u32, rowStride: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${o}>;
${
  n
    ? `@group(0) @binding(1) var<storage, read>       w: array<${s}>;
`
    : ``
}@group(0) @binding(${n ? 2 : 1}) var<storage, read_write> y: array<${c}>;
@group(0) @binding(${n ? 3 : 2}) var<uniform>             params: Params;

const DIM: u32 = ${e}u;
const EPS: f32 = ${t};
const WG: u32 = 64u;

var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowStride = select(params.rowStride, params.rows, params.rowStride == 0u);
  let row = wg.x + wg.y * rowStride;
  if (row >= params.rows) { return; }
  let tid = lid.x;
  let base = row * DIM;

  // Compute sum of squares.
  var acc: f32 = 0.0;
  var i: u32 = tid;
  loop {
    if (i >= DIM) { break; }
    let v = ${tn(`x[base + i]`, r)};
    acc = acc + v * v;
    i = i + WG;
  }
  partial[tid] = acc;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let scale = inverseSqrt(partial[0] / f32(DIM) + EPS);

  // Apply normalization (+ optional weight).
  var j: u32 = tid;
  loop {
    if (j >= DIM) { break; }
    let xv = ${tn(`x[base + j]`, r)};
    ${
      n
        ? `let wv = ${tn(`w[j]`, i)};
    y[base + j] = ${nn(`xv * scale * wv`, a)};`
        : `y[base + j] = ${nn(`xv * scale`, a)};`
    }
    j = j + WG;
  }
}
`;
}
function on({ headDim: e, activationDtype: t = `float32` }) {
  let n = e / 2,
    r = Math.min(64, n),
    i = $t(t);
  return `${en(t)}struct Params { seq: u32, heads: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> q: array<${i}>;
@group(0) @binding(1) var<storage, read>       cosTbl: array<f32>;
@group(0) @binding(2) var<storage, read>       sinTbl: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

const HEAD_DIM: u32 = ${e}u;
const HALF_DIM: u32 = ${n}u;
const WG: u32 = ${r}u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  if (t >= params.seq || h >= params.heads) { return; }
  let tid = lid.x;
  let qBase = (t * params.heads + h) * HEAD_DIM;
  let csBase = t * HALF_DIM;

  var k: u32 = tid;
  loop {
    if (k >= HALF_DIM) { break; }
    let c = cosTbl[csBase + k];
    let s = sinTbl[csBase + k];
    let x0 = ${tn(`q[qBase + k]`, t)};
    let x1 = ${tn(`q[qBase + k + HALF_DIM]`, t)};
    q[qBase + k] = ${nn(`x0 * c - x1 * s`, t)};
    q[qBase + k + HALF_DIM] = ${nn(`x1 * c + x0 * s`, t)};
    k = k + WG;
  }
}
`;
}
function sn({ dtype: e = `float32` } = {}) {
  let t = $t(e);
  return `${en(e)}struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<${t}>;
@group(0) @binding(1) var<uniform>             params: Params;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) { return; }
  let v = ${tn(`x[i]`, e)};
  x[i] = ${nn(`v / (1.0 + exp(-v))`, e)};
}
`;
}
function cn(e, t = 64) {
  let n = Math.ceil(e / t),
    r = Math.min(n, 1024);
  return { wgX: Math.ceil(n / r), wgY: r };
}
function ln({
  mlpInner: e,
  inputDtype: t = `float32`,
  outputDtype: n = `float32`,
}) {
  if (t === `float16` && n === `float16` && e % 4 == 0)
    return `enable f16;
struct Params { rows: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> y: array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params: Params;
const MLP_V4: u32 = ${e / 4}u;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= params.rows) { return; }
  let i0 = wg.y * WG + lid.x;
  if (i0 >= MLP_V4) { return; }
  let row_base = r * 2u * MLP_V4;
  let x1 = vec4<f32>(x[row_base + i0]);
  let x2 = vec4<f32>(x[row_base + MLP_V4 + i0]);
  y[r * MLP_V4 + i0] = vec4<f16>((x1 / (vec4<f32>(1.0) + exp(-x1))) * x2);
}
`;
  let r = $t(t),
    i = $t(n);
  return `${en(t, n)}struct Params { rows: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${r}>;
@group(0) @binding(1) var<storage, read_write> y: array<${i}>;
@group(0) @binding(2) var<uniform>             params: Params;
const MLP: u32 = ${e}u;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= params.rows) { return; }
  let i0 = wg.y * WG + lid.x;
  if (i0 >= MLP) { return; }
  let x1 = ${tn(`x[r * 2u * MLP + i0]`, t)};
  let x2 = ${tn(`x[r * 2u * MLP + MLP + i0]`, t)};
  y[r * MLP + i0] = ${nn(`(x1 / (1.0 + exp(-x1))) * x2`, n)};
}
`;
}
function un({ xDtype: e = `float32`, yDtype: t = `float32` } = {}) {
  let n = $t(e),
    r = $t(t);
  return `${en(e, t)}struct Params { count: u32, alpha: f32, wgY: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<${n}>;
@group(0) @binding(1) var<storage, read>       y: array<${r}>;
@group(0) @binding(2) var<uniform>             params: Params;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) { return; }
  x[i] = ${nn(`${tn(`x[i]`, e)} + params.alpha * ${tn(`y[i]`, t)}`, e)};
}
`;
}
function dn({ xDtype: e = `float32`, factorDtype: t = `float32` } = {}) {
  let n = $t(e),
    r = $t(t);
  return `${en(e, t)}struct Params { count: u32, period: u32, wgY: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<${n}>;
@group(0) @binding(1) var<storage, read>       factor: array<${r}>;
@group(0) @binding(2) var<uniform>             params: Params;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) { return; }
  let pIdx = select(i, i % params.period, params.period > 0u);
  x[i] = ${nn(`${tn(`x[i]`, e)} * ${tn(`factor[pIdx]`, t)}`, e)};
}
`;
}
function fn({
  headDim: e,
  inputDtype: t = `float32`,
  outputDtype: n = `float32`,
}) {
  let r = $t(t),
    i = $t(n);
  return `${en(t, n)}struct Params { seq: u32, qHeads: u32, kvHeads: u32, causal: u32 };
@group(0) @binding(0) var<storage, read>       q: array<${r}>;
@group(0) @binding(1) var<storage, read>       k: array<${r}>;
@group(0) @binding(2) var<storage, read>       v: array<${r}>;
@group(0) @binding(3) var<storage, read_write> out: array<${i}>;
@group(0) @binding(4) var<uniform>             params: Params;

const HEAD_DIM: u32 = ${e}u;
const WG: u32 = 64u;
const SCALE: f32 = ${(1 / Math.sqrt(e)).toFixed(8)};

var<workgroup> partial: array<f32, WG>;
var<workgroup> reduced_scalar: f32;
var<workgroup> running_max: f32;
var<workgroup> running_denom: f32;
var<workgroup> running_out: array<f32, ${e}>;

fn reduce_sum(tid: u32) -> f32 {
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;
  let h  = wg.y;
  if (qi >= params.seq || h >= params.qHeads) { return; }
  let tid = lid.x;
  let groupSize = params.qHeads / params.kvHeads;
  let hKv = h / groupSize;

  let qBase = (qi * params.qHeads + h) * HEAD_DIM;

  if (tid == 0u) {
    running_max = -3.4e38;
    running_denom = 0.0;
  }
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) { running_out[d] = 0.0; }
  workgroupBarrier();

  let maxKj = select(params.seq, qi + 1u, params.causal != 0u);

  for (var kj: u32 = 0u; kj < maxKj; kj = kj + 1u) {
    let kBase = (kj * params.kvHeads + hKv) * HEAD_DIM;
    // dot(Q[qi,h], K[kj,hKv])
    var acc: f32 = 0.0;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      acc = acc + ${tn(`q[qBase + d]`, t)} * ${tn(`k[kBase + d]`, t)};
    }
    partial[tid] = acc;
    let dot = reduce_sum(tid) * SCALE;

    // Online softmax update.
    if (tid == 0u) {
      let mNew = max(running_max, dot);
      let exp_old = exp(running_max - mNew);
      let exp_new = exp(dot - mNew);
      reduced_scalar = exp_old; // reuse: factor to scale previous outputs
      running_max = mNew;
      running_denom = running_denom * exp_old + exp_new;
    }
    workgroupBarrier();
    let scaleOld = reduced_scalar;
    let probNew = exp(dot - running_max);
    // Update running_out: scale prior accum + add probNew * V[kj]
    let vBase = (kj * params.kvHeads + hKv) * HEAD_DIM;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      running_out[d] = running_out[d] * scaleOld + probNew * ${tn(`v[vBase + d]`, t)};
    }
    workgroupBarrier();
  }

  // Write final.
  let outBase = (qi * params.qHeads + h) * HEAD_DIM;
  let inv = 1.0 / running_denom;
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    out[outBase + d] = ${nn(`running_out[d] * inv`, n)};
  }
}
`;
}
function pn({
  headDim: e,
  kTile: t = 64,
  inputDtype: n = `float32`,
  outputDtype: r = `float32`,
  useSubgroups: i = !1,
}) {
  if (e % 4 != 0) throw Error(`headDim must be divisible by 4`);
  let a = t;
  if (a > 256)
    throw Error(
      `flash attn kTile=${t} exceeds maxComputeInvocationsPerWorkgroup`,
    );
  let o = e / 4,
    s = $t(n),
    c = $t(r),
    l = i ? `, sg_size: u32` : ``,
    u = i ? `, sg_size` : ``,
    d = i
      ? `enable subgroups;
`
      : ``,
    f = i
      ? `  if (sg_size == WG) {
    return subgroupMax(value);
  }
`
      : ``,
    p = i
      ? `  if (sg_size == WG) {
    return subgroupAdd(value);
  }
`
      : ``;
  return `${en(n, r)}${d}struct Params { seq: u32, qHeads: u32, kvHeads: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       q: array<vec4<${s}>>;
@group(0) @binding(1) var<storage, read>       k: array<vec4<${s}>>;
@group(0) @binding(2) var<storage, read>       v: array<vec4<${s}>>;
@group(0) @binding(3) var<storage, read_write> out: array<${c}>;
@group(0) @binding(4) var<uniform>             params: Params;

const HEAD_DIM:    u32 = ${e}u;
const HEAD_DIM_V4: u32 = ${o}u;
const K_TILE:      u32 = ${t}u;
const WG:          u32 = ${a}u;
const SCALE:       f32 = ${(1 / Math.sqrt(e)).toFixed(8)};
const NEG_INF:     f32 = -3.4e38;

var<workgroup> q_shared:     array<vec4<f32>, HEAD_DIM_V4>;
var<workgroup> probs:        array<f32, K_TILE>;
var<workgroup> running_out:  array<f32, HEAD_DIM>;
var<workgroup> running_max:  f32;
var<workgroup> running_denom: f32;
var<workgroup> scale_old:    f32;
var<workgroup> reduce_buf:   array<f32, K_TILE>;

fn reduce_max(value: f32, tid: u32${l}) -> f32 {
${f}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + stride]);
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

fn reduce_sum(value: f32, tid: u32${l}) -> f32 {
${p}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>${i ? `, @builtin(subgroup_size) sg_size: u32` : ``}) {
  let qi = wg.x;
  let h  = wg.y;
  if (qi >= params.seq || h >= params.qHeads) { return; }
  let tid = lid.x;
  let kvGroupSize = params.qHeads / params.kvHeads;
  let hKv = h / kvGroupSize;

  // Load Q row (as vec4) into wgmem (cooperative).
  let qBaseV4 = (qi * params.qHeads + h) * HEAD_DIM_V4;
  for (var d: u32 = tid; d < HEAD_DIM_V4; d = d + WG) {
    q_shared[d] = ${rn(`q[qBaseV4 + d]`, n)};
  }
  // Init accumulators.
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    running_out[d] = 0.0;
  }
  if (tid == 0u) {
    running_max = NEG_INF;
    running_denom = 0.0;
  }
  workgroupBarrier();

  let seqLocal = params.seq;
  var kj_base: u32 = 0u;
  loop {
    if (kj_base >= seqLocal) { break; }
    let kj = kj_base + tid;
    var dot_val: f32 = NEG_INF;
    if (kj < seqLocal) {
      let kBaseV4 = (kj * params.kvHeads + hKv) * HEAD_DIM_V4;
      var acc: f32 = 0.0;
      for (var d: u32 = 0u; d < HEAD_DIM_V4; d = d + 1u) {
        acc = acc + dot(q_shared[d], ${rn(`k[kBaseV4 + d]`, n)});
      }
      dot_val = acc * SCALE;
    }
    let tile_max = reduce_max(dot_val, tid${u});
    if (tid == 0u) {
      let new_max = max(running_max, tile_max);
      scale_old = exp(running_max - new_max);
      running_max = new_max;
    }
    workgroupBarrier();

    var prob_val: f32 = 0.0;
    if (kj < seqLocal) {
      prob_val = exp(dot_val - running_max);
    }
    probs[tid] = prob_val;

    let tile_sum = reduce_sum(prob_val, tid${u});
    if (tid == 0u) {
      running_denom = running_denom * scale_old + tile_sum;
    }
    workgroupBarrier();

    // Update running_out using vec4 V loads. Each thread handles HEAD_DIM_V4 / WG groups of 4 dims.
    let tile_count = min(K_TILE, seqLocal - kj_base);
    let scale_old_v = scale_old;
    for (var d4: u32 = tid; d4 < HEAD_DIM_V4; d4 = d4 + WG) {
      var v_sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      for (var i: u32 = 0u; i < tile_count; i = i + 1u) {
        let vBaseV4 = ((kj_base + i) * params.kvHeads + hKv) * HEAD_DIM_V4;
        v_sum = v_sum + probs[i] * ${rn(`v[vBaseV4 + d4]`, n)};
      }
      let dBase = d4 * 4u;
      running_out[dBase + 0u] = running_out[dBase + 0u] * scale_old_v + v_sum.x;
      running_out[dBase + 1u] = running_out[dBase + 1u] * scale_old_v + v_sum.y;
      running_out[dBase + 2u] = running_out[dBase + 2u] * scale_old_v + v_sum.z;
      running_out[dBase + 3u] = running_out[dBase + 3u] * scale_old_v + v_sum.w;
    }
    workgroupBarrier();

    kj_base = kj_base + K_TILE;
  }

  let outBase = (qi * params.qHeads + h) * HEAD_DIM;
  let inv = 1.0 / running_denom;
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    out[outBase + d] = ${nn(`running_out[d] * inv`, r)};
  }
}
`;
}
function mn({
  headDim: e,
  kTile: t = 32,
  inputDtype: n = `float32`,
  outputDtype: r = `float32`,
  useSubgroups: i = !1,
  useHalfQk: a = !1,
}) {
  if (e % 4 != 0) throw Error(`headDim must be divisible by 4`);
  if (a && n !== `float16`)
    throw Error(`useHalfQk requires float16 attention inputs`);
  let o = t;
  if (o > 256)
    throw Error(
      `flash attn kTile=${t} exceeds maxComputeInvocationsPerWorkgroup`,
    );
  let s = e / 4,
    c = $t(n),
    l = $t(r),
    u = a ? `f16` : `f32`,
    d = (e) => (a ? e : rn(e, n)),
    f = (e, t) => (a ? `f32(dot(${e}, ${t}))` : `dot(${e}, ${t})`),
    p = i ? `, sg_size: u32` : ``,
    m = i ? `, sg_size` : ``,
    h = i
      ? `enable subgroups;
`
      : ``,
    g = i
      ? `  if (sg_size == WG) {
    return subgroupMax(value);
  }
`
      : ``,
    _ = i
      ? `  if (sg_size == WG) {
    return subgroupAdd(value);
  }
`
      : ``;
  return `${en(n, r)}${h}struct Params { seq: u32, qHeads: u32, kvHeads: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       q: array<vec4<${c}>>;
@group(0) @binding(1) var<storage, read>       k: array<vec4<${c}>>;
@group(0) @binding(2) var<storage, read>       v: array<vec4<${c}>>;
@group(0) @binding(3) var<storage, read_write> out: array<${l}>;
@group(0) @binding(4) var<uniform>             params: Params;

const HEAD_DIM:    u32 = ${e}u;
const HEAD_DIM_V4: u32 = ${s}u;
const K_TILE:      u32 = ${t}u;
const WG:          u32 = ${o}u;
const SCALE:       f32 = ${(1 / Math.sqrt(e)).toFixed(8)};
const NEG_INF:     f32 = -3.4e38;

var<workgroup> q0_shared:     array<vec4<${u}>, HEAD_DIM_V4>;
var<workgroup> q1_shared:     array<vec4<${u}>, HEAD_DIM_V4>;
var<workgroup> probs0:        array<f32, K_TILE>;
var<workgroup> probs1:        array<f32, K_TILE>;
var<workgroup> running_out0:  array<f32, HEAD_DIM>;
var<workgroup> running_out1:  array<f32, HEAD_DIM>;
var<workgroup> running_max0:  f32;
var<workgroup> running_max1:  f32;
var<workgroup> running_denom0: f32;
var<workgroup> running_denom1: f32;
var<workgroup> scale_old0:    f32;
var<workgroup> scale_old1:    f32;
var<workgroup> reduce_buf:    array<f32, K_TILE>;

fn reduce_max(value: f32, tid: u32${p}) -> f32 {
${g}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + stride]);
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

fn reduce_sum(value: f32, tid: u32${p}) -> f32 {
${_}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>${i ? `, @builtin(subgroup_size) sg_size: u32` : ``}) {
  let qi0 = wg.x * 2u;
  let qi1 = qi0 + 1u;
  let h = wg.y;
  if (qi0 >= params.seq || h >= params.qHeads) { return; }
  let tid = lid.x;
  let q1_valid = qi1 < params.seq;
  let kvGroupSize = params.qHeads / params.kvHeads;
  let hKv = h / kvGroupSize;

  let q0BaseV4 = (qi0 * params.qHeads + h) * HEAD_DIM_V4;
  let q1BaseV4 = (qi1 * params.qHeads + h) * HEAD_DIM_V4;
  for (var d: u32 = tid; d < HEAD_DIM_V4; d = d + WG) {
    q0_shared[d] = ${d(`q[q0BaseV4 + d]`)};
    if (q1_valid) {
      q1_shared[d] = ${d(`q[q1BaseV4 + d]`)};
    } else {
      q1_shared[d] = vec4<${u}>(0.0);
    }
  }
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    running_out0[d] = 0.0;
    running_out1[d] = 0.0;
  }
  if (tid == 0u) {
    running_max0 = NEG_INF;
    running_max1 = NEG_INF;
    running_denom0 = 0.0;
    running_denom1 = 0.0;
  }
  workgroupBarrier();

  let seqLocal = params.seq;
  var kj_base: u32 = 0u;
  loop {
    if (kj_base >= seqLocal) { break; }
    let kj = kj_base + tid;
    var dot0: f32 = NEG_INF;
    var dot1: f32 = NEG_INF;
    if (kj < seqLocal) {
      let kBaseV4 = (kj * params.kvHeads + hKv) * HEAD_DIM_V4;
      var acc0: f32 = 0.0;
      var acc1: f32 = 0.0;
      for (var d: u32 = 0u; d < HEAD_DIM_V4; d = d + 1u) {
        let kval = ${d(`k[kBaseV4 + d]`)};
        acc0 = acc0 + ${f(`q0_shared[d]`, `kval`)};
        acc1 = acc1 + ${f(`q1_shared[d]`, `kval`)};
      }
      dot0 = acc0 * SCALE;
      if (q1_valid) {
        dot1 = acc1 * SCALE;
      }
    }

    let tile_max0 = reduce_max(dot0, tid${m});
    let tile_max1 = reduce_max(dot1, tid${m});
    if (tid == 0u) {
      let new_max0 = max(running_max0, tile_max0);
      scale_old0 = exp(running_max0 - new_max0);
      running_max0 = new_max0;
      let new_max1 = max(running_max1, tile_max1);
      scale_old1 = exp(running_max1 - new_max1);
      running_max1 = new_max1;
    }
    workgroupBarrier();

    var prob0: f32 = 0.0;
    var prob1: f32 = 0.0;
    if (kj < seqLocal) {
      prob0 = exp(dot0 - running_max0);
      if (q1_valid) {
        prob1 = exp(dot1 - running_max1);
      }
    }
    probs0[tid] = prob0;
    probs1[tid] = prob1;

    let tile_sum0 = reduce_sum(prob0, tid${m});
    let tile_sum1 = reduce_sum(prob1, tid${m});
    if (tid == 0u) {
      running_denom0 = running_denom0 * scale_old0 + tile_sum0;
      running_denom1 = running_denom1 * scale_old1 + tile_sum1;
    }
    workgroupBarrier();

    let tile_count = min(K_TILE, seqLocal - kj_base);
    let scale_old_v0 = scale_old0;
    let scale_old_v1 = scale_old1;
    for (var d4: u32 = tid; d4 < HEAD_DIM_V4; d4 = d4 + WG) {
      var v_sum0: vec4<f32> = vec4<f32>(0.0);
      var v_sum1: vec4<f32> = vec4<f32>(0.0);
      for (var i: u32 = 0u; i < tile_count; i = i + 1u) {
        let vBaseV4 = ((kj_base + i) * params.kvHeads + hKv) * HEAD_DIM_V4;
        let vval = ${rn(`v[vBaseV4 + d4]`, n)};
        v_sum0 = v_sum0 + probs0[i] * vval;
        v_sum1 = v_sum1 + probs1[i] * vval;
      }
      let dBase = d4 * 4u;
      running_out0[dBase + 0u] = running_out0[dBase + 0u] * scale_old_v0 + v_sum0.x;
      running_out0[dBase + 1u] = running_out0[dBase + 1u] * scale_old_v0 + v_sum0.y;
      running_out0[dBase + 2u] = running_out0[dBase + 2u] * scale_old_v0 + v_sum0.z;
      running_out0[dBase + 3u] = running_out0[dBase + 3u] * scale_old_v0 + v_sum0.w;
      running_out1[dBase + 0u] = running_out1[dBase + 0u] * scale_old_v1 + v_sum1.x;
      running_out1[dBase + 1u] = running_out1[dBase + 1u] * scale_old_v1 + v_sum1.y;
      running_out1[dBase + 2u] = running_out1[dBase + 2u] * scale_old_v1 + v_sum1.z;
      running_out1[dBase + 3u] = running_out1[dBase + 3u] * scale_old_v1 + v_sum1.w;
    }
    workgroupBarrier();

    kj_base = kj_base + K_TILE;
  }

  let outBase0 = (qi0 * params.qHeads + h) * HEAD_DIM;
  let inv0 = 1.0 / running_denom0;
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    out[outBase0 + d] = ${nn(`running_out0[d] * inv0`, r)};
  }
  if (q1_valid) {
    let outBase1 = (qi1 * params.qHeads + h) * HEAD_DIM;
    let inv1 = 1.0 / running_denom1;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      out[outBase1 + d] = ${nn(`running_out1[d] * inv1`, r)};
    }
  }
}
`;
}
function hn({
  headDim: e,
  kTile: t = 32,
  inputDtype: n = `float16`,
  outputDtype: r = `float32`,
  useSubgroups: i = !1,
  useHalfQk: a = !0,
}) {
  if (e % 4 != 0) throw Error(`headDim must be divisible by 4`);
  if (n !== `float16` || !a)
    throw Error(`Q4 flash attention requires f16 Q/K inputs`);
  let o = t;
  if (o > 256)
    throw Error(
      `flash attn kTile=${t} exceeds maxComputeInvocationsPerWorkgroup`,
    );
  let s = e / 4,
    c = $t(n),
    l = $t(r),
    u = i ? `, sg_size: u32` : ``,
    d = i ? `, sg_size` : ``,
    f = i
      ? `enable subgroups;
`
      : ``,
    p = i
      ? `  if (sg_size == WG) {
    return subgroupMax(value);
  }
`
      : ``,
    m = i
      ? `  if (sg_size == WG) {
    return subgroupAdd(value);
  }
`
      : ``;
  return `${en(n, r)}${f}struct Params { seq: u32, qHeads: u32, kvHeads: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       q: array<vec4<${c}>>;
@group(0) @binding(1) var<storage, read>       k: array<vec4<${c}>>;
@group(0) @binding(2) var<storage, read>       v: array<vec4<${c}>>;
@group(0) @binding(3) var<storage, read_write> out: array<${l}>;
@group(0) @binding(4) var<uniform>             params: Params;

const HEAD_DIM:    u32 = ${e}u;
const HEAD_DIM_V4: u32 = ${s}u;
const K_TILE:      u32 = ${t}u;
const WG:          u32 = ${o}u;
const SCALE:       f32 = ${(1 / Math.sqrt(e)).toFixed(8)};
const NEG_INF:     f32 = -3.4e38;

var<workgroup> q0_shared:     array<vec4<f16>, HEAD_DIM_V4>;
var<workgroup> q1_shared:     array<vec4<f16>, HEAD_DIM_V4>;
var<workgroup> q2_shared:     array<vec4<f16>, HEAD_DIM_V4>;
var<workgroup> q3_shared:     array<vec4<f16>, HEAD_DIM_V4>;
var<workgroup> probs0:        array<f32, K_TILE>;
var<workgroup> probs1:        array<f32, K_TILE>;
var<workgroup> probs2:        array<f32, K_TILE>;
var<workgroup> probs3:        array<f32, K_TILE>;
var<workgroup> running_out0:  array<f32, HEAD_DIM>;
var<workgroup> running_out1:  array<f32, HEAD_DIM>;
var<workgroup> running_out2:  array<f32, HEAD_DIM>;
var<workgroup> running_out3:  array<f32, HEAD_DIM>;
var<workgroup> running_max0:  f32;
var<workgroup> running_max1:  f32;
var<workgroup> running_max2:  f32;
var<workgroup> running_max3:  f32;
var<workgroup> running_denom0: f32;
var<workgroup> running_denom1: f32;
var<workgroup> running_denom2: f32;
var<workgroup> running_denom3: f32;
var<workgroup> scale_old0:    f32;
var<workgroup> scale_old1:    f32;
var<workgroup> scale_old2:    f32;
var<workgroup> scale_old3:    f32;
var<workgroup> reduce_buf:    array<f32, K_TILE>;

fn reduce_max(value: f32, tid: u32${u}) -> f32 {
${p}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + stride]);
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

fn reduce_sum(value: f32, tid: u32${u}) -> f32 {
${m}  reduce_buf[tid] = value;
  workgroupBarrier();
  var stride: u32 = K_TILE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return reduce_buf[0];
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>${i ? `, @builtin(subgroup_size) sg_size: u32` : ``}) {
  let qi0 = wg.x * 4u;
  let qi1 = qi0 + 1u;
  let qi2 = qi0 + 2u;
  let qi3 = qi0 + 3u;
  let h = wg.y;
  if (qi0 >= params.seq || h >= params.qHeads) { return; }
  let tid = lid.x;
  let q1_valid = qi1 < params.seq;
  let q2_valid = qi2 < params.seq;
  let q3_valid = qi3 < params.seq;
  let kvGroupSize = params.qHeads / params.kvHeads;
  let hKv = h / kvGroupSize;

  let q0BaseV4 = (qi0 * params.qHeads + h) * HEAD_DIM_V4;
  let q1BaseV4 = (qi1 * params.qHeads + h) * HEAD_DIM_V4;
  let q2BaseV4 = (qi2 * params.qHeads + h) * HEAD_DIM_V4;
  let q3BaseV4 = (qi3 * params.qHeads + h) * HEAD_DIM_V4;
  for (var d: u32 = tid; d < HEAD_DIM_V4; d = d + WG) {
    q0_shared[d] = q[q0BaseV4 + d];
    if (q1_valid) { q1_shared[d] = q[q1BaseV4 + d]; } else { q1_shared[d] = vec4<f16>(0.0); }
    if (q2_valid) { q2_shared[d] = q[q2BaseV4 + d]; } else { q2_shared[d] = vec4<f16>(0.0); }
    if (q3_valid) { q3_shared[d] = q[q3BaseV4 + d]; } else { q3_shared[d] = vec4<f16>(0.0); }
  }
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    running_out0[d] = 0.0;
    running_out1[d] = 0.0;
    running_out2[d] = 0.0;
    running_out3[d] = 0.0;
  }
  if (tid == 0u) {
    running_max0 = NEG_INF;
    running_max1 = NEG_INF;
    running_max2 = NEG_INF;
    running_max3 = NEG_INF;
    running_denom0 = 0.0;
    running_denom1 = 0.0;
    running_denom2 = 0.0;
    running_denom3 = 0.0;
  }
  workgroupBarrier();

  let seqLocal = params.seq;
  var kj_base: u32 = 0u;
  loop {
    if (kj_base >= seqLocal) { break; }
    let kj = kj_base + tid;
    var dot0: f32 = NEG_INF;
    var dot1: f32 = NEG_INF;
    var dot2: f32 = NEG_INF;
    var dot3: f32 = NEG_INF;
    if (kj < seqLocal) {
      let kBaseV4 = (kj * params.kvHeads + hKv) * HEAD_DIM_V4;
      var acc0: f32 = 0.0;
      var acc1: f32 = 0.0;
      var acc2: f32 = 0.0;
      var acc3: f32 = 0.0;
      for (var d: u32 = 0u; d < HEAD_DIM_V4; d = d + 1u) {
        let kval = k[kBaseV4 + d];
        acc0 = acc0 + f32(dot(q0_shared[d], kval));
        acc1 = acc1 + f32(dot(q1_shared[d], kval));
        acc2 = acc2 + f32(dot(q2_shared[d], kval));
        acc3 = acc3 + f32(dot(q3_shared[d], kval));
      }
      dot0 = acc0 * SCALE;
      if (q1_valid) { dot1 = acc1 * SCALE; }
      if (q2_valid) { dot2 = acc2 * SCALE; }
      if (q3_valid) { dot3 = acc3 * SCALE; }
    }

    let tile_max0 = reduce_max(dot0, tid${d});
    let tile_max1 = reduce_max(dot1, tid${d});
    let tile_max2 = reduce_max(dot2, tid${d});
    let tile_max3 = reduce_max(dot3, tid${d});
    if (tid == 0u) {
      let new_max0 = max(running_max0, tile_max0);
      scale_old0 = exp(running_max0 - new_max0);
      running_max0 = new_max0;
      let new_max1 = max(running_max1, tile_max1);
      scale_old1 = exp(running_max1 - new_max1);
      running_max1 = new_max1;
      let new_max2 = max(running_max2, tile_max2);
      scale_old2 = exp(running_max2 - new_max2);
      running_max2 = new_max2;
      let new_max3 = max(running_max3, tile_max3);
      scale_old3 = exp(running_max3 - new_max3);
      running_max3 = new_max3;
    }
    workgroupBarrier();

    var prob0: f32 = 0.0;
    var prob1: f32 = 0.0;
    var prob2: f32 = 0.0;
    var prob3: f32 = 0.0;
    if (kj < seqLocal) {
      prob0 = exp(dot0 - running_max0);
      if (q1_valid) { prob1 = exp(dot1 - running_max1); }
      if (q2_valid) { prob2 = exp(dot2 - running_max2); }
      if (q3_valid) { prob3 = exp(dot3 - running_max3); }
    }
    probs0[tid] = prob0;
    probs1[tid] = prob1;
    probs2[tid] = prob2;
    probs3[tid] = prob3;

    let tile_sum0 = reduce_sum(prob0, tid${d});
    let tile_sum1 = reduce_sum(prob1, tid${d});
    let tile_sum2 = reduce_sum(prob2, tid${d});
    let tile_sum3 = reduce_sum(prob3, tid${d});
    if (tid == 0u) {
      running_denom0 = running_denom0 * scale_old0 + tile_sum0;
      running_denom1 = running_denom1 * scale_old1 + tile_sum1;
      running_denom2 = running_denom2 * scale_old2 + tile_sum2;
      running_denom3 = running_denom3 * scale_old3 + tile_sum3;
    }
    workgroupBarrier();

    let tile_count = min(K_TILE, seqLocal - kj_base);
    let scale_old_v0 = scale_old0;
    let scale_old_v1 = scale_old1;
    let scale_old_v2 = scale_old2;
    let scale_old_v3 = scale_old3;
    for (var d4: u32 = tid; d4 < HEAD_DIM_V4; d4 = d4 + WG) {
      var v_sum0: vec4<f32> = vec4<f32>(0.0);
      var v_sum1: vec4<f32> = vec4<f32>(0.0);
      var v_sum2: vec4<f32> = vec4<f32>(0.0);
      var v_sum3: vec4<f32> = vec4<f32>(0.0);
      for (var i: u32 = 0u; i < tile_count; i = i + 1u) {
        let vBaseV4 = ((kj_base + i) * params.kvHeads + hKv) * HEAD_DIM_V4;
        let vval = ${rn(`v[vBaseV4 + d4]`, n)};
        v_sum0 = v_sum0 + probs0[i] * vval;
        v_sum1 = v_sum1 + probs1[i] * vval;
        v_sum2 = v_sum2 + probs2[i] * vval;
        v_sum3 = v_sum3 + probs3[i] * vval;
      }
      let dBase = d4 * 4u;
      running_out0[dBase + 0u] = running_out0[dBase + 0u] * scale_old_v0 + v_sum0.x;
      running_out0[dBase + 1u] = running_out0[dBase + 1u] * scale_old_v0 + v_sum0.y;
      running_out0[dBase + 2u] = running_out0[dBase + 2u] * scale_old_v0 + v_sum0.z;
      running_out0[dBase + 3u] = running_out0[dBase + 3u] * scale_old_v0 + v_sum0.w;
      running_out1[dBase + 0u] = running_out1[dBase + 0u] * scale_old_v1 + v_sum1.x;
      running_out1[dBase + 1u] = running_out1[dBase + 1u] * scale_old_v1 + v_sum1.y;
      running_out1[dBase + 2u] = running_out1[dBase + 2u] * scale_old_v1 + v_sum1.z;
      running_out1[dBase + 3u] = running_out1[dBase + 3u] * scale_old_v1 + v_sum1.w;
      running_out2[dBase + 0u] = running_out2[dBase + 0u] * scale_old_v2 + v_sum2.x;
      running_out2[dBase + 1u] = running_out2[dBase + 1u] * scale_old_v2 + v_sum2.y;
      running_out2[dBase + 2u] = running_out2[dBase + 2u] * scale_old_v2 + v_sum2.z;
      running_out2[dBase + 3u] = running_out2[dBase + 3u] * scale_old_v2 + v_sum2.w;
      running_out3[dBase + 0u] = running_out3[dBase + 0u] * scale_old_v3 + v_sum3.x;
      running_out3[dBase + 1u] = running_out3[dBase + 1u] * scale_old_v3 + v_sum3.y;
      running_out3[dBase + 2u] = running_out3[dBase + 2u] * scale_old_v3 + v_sum3.z;
      running_out3[dBase + 3u] = running_out3[dBase + 3u] * scale_old_v3 + v_sum3.w;
    }
    workgroupBarrier();

    kj_base = kj_base + K_TILE;
  }

  let outBase0 = (qi0 * params.qHeads + h) * HEAD_DIM;
  let inv0 = 1.0 / running_denom0;
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    out[outBase0 + d] = ${nn(`running_out0[d] * inv0`, r)};
  }
  if (q1_valid) {
    let outBase1 = (qi1 * params.qHeads + h) * HEAD_DIM;
    let inv1 = 1.0 / running_denom1;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      out[outBase1 + d] = ${nn(`running_out1[d] * inv1`, r)};
    }
  }
  if (q2_valid) {
    let outBase2 = (qi2 * params.qHeads + h) * HEAD_DIM;
    let inv2 = 1.0 / running_denom2;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      out[outBase2 + d] = ${nn(`running_out2[d] * inv2`, r)};
    }
  }
  if (q3_valid) {
    let outBase3 = (qi3 * params.qHeads + h) * HEAD_DIM;
    let inv3 = 1.0 / running_denom3;
    for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
      out[outBase3 + d] = ${nn(`running_out3[d] * inv3`, r)};
    }
  }
}
`;
}
function gn({
  headDim: e,
  kStep: t = 32,
  inputDtype: n = `float16`,
  outputDtype: r = `float16`,
}) {
  if (e % 4 != 0) throw Error(`headDim must be divisible by 4`);
  if (n !== `float16`)
    throw Error(`Q32 flash attention requires f16 Q/K/V inputs`);
  if (t !== 32 && t !== 64)
    throw Error(`Q32 flash attention supports kStep=32 or 64`);
  let i = e / 4,
    a = $t(n),
    o = $t(r),
    s = [`x`, `y`, `z`, `w`],
    c = t / 4,
    l = Array.from(
      { length: c },
      (e, t) => `    var qk${t}: vec4<f32> = vec4<f32>(0.0);`,
    ).join(`
`),
    u = Array.from({ length: t }, (e, t) => {
      let n = Math.floor(t / 4),
        r = s[t % 4];
      return `        qk${n}.${r} = qk${n}.${r} + f32(dot(q_own, subgroupShuffle(${t < 32 ? `k_local0` : `k_local1`}, ${t % 32}u)));`;
    }).join(`
`),
    d = Array.from({ length: c }, (e, t) => {
      let n = [`    qk${t} = qk${t} * vec4<f32>(SCALE);`];
      for (let e = 0; e < 4; ++e) {
        let r = t * 4 + e;
        n.push(
          `    if (k_start + ${r}u >= seqLocal) { qk${t}.${s[e]} = NEG_INF; }`,
        );
      }
      return n.join(`
`);
    }).join(`
`),
    f = Array.from({ length: t }, (e, t) => {
      let n = Math.floor(t / 4),
        r = s[t % 4];
      return `      acc = acc + vec4<f32>(subgroupShuffle(${t < 32 ? `v_local0` : `v_local1`}, ${t % 32}u)) * qk${n}.${r};`;
    }).join(`
`),
    p =
      t === 32
        ? `      var k_local0: vec4<f16> = vec4<f16>(0.0h);
      if (k_start + sg_id < seqLocal) {
        let kBaseV4 = ((k_start + sg_id) * params.kvHeads + hKv) * HEAD_DIM_V4;
        k_local0 = k[kBaseV4 + d4];
      }`
        : `      var k_local0: vec4<f16> = vec4<f16>(0.0h);
      var k_local1: vec4<f16> = vec4<f16>(0.0h);
      if (k_start + sg_id < seqLocal) {
        let kBaseV4 = ((k_start + sg_id) * params.kvHeads + hKv) * HEAD_DIM_V4;
        k_local0 = k[kBaseV4 + d4];
      }
      if (k_start + sg_id + 32u < seqLocal) {
        let kBaseV4 = ((k_start + sg_id + 32u) * params.kvHeads + hKv) * HEAD_DIM_V4;
        k_local1 = k[kBaseV4 + d4];
      }`,
    m =
      t === 32
        ? `      var v_local0: vec4<f16> = vec4<f16>(0.0h);
      if (k_start + sg_id < seqLocal) {
        let vBaseV4 = ((k_start + sg_id) * params.kvHeads + hKv) * HEAD_DIM_V4;
        v_local0 = v[vBaseV4 + d4];
      }`
        : `      var v_local0: vec4<f16> = vec4<f16>(0.0h);
      var v_local1: vec4<f16> = vec4<f16>(0.0h);
      if (k_start + sg_id < seqLocal) {
        let vBaseV4 = ((k_start + sg_id) * params.kvHeads + hKv) * HEAD_DIM_V4;
        v_local0 = v[vBaseV4 + d4];
      }
      if (k_start + sg_id + 32u < seqLocal) {
        let vBaseV4 = ((k_start + sg_id + 32u) * params.kvHeads + hKv) * HEAD_DIM_V4;
        v_local1 = v[vBaseV4 + d4];
      }`,
    h =
      r === `float16`
        ? `    out[outBaseV4 + d4] = vec4<f16>(o_tile[d4]);`
        : `    out[outBaseV4 + d4] = o_tile[d4];`,
    g = Array.from(
      { length: c },
      (e, t) =>
        `    local_max = max(local_max, max(max(qk${t}.x, qk${t}.y), max(qk${t}.z, qk${t}.w)));`,
    ).join(`
`),
    _ = Array.from(
      { length: c },
      (e, t) => `    qk${t} = exp(qk${t} - vec4<f32>(new_max));`,
    ).join(`
`),
    v = Array.from(
      { length: c },
      (e, t) =>
        `    tile_sum = tile_sum + qk${t}.x + qk${t}.y + qk${t}.z + qk${t}.w;`,
    ).join(`
`),
    y = Array.from(
      { length: c },
      (e, t) => `    qk${t} = qk${t} / vec4<f32>(denom);`,
    ).join(`
`);
  return `${en(n, r)}enable subgroups;
struct Params { seq: u32, qHeads: u32, kvHeads: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       q: array<vec4<${a}>>;
@group(0) @binding(1) var<storage, read>       k: array<vec4<${a}>>;
@group(0) @binding(2) var<storage, read>       v: array<vec4<${a}>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<${o}>>;
@group(0) @binding(4) var<uniform>             params: Params;

const HEAD_DIM:    u32 = ${e}u;
const HEAD_DIM_V4: u32 = ${i}u;
const WG:          u32 = 32u;
const K_STEP:      u32 = ${t}u;
const SCALE:       f32 = ${(1 / Math.sqrt(e)).toFixed(8)};
const NEG_INF:     f32 = -3.4e38;

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_id: u32
) {
  let qi = wg.x * WG + lid.x;
  let h = wg.y;
  if (h >= params.qHeads) { return; }
  let valid_q = qi < params.seq;
  let kvGroupSize = params.qHeads / params.kvHeads;
  let hKv = h / kvGroupSize;
  let qBaseV4 = (qi * params.qHeads + h) * HEAD_DIM_V4;

  var q_tile: array<vec4<f16>, HEAD_DIM_V4>;
  var o_tile: array<vec4<f32>, HEAD_DIM_V4>;
  for (var d4: u32 = 0u; d4 < HEAD_DIM_V4; d4 = d4 + 1u) {
    if (valid_q) {
      q_tile[d4] = q[qBaseV4 + d4];
    } else {
      q_tile[d4] = vec4<f16>(0.0h);
    }
    o_tile[d4] = vec4<f32>(0.0);
  }

  var previous_max: f32 = NEG_INF;
  var previous_denom: f32 = 0.0;
  let seqLocal = params.seq;
  for (var k_start: u32 = 0u; k_start < seqLocal; k_start = k_start + K_STEP) {
${l}
    for (var d4: u32 = 0u; d4 < HEAD_DIM_V4; d4 = d4 + 1u) {
      let q_own = q_tile[d4];
${p}
${u}
    }
${d}

    var local_max: f32 = NEG_INF;
${g}
    let new_max = max(previous_max, local_max);
${_}
    var tile_sum: f32 = 0.0;
${v}
    let dleft = previous_denom * exp(previous_max - new_max);
    let denom = max(dleft + tile_sum, 0.0000001);
    let o_ratio = dleft / denom;
${y}
    previous_max = new_max;
    previous_denom = denom;

    for (var d4: u32 = 0u; d4 < HEAD_DIM_V4; d4 = d4 + 1u) {
${m}
      var acc: vec4<f32> = vec4<f32>(0.0);
${f}
      o_tile[d4] = o_tile[d4] * vec4<f32>(o_ratio) + acc;
    }
  }

  if (valid_q) {
    let outBaseV4 = (qi * params.qHeads + h) * HEAD_DIM_V4;
    for (var d4: u32 = 0u; d4 < HEAD_DIM_V4; d4 = d4 + 1u) {
${h}
    }
  }
}
`;
}
function _n({ dtype: e = `float32` } = {}) {
  let t = $t(e);
  return `${en(e)}struct Params { totalElems: u32, aElems: u32, wgY: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       a: array<${t}>;
@group(0) @binding(1) var<storage, read>       b: array<${t}>;
@group(0) @binding(2) var<storage, read_write> out: array<${t}>;
@group(0) @binding(3) var<uniform>             params: Params;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.totalElems) { return; }
  if (i < params.aElems) {
    out[i] = a[i];
  } else {
    out[i] = b[i - params.aElems];
  }
}
`;
}
function vn({
  bits: e,
  groupSize: t,
  inFeatures: n,
  outFeatures: r,
  label: i = `mlx_matmul`,
}) {
  if (![1, 2, 4, 8].includes(e)) throw Error(`unsupported bits=${e}`);
  if (n % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  let a = Math.floor(32 / e);
  if (t % a !== 0) throw Error(`groupSize must be divisible by valsPerWord`);
  let o = t / a,
    s = n / a;
  return `// MLX matmul, bits=${e}, group_size=${t}, in=${n}, out=${r}.
struct Params { M: u32 };
@group(0) @binding(0) var<storage, read>       a:         array<f32>;
@group(0) @binding(1) var<storage, read>       bits_buf:  array<u32>;
@group(0) @binding(2) var<storage, read>       scaleBias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:         array<f32>;
@group(0) @binding(4) var<uniform>             params:    Params;

const IN_FEATURES:     u32 = ${n}u;
const OUT_FEATURES:    u32 = ${r}u;
const GROUP_SIZE:      u32 = ${t}u;
const NUM_GROUPS:      u32 = ${n / t}u;
const WORDS_PER_ROW:   u32 = ${s}u;
const WORDS_PER_GROUP: u32 = ${o}u;
const VALS_PER_WORD:   u32 = ${a}u;
const BITS:            u32 = ${e}u;
const MASK:            u32 = ${(1 << e) - 1}u;
const WG_SIZE:         u32 = 64u;

var<workgroup> partial_acc: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row: u32 = wg.x;            // output feature
  let mrow: u32 = wg.y;           // input row (token index)
  if (row >= OUT_FEATURES || mrow >= params.M) { return; }
  let tid: u32 = lid.x;

  let row_words_base: u32 = row * WORDS_PER_ROW;
  let row_sb_base:    u32 = row * NUM_GROUPS * 2u;
  let a_row_base:     u32 = mrow * IN_FEATURES;

  var thread_acc: f32 = 0.0;
  var g: u32 = tid;
  loop {
    if (g >= NUM_GROUPS) { break; }
    let group_word_base: u32 = row_words_base + g * WORDS_PER_GROUP;
    let col_base: u32 = g * GROUP_SIZE;

    var sum_qa: f32 = 0.0;
    var sum_a:  f32 = 0.0;
    for (var w: u32 = 0u; w < WORDS_PER_GROUP; w = w + 1u) {
      let packed: u32 = bits_buf[group_word_base + w];
      let lane_base: u32 = col_base + w * VALS_PER_WORD;
      for (var v: u32 = 0u; v < VALS_PER_WORD; v = v + 1u) {
        let q: f32 = f32((packed >> (v * BITS)) & MASK);
        let ai: f32 = a[a_row_base + lane_base + v];
        sum_qa = sum_qa + q * ai;
        sum_a  = sum_a  + ai;
      }
    }
    let scale: f32 = scaleBias[row_sb_base + g * 2u];
    let bias:  f32 = scaleBias[row_sb_base + g * 2u + 1u];
    thread_acc = thread_acc + scale * sum_qa + bias * sum_a;

    g = g + WG_SIZE;
  }

  partial_acc[tid] = thread_acc;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partial_acc[tid] = partial_acc[tid] + partial_acc[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }

  if (tid == 0u) {
    y[mrow * OUT_FEATURES + row] = partial_acc[0];
  }
}
`;
}
function yn({
  bits: e,
  groupSize: t,
  inFeatures: n,
  outFeatures: r,
  mTile: i = 4,
  outTile: a = 64,
  kGroupsPerChunk: o = 2,
  nPerThread: s = 1,
  inputDtype: c = `float32`,
  outputDtype: l = `float32`,
  scaleBiasDtype: u = `float32`,
}) {
  if (![1, 2, 4, 8].includes(e)) throw Error(`unsupported bits=${e}`);
  if (n % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  let d = Math.floor(32 / e);
  if (t % d !== 0) throw Error(`groupSize must be divisible by valsPerWord`);
  if (t % 4 != 0)
    throw Error(`groupSize must be divisible by 4 for vec4 loads`);
  let f = t / d,
    p = n / d,
    m = n / t;
  if (m % o !== 0)
    throw Error(`numGroups=${m} not divisible by kGroupsPerChunk=${o}`);
  let h = m / o,
    g = t * o,
    _ = (1 << e) - 1,
    v = i * a,
    y = c === `float16` ? `f16` : `f32`,
    b = u === `float16` ? `f16` : `f32`,
    x = l === `float16` ? `f16` : `f32`,
    S =
      c === `float16` || l === `float16` || u === `float16`
        ? `enable f16;
`
        : ``,
    C = l === `float16` ? `f16(thread_accs[nn])` : `thread_accs[nn]`;
  if (v > 256)
    throw Error(
      `mTile * outTile = ${v} exceeds maxComputeInvocationsPerWorkgroup=256`,
    );
  let w = (i * g) / 4;
  if (d % 4 != 0) throw Error(`valsPerWord must be divisible by 4`);
  let T = d / 4,
    E = a * s,
    D = [];
  for (let t = 0; t < T; t++) {
    let n = (n) => `(packed >> ${(t * 4 + n) * e}u) & ${_}u`;
    D.push(
      `let q${t}: vec4<f32> = vec4<f32>(f32(${n(0)}), f32(${n(1)}), f32(${n(2)}), f32(${n(3)}));`,
    );
  }
  let O = ``;
  for (let e = 0; e < T; e++)
    ((O += `          ${D[e]}
`),
      (O += `          let a${e}: vec4<f32> = vec4<f32>(a_chunk[a_vec_off + ${e}u]);
`),
      (O += `          sum_qa = sum_qa + dot(q${e}, a${e});
`),
      (O += `          sum_a  = sum_a  + dot(vec4<f32>(1.0, 1.0, 1.0, 1.0), a${e});
`));
  return `// MLX matmul tiled, bits=${e}, group_size=${t}, in=${n}, out=${r}, mTile=${i}, outTile=${a}, kGroupsPerChunk=${o}, nPerThread=${s}, input=${c}, scaleBias=${u}, output=${l}.
${S}struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a:         array<vec4<${y}>>;
@group(0) @binding(1) var<storage, read>       bits_buf:  array<u32>;
@group(0) @binding(2) var<storage, read>       scaleBias: array<${b}>;
@group(0) @binding(3) var<storage, read_write> y:         array<${x}>;
@group(0) @binding(4) var<uniform>             params:    Params;

const IN_FEATURES:     u32 = ${n}u;
const IN_FEATURES_V4:  u32 = ${n / 4}u;
const OUT_FEATURES:    u32 = ${r}u;
const GROUP_SIZE:      u32 = ${t}u;
const GROUP_SIZE_V4:   u32 = ${t / 4}u;
const NUM_GROUPS:      u32 = ${m}u;
const WORDS_PER_ROW:   u32 = ${p}u;
const WORDS_PER_GROUP: u32 = ${f}u;
const VEC4_PER_WORD:   u32 = ${T}u;
const BITS:            u32 = ${e}u;
const MASK:            u32 = ${_}u;
const M_TILE:          u32 = ${i}u;
const OUT_TILE:        u32 = ${a}u;
const N_PER_THREAD:    u32 = ${s}u;
const OUT_TILE_TOTAL:  u32 = ${E}u;
const WG_SIZE:         u32 = ${v}u;
const A_CHUNK_VEC4:    u32 = ${w}u;
const NUM_CHUNKS:      u32 = ${h}u;
const K_GROUPS_PER_CHUNK: u32 = ${o}u;
const CHUNK_GROUP_SIZE_V4: u32 = ${g / 4}u;

var<workgroup> a_chunk: array<vec4<${y}>, A_CHUNK_VEC4>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n_block: u32 = wg.x * OUT_TILE_TOTAL;
  let m_block: u32 = wg.y * M_TILE;
  let tid: u32 = lid.x;

  let m_local: u32 = tid / OUT_TILE;   // 0..M_TILE-1
  let n_local0: u32 = tid % OUT_TILE;  // 0..OUT_TILE-1
  let m_row: u32 = m_block + m_local;

  let m_valid: bool = m_row < params.M;

  // Each thread computes N_PER_THREAD outputs, spaced by OUT_TILE within n_block.
  var thread_accs: array<f32, N_PER_THREAD>;
  for (var nn: u32 = 0u; nn < N_PER_THREAD; nn = nn + 1u) {
    thread_accs[nn] = 0.0;
  }

  for (var chunk: u32 = 0u; chunk < NUM_CHUNKS; chunk = chunk + 1u) {
    let chunk_k_vec_base: u32 = chunk * CHUNK_GROUP_SIZE_V4;
    for (var i: u32 = tid; i < A_CHUNK_VEC4; i = i + WG_SIZE) {
      let mi: u32 = i / CHUNK_GROUP_SIZE_V4;
      let kvi: u32 = i % CHUNK_GROUP_SIZE_V4;
      let src_m: u32 = m_block + mi;
      if (src_m < params.M) {
        a_chunk[i] = a[src_m * IN_FEATURES_V4 + chunk_k_vec_base + kvi];
      } else {
        a_chunk[i] = vec4<${y}>(0.0);
      }
    }
    workgroupBarrier();

    let chunk_g0: u32 = chunk * K_GROUPS_PER_CHUNK;
    for (var nn: u32 = 0u; nn < N_PER_THREAD; nn = nn + 1u) {
      let n_row: u32 = n_block + nn * OUT_TILE + n_local0;
      if (n_row < OUT_FEATURES) {
        let row_words_base: u32 = n_row * WORDS_PER_ROW;
        let row_sb_base: u32 = n_row * NUM_GROUPS * 2u;
        for (var gi: u32 = 0u; gi < K_GROUPS_PER_CHUNK; gi = gi + 1u) {
          let g: u32 = chunk_g0 + gi;
          let group_word_base: u32 = row_words_base + g * WORDS_PER_GROUP;
          let a_vec_base: u32 = m_local * CHUNK_GROUP_SIZE_V4 + gi * GROUP_SIZE_V4;
          var sum_qa: f32 = 0.0;
          var sum_a:  f32 = 0.0;
          for (var w: u32 = 0u; w < WORDS_PER_GROUP; w = w + 1u) {
            let packed: u32 = bits_buf[group_word_base + w];
            let a_vec_off: u32 = a_vec_base + w * VEC4_PER_WORD;
${O}        }
          let scale: f32 = f32(scaleBias[row_sb_base + g * 2u]);
          let bias:  f32 = f32(scaleBias[row_sb_base + g * 2u + 1u]);
          thread_accs[nn] = thread_accs[nn] + scale * sum_qa + bias * sum_a;
        }
      }
    }
    workgroupBarrier();
  }

  if (m_valid) {
    for (var nn: u32 = 0u; nn < N_PER_THREAD; nn = nn + 1u) {
      let n_row: u32 = n_block + nn * OUT_TILE + n_local0;
      if (n_row < OUT_FEATURES) {
        y[m_row * OUT_FEATURES + n_row] = ${C};
      }
    }
  }
}
`;
}
function bn({
  inFeatures: e,
  outFeatures: t,
  tileM: n = 10,
  tileN: r = 256,
  nPerThread: i = 1,
  assumeBiasNegHalfScale: a = !1,
  scaleOnly: o = !1,
}) {
  if (e % 128 != 0)
    throw Error(`binary LUT matmul requires K divisible by 128`);
  if (!Number.isInteger(i) || i < 1)
    throw Error(`binary LUT matmul requires nPerThread >= 1`);
  if (r % i !== 0)
    throw Error(`binary LUT matmul tileN must be divisible by nPerThread`);
  let s = r / i;
  if (s > 256)
    throw Error(
      `binary LUT matmul workgroup size exceeds max workgroup invocations`,
    );
  let c = e / 128,
    l = e / 4,
    u = n * 32 * 16,
    d = n * 32,
    f = u * 2 + (a ? 0 : n * 4);
  if (f > 16 * 1024)
    throw Error(`binary LUT matmul uses ${f} bytes of workgroup storage`);
  let p = Array.from({ length: i }, (e, t) =>
      Array.from({ length: n }, (e, n) => `  var acc${t}_${n}: f32 = 0.0;`)
        .join(`
`),
    ).join(`
`),
    m = (e) =>
      Array.from({ length: n }, (t, n) => `      var qa${e}_${n}: f32 = 0.0;`)
        .join(`
`),
    h = (e) =>
      Array.from({ length: 4 }, (t, r) =>
        Array.from({ length: 4 }, (t, i) => {
          let a = r * 8 + i * 2;
          return `      {
        let byte${r}_${i}: u32 = (p${r} >> ${i * 8}u) & 0xffu;
        let lo${r}_${i}: u32 = byte${r}_${i} & 0x0fu;
        let hi${r}_${i}: u32 = byte${r}_${i} >> 4u;
${Array.from(
  { length: n },
  (t, n) =>
    `        qa${e}_${n} = qa${e}_${n} + f32(lut[lutIndex(${n}u, ${a}u, lo${r}_${i})]) + f32(lut[lutIndex(${n}u, ${a + 1}u, hi${r}_${i})]);`,
).join(`
`)}
      }`;
        }).join(`
`),
      ).join(`
`),
    g = (e) =>
      a
        ? o
          ? `      let scale: f32 = f32(scaleBias[n${e} * NUM_GROUPS + g]);`
          : `      let scale: f32 = f32(scaleBias[sb]);`
        : `      let scale: f32 = f32(scaleBias[sb]);
      let bias: f32 = f32(scaleBias[sb + 1u]);`,
    _ = (e) =>
      Array.from({ length: n }, (t, n) =>
        a
          ? `      acc${e}_${n} = acc${e}_${n} + scale * qa${e}_${n};`
          : `      acc${e}_${n} = acc${e}_${n} + scale * qa${e}_${n} + bias * sumA[${n}u];`,
      ).join(`
`),
    v = Array.from(
      { length: i },
      (e, t) => `  let n${t}: u32 = n_base + ${t}u * WG_SIZE;
  let n${t}_valid: bool = n${t} < OUT_FEATURES;`,
    ).join(`
`),
    y = Array.from(
      { length: i },
      (e, t) => `    if (n${t}_valid) {
      let packed: vec4<u32> = bits_buf[n${t} * NUM_GROUPS + g];
      let p0: u32 = packed.x;
      let p1: u32 = packed.y;
      let p2: u32 = packed.z;
      let p3: u32 = packed.w;
      let sb: u32 = (n${t} * NUM_GROUPS + g) * 2u;
${g(t)}
${m(t)}
${h(t)}
${_(t)}
    }`,
    ).join(`
`),
    b = Array.from(
      { length: i },
      (e, t) => `  if (n${t}_valid) {
${Array.from(
  { length: n },
  (e, n) =>
    `    { let mr: u32 = m_block + ${n}u; if (mr < params.M) { y[mr * OUT_FEATURES + n${t}] = f16(acc${t}_${n}); } }`,
).join(`
`)}
  }`,
    ).join(`
`),
    x = a
      ? `      let c: f32 = 0.5 * (x + yv + z + wv);
      lut[base] = f16(-c);
      lut[base + 1u] = f16(x - c);
      lut[base + 2u] = f16(yv - c);
      lut[base + 3u] = f16(xy - c);
      lut[base + 4u] = f16(z - c);
      lut[base + 5u] = f16(xz - c);
      lut[base + 6u] = f16(yz - c);
      lut[base + 7u] = f16(xyz - c);
      lut[base + 8u] = f16(wv - c);
      lut[base + 9u] = f16(x + wv - c);
      lut[base + 10u] = f16(yv + wv - c);
      lut[base + 11u] = f16(xy + wv - c);
      lut[base + 12u] = f16(z + wv - c);
      lut[base + 13u] = f16(xz + wv - c);
      lut[base + 14u] = f16(yz + wv - c);
      lut[base + 15u] = f16(xyz + wv - c);`
      : `      lut[base] = 0.0h;
      lut[base + 1u] = f16(x);
      lut[base + 2u] = f16(yv);
      lut[base + 3u] = f16(xy);
      lut[base + 4u] = f16(z);
      lut[base + 5u] = f16(xz);
      lut[base + 6u] = f16(yz);
      lut[base + 7u] = f16(xyz);
      lut[base + 8u] = f16(wv);
      lut[base + 9u] = f16(x + wv);
      lut[base + 10u] = f16(yv + wv);
      lut[base + 11u] = f16(xy + wv);
      lut[base + 12u] = f16(z + wv);
      lut[base + 13u] = f16(xz + wv);
      lut[base + 14u] = f16(yz + wv);
      lut[base + 15u] = f16(xyz + wv);`;
  return `enable f16;
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a:         array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       bits_buf:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       scaleBias: array<f16>;
@group(0) @binding(3) var<storage, read_write> y:         array<f16>;
@group(0) @binding(4) var<uniform>             params:    Params;

const IN_FEATURES:    u32 = ${e}u;
const IN_FEATURES_V4: u32 = ${l}u;
const OUT_FEATURES:   u32 = ${t}u;
const NUM_GROUPS:     u32 = ${c}u;
const M_TILE:         u32 = ${n}u;
const N_TILE:         u32 = ${r}u;
const WG_SIZE:        u32 = ${s}u;
const LUT_ENTRIES:    u32 = ${u}u;
const LUT_NIBBLES:    u32 = ${d}u;

var<workgroup> lut: array<f16, ${u}>;
${
  a
    ? ``
    : `var<workgroup> sumA: array<f32, ${n}>;
`
}

fn lutIndex(m: u32, nibble: u32, mask: u32) -> u32 {
  return (m * 32u + nibble) * 16u + mask;
}

@compute @workgroup_size(${s}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  let n_block: u32 = wg.x * N_TILE;
  let m_block: u32 = wg.y * M_TILE;
  let n_base: u32 = n_block + tid;
${v}

${p}

  for (var g: u32 = 0u; g < NUM_GROUPS; g = g + 1u) {
    for (var i: u32 = tid; i < LUT_NIBBLES; i = i + WG_SIZE) {
      let nibble: u32 = i & 31u;
      let m_local: u32 = i >> 5u;
      let m: u32 = m_block + m_local;
      var v: vec4<f16> = vec4<f16>(0.0h);
      if (m < params.M) {
        v = a[m * IN_FEATURES_V4 + g * 32u + nibble];
      }
      let x: f32 = f32(v.x);
      let yv: f32 = f32(v.y);
      let z: f32 = f32(v.z);
      let wv: f32 = f32(v.w);
      let xy: f32 = x + yv;
      let xz: f32 = x + z;
      let yz: f32 = yv + z;
      let xyz: f32 = xy + z;
      let base: u32 = (m_local * 32u + nibble) * 16u;
${x}
    }
    workgroupBarrier();
${
  a
    ? ``
    : `
    for (var mi: u32 = tid; mi < M_TILE; mi = mi + WG_SIZE) {
      var s: f32 = 0.0;
      for (var nibble: u32 = 0u; nibble < 32u; nibble = nibble + 1u) {
        s = s + f32(lut[lutIndex(mi, nibble, 15u)]);
      }
      sumA[mi] = s;
    }
    workgroupBarrier();
`
}

${y}
    workgroupBarrier();
  }

${b}
}
`;
}
function xn({ inFeatures: e, groupSize: t, inputDtype: n = `float32` }) {
  if (e % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  if (t % 16 != 0)
    throw Error(`groupSize must be divisible by 16 (packs of 4 i8)`);
  if (t % 4 != 0) throw Error(`groupSize must be divisible by 4 (vec4 loads)`);
  if (n !== `float32` && n !== `float16`)
    throw Error(`unsupported quantize-A input dtype: ${n}`);
  let r = e / t,
    i = t / 4,
    a = t / 4,
    o = Math.min(32, t / 4),
    s = t / o,
    c = n === `float16` ? `f16` : `f32`,
    l =
      n === `float16`
        ? `enable f16;
`
        : ``,
    u = (e) => (n === `float16` ? `vec4<f32>(${e})` : e);
  return `${l}// Quantize A ${n} \u2192 i8, per-block scale + per-block sum.
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a:       array<vec4<${c}>>;
@group(0) @binding(1) var<storage, read_write> a_i8:    array<u32>;
@group(0) @binding(2) var<storage, read_write> scale_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> sum_a:   array<f32>;
@group(0) @binding(4) var<uniform>             params:  Params;

const IN_FEATURES:   u32 = ${e}u;
const IN_FEATURES_V4: u32 = ${e / 4}u;
const GROUP_SIZE:    u32 = ${t}u;
const GROUP_SIZE_V4: u32 = ${i}u;
const NUM_GROUPS:    u32 = ${r}u;
const U32_PER_GROUP: u32 = ${a}u;
const WG:            u32 = ${o}u;
const ELS_PER_THREAD: u32 = ${s}u;

var<workgroup> partial_max: array<f32, WG>;
var<workgroup> partial_sum: array<f32, WG>;
var<workgroup> group_scale: f32;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m_row: u32 = wg.y;
  let g: u32 = wg.x;
  let tid: u32 = lid.x;
  if (m_row >= params.M) { return; }

  let group_k_v4_base: u32 = m_row * IN_FEATURES_V4 + g * GROUP_SIZE_V4;
  let thread_v4_start: u32 = tid * (ELS_PER_THREAD / 4u);
  var local_max: f32 = 0.0;
  var local_sum: f32 = 0.0;
  var v0: vec4<f32>;
  var v1: vec4<f32>;
  v0 = ${u(`a[group_k_v4_base + thread_v4_start]`)};
  local_max = max(local_max, max(max(abs(v0.x), abs(v0.y)), max(abs(v0.z), abs(v0.w))));
  local_sum = local_sum + v0.x + v0.y + v0.z + v0.w;
${
  s === 8
    ? `  v1 = ${u(`a[group_k_v4_base + thread_v4_start + 1u]`)};
  local_max = max(local_max, max(max(abs(v1.x), abs(v1.y)), max(abs(v1.z), abs(v1.w))));
  local_sum = local_sum + v1.x + v1.y + v1.z + v1.w;`
    : ``
}

  partial_max[tid] = local_max;
  partial_sum[tid] = local_sum;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      let other = partial_max[tid + stride];
      if (other > partial_max[tid]) { partial_max[tid] = other; }
      partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    let m = partial_max[0];
    if (m > 0.0) {
      group_scale = m / 127.0;
    } else {
      group_scale = 1.0;
    }
    scale_a[m_row * NUM_GROUPS + g] = group_scale;
    sum_a[m_row * NUM_GROUPS + g] = partial_sum[0];
  }
  workgroupBarrier();
  let inv_scale: f32 = 1.0 / group_scale;
  let q0 = clamp(round(v0.x * inv_scale), -128.0, 127.0);
  let q1 = clamp(round(v0.y * inv_scale), -128.0, 127.0);
  let q2 = clamp(round(v0.z * inv_scale), -128.0, 127.0);
  let q3 = clamp(round(v0.w * inv_scale), -128.0, 127.0);
  let u32_0: u32 = pack4xI8(vec4<i32>(i32(q0), i32(q1), i32(q2), i32(q3)));
  let out_base: u32 = (m_row * NUM_GROUPS + g) * U32_PER_GROUP;
  a_i8[out_base + thread_v4_start] = u32_0;
${
  s === 8
    ? `  let q4 = clamp(round(v1.x * inv_scale), -128.0, 127.0);
  let q5 = clamp(round(v1.y * inv_scale), -128.0, 127.0);
  let q6 = clamp(round(v1.z * inv_scale), -128.0, 127.0);
  let q7 = clamp(round(v1.w * inv_scale), -128.0, 127.0);
  let u32_1: u32 = pack4xI8(vec4<i32>(i32(q4), i32(q5), i32(q6), i32(q7)));
  a_i8[out_base + thread_v4_start + 1u] = u32_1;`
    : ``
}
}
`;
}
function Sn({
  bits: e,
  groupSize: t,
  inFeatures: n,
  outFeatures: r,
  scaleBiasDtype: i = `float16`,
  scaleBiasLayout: a = `out-group`,
  outputLayout: o = `out-k`,
  assumeTernaryBias: s = !1,
}) {
  if (![1, 2, 4, 8].includes(e)) throw Error(`unsupported bits=${e}`);
  if (s && e !== 2)
    throw Error(`assumeTernaryBias is only valid for 2-bit MLX weights`);
  if (n % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  if (i !== `float32` && i !== `float16` && i !== `bfloat16`)
    throw Error(`unsupported scaleBias dtype: ${i}`);
  if (a !== `out-group` && a !== `group-out`)
    throw Error(`unsupported scaleBias layout: ${a}`);
  if (o !== `out-k` && o !== `k-out`)
    throw Error(`unsupported output layout: ${o}`);
  let c = Math.floor(32 / e),
    l = n / c,
    u = n / t,
    d = (1 << e) - 1,
    f = i === `bfloat16` ? `u32` : i === `float16` ? `f16` : `f32`,
    p = s ? `1u` : `2u`,
    m =
      a === `group-out`
        ? `let sb_idx: u32 = (g * OUT_FEATURES + row) * ${p};`
        : `let sb_idx: u32 = (row * NUM_GROUPS + g) * ${p};`;
  return `enable f16;
struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       bits_buf:  array<u32>;
@group(0) @binding(1) var<storage, read>       scaleBias: array<${f}>;
@group(0) @binding(2) var<storage, read_write> out:       array<f16>;
@group(0) @binding(3) var<uniform>             params:    Params;

${
  i === `bfloat16`
    ? `
fn loadBf16(idx: u32) -> f32 {
  let packed = scaleBias[idx / 2u];
  let half = select(packed & 0xffffu, packed >> 16u, (idx & 1u) == 1u);
  return bitcast<f32>(half << 16u);
}
`
    : ``
}

const IN_FEATURES:   u32 = ${n}u;
const OUT_FEATURES:  u32 = ${r}u;
const GROUP_SIZE:    u32 = ${t}u;
const NUM_GROUPS:    u32 = ${u}u;
const VALS_PER_WORD: u32 = ${c}u;
const WORDS_PER_ROW: u32 = ${l}u;
const MASK:          u32 = ${d}u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let idx = wg_idx * 64u + lid.x;
  if (idx >= params.count) { return; }
  let row = idx / WORDS_PER_ROW;
  let word = idx - row * WORDS_PER_ROW;
  let k_base = word * VALS_PER_WORD;
  let g = k_base / GROUP_SIZE;
  ${m}
  let scale_g: f32 = ${i === `bfloat16` ? `loadBf16(sb_idx)` : `f32(scaleBias[sb_idx])`};
  ${s ? `let bias_g: f32 = -scale_g;` : `let bias_g: f32 = ${i === `bfloat16` ? `loadBf16(sb_idx + 1u)` : `f32(scaleBias[sb_idx + 1u])`};`}
  let packed = bits_buf[idx];
${Array.from(
  { length: c },
  (t, n) =>
    `  out[${o === `k-out` ? `(k_base + ${n}u) * OUT_FEATURES + row` : `row * IN_FEATURES + k_base + ${n}u`}] = f16(scale_g * f32((packed >> ${n * e}u) & MASK) + bias_g);`,
).join(`
`)}
}
`;
}
function Cn({
  bits: e,
  groupSize: t,
  inFeatures: n,
  outFeatures: r,
  tileM: i = 2,
  tileN: a = 64,
  outputDtype: o = `float32`,
  scaleBiasDtype: s = `float32`,
}) {
  if (e !== 2) throw Error(`DP4A kernel currently only supports bits=2`);
  if (t !== 128)
    throw Error(`DP4A kernel currently only supports groupSize=128`);
  if (n % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  if (o !== `float32` && o !== `float16`)
    throw Error(`unsupported DP4A output dtype: ${o}`);
  if (s !== `float32` && s !== `float16`)
    throw Error(`unsupported DP4A scaleBias dtype: ${s}`);
  let c = n / t,
    l = t / 4;
  t / 16;
  let u = n / 16,
    d = u / 4;
  if (u % 4 != 0)
    throw Error(`wordsPerRow must be divisible by 4 for vec4<u32> loads`);
  let f = l / 4,
    p = a;
  if (p > 256) throw Error(`tileN=${a} exceeds 256`);
  let m = o === `float16` ? `f16` : `f32`,
    h = s === `float16` ? `f16` : `f32`,
    g =
      o === `float16` || s === `float16`
        ? `enable f16;
`
        : ``,
    _ = (e) => (o === `float16` ? `f16(${e})` : e);
  i * f;
  let v = Array.from({ length: i }, (e, t) => `  var r${t}: i32 = 0;`).join(`
`),
    y = Array.from({ length: i }, (e, t) => `  var y${t}: f32 = 0.0;`).join(`
`);
  function b(e) {
    let t = ``;
    ((t += `        let b_vec_h${e}: vec4<u32> = b_quad_${e};
`),
      (t += `        let p${e}_0: u32 = b_vec_h${e}[0u];
`),
      (t += `        let p${e}_1: u32 = b_vec_h${e}[1u];
`),
      (t += `        let p${e}_2: u32 = b_vec_h${e}[2u];
`),
      (t += `        let p${e}_3: u32 = b_vec_h${e}[3u];
`));
    for (let n = 0; n < 4; n++)
      for (let r = 0; r < 4; r++)
        ((t += `        let b_byte_${e}_${n}_${r}: u32 = (p${e}_${n} >> ${r * 8}u) & 0xFFu;
`),
          (t += `        let qb_${e}_${n}_${r}: u32 = (b_byte_${e}_${n}_${r} & 0x03u) | ((b_byte_${e}_${n}_${r} & 0x0Cu) << 6u) | ((b_byte_${e}_${n}_${r} & 0x30u) << 12u) | ((b_byte_${e}_${n}_${r} & 0xC0u) << 18u);
`));
    for (let n = 0; n < i; n++) {
      let r = `(${n}u * AU32_PER_GROUP) + ${e}u * 16u`;
      for (let i = 0; i < 4; i++)
        for (let a = 0; a < 4; a++)
          t += `        r${n} = r${n} + dot4I8Packed(a_tile[${r} + ${i * 4 + a}u], qb_${e}_${i}_${a});
`;
    }
    return t;
  }
  let x = Array.from(
      { length: i },
      (e, t) => `      {
        let sa_mg: f32 = scale_a[(m_block + ${t}u) * NUM_GROUPS + g];
        let raw_sum: f32 = sum_a[(m_block + ${t}u) * NUM_GROUPS + g];
        y${t} = y${t} + sa_mg * scale_g * f32(r${t}) + bias_g * raw_sum;
        r${t} = 0;
      }`,
    ).join(`
`),
    S = Array.from(
      { length: i },
      (e, t) =>
        `    { let mr: u32 = m_block + ${t}u; if (mr < params.M) { y[mr * OUT_FEATURES + n_row] = ${_(`y${t}`)}; } }`,
    ).join(`
`);
  return `${g}// DP4A MLX matmul, bits=2, gs=128, in=${n}, out=${r}, tileM=${i}, tileN=${a}, scaleBias=${s}, output=${o}.
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a_i8:     array<u32>;
@group(0) @binding(1) var<storage, read>       scale_a:  array<f32>;
@group(0) @binding(2) var<storage, read>       sum_a:    array<f32>;
@group(0) @binding(3) var<storage, read>       bits_buf: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       scaleBias: array<${h}>;
@group(0) @binding(5) var<storage, read_write> y:        array<${m}>;
@group(0) @binding(6) var<uniform>             params:   Params;

const IN_FEATURES:       u32 = ${n}u;
const OUT_FEATURES:      u32 = ${r}u;
const GROUP_SIZE:        u32 = ${t}u;
const NUM_GROUPS:        u32 = ${c}u;
const WORDS_PER_ROW_V4:  u32 = ${d}u;
const AU32_PER_GROUP:    u32 = ${l}u;
const A_TILE_U32:        u32 = ${i * l}u;
const TILE_M:            u32 = ${i}u;
const TILE_N:            u32 = ${a}u;
const WG_SIZE:           u32 = ${p}u;

var<workgroup> a_tile: array<u32, A_TILE_U32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n_block: u32 = wg.x * TILE_N;
  let m_block: u32 = wg.y * TILE_M;
  let tid: u32 = lid.x;
  let n_row: u32 = n_block + tid;
  let n_valid: bool = n_row < OUT_FEATURES;

${v}
${y}

  for (var g: u32 = 0u; g < NUM_GROUPS; g = g + 1u) {
    // Load A tile: TILE_M rows \xD7 AU32_PER_GROUP u32 from a_i8.
    let a_global_base: u32 = m_block * NUM_GROUPS * AU32_PER_GROUP + g * AU32_PER_GROUP;
    let a_group_stride: u32 = NUM_GROUPS * AU32_PER_GROUP;
    for (var i: u32 = tid; i < A_TILE_U32; i = i + WG_SIZE) {
      let mi: u32 = i / AU32_PER_GROUP;
      let ki: u32 = i % AU32_PER_GROUP;
      let src_m: u32 = m_block + mi;
      if (src_m < params.M) {
        a_tile[i] = a_i8[src_m * a_group_stride + g * AU32_PER_GROUP + ki];
      } else {
        a_tile[i] = 0u;
      }
    }
    workgroupBarrier();

    if (n_valid) {
      let row_base_v4: u32 = n_row * WORDS_PER_ROW_V4 + g * 2u;
      let b_quad_0: vec4<u32> = bits_buf[row_base_v4];
      let b_quad_1: vec4<u32> = bits_buf[row_base_v4 + 1u];
      let sb_idx: u32 = (n_row * NUM_GROUPS + g) * 2u;
      let scale_g: f32 = f32(scaleBias[sb_idx]);
      let bias_g: f32 = f32(scaleBias[sb_idx + 1u]);

${b(0)}
${b(1)}

${x}
    }
    workgroupBarrier();
  }

  if (n_valid) {
${S}
  }
}
`;
}
function wn({
  bits: e,
  groupSize: t,
  inFeatures: n,
  outFeatures: r,
  useF16: i = !1,
  tileM: a = 32,
  tileN: o = 64,
  inputDtype: s = `float32`,
  outputDtype: c = `float32`,
  scaleBiasDtype: l = `float32`,
  scaleBiasLayout: u = `out-group`,
  assumeFullM: d = !1,
  assumeTernaryBias: f = !1,
  useARowOffset: p = !1,
}) {
  if (e !== 1 && e !== 2 && e !== 4)
    throw Error(
      `subgroup-matrix kernel currently only supports bits=1, bits=2, or bits=4`,
    );
  if (f && e !== 2)
    throw Error(`assumeTernaryBias is only valid for 2-bit MLX weights`);
  if (a !== 32 && a !== 64)
    throw Error(
      `subgroup-matrix kernel currently supports tileM=32 or tileM=64`,
    );
  if (o !== 64 && o !== 128)
    throw Error(
      `subgroup-matrix kernel currently supports tileN=64 or tileN=128`,
    );
  if (a === 64 && o !== 64) throw Error(`tileM=64 currently requires tileN=64`);
  if (a === 64 && !i)
    throw Error(
      `tileM=64 requires f16 compute to stay within workgroup storage limits`,
    );
  if (o > 64 && !i)
    throw Error(
      `tileN=128 requires f16 compute to stay within workgroup storage limits`,
    );
  if (n % t !== 0) throw Error(`inFeatures must be divisible by groupSize`);
  if (t % 32 != 0)
    throw Error(`groupSize must be divisible by subgroup K tile=32`);
  if (n % 32 != 0)
    throw Error(`inFeatures must be divisible by subgroup K tile=32`);
  if (r % o !== 0)
    throw Error(`outFeatures must be divisible by subgroup N tile=${o}`);
  if (u !== `out-group` && u !== `group-out`)
    throw Error(`unsupported scaleBiasLayout: ${u}`);
  let m = 32 / e,
    h = n / m,
    g = n / t,
    _ = a === 64 ? o : o / 2,
    v = _ / 8,
    y = i ? `f16` : `f32`,
    b = s === `float16` ? `f16` : `f32`,
    x = l === `float16` ? `f16` : `f32`,
    S = c === `float16` ? `f16` : `f32`,
    C =
      i || s === `float16` || c === `float16` || l === `float16`
        ? `enable f16;
`
        : ``,
    w = p ? `(params.aRowOffset + a_global)` : `a_global`,
    T = i
      ? s === `float16`
        ? `a[${w} * IN_FEATURES + k]`
        : `f16(a[${w} * IN_FEATURES + k])`
      : s === `float16`
        ? `f32(a[${w} * IN_FEATURES + k])`
        : `a[${w} * IN_FEATURES + k]`,
    E = i ? `0.0h` : `0.0`,
    D = f
      ? i
        ? l === `float16`
          ? `  let scale_g: f16 = scaleBias[sb_idx];
  let bias_g: f16 = -scale_g;`
          : `  let scale_g: f16 = f16(scaleBias[sb_idx]);
  let bias_g: f16 = -scale_g;`
        : `  let scale_g: f32 = f32(scaleBias[sb_idx]);
  let bias_g: f32 = -scale_g;`
      : i
        ? l === `float16`
          ? `  let scale_g: f16 = scaleBias[sb_idx];
  let bias_g: f16 = scaleBias[sb_idx + 1u];`
          : `  let scale_g: f16 = f16(scaleBias[sb_idx]);
  let bias_g: f16 = f16(scaleBias[sb_idx + 1u]);`
        : `  let scale_g: f32 = f32(scaleBias[sb_idx]);
  let bias_g: f32 = f32(scaleBias[sb_idx + 1u]);`,
    O =
      u === `group-out`
        ? `    let g: u32 = k_base / GROUP_SIZE;
    let sb_idx: u32 = (g * OUT_FEATURES + b_global) * 2u;`
        : `    let g: u32 = k_base / GROUP_SIZE;
    let sb_idx: u32 = (b_global * NUM_GROUPS + g) * 2u;`,
    k = `vec4<${y}>`,
    A = (e, t) => `    tile_B[tile_b_base + ${t}u] = ${e}.x;
    tile_B[tile_b_base + ${t + 1}u] = ${e}.y;
    tile_B[tile_b_base + ${t + 2}u] = ${e}.z;
    tile_B[tile_b_base + ${t + 3}u] = ${e}.w;`,
    j = Array.from(
      { length: 16 },
      (e, t) => `  {
    ${`let q: bool = ((packed >> (col + ${t}u)) & 0x1u) != 0u;`}
    tile_B[b_row * TILE_K + col + ${t}u] = select(bias_g, bias_g + scale_g, q);
  }`,
    ).join(`
`),
    M = Array.from({ length: 16 }, (e, t) => {
      let n = t * 2;
      return `  {
    ${i ? `let q: f16 = f16((packed >> ${n}u) & 0x3u);` : `let q: f32 = f32((packed >> ${n}u) & 0x3u);`}
    tile_B[b_row * TILE_K + col + ${t}u] = scale_g * q + bias_g;
  }`;
    }).join(`
`),
    N = `  let lower0: ${k} = ${k}(unpack4xU8(packed0 & 0x0F0F0F0Fu));
  let upper0: ${k} = ${k}(unpack4xU8((packed0 >> 4u) & 0x0F0F0F0Fu));
  let lower1: ${k} = ${k}(unpack4xU8(packed1 & 0x0F0F0F0Fu));
  let upper1: ${k} = ${k}(unpack4xU8((packed1 >> 4u) & 0x0F0F0F0Fu));
  let b0: ${k} = ${k}(lower0.x, upper0.x, lower0.y, upper0.y) * scale_g + ${k}(bias_g);
  let b1: ${k} = ${k}(lower0.z, upper0.z, lower0.w, upper0.w) * scale_g + ${k}(bias_g);
  let b2: ${k} = ${k}(lower1.x, upper1.x, lower1.y, upper1.y) * scale_g + ${k}(bias_g);
  let b3: ${k} = ${k}(lower1.z, upper1.z, lower1.w, upper1.w) * scale_g + ${k}(bias_g);
  let tile_b_base: u32 = b_row * TILE_K + col;
${A(`b0`, 0)}
${A(`b1`, 4)}
${A(`b2`, 8)}
${A(`b3`, 12)}`,
    P =
      e === 1 || e === 2
        ? `  let packed: u32 = bits_buf[b_global * WORDS_PER_ROW + k_base / VALS_PER_WORD];`
        : `  let word_idx: u32 = b_global * WORDS_PER_ROW + k_base / VALS_PER_WORD;
  let packed0: u32 = bits_buf[word_idx];
  let packed1: u32 = bits_buf[word_idx + 1u];`,
    F = e === 1 ? j : e === 2 ? M : N,
    I = (e) => (c === `float16` ? (i ? e : `f16(${e})`) : `f32(${e})`),
    ee = (i && c === `float16`) || (!i && c === `float32`),
    L = !(ee && d),
    te = d
      ? `    tile_A[row * TILE_K + col + col_offset] = ${T};`
      : `    if (a_global < params.M) {
      tile_A[row * TILE_K + col + col_offset] = ${T};
    } else {
      tile_A[row * TILE_K + col + col_offset] = ${E};
    }`,
    R =
      a === 64
        ? `  let subtile_idx: u32 = 0u;
  let subtile_idy: u32 = subtile_id;`
        : `  let subtile_idx: u32 = subtile_id / 2u;
  let subtile_idy: u32 = subtile_id % 2u;`,
    ne =
      a === 64
        ? `    loadSHMA(a_global_base, kidx, local_idx / 4u, local_idx % 4u);
    loadSHMA(a_global_base, kidx, local_idx / 4u + 32u, local_idx % 4u);`
        : `    loadSHMA(a_global_base, kidx, local_idx / 4u, local_idx % 4u);`,
    re = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: v },
        (e, n) => `  var matC${t}${n}: subgroup_matrix_result<${y}, 8, 8>;`,
      ).join(`
`),
    ).join(`
`),
    ie = Array.from(
      { length: v },
      (e, t) =>
        `      var matB${t}: subgroup_matrix_right<${y}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<${y}, 8, 8>>(&tile_B, matrix_b_offset + ${t * 8}u * TILE_K, true, TILE_K);`,
    ).join(`
`),
    ae = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: v },
        (e, n) =>
          `      matC${t}${n} = subgroupMatrixMultiplyAccumulate(matA${t}, matB${n}, matC${t}${n});`,
      ).join(`
`),
    ).join(`
`),
    oe = (e) =>
      Array.from(
        { length: v },
        (t, n) =>
          `  subgroupMatrixStore(&scratch[subtile_id][${n}], 0u, matC${e}${n}, false, 8u);`,
      ).join(`
`),
    se = (e) =>
      Array.from(
        { length: v },
        (t, n) =>
          `    subgroupMatrixStore(&y, matrix_c_offset + ${e === 0 ? `` : `8u * OUT_FEATURES + `}${n * 8}u, matC${e}${n}, false, OUT_FEATURES);`,
      ).join(`
`),
    ce = `${oe(0)}
  let row: u32 = sg_id / 4u;
  let col: u32 = (sg_id % 4u) * 2u;
${d ? `` : `  var row_limit: i32 = i32(params.M) - i32(a_global_base + base_A);`}
  storeOutput(matrix_c_offset, row, col, subtile_id${d ? `` : `, row_limit, full_m_tile`});

${oe(1)}
${d ? `` : `  row_limit = i32(params.M) - i32(a_global_base + base_A + 8u);`}
  storeOutput(matrix_c_offset + 8u * OUT_FEATURES, row, col, subtile_id${d ? `` : `, row_limit, full_m_tile`});`,
    le = ee
      ? d
        ? `${se(0)}
${se(1)}`
        : `  if (full_m_tile) {
${se(0)}
${se(1)}
  } else {
${ce}
  }`
      : ce,
    ue = Array.from({ length: v }, (e, t) => {
      let n = t * 8;
      return `    y[offset + row * OUT_FEATURES + col + ${n}u] = ${I(`scratch[src_slot][${t}][row * 8u + col]`)};
    y[offset + row * OUT_FEATURES + col + ${n + 1}u] = ${I(`scratch[src_slot][${t}][row * 8u + col2]`)};`;
    }).join(`
`);
  return `// Subgroup-matrix MLX matmul, bits=${e}, gs=${t}, in=${n}, out=${r}, tileM=${a}, tileN=${o}, precision=${y}, input=${s}, scaleBias=${l}/${u}, output=${c}, fullM=${d}, ternaryBias=${f}, aRowOffset=${p}.
enable subgroups;
${C}enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

struct Params { M: u32, aRowOffset: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       a:         array<${b}>;
@group(0) @binding(1) var<storage, read>       bits_buf:  array<u32>;
@group(0) @binding(2) var<storage, read>       scaleBias: array<${x}>;
@group(0) @binding(3) var<storage, read_write> y:         array<${S}>;
@group(0) @binding(4) var<uniform>             params:    Params;

const IN_FEATURES:    u32 = ${n}u;
const OUT_FEATURES:   u32 = ${r}u;
const GROUP_SIZE:     u32 = ${t}u;
const NUM_GROUPS:     u32 = ${g}u;
const VALS_PER_WORD:  u32 = ${m}u;
const WORDS_PER_ROW:  u32 = ${h}u;

const TILE_COLS:      u32 = ${o}u;
const TILE_ROWS:      u32 = ${a}u;
const TILE_K:         u32 = 32u;
const SUBTILE_COLS:   u32 = ${_}u;
const SUBTILE_ROWS:   u32 = 16u;

var<workgroup> tile_A: array<${y}, ${a} * 32>;
var<workgroup> tile_B: array<${y}, ${o} * 32>;
${L ? `var<workgroup> scratch: array<array<array<${y}, 64>, ${v}>, 4>;` : ``}

fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let a_global: u32 = tile_base + row;
  let col: u32 = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let k: u32 = k_idx + col + col_offset;
${te}
  }
}

fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let col: u32 = c_idx * 16u;
  for (var row_offset: u32 = 0u; row_offset < TILE_COLS; row_offset += 64u) {
    let b_row: u32 = row + row_offset;
    let b_global: u32 = tile_base + b_row;
    let k_base: u32 = k_idx + col;
${O}
${P}
${D}
${F}
  }
}

${
  L
    ? `fn storeOutput(offset: u32, row: u32, col: u32, src_slot: u32${d ? `` : `, row_limit: i32, full_m_tile: bool`}) {
${d ? `` : `  if (full_m_tile || (row_limit > 0 && row < u32(row_limit))) {`}
    let col2: u32 = col + 1u;
${ue}
${d ? `` : `  }`}
}`
    : ``
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let a_global_base: u32 = workgroup_id.y * TILE_ROWS;
  let b_global_base: u32 = workgroup_id.x * TILE_COLS;

  let subtile_id: u32 = local_idx / sg_size;
${R}
  let base_A: u32 = subtile_idy * SUBTILE_ROWS;
  let base_B: u32 = subtile_idx * SUBTILE_COLS;
${d ? `` : `  let full_m_tile: bool = a_global_base + TILE_ROWS <= params.M;`}

${re}

  for (var kidx: u32 = 0u; kidx < IN_FEATURES; kidx += TILE_K) {
${ne}
    loadSHMB(b_global_base, kidx, local_idx / 2u, local_idx % 2u);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step += 8u) {
      let matrix_a_offset: u32 = subtile_idy * SUBTILE_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<${y}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<${y}, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<${y}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<${y}, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset: u32 = subtile_idx * SUBTILE_COLS * TILE_K + step;
${ie}

${ae}
    }
    workgroupBarrier();
  }

  var matrix_c_offset: u32 = (a_global_base + base_A) * OUT_FEATURES + b_global_base + base_B;
${le}
}
`;
}
function Tn({
  inFeatures: e,
  outFeatures: t,
  mTile: n = 88,
  nTile: r = 16,
  rowPerThread: i = 11,
  kTile: a = 32,
  assumeFullN: o = !1,
}) {
  if (e % 4 != 0 || a % 4 != 0 || e % a !== 0)
    throw Error(
      `packed dense dual-N matmul requires K divisible by 4 and kTile`,
    );
  if (t % 4 != 0)
    throw Error(`packed dense dual-N matmul requires N divisible by 4`);
  if (n % i !== 0)
    throw Error(
      `packed dense dual-N matmul requires mTile divisible by rowPerThread`,
    );
  let s = r,
    c = n / i,
    l = s * c;
  if (l > 256)
    throw Error(`packed dense dual-N matmul exceeds max workgroup invocations`);
  let u = a / 4;
  if ((n * u + 2 * a * s) * 8 > 16 * 1024)
    throw Error(`packed dense dual-N matmul exceeds 16KB workgroup storage`);
  let d = s * 8,
    f = Array.from(
      { length: i },
      (e, t) => `  var acc0_${t}: vec4<f16> = vec4<f16>(0.0h);
  var acc1_${t}: vec4<f16> = vec4<f16>(0.0h);`,
    ).join(`
`),
    p = Array.from(
      { length: i },
      (
        e,
        t,
      ) => `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc0_${t} = fma(b0_3, vec4<f16>(a_vec${t}.w), fma(b0_2, vec4<f16>(a_vec${t}.z), fma(b0_1, vec4<f16>(a_vec${t}.y), fma(b0_0, vec4<f16>(a_vec${t}.x), acc0_${t}))));
        acc1_${t} = fma(b1_3, vec4<f16>(a_vec${t}.w), fma(b1_2, vec4<f16>(a_vec${t}.z), fma(b1_1, vec4<f16>(a_vec${t}.y), fma(b1_0, vec4<f16>(a_vec${t}.x), acc1_${t}))));`,
    ).join(`
`),
    m = o ? `` : `n_group0 < N_V4 && `,
    h = o ? `` : `n_group1 < N_V4 && `,
    g = Array.from(
      { length: i },
      (
        e,
        t,
      ) => `  if (${m}m_base + ${t}u < params.M) { y[(m_base + ${t}u) * N_V4 + n_group0] = acc0_${t}; }
  if (${h}m_base + ${t}u < params.M) { y[(m_base + ${t}u) * N_V4 + n_group1] = acc1_${t}; }`,
    ).join(`
`),
    _ = o
      ? `      bTile0[i] = w[(k_base + kk) * N_V4 + b_group0];
      bTile1[i] = w[(k_base + kk) * N_V4 + b_group1];`
      : `      if (b_group0 < N_V4) {
        bTile0[i] = w[(k_base + kk) * N_V4 + b_group0];
      } else {
        bTile0[i] = vec4<f16>(0.0h);
      }
      if (b_group1 < N_V4) {
        bTile1[i] = w[(k_base + kk) * N_V4 + b_group1];
      } else {
        bTile1[i] = vec4<f16>(0.0h);
      }`;
  return `enable f16;
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             params: Params;

const K: u32 = ${e}u;
const N: u32 = ${t}u;
const K_V4: u32 = ${e / 4}u;
const N_V4: u32 = ${t / 4}u;
const M_TILE: u32 = ${n}u;
const WG_X: u32 = ${s}u;
const WG_Y: u32 = ${c}u;
const ROW_PER_THREAD: u32 = ${i}u;
const OUT_TILE: u32 = ${d}u;
const K_TILE: u32 = ${a}u;
const K_TILE_V4: u32 = ${u}u;
const WG: u32 = ${l}u;

var<workgroup> aTile: array<vec4<f16>, ${n * u}>;
var<workgroup> bTile0: array<vec4<f16>, ${a * s}>;
var<workgroup> bTile1: array<vec4<f16>, ${a * s}>;

@compute @workgroup_size(${s}, ${c}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * M_TILE + row_base;
  let n_group0 = wg.x * (WG_X * 2u) + lx;
  let n_group1 = n_group0 + WG_X;
${f}

  for (var k_base: u32 = 0u; k_base < K; k_base = k_base + K_TILE) {
    let k_base_v4 = k_base / 4u;
    for (var i: u32 = tid; i < ${n * u}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let gm = wg.y * M_TILE + tm;
      if (gm < params.M) {
        aTile[i] = a[gm * K_V4 + k_base_v4 + kv];
      } else {
        aTile[i] = vec4<f16>(0.0h);
      }
    }
    for (var i: u32 = tid; i < ${a * s}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let b_group0 = wg.x * (WG_X * 2u) + nx;
      let b_group1 = b_group0 + WG_X;
${_}
    }
    workgroupBarrier();

    if (m_base < params.M) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let b0_0 = bTile0[(kv * 4u + 0u) * WG_X + lx];
        let b0_1 = bTile0[(kv * 4u + 1u) * WG_X + lx];
        let b0_2 = bTile0[(kv * 4u + 2u) * WG_X + lx];
        let b0_3 = bTile0[(kv * 4u + 3u) * WG_X + lx];
        let b1_0 = bTile1[(kv * 4u + 0u) * WG_X + lx];
        let b1_1 = bTile1[(kv * 4u + 1u) * WG_X + lx];
        let b1_2 = bTile1[(kv * 4u + 2u) * WG_X + lx];
        let b1_3 = bTile1[(kv * 4u + 3u) * WG_X + lx];
${p}
      }
    }
    workgroupBarrier();
  }

${g}
}
`;
}
async function En(
  e,
  {
    aT: t,
    wT: n,
    outT: r,
    M: i,
    inFeatures: a,
    outFeatures: o,
    mTile: s = null,
    nTile: c = null,
    rowPerThread: l = null,
    kTile: u = null,
  },
) {
  if (t.dtype !== `float16` || n.dtype !== `float16` || r.dtype !== `float16`)
    throw Error(
      `gpuDenseF16PackedVec4NDualNMatmul requires f16 input, weight, and output tensors`,
    );
  let d =
    i >= 128
      ? { mTile: 88, nTile: 16, rowPerThread: 11, kTile: 32 }
      : { mTile: 32, nTile: 16, rowPerThread: 4, kTile: 32 };
  ((s ??= d.mTile), (c ??= d.nTile), (l ??= d.rowPerThread), (u ??= d.kTile));
  let f = o % (c * 8) == 0,
    p = `dense_f16_packed_vec4n_dual_${a}_${o}_tm${s}_tn${c}_rpt${l}_tk${u}_fn${+!!f}`,
    m = Tn({
      inFeatures: a,
      outFeatures: o,
      mTile: s,
      nTile: c,
      rowPerThread: l,
      kTile: u,
      assumeFullN: f,
    }),
    h = Qt(e, [{ u32: i }], `dense-f16-packed-dual-params`);
  await e.runProgram({
    name: `dense_f16_packed_dual`,
    source: m,
    cacheKey: p,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: h, type: `uniform` },
    ],
    workgroups: [Math.ceil(o / (c * 8)), Math.ceil(i / s), 1],
  });
}
function Dn({
  inFeatures: e,
  innerFeatures: t,
  mTile: n = 64,
  nTile: r = 16,
  rowPerThread: i = 8,
  kTile: a = 32,
  assumeFullN: o = !1,
}) {
  if (e % 4 != 0 || a % 4 != 0 || e % a !== 0)
    throw Error(
      `packed dense swiglu matmul requires K divisible by 4 and kTile`,
    );
  if (t % 4 != 0)
    throw Error(
      `packed dense swiglu matmul requires innerFeatures divisible by 4`,
    );
  if (n % i !== 0)
    throw Error(
      `packed dense swiglu matmul requires mTile divisible by rowPerThread`,
    );
  let s = r,
    c = n / i,
    l = s * c;
  if (l > 256)
    throw Error(`packed dense swiglu matmul exceeds max workgroup invocations`);
  let u = a / 4;
  if ((n * u + 2 * a * s) * 8 > 16 * 1024)
    throw Error(`packed dense swiglu matmul exceeds 16KB workgroup storage`);
  let d = s * 4,
    f = Array.from(
      { length: i },
      (e, t) => `  var accA${t}: vec4<f16> = vec4<f16>(0.0h);
  var accB${t}: vec4<f16> = vec4<f16>(0.0h);`,
    ).join(`
`),
    p = Array.from(
      { length: i },
      (
        e,
        t,
      ) => `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        accA${t} = fma(bA3, vec4<f16>(a_vec${t}.w), fma(bA2, vec4<f16>(a_vec${t}.z), fma(bA1, vec4<f16>(a_vec${t}.y), fma(bA0, vec4<f16>(a_vec${t}.x), accA${t}))));
        accB${t} = fma(bB3, vec4<f16>(a_vec${t}.w), fma(bB2, vec4<f16>(a_vec${t}.z), fma(bB1, vec4<f16>(a_vec${t}.y), fma(bB0, vec4<f16>(a_vec${t}.x), accB${t}))));`,
    ).join(`
`),
    m = o ? `` : `n_group < INNER_V4 && `,
    h = Array.from(
      { length: i },
      (e, t) => `  if (${m}m_base + ${t}u < params.M) {
    let x${t} = vec4<f32>(accA${t});
    let yv${t} = vec4<f32>(accB${t});
    y[(m_base + ${t}u) * INNER_V4 + n_group] = vec4<f16>((x${t} / (vec4<f32>(1.0) + exp(-x${t}))) * yv${t});
  }`,
    ).join(`
`),
    g = o
      ? ``
      : `      if (b_group >= INNER_V4) {
        bTileA[i] = vec4<f16>(0.0h);
        bTileB[i] = vec4<f16>(0.0h);
        continue;
      }`;
  return `enable f16;
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             params: Params;

const K: u32 = ${e}u;
const INNER: u32 = ${t}u;
const K_V4: u32 = ${e / 4}u;
const INNER_V4: u32 = ${t / 4}u;
const FULL_N_V4: u32 = ${t / 2}u;
const M_TILE: u32 = ${n}u;
const WG_X: u32 = ${s}u;
const WG_Y: u32 = ${c}u;
const ROW_PER_THREAD: u32 = ${i}u;
const OUT_TILE: u32 = ${d}u;
const K_TILE: u32 = ${a}u;
const K_TILE_V4: u32 = ${u}u;
const WG: u32 = ${l}u;

var<workgroup> aTile: array<vec4<f16>, ${n * u}>;
var<workgroup> bTileA: array<vec4<f16>, ${a * s}>;
var<workgroup> bTileB: array<vec4<f16>, ${a * s}>;

@compute @workgroup_size(${s}, ${c}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * M_TILE + row_base;
  let n_group = wg.x * WG_X + lx;
${f}

  for (var k_base: u32 = 0u; k_base < K; k_base = k_base + K_TILE) {
    let k_base_v4 = k_base / 4u;
    for (var i: u32 = tid; i < ${n * u}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let gm = wg.y * M_TILE + tm;
      if (gm < params.M) {
        aTile[i] = a[gm * K_V4 + k_base_v4 + kv];
      } else {
        aTile[i] = vec4<f16>(0.0h);
      }
    }
    for (var i: u32 = tid; i < ${a * s}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let b_group = wg.x * WG_X + nx;
${g}
      bTileA[i] = w[(k_base + kk) * FULL_N_V4 + b_group];
      bTileB[i] = w[(k_base + kk) * FULL_N_V4 + b_group + INNER_V4];
    }
    workgroupBarrier();

    if (m_base < params.M) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let bA0 = bTileA[(kv * 4u + 0u) * WG_X + lx];
        let bA1 = bTileA[(kv * 4u + 1u) * WG_X + lx];
        let bA2 = bTileA[(kv * 4u + 2u) * WG_X + lx];
        let bA3 = bTileA[(kv * 4u + 3u) * WG_X + lx];
        let bB0 = bTileB[(kv * 4u + 0u) * WG_X + lx];
        let bB1 = bTileB[(kv * 4u + 1u) * WG_X + lx];
        let bB2 = bTileB[(kv * 4u + 2u) * WG_X + lx];
        let bB3 = bTileB[(kv * 4u + 3u) * WG_X + lx];
${p}
      }
    }
    workgroupBarrier();
  }

${h}
}
`;
}
async function On(
  e,
  {
    aT: t,
    wT: n,
    outT: r,
    M: i,
    inFeatures: a,
    innerFeatures: o,
    mTile: s = null,
    nTile: c = null,
    rowPerThread: l = null,
    kTile: u = null,
  },
) {
  if (t.dtype !== `float16` || n.dtype !== `float16` || r.dtype !== `float16`)
    throw Error(
      `gpuDenseF16PackedVec4NSwiGluMatmul requires f16 input, weight, and output tensors`,
    );
  let d =
    i >= 128
      ? { mTile: 88, nTile: 16, rowPerThread: 11, kTile: 32 }
      : { mTile: 32, nTile: 16, rowPerThread: 4, kTile: 32 };
  ((s ??= d.mTile), (c ??= d.nTile), (l ??= d.rowPerThread), (u ??= d.kTile));
  let f = o % (c * 4) == 0,
    p = `dense_f16_packed_swiglu_${a}_${o}_tm${s}_tn${c}_rpt${l}_tk${u}_fn${+!!f}`,
    m = Dn({
      inFeatures: a,
      innerFeatures: o,
      mTile: s,
      nTile: c,
      rowPerThread: l,
      kTile: u,
      assumeFullN: f,
    }),
    h = Qt(e, [{ u32: i }], `dense-f16-packed-swiglu-params`);
  await e.runProgram({
    name: `dense_f16_packed_swiglu`,
    source: m,
    cacheKey: p,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: h, type: `uniform` },
    ],
    workgroups: [Math.ceil(o / (c * 4)), Math.ceil(i / s), 1],
  });
}
function kn(e) {
  return e === `float16` ? `f16` : `f32`;
}
function An(...e) {
  return e.includes(`float16`)
    ? `enable f16;
`
    : ``;
}
function jn(e, t) {
  return t === `float16` ? `f32(${e})` : e;
}
function Mn(e, t) {
  return t === `float16` ? `f16(${e})` : e;
}
async function Nn(e, { xT: t, wT: n, yT: r, rows: i, dim: a, eps: o = 1e-6 }) {
  let s = t.dtype,
    c = n?.dtype ?? `float32`,
    l = r.dtype,
    u = `rmsnorm_d${a}_e${o}_${n ? `w` : `nw`}_${s}_${c}_${l}`,
    d = an({
      dim: a,
      eps: o,
      withWeight: !!n,
      inputDtype: s,
      weightDtype: c,
      outputDtype: l,
    }),
    f = Math.min(i, 65535),
    p = Math.ceil(i / f),
    m = Qt(e, [{ u32: i }, { u32: f }], `rmsnorm-params`),
    h = [
      { tensor: t, type: `read-only-storage` },
      ...(n ? [{ tensor: n, type: `read-only-storage` }] : []),
      { tensor: r, type: `storage` },
      { buffer: m, type: `uniform` },
    ];
  await e.runProgram({
    name: `rmsnorm`,
    source: d,
    cacheKey: u,
    bindings: h,
    workgroups: [f, p, 1],
  });
}
async function Pn(
  e,
  { xT: t, cosT: n, sinT: r, seq: i, heads: a, headDim: o },
) {
  let s = t.dtype,
    c = `rope1d_hd${o}_${s}`,
    l = on({ headDim: o, activationDtype: s }),
    u = Qt(e, [{ u32: i }, { u32: a }], `rope-params`);
  await e.runProgram({
    name: `rope1d`,
    source: l,
    cacheKey: c,
    bindings: [
      { tensor: t, type: `storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { buffer: u, type: `uniform` },
    ],
    workgroups: [i, a, 1],
  });
}
async function Fn(e, { xT: t, count: n }) {
  let r = t.dtype,
    i = `silu_2d_${r}`,
    a = sn({ dtype: r }),
    { wgX: o, wgY: s } = cn(n),
    c = Qt(e, [{ u32: n }, { u32: s }], `silu-params`);
  await e.runProgram({
    name: `silu`,
    source: a,
    cacheKey: i,
    bindings: [
      { tensor: t, type: `storage` },
      { buffer: c, type: `uniform` },
    ],
    workgroups: [s, o, 1],
  });
}
async function In(e, { xT: t, yT: n, rows: r, mlpInner: i }) {
  let a = t.dtype,
    o = n.dtype,
    s = a === `float16` && o === `float16` && i % 4 == 0,
    c = `swiglu_m${i}_${a}_${o}${s ? `_v4` : ``}`,
    l = ln({ mlpInner: i, inputDtype: a, outputDtype: o }),
    u = Qt(e, [{ u32: r }], `swiglu-params`);
  await e.runProgram({
    name: `swiglu`,
    source: l,
    cacheKey: c,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: u, type: `uniform` },
    ],
    workgroups: [r, Math.ceil((s ? i / 4 : i) / 64), 1],
  });
}
async function Ln(e, { xT: t, yT: n, count: r, alpha: i }) {
  let a = t.dtype,
    o = n.dtype,
    s = `axpy_2d_${a}_${o}`,
    c = un({ xDtype: a, yDtype: o }),
    { wgX: l, wgY: u } = cn(r),
    d = Qt(e, [{ u32: r }, { f32: i }, { u32: u }], `axpy-params`);
  await e.runProgram({
    name: `axpy`,
    source: c,
    cacheKey: s,
    bindings: [
      { tensor: t, type: `storage` },
      { tensor: n, type: `read-only-storage` },
      { buffer: d, type: `uniform` },
    ],
    workgroups: [u, l, 1],
  });
}
async function Rn(
  e,
  {
    packedT: t,
    meanT: n,
    stdT: r,
    outT: i = null,
    outputDtype: a = `float32`,
    latentC: o,
    latentH: s,
    latentW: c,
  },
) {
  let l = s / 2,
    u = c / 2;
  if (!Number.isInteger(l) || !Number.isInteger(u))
    throw Error(`latentH and latentW must be divisible by 2`);
  let d = o * 4,
    f = o * s * c,
    p = i?.dtype ?? a,
    m = kn(p),
    h = i ?? e.empty(p, [o, s, c], `flux-vae-latents`),
    g = `flux_pack_to_vae_lc${o}_h${s}_w${c}_${p}`,
    _ = `${An(p)}struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       packed: array<f32>;
@group(0) @binding(1) var<storage, read>       mean: array<f32>;
@group(0) @binding(2) var<storage, read>       bnStd: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<${m}>;
@group(0) @binding(4) var<uniform>             params: Params;

const LATENT_C: u32 = ${o}u;
const LATENT_H: u32 = ${s}u;
const LATENT_W: u32 = ${c}u;
const PATCH_W:  u32 = ${u}u;
const PATCHED_C: u32 = ${d}u;
const HW: u32 = ${s * c}u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= params.count) { return; }

  let c = i / HW;
  let rem = i - c * HW;
  let y = rem / LATENT_W;
  let x = rem - y * LATENT_W;
  let py = y & 1u;
  let px = x & 1u;
  let in_chan = c * 4u + py * 2u + px;
  let seq = (y / 2u) * PATCH_W + (x / 2u);
  let v = packed[seq * PATCHED_C + in_chan];
  out[i] = ${Mn(`v * bnStd[in_chan] + mean[in_chan]`, p)};
}
`,
    { wgX: v, wgY: y } = cn(f),
    b = Qt(e, [{ u32: f }, { u32: y }], `flux-vae-latents-params`);
  return (
    await e.runProgram({
      name: `flux_pack_to_vae`,
      source: _,
      cacheKey: g,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: r, type: `read-only-storage` },
        { tensor: h, type: `storage` },
        { buffer: b, type: `uniform` },
      ],
      workgroups: [y, v, 1],
    }),
    h
  );
}
function zn({ inputDtype: e = `float32`, outputDtype: t = `float16` }) {
  let n = kn(e),
    r = kn(t);
  return `${An(e, t)}struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${n}>;
@group(0) @binding(1) var<storage, read_write> y: array<${r}>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= params.count) { return; }
  y[i] = ${Mn(jn(`x[i]`, e), t)};
}
`;
}
async function Bn(e, { xT: t, yT: n, count: r }) {
  let i = t.dtype,
    a = n.dtype,
    o = `cast_${i}_${a}`,
    s = zn({ inputDtype: i, outputDtype: a }),
    { wgX: c, wgY: l } = cn(r),
    u = Qt(e, [{ u32: r }, { u32: l }], `cast-params`);
  await e.runProgram({
    name: `cast`,
    source: s,
    cacheKey: o,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: u, type: `uniform` },
    ],
    workgroups: [l, c, 1],
  });
}
function Vn({ dtype: e = `float16` }) {
  let t = kn(e);
  return `${An(e)}struct Params { rows: u32, cols: u32, scale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<${t}>;
@group(0) @binding(1) var<uniform>             params: Params;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

fn reduce_max(v: f32, tid: u32) -> f32 {
  partial[tid] = v;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partial[tid] = max(partial[tid], partial[tid + stride]);
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}

fn reduce_sum(v: f32, tid: u32) -> f32 {
  partial[tid] = v;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partial[tid] = partial[tid] + partial[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= params.rows) { return; }
  let tid = lid.x;
  let base = row * params.cols;
  var local_max = -3.4e38;
  for (var c: u32 = tid; c < params.cols; c = c + WG) {
    local_max = max(local_max, ${jn(`x[base + c]`, e)} * params.scale);
  }
  let row_max = reduce_max(local_max, tid);

  var local_sum = 0.0;
  for (var c: u32 = tid; c < params.cols; c = c + WG) {
    local_sum = local_sum + exp(${jn(`x[base + c]`, e)} * params.scale - row_max);
  }
  let inv_sum = 1.0 / reduce_sum(local_sum, tid);

  for (var c: u32 = tid; c < params.cols; c = c + WG) {
    let p = exp(${jn(`x[base + c]`, e)} * params.scale - row_max) * inv_sum;
    x[base + c] = ${Mn(`p`, e)};
  }
}
`;
}
async function Hn(e, { xT: t, rows: n, cols: r, scale: i = 1 }) {
  let a = t.dtype,
    o = `row_softmax_inplace_${a}`,
    s = Vn({ dtype: a }),
    c = Qt(e, [{ u32: n }, { u32: r }, { f32: i }], `row-softmax-params`);
  await e.runProgram({
    name: `row_softmax`,
    source: s,
    cacheKey: o,
    bindings: [
      { tensor: t, type: `storage` },
      { buffer: c, type: `uniform` },
    ],
    workgroups: [n, 1, 1],
  });
}
async function Un(e, { xT: t, factorT: n, count: r, period: i = 0 }) {
  let a = t.dtype,
    o = n.dtype,
    s = `mulbcast_2d_${a}_${o}`,
    c = dn({ xDtype: a, factorDtype: o }),
    { wgX: l, wgY: u } = cn(r),
    d = Qt(e, [{ u32: r }, { u32: i }, { u32: u }], `mulbcast-params`);
  await e.runProgram({
    name: `mulbcast`,
    source: c,
    cacheKey: s,
    bindings: [
      { tensor: t, type: `storage` },
      { tensor: n, type: `read-only-storage` },
      { buffer: d, type: `uniform` },
    ],
    workgroups: [u, l, 1],
  });
}
async function Wn(
  e,
  {
    qT: t,
    kT: n,
    vT: r,
    outT: i,
    seq: a,
    qHeads: o,
    kvHeads: s,
    headDim: c,
    causal: l,
  },
) {
  let u = t.dtype,
    d = i.dtype;
  if (!l && a >= 8) {
    let l = !!e.caps().subgroups,
      f =
        l && u === `float16` && d === `float16` && c === 128
          ? 32
          : u === `float16` && d === `float16` && c === 512
            ? 4
            : (u === `float16` && d === `float16` && c <= 512) ||
                (!l && u === `float32` && d === `float32` && c === 128)
              ? 2
              : 1,
      p = Number({}.BONSAI_FLASH_Q32_KSTEP ?? 64) === 64 ? 64 : 32,
      m = f === 32 ? p : 32,
      h = f > 1 && u === `float16`,
      g = `bonsai_flash_attn_q${f}_hd${c}_kt${m}_${u}_${d}_${l ? `sg` : `nosg`}_${h ? `qkh` : `qkf`}`,
      _ =
        f === 32
          ? gn({ headDim: c, kStep: m, inputDtype: u, outputDtype: d })
          : f === 4
            ? hn({
                headDim: c,
                kTile: m,
                inputDtype: u,
                outputDtype: d,
                useSubgroups: l,
                useHalfQk: h,
              })
            : f === 2
              ? mn({
                  headDim: c,
                  kTile: m,
                  inputDtype: u,
                  outputDtype: d,
                  useSubgroups: l,
                  useHalfQk: h,
                })
              : pn({
                  headDim: c,
                  kTile: m,
                  inputDtype: u,
                  outputDtype: d,
                  useSubgroups: l,
                }),
      v = Qt(
        e,
        [{ u32: a }, { u32: o }, { u32: s }, { u32: 0 }],
        `flash-attn-params`,
      );
    await e.runProgram({
      name: `flash_attention`,
      source: _,
      cacheKey: g,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: r, type: `read-only-storage` },
        { tensor: i, type: `storage` },
        { buffer: v, type: `uniform` },
      ],
      workgroups: [Math.ceil(a / f), o, 1],
    });
    return;
  }
  let f = `bonsai_attn_hd${c}_${u}_${d}`,
    p = fn({ headDim: c, inputDtype: u, outputDtype: d }),
    m = Qt(
      e,
      [{ u32: a }, { u32: o }, { u32: s }, { u32: +!!l }],
      `attn-params`,
    );
  await e.runProgram({
    name: `attention`,
    source: p,
    cacheKey: f,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: m, type: `uniform` },
    ],
    workgroups: [a, o, 1],
  });
}
async function Gn(e, { aT: t, bT: n, outT: r, aElems: i, totalElems: a }) {
  let o = r.dtype,
    s = `concat_${o}`,
    c = _n({ dtype: o }),
    { wgX: l, wgY: u } = cn(a),
    d = Qt(e, [{ u32: a }, { u32: i }, { u32: u }], `concat-params`);
  await e.runProgram({
    name: `concat`,
    source: c,
    cacheKey: s,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: d, type: `uniform` },
    ],
    workgroups: [u, l, 1],
  });
}
function Kn(e, { M: t, inFeatures: n, groupSize: r }) {
  let i = n / r;
  return {
    aQT: e.empty(`uint32`, [t, n / 4], `a-quantized`),
    scaleAT: e.empty(`float32`, [t, i], `a-scale`),
    sumAT: e.empty(`float32`, [t, i], `a-sum`),
  };
}
async function qn(
  e,
  { aT: t, aQT: n, scaleAT: r, sumAT: i, M: a, inFeatures: o, groupSize: s },
) {
  let c = t.dtype,
    l = `quantize_a_i8_${c}_g${s}_k${o}`,
    u = xn({ inFeatures: o, groupSize: s, inputDtype: c }),
    d = Qt(e, [{ u32: a }], `quantizeA-params`),
    f = o / s;
  await e.runProgram({
    name: `quantize_a_i8`,
    source: u,
    cacheKey: l,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { tensor: r, type: `storage` },
      { tensor: i, type: `storage` },
      { buffer: d, type: `uniform` },
    ],
    workgroups: [f, a, 1],
  });
}
async function Jn(
  e,
  {
    aQT: t,
    scaleAT: n,
    sumAT: r,
    bitsT: i,
    sbT: a,
    outT: o,
    M: s,
    inFeatures: c,
    outFeatures: l,
    bits: u,
    groupSize: d,
  },
) {
  if (s < 2) throw Error(`gpuMlxMatmulDP4A requires M >= 2`);
  if (u !== 2 || d !== 128)
    throw Error(`DP4A path requires bits=2, groupSize=128`);
  let f = 64;
  for (let e of [128, 64, 32])
    if (l % e === 0) {
      f = e;
      break;
    }
  let p = c <= 9216 ? 3 : 2;
  s < p && (p = 1);
  let m = o.dtype,
    h = a.dtype,
    g = `mlxmatmul_dp4a_${m}_${h}_b${u}_g${d}_i${c}_o${l}_tm${p}_tn${f}`,
    _ = Cn({
      bits: u,
      groupSize: d,
      inFeatures: c,
      outFeatures: l,
      tileM: p,
      tileN: f,
      outputDtype: m,
      scaleBiasDtype: h,
    }),
    v = Qt(e, [{ u32: s }], `mlxmatmul-dp4a-params`),
    y = Math.ceil(l / f),
    b = Math.ceil(s / p);
  await e.runProgram({
    name: `mlx_matmul`,
    source: _,
    cacheKey: g,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `read-only-storage` },
      { tensor: a, type: `read-only-storage` },
      { tensor: o, type: `storage` },
      { buffer: v, type: `uniform` },
    ],
    workgroups: [y, b, 1],
  });
}
async function Yn(
  e,
  {
    bitsT: t,
    sbT: n,
    outT: r,
    inFeatures: i,
    outFeatures: a,
    bits: o,
    groupSize: s,
    outputLayout: c = `out-k`,
  },
) {
  if (r.dtype !== `float16`)
    throw Error(`gpuMlxDequantizeToF16 requires a float16 output tensor`);
  let l = n.scaleBiasDtype ?? n.dtype,
    u = n.scaleBiasLayout ?? `out-group`,
    d = !!n.ternaryBiasFromScale,
    f = `mlx_dequant_f16_${l}_${u}_${c}_b${o}_g${s}_i${i}_o${a}_tb${+!!d}`,
    p = Sn({
      bits: o,
      groupSize: s,
      inFeatures: i,
      outFeatures: a,
      scaleBiasDtype: l,
      scaleBiasLayout: u,
      outputLayout: c,
      assumeTernaryBias: d,
    }),
    m = a * (i / (32 / o)),
    { wgX: h, wgY: g } = cn(m),
    _ = Qt(e, [{ u32: m }, { u32: g }], `mlx-dequant-f16-params`);
  await e.runProgram({
    name: `mlx_dequant_f16`,
    source: p,
    cacheKey: f,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: _, type: `uniform` },
    ],
    workgroups: [g, h, 1],
  });
}
async function Xn(
  e,
  {
    aT: t,
    bitsT: n,
    sbT: r,
    outT: i,
    M: a,
    inFeatures: o,
    outFeatures: s,
    bits: c,
    groupSize: l,
    useF16: u = !1,
    assumeTernaryBias: d = !1,
    aRowOffset: f = 0,
    scaleBiasLayout: p = `out-group`,
    tileM: m = 32,
    tileN: h = null,
  },
) {
  if (!e.caps().subgroupMatrix)
    throw Error(
      `gpuMlxMatmulSubgroupMatrix requires chromium-experimental-subgroup-matrix`,
    );
  if (
    !((c === 1 && l === 128) || (c === 2 && l === 128) || (c === 4 && l === 64))
  )
    throw Error(
      `subgroup-matrix path requires bits=1/groupSize=128, bits=2/groupSize=128, or bits=4/groupSize=64`,
    );
  if (s % 64 != 0)
    throw Error(`subgroup-matrix path requires outFeatures divisible by 64`);
  if (o % 32 != 0)
    throw Error(`subgroup-matrix path requires inFeatures divisible by 32`);
  let g = m,
    _ = u && g === 32 && s % 128 == 0 && (a >= g || (o === 3072 && s >= 18432)),
    v = h ?? (_ ? 128 : 64),
    y = t.dtype,
    b = r.dtype,
    x = i.dtype,
    S = u ? `f16` : `f32`,
    C = a % g === 0,
    w = f !== 0,
    T = `mlxmatmul_sgmat_${S}_${y}_${b}_${x}_${p === `group-out` ? `go` : `og`}_b${c}_g${l}_i${o}_o${s}_tm${g}_tn${v}_fm${+!!C}_tb${+!!d}_ao${+!!w}`,
    E = wn({
      bits: c,
      groupSize: l,
      inFeatures: o,
      outFeatures: s,
      useF16: u,
      tileM: g,
      tileN: v,
      inputDtype: y,
      scaleBiasDtype: b,
      outputDtype: x,
      scaleBiasLayout: p,
      assumeFullM: C,
      assumeTernaryBias: d,
      useARowOffset: w,
    }),
    D = Qt(e, [{ u32: a }, { u32: f }], `mlxmatmul-sgmat-params`);
  await e.runProgram({
    name: `mlx_matmul`,
    source: E,
    cacheKey: T,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: D, type: `uniform` },
    ],
    workgroups: [Math.ceil(s / v), Math.ceil(a / g), 1],
  });
}
async function Zn(
  e,
  {
    aT: t,
    bitsT: n,
    sbT: r,
    outT: i,
    M: a,
    inFeatures: o,
    outFeatures: s,
    tileM: c = null,
    tileN: l = 256,
    nPerThread: u = null,
    assumeBiasNegHalfScale: d = !1,
    scaleOnly: f = !1,
  },
) {
  if (t.dtype !== `float16` || r.dtype !== `float16` || i.dtype !== `float16`)
    throw Error(
      `gpuMlxMatmulBinaryLut4 requires f16 A, scaleBias, and output tensors`,
    );
  if (o % 128 != 0)
    throw Error(`gpuMlxMatmulBinaryLut4 requires K divisible by 128`);
  ((c ??=
    a < 128
      ? a >= 8
        ? 10
        : Math.max(1, a)
      : d
        ? s === 3072 && o === 3072
          ? 10
          : s >= 27648
            ? 14
            : 16
        : s === 3072 && o === 3072
          ? 6
          : s === 3072 && o >= 9216
            ? 15
            : 13),
    u ??
      (d && a >= 128 && o === 3072 && s >= 9216
        ? ((u = 2), l === 256 && (l = 512))
        : (u = 1)));
  let p = !!(f && d),
    m = `mlxmatmul_binary_lut4_${o}_${s}_tm${c}_tn${l}_npt${u}_nh${+!!d}_so${+!!p}`,
    h = bn({
      inFeatures: o,
      outFeatures: s,
      tileM: c,
      tileN: l,
      nPerThread: u,
      assumeBiasNegHalfScale: d,
      scaleOnly: p,
    }),
    g = Qt(e, [{ u32: a }], `mlxmatmul-binary-lut4-params`);
  await e.runProgram({
    name: `mlx_matmul_binary_lut4`,
    source: h,
    cacheKey: m,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: g, type: `uniform` },
    ],
    workgroups: [Math.ceil(s / l), Math.ceil(a / c), 1],
  });
}
async function Qn(
  e,
  {
    aT: t,
    bitsT: n,
    sbT: r,
    outT: i,
    M: a,
    inFeatures: o,
    outFeatures: s,
    bits: c,
    groupSize: l,
  },
) {
  let u = t.dtype,
    d = r.dtype,
    f = i.dtype;
  if (
    a >= 16 &&
    e.caps().subgroupMatrix &&
    e.caps().f16 &&
    u === `float16` &&
    f === `float16` &&
    ((c === 1 && l === 128) ||
      (c === 2 && l === 128) ||
      (c === 4 && l === 64)) &&
    o % 32 == 0 &&
    s % 64 == 0
  ) {
    await Xn(e, {
      aT: t,
      bitsT: n,
      sbT: r,
      outT: i,
      M: a,
      inFeatures: o,
      outFeatures: s,
      bits: c,
      groupSize: l,
      useF16: !0,
      assumeTernaryBias: c === 2,
    });
    return;
  }
  let p, m;
  if (
    (a === 1
      ? u === `float32` && d === `float32` && f === `float32`
        ? (p = 0)
        : s % 64 == 0
          ? ((p = 1), (m = 64))
          : s % 32 == 0
            ? ((p = 1), (m = 32))
            : s % 16 == 0
              ? ((p = 1), (m = 16))
              : (p = 0)
      : s % 64 == 0
        ? ((p = a >= 4 ? 4 : a >= 2 ? 2 : 1), (m = 64))
        : s % 32 == 0
          ? ((p = a >= 8 ? 8 : a >= 4 ? 4 : a >= 2 ? 2 : 1), (m = 32))
          : s % 16 == 0
            ? ((p = a >= 16 ? 16 : a >= 8 ? 8 : 4), (m = 16))
            : (p = 0),
    p === 0)
  ) {
    if (t.dtype !== `float32` || r.dtype !== `float32` || i.dtype !== `float32`)
      throw Error(`MLX matmul fallback path only supports float32 tensors`);
    let u = `mlxmatmul_b${c}_g${l}_i${o}_o${s}`,
      d = vn({ bits: c, groupSize: l, inFeatures: o, outFeatures: s }),
      f = Qt(e, [{ u32: a }], `mlxmatmul-params`);
    await e.runProgram({
      name: `mlx_matmul`,
      source: d,
      cacheKey: u,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: r, type: `read-only-storage` },
        { tensor: i, type: `storage` },
        { buffer: f, type: `uniform` },
      ],
      workgroups: [s, a, 1],
    });
    return;
  }
  let h = o / l,
    g = 1;
  for (let e of [4, 2])
    if (h % e === 0 && p * e * l * 4 <= 8192) {
      g = e;
      break;
    }
  let _ = 1;
  for (let e of [4, 2])
    if (s % (m * e) === 0) {
      _ = e;
      break;
    }
  let v = m * _,
    y = `mlxmatmul_tiled_${u}_${d}_${f}_b${c}_g${l}_i${o}_o${s}_m${p}_n${m}_kgc${g}_npt${_}`,
    b = yn({
      bits: c,
      groupSize: l,
      inFeatures: o,
      outFeatures: s,
      mTile: p,
      outTile: m,
      kGroupsPerChunk: g,
      nPerThread: _,
      inputDtype: u,
      outputDtype: f,
      scaleBiasDtype: d,
    }),
    x = Qt(e, [{ u32: a }], `mlxmatmul-params`),
    S = Math.ceil(s / v),
    C = Math.ceil(a / p);
  await e.runProgram({
    name: `mlx_matmul`,
    source: b,
    cacheKey: y,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: x, type: `uniform` },
    ],
    workgroups: [S, C, 1],
  });
}
async function $n(
  e,
  {
    aT: t,
    bitsT: n,
    sbT: r,
    outT: i,
    M: a,
    inFeatures: o,
    outFeatures: s,
    bits: c,
    groupSize: l,
  },
) {
  let u = t.dtype,
    d = r.dtype,
    f = i.dtype;
  if (
    a >= 16 &&
    e.caps().subgroupMatrix &&
    e.caps().f16 &&
    u === `float16` &&
    f === `float16` &&
    c === 4 &&
    l === 64 &&
    o % 32 == 0 &&
    s % 64 == 0
  ) {
    await Xn(e, {
      aT: t,
      bitsT: n,
      sbT: r,
      outT: i,
      M: a,
      inFeatures: o,
      outFeatures: s,
      bits: c,
      groupSize: l,
      useF16: !0,
      assumeTernaryBias: !1,
    });
    return;
  }
  let p, m;
  if (s % 64 == 0) ((p = a >= 4 ? 4 : a >= 2 ? 2 : 1), (m = 64));
  else if (s % 32 == 0)
    ((p = a >= 8 ? 8 : a >= 4 ? 4 : a >= 2 ? 2 : 1), (m = 32));
  else if (s % 16 == 0)
    ((p = a >= 16 ? 16 : a >= 8 ? 8 : a >= 4 ? 4 : 1), (m = 16));
  else
    throw Error(`Bonsai text MLX matmul requires outFeatures divisible by 16`);
  let h = o / l,
    g = 1;
  for (let e of [4, 2])
    if (h % e === 0 && p * e * l * 4 <= 8192) {
      g = e;
      break;
    }
  let _ = 1;
  for (let e of [4, 2])
    if (s % (m * e) === 0) {
      _ = e;
      break;
    }
  let v = m * _,
    y = `bonsai_mlxmatmul_tiled_${u}_${d}_${f}_b${c}_g${l}_i${o}_o${s}_m${p}_n${m}_kgc${g}_npt${_}`,
    b = yn({
      bits: c,
      groupSize: l,
      inFeatures: o,
      outFeatures: s,
      mTile: p,
      outTile: m,
      kGroupsPerChunk: g,
      nPerThread: _,
      inputDtype: u,
      outputDtype: f,
      scaleBiasDtype: d,
    }),
    x = Qt(e, [{ u32: a }], `mlxmatmul-params`);
  await e.runProgram({
    name: `mlx_matmul`,
    source: b,
    cacheKey: y,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: x, type: `uniform` },
    ],
    workgroups: [Math.ceil(s / v), Math.ceil(a / p), 1],
  });
}
function er({
  inFeatures: e,
  outFeatures: t,
  hasBias: n,
  inputDtype: r = `float32`,
  weightDtype: i = `float32`,
  biasDtype: a = `float32`,
  outputDtype: o = `float32`,
}) {
  let s = kn(r),
    c = kn(i),
    l = kn(a),
    u = kn(o);
  return `${An(r, i, a, o)}struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${s}>;
@group(0) @binding(1) var<storage, read>       w: array<${c}>;
${
  n
    ? `@group(0) @binding(2) var<storage, read>       b: array<${l}>;
`
    : ``
}@group(0) @binding(${n ? 3 : 2}) var<storage, read_write> y: array<${u}>;
@group(0) @binding(${n ? 4 : 3}) var<uniform>             params: Params;

const IN_F: u32 = ${e}u;
const OUT_F: u32 = ${t}u;
const WG: u32 = 64u;

var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x;
  let m = wg.y;
  if (n >= OUT_F || m >= params.M) { return; }
  let tid = lid.x;
  let wBase = n * IN_F;
  let xBase = m * IN_F;

  var acc: f32 = 0.0;
  for (var i: u32 = tid; i < IN_F; i = i + WG) {
    acc = acc + ${jn(`x[xBase + i]`, r)} * ${jn(`w[wBase + i]`, i)};
  }
  partial[tid] = acc;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    ${n ? `y[m * OUT_F + n] = ${Mn(`partial[0] + ${jn(`b[n]`, a)}`, o)};` : `y[m * OUT_F + n] = ${Mn(`partial[0]`, o)};`}
  }
}
`;
}
function tr({
  inFeatures: e,
  outFeatures: t,
  outputDtype: n = `float16`,
  mTile: r = 16,
  nTile: i = 8,
  nPerThread: a = 8,
  kTile: o = 64,
}) {
  if (e % 4 != 0 || o % 4 != 0)
    throw Error(`dense f16 tiled matmul requires K and kTile divisible by 4`);
  if (r * i > 256)
    throw Error(
      `dense f16 tiled matmul tile exceeds max workgroup invocations`,
    );
  if (n !== `float16` && n !== `float32`)
    throw Error(`dense f16 tiled matmul output must be f16/f32`);
  let s = r * i,
    c = o / 4,
    l = i * a;
  if ((r * c + l * c) * 8 > 16 * 1024)
    throw Error(`dense f16 tiled matmul exceeds 16KB workgroup storage`);
  let u = kn(n),
    d = Array.from({ length: a }, (e, t) => `  var acc${t}: f32 = 0.0;`).join(`
`),
    f = Array.from(
      { length: a },
      (e, t) => `        let b_idx${t} = (n_local + ${t * i}u) * K_TILE_V4 + kv;
        acc${t} = acc${t} + dot(vec4<f32>(a_vec), vec4<f32>(bTile[b_idx${t}]));`,
    ).join(`
`),
    p = Array.from({ length: a }, (e, t) => {
      let r = `n0 + ${t * i}u`;
      return `    if (${r} < N) { y[m * N + ${r}] = ${Mn(`acc${t}`, n)}; }`;
    }).join(`
`);
  return `enable f16;
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<${u}>;
@group(0) @binding(3) var<uniform>             params: Params;

const K: u32 = ${e}u;
const N: u32 = ${t}u;
const K_V4: u32 = ${e / 4}u;
const M_TILE: u32 = ${r}u;
const N_TILE: u32 = ${i}u;
const OUT_TILE: u32 = ${l}u;
const K_TILE: u32 = ${o}u;
const K_TILE_V4: u32 = ${c}u;
const WG: u32 = ${s}u;

var<workgroup> aTile: array<vec4<f16>, ${r * c}>;
var<workgroup> bTile: array<vec4<f16>, ${l * c}>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let m_local = tid / N_TILE;
  let n_local = tid - m_local * N_TILE;
  let m = wg.y * M_TILE + m_local;
  let n0 = wg.x * OUT_TILE + n_local;
${d}

  for (var k_base: u32 = 0u; k_base < K; k_base = k_base + K_TILE) {
    let k_base_v4 = k_base / 4u;
    for (var i: u32 = tid; i < ${r * c}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let gm = wg.y * M_TILE + tm;
      if (gm < params.M) {
        aTile[i] = a[gm * K_V4 + k_base_v4 + kv];
      } else {
        aTile[i] = vec4<f16>(0.0h);
      }
    }
    for (var i: u32 = tid; i < ${l * c}u; i = i + WG) {
      let tn = i / K_TILE_V4;
      let kv = i - tn * K_TILE_V4;
      let gn = wg.x * OUT_TILE + tn;
      if (gn < N) {
        bTile[i] = w[gn * K_V4 + k_base_v4 + kv];
      } else {
        bTile[i] = vec4<f16>(0.0h);
      }
    }
    workgroupBarrier();

    if (m < params.M) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let a_idx = m_local * K_TILE_V4 + kv;
        let a_vec = aTile[a_idx];
${f}
      }
    }
    workgroupBarrier();
  }

  if (m < params.M) {
${p}
  }
}
`;
}
async function nr(
  e,
  { aT: t, wT: n, outT: r, M: i, inFeatures: a, outFeatures: o },
) {
  if (t.dtype !== `float16` || n.dtype !== `float16`)
    throw Error(`gpuDenseF16TiledMatmul requires f16 input and weight tensors`);
  let s = r.dtype,
    c = `dense_f16_tiled_${a}_${o}_${s}_tm32_tn4_npt16_tk64`,
    l = tr({
      inFeatures: a,
      outFeatures: o,
      outputDtype: s,
      mTile: 32,
      nTile: 4,
      nPerThread: 16,
      kTile: 64,
    }),
    u = Qt(e, [{ u32: i }], `dense-f16-tiled-params`);
  await e.runProgram({
    name: `dense_f16_tiled`,
    source: l,
    cacheKey: c,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: u, type: `uniform` },
    ],
    workgroups: [Math.ceil(o / 64), Math.ceil(i / 32), 1],
  });
}
function rr({
  inFeatures: e,
  outFeatures: t,
  mTile: n = 32,
  nTile: r = 16,
  rowPerThread: i = 4,
  kTile: a = 64,
  accumDtype: o = `float16`,
  assumeFullN: s = !1,
}) {
  if (e % 4 != 0 || a % 4 != 0 || e % a !== 0)
    throw Error(`packed dense f16 matmul requires K divisible by 4 and kTile`);
  if (t % 4 != 0)
    throw Error(`packed dense f16 matmul requires N divisible by 4`);
  if (n % i !== 0)
    throw Error(
      `packed dense f16 matmul requires mTile divisible by rowPerThread`,
    );
  let c = r,
    l = n / i,
    u = c * l;
  if (u > 256)
    throw Error(`packed dense f16 matmul exceeds max workgroup invocations`);
  let d = a / 4;
  if ((n * d + a * c) * 8 > 16 * 1024)
    throw Error(`packed dense f16 matmul exceeds 16KB workgroup storage`);
  let f = c * 4;
  if (o !== `float16` && o !== `float32`)
    throw Error(`packed dense f16 matmul accumulation must be f16 or f32`);
  let p = o === `float32`,
    m = p ? `f32` : `f16`,
    h = p ? `vec4<f32>(0.0)` : `vec4<f16>(0.0h)`,
    g = Array.from({ length: i }, (e, t) => `  var acc${t}: vec4<${m}> = ${h};`)
      .join(`
`),
    _ = Array.from({ length: i }, (e, t) =>
      p
        ? `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(vec4<f32>(b3), vec4<f32>(f32(a_vec${t}.w)), fma(vec4<f32>(b2), vec4<f32>(f32(a_vec${t}.z)), fma(vec4<f32>(b1), vec4<f32>(f32(a_vec${t}.y)), fma(vec4<f32>(b0), vec4<f32>(f32(a_vec${t}.x)), acc${t}))));`
        : `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(b3, vec4<f16>(a_vec${t}.w), fma(b2, vec4<f16>(a_vec${t}.z), fma(b1, vec4<f16>(a_vec${t}.y), fma(b0, vec4<f16>(a_vec${t}.x), acc${t}))));`,
    ).join(`
`),
    v = s ? `` : `n_group < N_V4 && `,
    y = Array.from(
      { length: i },
      (e, t) =>
        `  if (${v}m_base + ${t}u < params.M) { y[(m_base + ${t}u) * N_V4 + n_group] = ${p ? `vec4<f16>(acc${t})` : `acc${t}`}; }`,
    ).join(`
`),
    b = s
      ? `      bTile[i] = w[(k_base + kk) * N_V4 + b_group];`
      : `      if (b_group < N_V4) {
        bTile[i] = w[(k_base + kk) * N_V4 + b_group];
      } else {
        bTile[i] = vec4<f16>(0.0h);
      }`;
  return `enable f16;
struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             params: Params;

const K: u32 = ${e}u;
const N: u32 = ${t}u;
const K_V4: u32 = ${e / 4}u;
const N_V4: u32 = ${t / 4}u;
const M_TILE: u32 = ${n}u;
const WG_X: u32 = ${c}u;
const WG_Y: u32 = ${l}u;
const ROW_PER_THREAD: u32 = ${i}u;
const OUT_TILE: u32 = ${f}u;
const K_TILE: u32 = ${a}u;
const K_TILE_V4: u32 = ${d}u;
const WG: u32 = ${u}u;

var<workgroup> aTile: array<vec4<f16>, ${n * d}>;
var<workgroup> bTile: array<vec4<f16>, ${a * c}>;

@compute @workgroup_size(${c}, ${l}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * M_TILE + row_base;
  let n_group = wg.x * WG_X + lx;
${g}

  for (var k_base: u32 = 0u; k_base < K; k_base = k_base + K_TILE) {
    let k_base_v4 = k_base / 4u;
    for (var i: u32 = tid; i < ${n * d}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let gm = wg.y * M_TILE + tm;
      if (gm < params.M) {
        aTile[i] = a[gm * K_V4 + k_base_v4 + kv];
      } else {
        aTile[i] = vec4<f16>(0.0h);
      }
    }
    for (var i: u32 = tid; i < ${a * c}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let b_group = wg.x * WG_X + nx;
${b}
    }
    workgroupBarrier();

    if (m_base < params.M) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let b0 = bTile[(kv * 4u + 0u) * WG_X + lx];
        let b1 = bTile[(kv * 4u + 1u) * WG_X + lx];
        let b2 = bTile[(kv * 4u + 2u) * WG_X + lx];
        let b3 = bTile[(kv * 4u + 3u) * WG_X + lx];
${_}
      }
    }
    workgroupBarrier();
  }

${y}
}
`;
}
async function ir(
  e,
  {
    aT: t,
    wT: n,
    outT: r,
    M: i,
    inFeatures: a,
    outFeatures: o,
    mTile: s = null,
    nTile: c = null,
    rowPerThread: l = null,
    kTile: u = null,
    accumDtype: d = null,
  },
) {
  if (t.dtype !== `float16` || n.dtype !== `float16` || r.dtype !== `float16`)
    throw Error(
      `gpuDenseF16PackedVec4NMatmul requires f16 input, weight, and output tensors`,
    );
  let f =
    i >= 128
      ? {
          mTile: 88,
          nTile: 16,
          rowPerThread: 11,
          kTile: 32,
          accumDtype: `float16`,
        }
      : {
          mTile: 32,
          nTile: 16,
          rowPerThread: 4,
          kTile: 64,
          accumDtype: `float16`,
        };
  ((s ??= f.mTile),
    (c ??= f.nTile),
    (l ??= f.rowPerThread),
    (u ??= f.kTile),
    (d ??= f.accumDtype));
  let p = o % (c * 4) == 0,
    m = `dense_f16_packed_vec4n_${d}_${a}_${o}_tm${s}_tn${c}_rpt${l}_tk${u}_fn${+!!p}`,
    h = rr({
      inFeatures: a,
      outFeatures: o,
      mTile: s,
      nTile: c,
      rowPerThread: l,
      kTile: u,
      accumDtype: d,
      assumeFullN: p,
    }),
    g = Qt(e, [{ u32: i }], `dense-f16-packed-params`);
  await e.runProgram({
    name: `dense_f16_packed`,
    source: h,
    cacheKey: m,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: g, type: `uniform` },
    ],
    workgroups: [Math.ceil(o / (c * 4)), Math.ceil(i / s), 1],
  });
}
function ar({
  inFeatures: e,
  outFeatures: t,
  inputDtype: n = `float32`,
  weightDtype: r = `float32`,
  outputDtype: i = `float16`,
}) {
  if (e % 32 != 0)
    throw Error(`dense subgroup matmul requires inFeatures divisible by 32`);
  if (t % 64 != 0)
    throw Error(`dense subgroup matmul requires outFeatures divisible by 64`);
  if (i !== `float16` && i !== `float32`)
    throw Error(`dense subgroup matmul requires f16/f32 output`);
  let a = kn(n),
    o = kn(r),
    s = kn(i),
    c = s,
    l = i === `float16` ? `0.0h` : `0.0`,
    u =
      i === `float16`
        ? n === `float16`
          ? `x[a_global * IN_F + k]`
          : `f16(x[a_global * IN_F + k])`
        : jn(`x[a_global * IN_F + k]`, n),
    d =
      i === `float16`
        ? r === `float16`
          ? `w[w_global * IN_F + k]`
          : `f16(w[w_global * IN_F + k])`
        : jn(`w[w_global * IN_F + k]`, r);
  return `${An(n, r, i)}enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${a}>;
@group(0) @binding(1) var<storage, read>       w: array<${o}>;
@group(0) @binding(2) var<storage, read_write> y: array<${s}>;
@group(0) @binding(3) var<uniform>             params: Params;

const IN_F:       u32 = ${e}u;
const OUT_F:      u32 = ${t}u;
const TILE_COLS:  u32 = 64u;
const TILE_ROWS:  u32 = 32u;
const TILE_K:     u32 = 32u;
const SUB_COLS:   u32 = 32u;
const SUB_ROWS:   u32 = 16u;

var<workgroup> tile_A: array<${c}, 32 * 32>;
var<workgroup> tile_B: array<${c}, 64 * 32>;
var<workgroup> scratch: array<array<array<${c}, 64>, 4>, 4>;

fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let a_global: u32 = tile_base + row;
  let col: u32 = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let k: u32 = k_idx + col + col_offset;
    if (a_global < params.M) {
      tile_A[row * TILE_K + col + col_offset] = ${u};
    } else {
      tile_A[row * TILE_K + col + col_offset] = ${l};
    }
  }
}

fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let w_global: u32 = tile_base + row;
  let col: u32 = c_idx * 16u;
  for (var i: u32 = 0u; i < 16u; i++) {
    let k: u32 = k_idx + col + i;
    tile_B[row * TILE_K + col + i] = ${d};
  }
}

fn storeOutput(offset: u32, row: u32, col: u32, src_slot: u32, row_limit: i32) {
  if (row_limit > 0 && row < u32(row_limit)) {
    let col2: u32 = col + 1u;
    y[offset + row * OUT_F + col]       = scratch[src_slot][0][row * 8u + col];
    y[offset + row * OUT_F + col + 8u]  = scratch[src_slot][1][row * 8u + col];
    y[offset + row * OUT_F + col + 16u] = scratch[src_slot][2][row * 8u + col];
    y[offset + row * OUT_F + col + 24u] = scratch[src_slot][3][row * 8u + col];

    y[offset + row * OUT_F + col2]       = scratch[src_slot][0][row * 8u + col2];
    y[offset + row * OUT_F + col2 + 8u]  = scratch[src_slot][1][row * 8u + col2];
    y[offset + row * OUT_F + col2 + 16u] = scratch[src_slot][2][row * 8u + col2];
    y[offset + row * OUT_F + col2 + 24u] = scratch[src_slot][3][row * 8u + col2];
  }
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let a_global_base: u32 = workgroup_id.y * TILE_ROWS;
  let w_global_base: u32 = workgroup_id.x * TILE_COLS;

  let subtile_id: u32 = local_idx / sg_size;
  let subtile_idx: u32 = subtile_id / 2u;
  let subtile_idy: u32 = subtile_id % 2u;
  let base_A: u32 = subtile_idy * SUB_ROWS;
  let base_B: u32 = subtile_idx * SUB_COLS;

  var matC00: subgroup_matrix_result<${c}, 8, 8>;
  var matC01: subgroup_matrix_result<${c}, 8, 8>;
  var matC02: subgroup_matrix_result<${c}, 8, 8>;
  var matC03: subgroup_matrix_result<${c}, 8, 8>;
  var matC10: subgroup_matrix_result<${c}, 8, 8>;
  var matC11: subgroup_matrix_result<${c}, 8, 8>;
  var matC12: subgroup_matrix_result<${c}, 8, 8>;
  var matC13: subgroup_matrix_result<${c}, 8, 8>;

  for (var kidx: u32 = 0u; kidx < IN_F; kidx += TILE_K) {
    loadSHMA(a_global_base, kidx, local_idx / 4u, local_idx % 4u);
    loadSHMB(w_global_base, kidx, local_idx / 2u, local_idx % 2u);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step += 8u) {
      let matrix_a_offset: u32 = subtile_idy * SUB_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<${c}, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<${c}, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset: u32 = subtile_idx * SUB_COLS * TILE_K + step;
      var matB0: subgroup_matrix_right<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<${c}, 8, 8>>(&tile_B, matrix_b_offset, true, TILE_K);
      var matB1: subgroup_matrix_right<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<${c}, 8, 8>>(&tile_B, matrix_b_offset +  8u * TILE_K, true, TILE_K);
      var matB2: subgroup_matrix_right<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<${c}, 8, 8>>(&tile_B, matrix_b_offset + 16u * TILE_K, true, TILE_K);
      var matB3: subgroup_matrix_right<${c}, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<${c}, 8, 8>>(&tile_B, matrix_b_offset + 24u * TILE_K, true, TILE_K);

      matC00 = subgroupMatrixMultiplyAccumulate(matA0, matB0, matC00);
      matC01 = subgroupMatrixMultiplyAccumulate(matA0, matB1, matC01);
      matC02 = subgroupMatrixMultiplyAccumulate(matA0, matB2, matC02);
      matC03 = subgroupMatrixMultiplyAccumulate(matA0, matB3, matC03);
      matC10 = subgroupMatrixMultiplyAccumulate(matA1, matB0, matC10);
      matC11 = subgroupMatrixMultiplyAccumulate(matA1, matB1, matC11);
      matC12 = subgroupMatrixMultiplyAccumulate(matA1, matB2, matC12);
      matC13 = subgroupMatrixMultiplyAccumulate(matA1, matB3, matC13);
    }
    workgroupBarrier();
  }

  subgroupMatrixStore(&scratch[subtile_id][0], 0u, matC00, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][1], 0u, matC01, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][2], 0u, matC02, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][3], 0u, matC03, false, 8u);
  let row: u32 = sg_id / 4u;
  let col: u32 = (sg_id % 4u) * 2u;
  var matrix_c_offset: u32 = (a_global_base + base_A) * OUT_F + w_global_base + base_B;
  var row_limit: i32 = i32(params.M) - i32(a_global_base + base_A);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit);

  subgroupMatrixStore(&scratch[subtile_id][0], 0u, matC10, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][1], 0u, matC11, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][2], 0u, matC12, false, 8u);
  subgroupMatrixStore(&scratch[subtile_id][3], 0u, matC13, false, 8u);
  matrix_c_offset = matrix_c_offset + 8u * OUT_F;
  row_limit = i32(params.M) - i32(a_global_base + base_A + 8u);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit);
}
`;
}
async function or(
  e,
  { aT: t, wT: n, bT: r, outT: i, M: a, inFeatures: o, outFeatures: s },
) {
  let c = t.dtype,
    l = n.dtype,
    u = r?.dtype ?? `float32`,
    d = i.dtype;
  if (
    !r &&
    e.caps().subgroupMatrix &&
    ((d === `float16` && e.caps().f16) || (d === `float32` && a >= 32)) &&
    o % 32 == 0 &&
    s % 64 == 0
  ) {
    let r = `dense_sgmat_${d === `float16` ? `f16` : `f32`}_${o}_${s}_${c}_${l}_${d}_tm32_tn64`,
      u = ar({
        inFeatures: o,
        outFeatures: s,
        inputDtype: c,
        weightDtype: l,
        outputDtype: d,
      }),
      f = Qt(e, [{ u32: a }], `dense-sgmat-params`);
    await e.runProgram({
      name: `dense_matmul`,
      source: u,
      cacheKey: r,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: i, type: `storage` },
        { buffer: f, type: `uniform` },
      ],
      workgroups: [Math.ceil(s / 64), Math.ceil(a / 32), 1],
    });
    return;
  }
  let f = `dense_m_${o}_${s}_${r ? `b` : `nb`}_${c}_${l}_${u}_${d}`,
    p = er({
      inFeatures: o,
      outFeatures: s,
      hasBias: !!r,
      inputDtype: c,
      weightDtype: l,
      biasDtype: u,
      outputDtype: d,
    }),
    m = Qt(e, [{ u32: a }], `dense-params`),
    h = [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      ...(r ? [{ tensor: r, type: `read-only-storage` }] : []),
      { tensor: i, type: `storage` },
      { buffer: m, type: `uniform` },
    ];
  await e.runProgram({
    name: `dense_matmul`,
    source: p,
    cacheKey: f,
    bindings: h,
    workgroups: [s, a, 1],
  });
}
var sr = [8, 17, 26],
  cr = class e {
    constructor({
      rt: e,
      config: t,
      embedBitsT: n,
      embedScaleBiasT: r,
      layers: i,
      finalNormT: a,
      hiddenLayerIndices: o,
    }) {
      ((this.rt = e),
        (this.config = t),
        (this.embedBitsT = n),
        (this.embedScaleBiasT = r),
        (this.layers = i),
        (this.finalNormT = a),
        (this.hiddenLayerIndices = o));
    }
    static async fromMlxSafeTensors({
      rt: t,
      config: n,
      safeTensors: r,
      hiddenLayerIndices: i = sr,
      onProgress: a = null,
      concurrency: o,
      chunkMaxBytes: s,
      signal: c,
    }) {
      let l =
          n.quantization?.group_size ?? n.quantization_config?.group_size ?? 64,
        u = Kt(n.quantization?.bits ?? n.quantization_config?.bits ?? 4),
        d = !!t.caps().f16,
        f = n.hidden_size,
        p = n.intermediate_size,
        m = n.num_attention_heads,
        h = n.num_key_value_heads,
        g = n.head_dim,
        _ = m * g,
        v = h * g,
        y = Math.max(...i);
      if (!Number.isInteger(y) || y < 0 || y >= n.num_hidden_layers)
        throw Error(
          `hiddenLayerIndices must be within [0, ${n.num_hidden_layers - 1}]`,
        );
      let b = y + 1,
        x = { embedQ: null, layers: Array(b) };
      for (let e = 0; e < b; ++e) x.layers[e] = {};
      let S = Zt(),
        C = (e, n, r, i) => {
          S.group(
            [`${e}.weight`, `${e}.scales`, `${e}.biases`],
            async ({
              [`${e}.weight`]: a,
              [`${e}.scales`]: o,
              [`${e}.biases`]: s,
            }) => {
              let c = t.allocateWeightsBuffer({
                byteLength: a.byteLength,
                dtype: `uint32`,
                shape: [n, r / u],
                label: `${e}.bits`,
              });
              t.writeWeightsRange(c, 0, a);
              let f = Yt({
                  scalesBytes: o,
                  biasesBytes: s,
                  outFeatures: n,
                  inFeatures: r,
                  groupSize: l,
                  dtype: d ? `f16` : `f32`,
                }),
                p = t.tensorFromTypedArray(
                  d ? `float16` : `float32`,
                  [n, r / l, 2],
                  f,
                );
              ((i.bitsT = c), (i.sbT = p));
            },
          );
        },
        w = (e, n) => {
          S.tensor(e, async (e) => {
            let r = jt(e);
            n(t.tensorFromTypedArray(`float32`, [r.length], r));
          });
        };
      ((x.embedQ = {}),
        C(`model.embed_tokens`, n.vocab_size, n.hidden_size, x.embedQ));
      for (let e = 0; e < b; ++e) {
        let t = `model.layers.${e}`,
          n = x.layers[e];
        ((n.qProj = {}),
          (n.kProj = {}),
          (n.vProj = {}),
          (n.oProj = {}),
          (n.gateProj = {}),
          (n.upProj = {}),
          (n.downProj = {}),
          w(`${t}.input_layernorm.weight`, (e) => {
            n.inputLn = e;
          }),
          w(`${t}.post_attention_layernorm.weight`, (e) => {
            n.postAttnLn = e;
          }),
          w(`${t}.self_attn.q_norm.weight`, (e) => {
            n.qNorm = e;
          }),
          w(`${t}.self_attn.k_norm.weight`, (e) => {
            n.kNorm = e;
          }),
          C(`${t}.self_attn.q_proj`, _, f, n.qProj),
          C(`${t}.self_attn.k_proj`, v, f, n.kProj),
          C(`${t}.self_attn.v_proj`, v, f, n.vProj),
          C(`${t}.self_attn.o_proj`, f, _, n.oProj),
          C(`${t}.mlp.gate_proj`, p, f, n.gateProj),
          C(`${t}.mlp.up_proj`, p, f, n.upProj),
          C(`${t}.mlp.down_proj`, f, p, n.downProj));
      }
      return (
        await r.streamAll(S.onChunk, {
          concurrency: o,
          chunkMaxBytes: s,
          names: S.names(),
          onProgress: a,
          signal: c,
        }),
        S.assertComplete(),
        new e({
          rt: t,
          config: n,
          embedBitsT: x.embedQ.bitsT,
          embedScaleBiasT: x.embedQ.sbT,
          layers: x.layers,
          finalNormT: null,
          hiddenLayerIndices: i,
        })
      );
    }
    async encode(e, { scope: t = null } = {}) {
      let n = !t,
        r = t ?? Vt(),
        i = Ht(this.rt, r);
      try {
        let t = await this._encodeWithRuntime(e, i);
        return (n && r.keep(t.hiddenStackT), t);
      } finally {
        n && r.destroy();
      }
    }
    async _encodeWithRuntime(e, t) {
      let n = this.config,
        r = e.length,
        i = n.hidden_size,
        a = n.num_attention_heads,
        o = n.num_key_value_heads,
        s = n.head_dim,
        c = a * s,
        l = o * s,
        u = n.intermediate_size,
        d = n.rope_theta ?? 1e6,
        f = n.rms_norm_eps ?? 1e-6,
        p = new Set(this.hiddenLayerIndices),
        m = this.hiddenLayerIndices.length,
        h = t.caps().f16 ? `float16` : `float32`,
        g = t.tensorFromTypedArray(`uint32`, [r], e),
        _ = t.empty(h, [r, i], `qwen3-hidden`);
      await lr(t, {
        idsT: g,
        bitsT: this.embedBitsT,
        sbT: this.embedScaleBiasT,
        yT: _,
        seq: r,
        hidden: i,
        vocab: n.vocab_size,
        bits: n.quantization?.bits ?? n.quantization_config?.bits ?? 4,
        groupSize:
          n.quantization?.group_size ?? n.quantization_config?.group_size ?? 64,
      });
      let v = s / 2,
        y = new Float32Array(r * v),
        b = new Float32Array(r * v);
      for (let e = 0; e < r; ++e)
        for (let t = 0; t < v; ++t) {
          let n = 1 / d ** ((2 * t) / s),
            r = e * n;
          ((y[e * v + t] = Math.cos(r)), (b[e * v + t] = Math.sin(r)));
        }
      let x = t.tensorFromTypedArray(`float32`, [r, v], y),
        S = t.tensorFromTypedArray(`float32`, [r, v], b),
        C = t.empty(h, [r, i], `qwen3-normed`),
        w = t.empty(h, [r, c], `qwen3-q`),
        T = t.empty(h, [r, l], `qwen3-k`),
        E = t.empty(h, [r, l], `qwen3-v`),
        D = t.empty(h, [r, c], `qwen3-q-normed`),
        O = t.empty(h, [r, l], `qwen3-k-normed`),
        k = t.empty(h, [r, c], `qwen3-attn`),
        A = t.empty(h, [r, i], `qwen3-oproj`),
        j = t.empty(h, [r, u], `qwen3-gate`),
        M = t.empty(h, [r, u], `qwen3-up`),
        N = t.empty(h, [r, i], `qwen3-ff`),
        P = t.empty(h, [r, m * i], `qwen3-stack`),
        F = n.quantization?.bits ?? n.quantization_config?.bits ?? 4,
        I =
          n.quantization?.group_size ?? n.quantization_config?.group_size ?? 64;
      for (let e = 0; e < this.layers.length; ++e) {
        let n = this.layers[e];
        if (
          (await Nn(t, {
            xT: _,
            wT: n.inputLn,
            yT: C,
            rows: r,
            dim: i,
            eps: f,
          }),
          await $n(t, {
            aT: C,
            bitsT: n.qProj.bitsT,
            sbT: n.qProj.sbT,
            outT: w,
            M: r,
            inFeatures: i,
            outFeatures: c,
            bits: F,
            groupSize: I,
          }),
          await $n(t, {
            aT: C,
            bitsT: n.kProj.bitsT,
            sbT: n.kProj.sbT,
            outT: T,
            M: r,
            inFeatures: i,
            outFeatures: l,
            bits: F,
            groupSize: I,
          }),
          await $n(t, {
            aT: C,
            bitsT: n.vProj.bitsT,
            sbT: n.vProj.sbT,
            outT: E,
            M: r,
            inFeatures: i,
            outFeatures: l,
            bits: F,
            groupSize: I,
          }),
          await Nn(t, {
            xT: w,
            wT: n.qNorm,
            yT: D,
            rows: r * a,
            dim: s,
            eps: f,
          }),
          await Nn(t, {
            xT: T,
            wT: n.kNorm,
            yT: O,
            rows: r * o,
            dim: s,
            eps: f,
          }),
          await Pn(t, {
            xT: D,
            cosT: x,
            sinT: S,
            seq: r,
            heads: a,
            headDim: s,
          }),
          await Pn(t, {
            xT: O,
            cosT: x,
            sinT: S,
            seq: r,
            heads: o,
            headDim: s,
          }),
          await Wn(t, {
            qT: D,
            kT: O,
            vT: E,
            outT: k,
            seq: r,
            qHeads: a,
            kvHeads: o,
            headDim: s,
            causal: !0,
          }),
          await $n(t, {
            aT: k,
            bitsT: n.oProj.bitsT,
            sbT: n.oProj.sbT,
            outT: A,
            M: r,
            inFeatures: c,
            outFeatures: i,
            bits: F,
            groupSize: I,
          }),
          await Ln(t, { xT: _, yT: A, count: r * i, alpha: 1 }),
          await Nn(t, {
            xT: _,
            wT: n.postAttnLn,
            yT: C,
            rows: r,
            dim: i,
            eps: f,
          }),
          await $n(t, {
            aT: C,
            bitsT: n.gateProj.bitsT,
            sbT: n.gateProj.sbT,
            outT: j,
            M: r,
            inFeatures: i,
            outFeatures: u,
            bits: F,
            groupSize: I,
          }),
          await $n(t, {
            aT: C,
            bitsT: n.upProj.bitsT,
            sbT: n.upProj.sbT,
            outT: M,
            M: r,
            inFeatures: i,
            outFeatures: u,
            bits: F,
            groupSize: I,
          }),
          await Fn(t, { xT: j, count: r * u }),
          await Un(t, { xT: j, factorT: M, count: r * u, period: 0 }),
          await $n(t, {
            aT: j,
            bitsT: n.downProj.bitsT,
            sbT: n.downProj.sbT,
            outT: N,
            M: r,
            inFeatures: u,
            outFeatures: i,
            bits: F,
            groupSize: I,
          }),
          await Ln(t, { xT: _, yT: N, count: r * i, alpha: 1 }),
          p.has(e))
        ) {
          let n = this.hiddenLayerIndices.indexOf(e);
          await ur(t, {
            srcT: _,
            dstT: P,
            rows: r,
            srcRowStrideEl: i,
            srcStartCol: 0,
            dstRowStrideEl: m * i,
            dstStartCol: n * i,
            copyCols: i,
          });
        }
      }
      return { hiddenStackT: P, seq: r, stackDim: m * i };
    }
    destroy() {
      (Bt({
        embedBitsT: this.embedBitsT,
        embedScaleBiasT: this.embedScaleBiasT,
        layers: this.layers,
        finalNormT: this.finalNormT,
      }),
        (this.embedBitsT = null),
        (this.embedScaleBiasT = null),
        (this.layers = []),
        (this.finalNormT = null));
    }
  };
async function lr(
  e,
  {
    idsT: t,
    bitsT: n,
    sbT: r,
    yT: i,
    seq: a,
    hidden: o,
    vocab: s,
    bits: c,
    groupSize: l,
  },
) {
  if (c !== 4) throw Error(`only bits=4 supported in mlxEmbedGather for now`);
  let u = o / 8,
    d = o / l,
    f = r.dtype === `float16` ? `f16` : `f32`,
    p = i.dtype === `float16` ? `f16` : `f32`,
    m = `${
      f === `f16` || p === `f16`
        ? `enable f16;
`
        : ``
    }struct Params { seq: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read>       ids: array<u32>;
@group(0) @binding(1) var<storage, read>       bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read>       scaleBias: array<${f}>;
@group(0) @binding(3) var<storage, read_write> y: array<${p}>;
@group(0) @binding(4) var<uniform>             params: Params;

const HIDDEN:        u32 = ${o}u;
const VOCAB:         u32 = ${s}u;
const GROUP_SIZE:    u32 = ${l}u;
const NUM_GROUPS:    u32 = ${d}u;
const WORDS_PER_ROW: u32 = ${u}u;
const VALS_PER_WORD: u32 = 8u;
const BITS:          u32 = ${c}u;
const MASK:          u32 = 0xfu;
const WG: u32 = 64u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  if (t >= params.seq) { return; }
  let id = ids[t];
  if (id >= VOCAB) { return; }

  let row_words_base: u32 = id * WORDS_PER_ROW;
  let row_sb_base:    u32 = id * NUM_GROUPS * 2u;

  // Each thread fills VALS_PER_WORD output positions per word it handles.
  var w: u32 = lid.x;
  loop {
    if (w >= WORDS_PER_ROW) { break; }
    let packed: u32 = bits_buf[row_words_base + w];
    let colBase: u32 = w * VALS_PER_WORD;
    let g: u32 = colBase / GROUP_SIZE;
    let scale: f32 = f32(scaleBias[row_sb_base + g * 2u]);
    let bias:  f32 = f32(scaleBias[row_sb_base + g * 2u + 1u]);
    for (var v: u32 = 0u; v < VALS_PER_WORD; v = v + 1u) {
      let q: f32 = f32((packed >> (v * BITS)) & MASK);
      y[t * HIDDEN + colBase + v] = ${p === `f16` ? `f16(scale * q + bias)` : `scale * q + bias`};
    }
    w = w + WG;
  }
}
`,
    h = e.createUniformU32([a, 0, 0, 0], `embed-gather-params`);
  await e.runProgram({
    name: `mlx_embed_gather`,
    source: m,
    entryPoint: `main`,
    cacheKey: `mlx_embed_gather_${f}_${p}_h${o}_v${s}_g${l}_b${c}`,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `storage` },
      { buffer: h, type: `uniform` },
    ],
    workgroups: [a, 1, 1],
  });
}
async function ur(
  e,
  {
    srcT: t,
    dstT: n,
    rows: r,
    srcRowStrideEl: i,
    srcStartCol: a,
    dstRowStrideEl: o,
    dstStartCol: s,
    copyCols: c,
  },
) {
  let l = t.dtype === `float16` ? `f16` : `f32`,
    u = n.dtype === `float16` ? `f16` : `f32`,
    d =
      l === `f16` || u === `f16`
        ? `enable f16;
`
        : ``,
    f =
      l === `f16`
        ? `f32(s[r * p.srcStride + p.srcStart + i])`
        : `s[r * p.srcStride + p.srcStart + i]`,
    p = u === `f16` ? `f16(${f})` : f,
    m = `strided_copy_${t.dtype}_${n.dtype}`,
    h = `${d}struct Params { rows: u32, copyCols: u32, srcStride: u32, srcStart: u32, dstStride: u32, dstStart: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       s: array<${l}>;
@group(0) @binding(1) var<storage, read_write> d: array<${u}>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= p.rows) { return; }
  var i: u32 = lid.x;
  loop {
    if (i >= p.copyCols) { break; }
    d[r * p.dstStride + p.dstStart + i] = ${p};
    i = i + 64u;
  }
}
`,
    g = new ArrayBuffer(32),
    _ = new Uint32Array(g);
  ((_[0] = r), (_[1] = c), (_[2] = i), (_[3] = a), (_[4] = o), (_[5] = s));
  let v = e.createUniformU32(new Uint32Array(g), `strided-copy-params`);
  await e.runProgram({
    name: `strided_copy`,
    source: h,
    entryPoint: `main`,
    cacheKey: m,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: v, type: `uniform` },
    ],
    workgroups: [r, 1, 1],
  });
}
function dr(e, t) {
  let n = 16927e-8,
    r = 0.45666666;
  if (e > 4300) return n * e + r;
  let i = n * e + r,
    a = (i - (873809524e-13 * e + 1.89833333)) / 190,
    o = i - 200 * a;
  return a * t + o;
}
function fr(e, t, n = 1) {
  let r = Math.exp(e);
  if (typeof t == `number`) return r / (r + (1 / t - 1) ** n);
  let i = new Float32Array(t.length);
  for (let e = 0; e < t.length; ++e) i[e] = r / (r + (1 / t[e] - 1) ** n);
  return i;
}
var pr = class e {
    constructor(e = {}) {
      if (
        ((this.num_train_timesteps = e.num_train_timesteps ?? 1e3),
        (this.shift = e.shift ?? 1),
        (this.use_dynamic_shifting = e.use_dynamic_shifting ?? !1),
        (this.base_image_seq_len = e.base_image_seq_len ?? 256),
        (this.max_image_seq_len = e.max_image_seq_len ?? 4096),
        (this.base_shift = e.base_shift ?? 0.5),
        (this.max_shift = e.max_shift ?? 1.15),
        (this.shift_terminal = e.shift_terminal ?? null),
        (this.time_shift_type = e.time_shift_type ?? `exponential`),
        (this.invert_sigmas = e.invert_sigmas ?? !1),
        (this.stochastic_sampling = e.stochastic_sampling ?? !1),
        (this.use_beta_sigmas = e.use_beta_sigmas ?? !1),
        (this.use_exponential_sigmas = e.use_exponential_sigmas ?? !1),
        (this.use_karras_sigmas = e.use_karras_sigmas ?? !1),
        this.time_shift_type !== `exponential`)
      )
        throw Error(`Unsupported time_shift_type: ${this.time_shift_type}`);
      if (
        this.use_beta_sigmas ||
        this.use_exponential_sigmas ||
        this.use_karras_sigmas
      )
        throw Error(`Alternative sigma schedules not implemented`);
      if (this.shift_terminal !== null)
        throw Error(`shift_terminal stretch not implemented`);
      if (this.invert_sigmas) throw Error(`invert_sigmas not implemented`);
      ((this.sigmas = null),
        (this.timesteps = null),
        (this.numInferenceSteps = null),
        (this._stepIndex = null));
    }
    static fromConfig(t) {
      return new e(t);
    }
    setTimesteps({ numInferenceSteps: e, mu: t }) {
      if (this.use_dynamic_shifting && t == null)
        throw Error(`mu is required when use_dynamic_shifting=true`);
      let n = 1 / this.num_train_timesteps,
        r = this.num_train_timesteps,
        i = new Float32Array(e);
      if (e === 1) i[0] = 1 * r;
      else {
        let t = 1 * r,
          a = (n * r - t) / (e - 1);
        for (let n = 0; n < e; ++n) i[n] = t + n * a;
      }
      let a = new Float32Array(e);
      for (let t = 0; t < e; ++t) a[t] = i[t] / r;
      if (this.use_dynamic_shifting) a = fr(t, a, 1);
      else if (this.shift !== 1) {
        let t = new Float32Array(e);
        for (let n = 0; n < e; ++n) {
          let e = a[n];
          t[n] = (this.shift * e) / (1 + (this.shift - 1) * e);
        }
        a = t;
      }
      let o = new Float32Array(e);
      for (let t = 0; t < e; ++t) o[t] = a[t] * r;
      let s = new Float32Array(e + 1);
      (s.set(a),
        (s[e] = 0),
        (this.sigmas = s),
        (this.timesteps = o),
        (this.numInferenceSteps = e),
        (this._stepIndex = null));
    }
    stepDelta(e) {
      if (this.sigmas === null) throw Error(`setTimesteps not called`);
      if (e < 0 || e >= this.numInferenceSteps)
        throw Error(`stepIndex ${e} out of range`);
      return this.sigmas[e + 1] - this.sigmas[e];
    }
    stepCpu(e, t, n) {
      let r = this.stepDelta(t);
      if (e.length !== n.length)
        throw Error(`modelOutput and sample length mismatch`);
      let i = new Float32Array(n.length);
      for (let t = 0; t < n.length; ++t) i[t] = n[t] + r * e[t];
      return i;
    }
  },
  mr = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
  hr = null;
function gr() {
  let e = new Uint32Array(256);
  for (let t = 0; t < 256; ++t) {
    let n = t;
    for (let e = 0; e < 8; ++e)
      n = n & 1 ? (3988292384 ^ (n >>> 1)) >>> 0 : n >>> 1;
    e[t] = n >>> 0;
  }
  return e;
}
function _r(e) {
  hr ||= gr();
  let t = 4294967295;
  for (let n = 0; n < e.byteLength; ++n)
    t = (hr[(t ^ e[n]) & 255] ^ (t >>> 8)) >>> 0;
  return (t ^ 4294967295) >>> 0;
}
function vr(e, t) {
  let n = new TextEncoder().encode(e);
  if (n.length !== 4) throw Error(`chunk type must be 4 bytes: ${e}`);
  let r = new Uint8Array(8 + t.byteLength + 4),
    i = new DataView(r.buffer);
  (i.setUint32(0, t.byteLength, !1), r.set(n, 4), r.set(t, 8));
  let a = new Uint8Array(4 + t.byteLength);
  return (
    a.set(n, 0),
    a.set(t, 4),
    i.setUint32(8 + t.byteLength, _r(a), !1),
    r
  );
}
function yr(e, t) {
  let n = new Uint8Array(13),
    r = new DataView(n.buffer);
  return (
    r.setUint32(0, e, !1),
    r.setUint32(4, t, !1),
    (n[8] = 8),
    (n[9] = 2),
    (n[10] = 0),
    (n[11] = 0),
    (n[12] = 0),
    n
  );
}
function br(e, t, n) {
  if (!Number.isInteger(e) || e <= 0)
    throw Error(`width must be positive integer`);
  if (!Number.isInteger(t) || t <= 0)
    throw Error(`height must be positive integer`);
  if (!(n instanceof Uint8Array)) throw Error(`rgb must be Uint8Array`);
  if (n.byteLength !== e * t * 3)
    throw Error(`rgb length ${n.byteLength} != width*height*3 = ${e * t * 3}`);
  let r = e * 3,
    i = new Uint8Array(t * (1 + r));
  for (let e = 0; e < t; ++e)
    ((i[e * (1 + r)] = 0),
      i.set(n.subarray(e * r, (e + 1) * r), e * (1 + r) + 1));
  let a = xr(i),
    o = [mr, vr(`IHDR`, yr(e, t)), vr(`IDAT`, a), vr(`IEND`, new Uint8Array())],
    s = 0;
  for (let e of o) s += e.byteLength;
  let c = new Uint8Array(s),
    l = 0;
  for (let e of o) (c.set(e, l), (l += e.byteLength));
  return c;
}
function xr(e) {
  let t = Math.ceil(e.byteLength / 65535),
    n = new Uint8Array(2 + t * 5 + e.byteLength + 4),
    r = 0;
  ((n[r++] = 120), (n[r++] = 1));
  let i = 0;
  for (; i < e.byteLength; ) {
    let t = Math.min(65535, e.byteLength - i),
      a = i + t === e.byteLength;
    ((n[r++] = +!!a), (n[r++] = t & 255), (n[r++] = t >>> 8));
    let o = ~t & 65535;
    ((n[r++] = o & 255),
      (n[r++] = o >>> 8),
      n.set(e.subarray(i, i + t), r),
      (r += t),
      (i += t));
  }
  let a = Sr(e);
  return (
    (n[r++] = (a >>> 24) & 255),
    (n[r++] = (a >>> 16) & 255),
    (n[r++] = (a >>> 8) & 255),
    (n[r++] = a & 255),
    n
  );
}
function Sr(e) {
  let t = 1,
    n = 0;
  for (let r = 0; r < e.byteLength; ++r)
    ((t += e[r]), (n += t), (r & 4095) == 4095 && ((t %= 65521), (n %= 65521)));
  return ((t %= 65521), (n %= 65521), ((n << 16) | t) >>> 0);
}
function Cr(e) {
  let t = {}.BONSAI_FORCE_DENSE_F16_MATMUL;
  return t === `1` || t === `true`
    ? !0
    : t === `0` || t === `false`
      ? !1
      : e.caps().f16 && e.caps().adapter?.vendor === `apple`;
}
function wr(e, t, n) {
  let r = e.quantization_config ?? e.quantization ?? {},
    i = Tr(t, `transformer_blocks.0.attn.to_q.weight`, n),
    a = Number(r.bits ?? r.nbits ?? i ?? 2),
    o = Number(r.group_size ?? r.groupSize ?? 128);
  if (![1, 2].includes(a))
    throw Error(
      `Flux2 transformer MLX weights require 1-bit or 2-bit quantization, got bits=${a}`,
    );
  if (o !== 128)
    throw Error(
      `Flux2 transformer MLX weights require group_size=128, got ${o}`,
    );
  if (i != null && i !== a)
    throw Error(
      `Flux2 transformer weight shape implies ${i}-bit packing, but quantization_config says ${a}-bit`,
    );
  return { bits: a, groupSize: o, quantSolver: r.solver ?? null };
}
function Tr(e, t, n) {
  if (!e.has?.(t)) return null;
  let r = e.info(t).shape;
  if (r.length !== 2) return null;
  let i = r[1] * 32;
  if (i % n !== 0) return null;
  let a = i / n;
  return Number.isInteger(a) ? a : null;
}
function Er(e, t) {
  if (e.byteLength !== t.byteLength) return !1;
  for (let n = 0; n < e.byteLength; n += 2) {
    let r = e[n] | (e[n + 1] << 8),
      i = r & 32640;
    if (i === 0 || i === 32640) return !1;
    let a = (((r & 32767) - 128) | 32768) & 65535;
    if ((t[n] | (t[n + 1] << 8)) !== a) return !1;
  }
  return !0;
}
function Dr(e, t, n) {
  let r = (t.byteLength + 3) & -4,
    i = e.allocateWeightsBuffer({
      byteLength: r,
      dtype: `uint32`,
      shape: [r / 4],
      label: n,
    });
  return (
    e.writeWeightsRange(i, 0, t),
    (i.scaleBiasDtype = `bfloat16`),
    (i.scaleBiasLayout = `out-group`),
    i
  );
}
function Or(e) {
  return e === `float16` ? 2 : 4;
}
function kr(e) {
  return e === `float16` ? `f16` : `f32`;
}
function Ar(...e) {
  return e.includes(`float16`)
    ? `enable f16;
`
    : ``;
}
function jr(e, t) {
  return t === `float16` ? `f32(${e})` : e;
}
function Mr(e, t) {
  return t === `float16` ? `f16(${e})` : e;
}
var Nr = class e {
  constructor({ rt: e, config: t, w: n }) {
    ((this.rt = e),
      (this.config = t),
      (this.w = n),
      (this.ropeCache = new Map()));
  }
  destroy() {
    (Bt(this.w),
      Bt(this.ropeCache),
      (this.w = null),
      (this.ropeCache = new Map()));
  }
  static async fromMlxSafeTensors({
    rt: t,
    config: n,
    safeTensors: r,
    onProgress: i = null,
    concurrency: a,
    chunkMaxBytes: o,
    signal: s,
  }) {
    let c = n.num_layers,
      l = n.num_single_layers,
      u = n.num_attention_heads * n.attention_head_dim,
      { bits: d, groupSize: f, quantSolver: p } = wr(n, r, u),
      m = Kt(d),
      h = !!t.caps().f16,
      g = n.joint_attention_dim,
      _ = n.in_channels,
      v = n.out_channels ?? _,
      y = n.timestep_guidance_channels ?? 256,
      b = n.mlp_ratio ?? 3,
      x = Math.floor(u * b),
      S = 3 * u + x * 2,
      C = u + x,
      w = Cr(t),
      T = !!(d === 1 && f === 128 && t.caps().f16),
      E = !!(d === 2 && f === 128 && p === `ternary` && t.caps().f16),
      D = t.caps().subgroupMatrix && !w && !T && h ? `group-out` : `out-group`,
      O = !!(t.caps().f16 && !T && (!t.caps().subgroupMatrix || w)),
      k = Zt(),
      A = (e, n, r, i, a = {}) => {
        let o = !!(a.denseF16 && E),
          s = o
            ? [`${e}.weight`, `${e}.scales`]
            : [`${e}.weight`, `${e}.scales`, `${e}.biases`];
        k.group(s, async (s) => {
          let c = s[`${e}.weight`],
            l = s[`${e}.scales`],
            u = s[`${e}.biases`],
            p = t.allocateWeightsBuffer({
              byteLength: c.byteLength,
              dtype: `uint32`,
              shape: [n, r / m],
              label: `${e}.bits`,
            });
          t.writeWeightsRange(p, 0, c);
          let g;
          if (o)
            ((g = Dr(t, l, `${e}.scales_bf16`)), (g.ternaryBiasFromScale = !0));
          else if (T && Er(l, u)) {
            let e = Mt(l);
            ((g = t.tensorFromTypedArray(`float16`, [n, r / f], e)),
              (g.scaleBiasLayout = `out-group`),
              (g.binaryBiasIsNegHalfScale = !0),
              (g.binaryScaleOnly = !0));
          } else {
            let e = (D === `group-out` ? Jt : Yt)({
                scalesBytes: l,
                biasesBytes: u,
                outFeatures: n,
                inFeatures: r,
                groupSize: f,
                dtype: h ? `f16` : `f32`,
              }),
              i = D === `group-out` ? [r / f, n, 2] : [n, r / f, 2];
            ((g = t.tensorFromTypedArray(h ? `float16` : `float32`, i, e)),
              (g.scaleBiasLayout = D),
              T && (g.binaryBiasIsNegHalfScale = !1));
          }
          if (((i.bitsT = p), (i.sbT = g), a.denseF16)) {
            let a = t.empty(`float16`, [r, n], `${e}.dense_f16_kn`);
            (await Yn(t, {
              bitsT: p,
              sbT: g,
              outT: a,
              inFeatures: r,
              outFeatures: n,
              bits: d,
              groupSize: f,
              outputLayout: `k-out`,
            }),
              (a.denseLayout = `k-out-vec4n`),
              (i.denseT = a),
              (i.denseLayout = `k-out-vec4n`));
          }
        });
      },
      j = (e, n, r, i, a = {}) => {
        let o = !!(a.denseF16 && E),
          s = r / m,
          c = r / f,
          l = n * e.length,
          u = Array(e.length),
          p = 0,
          g = null,
          _ = null,
          v = null,
          y = null,
          b = null,
          x = null,
          S = null,
          C = T,
          w = (n, r = null) => {
            if (g) return;
            let i = n.byteLength * e.length;
            if (
              ((g = t.allocateWeightsBuffer({
                byteLength: i,
                dtype: `uint32`,
                shape: [l, s],
                label: `${e[0]}.fused_bits`,
              })),
              T)
            )
              ((y = new Uint16Array(l * c)),
                (x = Array(e.length)),
                (S = Array(e.length)));
            else if (o) {
              let n = (r.byteLength * e.length + 3) & -4;
              ((b = t.allocateWeightsBuffer({
                byteLength: n,
                dtype: `uint32`,
                shape: [n / 4],
                label: `${e[0]}.fused_scales_bf16`,
              })),
                (b.scaleBiasDtype = `bfloat16`),
                (b.scaleBiasLayout = `out-group`),
                (b.ternaryBiasFromScale = !0));
            } else
              ((_ = h ? Uint16Array : Float32Array), (v = new _(l * c * 2)));
          };
        e.forEach((s, m) => {
          let E = o
            ? [`${s}.weight`, `${s}.scales`]
            : [`${s}.weight`, `${s}.scales`, `${s}.biases`];
          k.group(E, async (E) => {
            let O = E[`${s}.weight`],
              k = E[`${s}.scales`],
              A = E[`${s}.biases`];
            if ((w(O, k), t.writeWeightsRange(g, O.byteLength * m, O), o))
              t.writeWeightsRange(b, k.byteLength * m, k);
            else if (T) {
              ((x[m] = new Uint8Array(k)),
                (S[m] = new Uint8Array(A)),
                Er(k, A) || (C = !1));
              let e = Mt(k);
              for (let t = 0; t < n; ++t) {
                let r = t * c,
                  i = (m * n + t) * c;
                y.set(e.subarray(r, r + c), i);
              }
            } else if (D === `group-out`) {
              let e = Jt({
                scalesBytes: k,
                biasesBytes: A,
                outFeatures: n,
                inFeatures: r,
                groupSize: f,
                dtype: h ? `f16` : `f32`,
              });
              for (let t = 0; t < c; ++t) {
                let r = (t * l + m * n) * 2,
                  i = t * n * 2;
                v.set(e.subarray(i, i + n * 2), r);
              }
            } else
              Xt({
                scalesBytes: k,
                biasesBytes: A,
                out: v,
                outFeatures: n,
                inFeatures: r,
                groupSize: f,
                dtype: h ? `f16` : `f32`,
                dstElementOffset: m * n * c * 2,
              });
            if (((u[m] = !0), ++p === e.length)) {
              let s;
              if (o) s = b;
              else if (T && C)
                ((s = t.tensorFromTypedArray(`float16`, [l, c], y)),
                  (s.scaleBiasLayout = `out-group`),
                  (s.binaryBiasIsNegHalfScale = !0),
                  (s.binaryScaleOnly = !0));
              else {
                if (T) {
                  ((_ = h ? Uint16Array : Float32Array),
                    (v = new _(l * c * 2)));
                  for (let t = 0; t < e.length; ++t)
                    if (D === `group-out`) {
                      let e = Jt({
                        scalesBytes: x[t],
                        biasesBytes: S[t],
                        outFeatures: n,
                        inFeatures: r,
                        groupSize: f,
                        dtype: h ? `f16` : `f32`,
                      });
                      for (let r = 0; r < c; ++r) {
                        let i = (r * l + t * n) * 2,
                          a = r * n * 2;
                        v.set(e.subarray(a, a + n * 2), i);
                      }
                    } else
                      Xt({
                        scalesBytes: x[t],
                        biasesBytes: S[t],
                        out: v,
                        outFeatures: n,
                        inFeatures: r,
                        groupSize: f,
                        dtype: h ? `f16` : `f32`,
                        dstElementOffset: t * n * c * 2,
                      });
                }
                let i = D === `group-out` ? [c, l, 2] : [l, c, 2];
                ((s = t.tensorFromTypedArray(h ? `float16` : `float32`, i, v)),
                  (s.scaleBiasLayout = D),
                  T && (s.binaryBiasIsNegHalfScale = !1));
              }
              if (((i.bitsT = g), (i.sbT = s), a.denseF16)) {
                let n = t.empty(
                  `float16`,
                  [r, l],
                  `${e[0]}.fused_dense_f16_kn`,
                );
                (await Yn(t, {
                  bitsT: g,
                  sbT: s,
                  outT: n,
                  inFeatures: r,
                  outFeatures: l,
                  bits: d,
                  groupSize: f,
                  outputLayout: `k-out`,
                }),
                  (n.denseLayout = `k-out-vec4n`),
                  (i.denseT = n),
                  (i.denseLayout = `k-out-vec4n`));
              }
              ((v = null), (y = null), (b = null), (x = null), (S = null));
            }
          });
        });
      },
      M = (e, n) => {
        k.tensor(e, async (e) => {
          let r = jt(e);
          n(t.tensorFromTypedArray(`float32`, [r.length], r));
        });
      },
      N = (e, t) => {
        r.has(e) ? M(e, t) : t(null);
      },
      P = {},
      F = (e, n, i, a) => {
        k.tensor(e, async (o) => {
          let s = r.has(e.replace(/\.weight$/, `.bias`)),
            c = t.caps().f16 && !s && i % 4 == 0 && a % 4 == 0;
          if (!c) {
            let e = jt(o);
            n.weight = t.tensorFromTypedArray(`float32`, [e.length], e);
          }
          if (c) {
            let e = Pt(o, a, i);
            n.weightPackedKOutF16 = t.tensorFromTypedArray(
              `float16`,
              [e.length],
              e,
            );
          }
        });
      };
    ((P.x_embedder = {}),
      F(`x_embedder.weight`, P.x_embedder, _, u),
      N(`x_embedder.bias`, (e) => {
        P.x_embedder.bias = e;
      }),
      (P.context_embedder = {}),
      F(`context_embedder.weight`, P.context_embedder, g, u),
      N(`context_embedder.bias`, (e) => {
        P.context_embedder.bias = e;
      }),
      (P.time_text_embed = { linear_1: {}, linear_2: {} }),
      M(`time_guidance_embed.timestep_embedder.linear_1.weight`, (e) => {
        P.time_text_embed.linear_1.weight = e;
      }),
      N(`time_guidance_embed.timestep_embedder.linear_1.bias`, (e) => {
        P.time_text_embed.linear_1.bias = e;
      }),
      M(`time_guidance_embed.timestep_embedder.linear_2.weight`, (e) => {
        P.time_text_embed.linear_2.weight = e;
      }),
      N(`time_guidance_embed.timestep_embedder.linear_2.bias`, (e) => {
        P.time_text_embed.linear_2.bias = e;
      }),
      (P.double_stream_modulation_img = {}),
      M(`double_stream_modulation_img.linear.weight`, (e) => {
        P.double_stream_modulation_img.weight = e;
      }),
      N(`double_stream_modulation_img.linear.bias`, (e) => {
        P.double_stream_modulation_img.bias = e;
      }),
      (P.double_stream_modulation_txt = {}),
      M(`double_stream_modulation_txt.linear.weight`, (e) => {
        P.double_stream_modulation_txt.weight = e;
      }),
      N(`double_stream_modulation_txt.linear.bias`, (e) => {
        P.double_stream_modulation_txt.bias = e;
      }),
      (P.single_stream_modulation = {}),
      M(`single_stream_modulation.linear.weight`, (e) => {
        P.single_stream_modulation.weight = e;
      }),
      N(`single_stream_modulation.linear.bias`, (e) => {
        P.single_stream_modulation.bias = e;
      }),
      (P.norm_out = {}),
      M(`norm_out.linear.weight`, (e) => {
        P.norm_out.weight = e;
      }),
      N(`norm_out.linear.bias`, (e) => {
        P.norm_out.bias = e;
      }),
      (P.proj_out = {}),
      M(`proj_out.weight`, (e) => {
        P.proj_out.weight = e;
      }),
      N(`proj_out.bias`, (e) => {
        P.proj_out.bias = e;
      }),
      (P.joint = []));
    for (let e = 0; e < c; ++e) {
      let t = `transformer_blocks.${e}`,
        n = {
          attn: { to_qkv: {}, to_out_0: {}, add_qkv: {}, to_add_out: {} },
          ff: { linear_in: {}, linear_out: {} },
          ff_context: { linear_in: {}, linear_out: {} },
        };
      (P.joint.push(n),
        j(
          [`${t}.attn.to_q`, `${t}.attn.to_k`, `${t}.attn.to_v`],
          u,
          u,
          n.attn.to_qkv,
          { denseF16: O },
        ),
        A(`${t}.attn.to_out.0`, u, u, n.attn.to_out_0, { denseF16: O }),
        j(
          [
            `${t}.attn.add_q_proj`,
            `${t}.attn.add_k_proj`,
            `${t}.attn.add_v_proj`,
          ],
          u,
          u,
          n.attn.add_qkv,
          { denseF16: O },
        ),
        A(`${t}.attn.to_add_out`, u, u, n.attn.to_add_out, { denseF16: O }),
        M(`${t}.attn.norm_q.weight`, (e) => {
          n.attn.norm_q = e;
        }),
        M(`${t}.attn.norm_k.weight`, (e) => {
          n.attn.norm_k = e;
        }),
        M(`${t}.attn.norm_added_q.weight`, (e) => {
          n.attn.norm_added_q = e;
        }),
        M(`${t}.attn.norm_added_k.weight`, (e) => {
          n.attn.norm_added_k = e;
        }),
        A(`${t}.ff.linear_in`, x * 2, u, n.ff.linear_in, { denseF16: O }),
        A(`${t}.ff.linear_out`, u, x, n.ff.linear_out, { denseF16: O }),
        A(`${t}.ff_context.linear_in`, x * 2, u, n.ff_context.linear_in, {
          denseF16: O,
        }),
        A(`${t}.ff_context.linear_out`, u, x, n.ff_context.linear_out, {
          denseF16: O,
        }));
    }
    P.single = [];
    for (let e = 0; e < l; ++e) {
      let t = `single_transformer_blocks.${e}`,
        n = { attn: { to_qkv_mlp_proj: {}, to_out: {} } };
      (P.single.push(n),
        A(`${t}.attn.to_qkv_mlp_proj`, S, u, n.attn.to_qkv_mlp_proj, {
          denseF16: O,
        }),
        A(`${t}.attn.to_out`, u, C, n.attn.to_out, { denseF16: O }),
        M(`${t}.attn.norm_q.weight`, (e) => {
          n.attn.norm_q = e;
        }),
        M(`${t}.attn.norm_k.weight`, (e) => {
          n.attn.norm_k = e;
        }));
    }
    return (
      await r.streamAll(k.onChunk, {
        concurrency: a,
        chunkMaxBytes: o,
        names: k.names(),
        onProgress: i,
        signal: s,
      }),
      k.assertComplete(),
      new e({
        rt: t,
        config: {
          ...n,
          inner_dim: u,
          mlp_inner: x,
          ts_channels: y,
          out_channels: v,
          bits: d,
          groupSize: f,
          quantSolver: p,
          fusedSingleOutRows: S,
          fusedSingleAttnOutInF: C,
        },
        w: P,
      })
    );
  }
  getRopeTensors({
    textSeq: e,
    imageSeq: t,
    totalSeq: n,
    headDim: r,
    txtIds: i,
    imgIds: a,
  }) {
    let o = this.config.axes_dims_rope ?? [32, 32, 32, 32],
      s = this.config.rope_theta ?? 2e3,
      c = `${n}:${r}:${o.join(`,`)}:${s}:${Ir(i)}:${Ir(a)}`,
      l = this.ropeCache.get(c);
    if (l) return l;
    let u = new Float32Array((e + t) * 4);
    (u.set(i, 0), u.set(a, e * 4));
    let { cos: d, sin: f } = Fr(u, n, o, s);
    return (
      (l = {
        cosT: this.rt.tensorFromTypedArray(`float32`, [n, r], d),
        sinT: this.rt.tensorFromTypedArray(`float32`, [n, r], f),
      }),
      this.ropeCache.set(c, l),
      l
    );
  }
  async forward({
    hiddenStatesT: e,
    encoderHiddenStatesT: t,
    timestep: n,
    imgIds: r,
    txtIds: i,
    scope: a = null,
  }) {
    let o = !a,
      s = a ?? Vt(),
      c = Ht(this.rt, s);
    try {
      let a = await this._forwardWithRuntime(
        {
          hiddenStatesT: e,
          encoderHiddenStatesT: t,
          timestep: n,
          imgIds: r,
          txtIds: i,
        },
        c,
      );
      return (o && s.keep(a), a);
    } finally {
      o && s.destroy();
    }
  }
  async _forwardWithRuntime(
    {
      hiddenStatesT: e,
      encoderHiddenStatesT: t,
      timestep: n,
      imgIds: r,
      txtIds: i,
    },
    a,
  ) {
    let o = this.config,
      s = this.w,
      c = o.inner_dim,
      l = o.num_attention_heads,
      u = o.attention_head_dim,
      d = o.mlp_inner,
      f = o.eps ?? 1e-6,
      p = o.bits,
      m = o.groupSize,
      h = e.shape[0],
      g = t.shape[0],
      _ = g + h,
      v = !!(p === 1 && m === 128 && a.caps().f16),
      y = !!(
        !v &&
        !Cr(a) &&
        a.caps().subgroupMatrix &&
        (p === 1 || p === 2) &&
        m === 128
      ),
      b = y && a.caps().f16,
      x = p === 2 && o.quantSolver === `ternary`,
      S = a.caps().f16 ? `float16` : `float32`,
      C = Or(S),
      w = y ? Math.ceil(_ / 32) * 32 : _,
      T = Pr(n * 1e3, o.ts_channels),
      E = a.tensorFromTypedArray(`float32`, [1, o.ts_channels], T),
      D = a.empty(`float32`, [1, c], `ts-h1`);
    (await or(a, {
      aT: E,
      wT: s.time_text_embed.linear_1.weight,
      bT: s.time_text_embed.linear_1.bias,
      outT: D,
      M: 1,
      inFeatures: o.ts_channels,
      outFeatures: c,
    }),
      await Fn(a, { xT: D, count: c }));
    let O = a.empty(`float32`, [1, c], `temb`);
    await or(a, {
      aT: D,
      wT: s.time_text_embed.linear_2.weight,
      bT: s.time_text_embed.linear_2.bias,
      outT: O,
      M: 1,
      inFeatures: c,
      outFeatures: c,
    });
    let k = a.empty(`float32`, [1, c], `temb-silu`);
    (await a.copyBufferToBuffer({
      src: O.buffer,
      dst: k.buffer,
      byteLength: c * 4,
    }),
      await Fn(a, { xT: k, count: c }));
    let A = a.empty(`float32`, [1, c * 6], `dbl-img-mod`);
    await or(a, {
      aT: k,
      wT: s.double_stream_modulation_img.weight,
      bT: s.double_stream_modulation_img.bias,
      outT: A,
      M: 1,
      inFeatures: c,
      outFeatures: c * 6,
    });
    let j = a.empty(`float32`, [1, c * 6], `dbl-txt-mod`);
    await or(a, {
      aT: k,
      wT: s.double_stream_modulation_txt.weight,
      bT: s.double_stream_modulation_txt.bias,
      outT: j,
      M: 1,
      inFeatures: c,
      outFeatures: c * 6,
    });
    let M = a.empty(`float32`, [1, c * 3], `single-mod`);
    await or(a, {
      aT: k,
      wT: s.single_stream_modulation.weight,
      bT: s.single_stream_modulation.bias,
      outT: M,
      M: 1,
      inFeatures: c,
      outFeatures: c * 3,
    });
    let N = a.empty(S, [h, c], `hs`);
    if (
      S === `float16` &&
      !s.x_embedder.bias &&
      s.x_embedder.weightPackedKOutF16
    ) {
      let t =
        e.dtype === `float16`
          ? e
          : a.empty(`float16`, [h, o.in_channels], `hidden-states-f16`);
      (t !== e && (await Bn(a, { xT: e, yT: t, count: h * o.in_channels })),
        await ir(a, {
          aT: t,
          wT: s.x_embedder.weightPackedKOutF16,
          outT: N,
          M: h,
          inFeatures: o.in_channels,
          outFeatures: c,
        }));
    } else
      await or(a, {
        aT: e,
        wT: s.x_embedder.weight,
        bT: s.x_embedder.bias,
        outT: N,
        M: h,
        inFeatures: o.in_channels,
        outFeatures: c,
      });
    let P = a.empty(S, [g, c], `ehs`);
    if (
      S === `float16` &&
      !s.context_embedder.bias &&
      s.context_embedder.weightPackedKOutF16
    ) {
      let e =
        t.dtype === `float16`
          ? t
          : a.empty(
              `float16`,
              [g, o.joint_attention_dim],
              `encoder-hidden-states-f16`,
            );
      (e !== t &&
        (await Bn(a, { xT: t, yT: e, count: g * o.joint_attention_dim })),
        await ir(a, {
          aT: e,
          wT: s.context_embedder.weightPackedKOutF16,
          outT: P,
          M: g,
          inFeatures: o.joint_attention_dim,
          outFeatures: c,
        }));
    } else
      await or(a, {
        aT: t,
        wT: s.context_embedder.weight,
        bT: s.context_embedder.bias,
        outT: P,
        M: g,
        inFeatures: o.joint_attention_dim,
        outFeatures: c,
      });
    let { cosT: F, sinT: I } = this.getRopeTensors({
        textSeq: g,
        imageSeq: h,
        totalSeq: _,
        headDim: u,
        txtIds: i,
        imgIds: r,
      }),
      ee = a.empty(S, [h, c], `norm-img`),
      L = a.empty(S, [g, c], `norm-txt`),
      te = a.empty(S, [h, c * 3], `qkv-img`),
      R = a.empty(S, [g, c * 3], `qkv-txt`),
      ne = a.empty(S, [h, c], `vi`),
      re = a.empty(S, [g, c], `vt`),
      ie = a.empty(S, [_, c], `q-full`),
      ae = a.empty(S, [_, c], `k-full`),
      oe = a.empty(S, [_, c], `v-full`),
      se = a.empty(S, [_, c], `attn-full`),
      ce = a.empty(S, [h, c], `o-img`),
      le = a.empty(S, [g, c], `o-txt`),
      ue = null,
      de = null,
      fe = () => (ue ??= a.empty(S, [h, d * 2], `ff-pre-img`)),
      pe = () => (de ??= a.empty(S, [g, d * 2], `ff-pre-txt`)),
      me = a.empty(S, [h, d], `ff-post-img`),
      he = a.empty(S, [g, d], `ff-post-txt`),
      z = a.empty(S, [h, c], `ff-out-img`),
      ge = a.empty(S, [g, c], `ff-out-txt`),
      _e = Lr(A, c),
      ve = Lr(j, c),
      B = Rr(M, c),
      ye = (e, t) => ({
        shiftT: e.t,
        scaleT: e.t,
        shiftOffset: e[`${t}ShiftOffset`],
        scaleOffset: e[`${t}ScaleOffset`],
      }),
      V = (e, t) => ({ gateT: e.t, gateOffset: e[`${t}GateOffset`] }),
      H = async ({ aT: e, q: t, M: n, inFeatures: r }) => {
        y ||
          p !== 2 ||
          (await qn(a, {
            aT: e,
            aQT: t.aQT,
            scaleAT: t.scaleAT,
            sumAT: t.sumAT,
            M: n,
            inFeatures: r,
            groupSize: m,
          }));
      },
      be = async ({
        aT: e,
        q: t,
        bitsT: n,
        sbT: r,
        denseT: i,
        outT: o,
        M: s,
        inFeatures: l,
        outFeatures: u,
        aRowOffset: d = 0,
      }) => {
        if (y) {
          await Xn(a, {
            aT: e,
            bitsT: n,
            sbT: r,
            outT: o,
            M: s,
            inFeatures: l,
            outFeatures: u,
            bits: p,
            groupSize: m,
            useF16: b,
            assumeTernaryBias: x,
            aRowOffset: d,
            scaleBiasLayout: r.scaleBiasLayout ?? `out-group`,
          });
          return;
        }
        if (
          v &&
          d === 0 &&
          e.dtype === `float16` &&
          r.dtype === `float16` &&
          o.dtype === `float16` &&
          (r.scaleBiasLayout ?? `out-group`) === `out-group` &&
          l % 128 == 0
        ) {
          await Zn(a, {
            aT: e,
            bitsT: n,
            sbT: r,
            outT: o,
            M: s,
            inFeatures: l,
            outFeatures: u,
            assumeBiasNegHalfScale: !!r.binaryBiasIsNegHalfScale,
            scaleOnly: !!r.binaryScaleOnly,
          });
          return;
        }
        if (i && e.dtype === `float16` && o.dtype === `float16`) {
          i.denseLayout === `k-out-vec4n`
            ? s >= 128 && l === c && u >= c * 3
              ? await En(a, {
                  aT: e,
                  wT: i,
                  outT: o,
                  M: s,
                  inFeatures: l,
                  outFeatures: u,
                })
              : await ir(a, {
                  aT: e,
                  wT: i,
                  outT: o,
                  M: s,
                  inFeatures: l,
                  outFeatures: u,
                })
            : await nr(a, {
                aT: e,
                wT: i,
                outT: o,
                M: s,
                inFeatures: l,
                outFeatures: u,
              });
          return;
        }
        p === 2 && t
          ? await Jn(a, {
              aQT: t.aQT,
              scaleAT: t.scaleAT,
              sumAT: t.sumAT,
              bitsT: n,
              sbT: r,
              outT: o,
              M: s,
              inFeatures: l,
              outFeatures: u,
              bits: p,
              groupSize: m,
            })
          : await Qn(a, {
              aT: e,
              bitsT: n,
              sbT: r,
              outT: o,
              M: s,
              inFeatures: l,
              outFeatures: u,
              bits: p,
              groupSize: m,
            });
      },
      xe = p === 2 && !y,
      Se = xe ? Kn(a, { M: h, inFeatures: c, groupSize: m }) : null,
      Ce = xe ? Kn(a, { M: g, inFeatures: c, groupSize: m }) : null,
      we = xe ? Kn(a, { M: h, inFeatures: c, groupSize: m }) : null,
      Te = xe ? Kn(a, { M: g, inFeatures: c, groupSize: m }) : null,
      Ee = xe ? Kn(a, { M: h, inFeatures: d, groupSize: m }) : null,
      De = xe ? Kn(a, { M: g, inFeatures: d, groupSize: m }) : null;
    for (let e = 0; e < o.num_layers; ++e) {
      let t = s.joint[e];
      if (
        (await zr(a, {
          xT: N,
          yT: ee,
          rows: h,
          dim: c,
          eps: f,
          ...ye(_e, `msa`),
        }),
        await zr(a, {
          xT: P,
          yT: L,
          rows: g,
          dim: c,
          eps: f,
          ...ye(ve, `msa`),
        }),
        t.attn.to_qkv.denseT ||
          (await H({ aT: ee, q: Se, M: h, inFeatures: c })),
        t.attn.add_qkv.denseT ||
          (await H({ aT: L, q: Ce, M: g, inFeatures: c })),
        await be({
          aT: ee,
          q: Se,
          bitsT: t.attn.to_qkv.bitsT,
          sbT: t.attn.to_qkv.sbT,
          denseT: t.attn.to_qkv.denseT,
          outT: te,
          M: h,
          inFeatures: c,
          outFeatures: c * 3,
        }),
        await be({
          aT: L,
          q: Ce,
          bitsT: t.attn.add_qkv.bitsT,
          sbT: t.attn.add_qkv.sbT,
          denseT: t.attn.add_qkv.denseT,
          outT: R,
          M: g,
          inFeatures: c,
          outFeatures: c * 3,
        }),
        await Wr(a, {
          srcT: R,
          wT: t.attn.norm_added_q,
          cosT: F,
          sinT: I,
          yT: ie,
          seq: g,
          heads: l,
          headDim: u,
          srcStride: c * 3,
          srcStart: 0,
          dstRowOffset: 0,
          eps: f,
        }),
        await Wr(a, {
          srcT: te,
          wT: t.attn.norm_q,
          cosT: F,
          sinT: I,
          yT: ie,
          seq: h,
          heads: l,
          headDim: u,
          srcStride: c * 3,
          srcStart: 0,
          dstRowOffset: g,
          eps: f,
        }),
        await Wr(a, {
          srcT: R,
          wT: t.attn.norm_added_k,
          cosT: F,
          sinT: I,
          yT: ae,
          seq: g,
          heads: l,
          headDim: u,
          srcStride: c * 3,
          srcStart: c,
          dstRowOffset: 0,
          eps: f,
        }),
        await Wr(a, {
          srcT: te,
          wT: t.attn.norm_k,
          cosT: F,
          sinT: I,
          yT: ae,
          seq: h,
          heads: l,
          headDim: u,
          srcStride: c * 3,
          srcStart: c,
          dstRowOffset: g,
          eps: f,
        }),
        await Hr(a, {
          srcT: te,
          dstT: ne,
          rows: h,
          srcStride: c * 3,
          srcStart: 2 * c,
          dstStride: c,
          copyCols: c,
        }),
        await Hr(a, {
          srcT: R,
          dstT: re,
          rows: g,
          srcStride: c * 3,
          srcStart: 2 * c,
          dstStride: c,
          copyCols: c,
        }),
        await Br(a, {
          aT: re,
          bT: ne,
          outT: oe,
          aElems: g * c,
          totalElems: _ * c,
        }),
        await Wn(a, {
          qT: ie,
          kT: ae,
          vT: oe,
          outT: se,
          seq: _,
          qHeads: l,
          kvHeads: l,
          headDim: u,
          causal: !1,
        }),
        y)
      )
        (await be({
          aT: se,
          q: we,
          bitsT: t.attn.to_out_0.bitsT,
          sbT: t.attn.to_out_0.sbT,
          outT: ce,
          M: h,
          inFeatures: c,
          outFeatures: c,
          aRowOffset: g,
        }),
          await be({
            aT: se,
            q: Te,
            bitsT: t.attn.to_add_out.bitsT,
            sbT: t.attn.to_add_out.sbT,
            outT: le,
            M: g,
            inFeatures: c,
            outFeatures: c,
          }));
      else {
        let e = a.empty(S, [g, c], `attn-txt`),
          n = a.empty(S, [h, c], `attn-img`);
        (await a.copyBufferToBuffer({
          src: se.buffer,
          dst: e.buffer,
          srcOffset: 0,
          byteLength: g * c * C,
        }),
          await a.copyBufferToBuffer({
            src: se.buffer,
            dst: n.buffer,
            srcOffset: g * c * C,
            byteLength: h * c * C,
          }),
          t.attn.to_out_0.denseT ||
            (await H({ aT: n, q: we, M: h, inFeatures: c })),
          t.attn.to_add_out.denseT ||
            (await H({ aT: e, q: Te, M: g, inFeatures: c })),
          await be({
            aT: n,
            q: we,
            bitsT: t.attn.to_out_0.bitsT,
            sbT: t.attn.to_out_0.sbT,
            denseT: t.attn.to_out_0.denseT,
            outT: ce,
            M: h,
            inFeatures: c,
            outFeatures: c,
          }),
          await be({
            aT: e,
            q: Te,
            bitsT: t.attn.to_add_out.bitsT,
            sbT: t.attn.to_add_out.sbT,
            denseT: t.attn.to_add_out.denseT,
            outT: le,
            M: g,
            inFeatures: c,
            outFeatures: c,
          }));
      }
      if (
        (await Vr(a, {
          xT: N,
          addT: ce,
          count: h * c,
          ...V(_e, `msa`),
          inner: c,
        }),
        await Vr(a, {
          xT: P,
          addT: le,
          count: g * c,
          ...V(ve, `msa`),
          inner: c,
        }),
        await zr(a, {
          xT: N,
          yT: ee,
          rows: h,
          dim: c,
          eps: f,
          ...ye(_e, `mlp`),
        }),
        await zr(a, {
          xT: P,
          yT: L,
          rows: g,
          dim: c,
          eps: f,
          ...ye(ve, `mlp`),
        }),
        t.ff.linear_in.denseT ||
          (await H({ aT: ee, q: Se, M: h, inFeatures: c })),
        t.ff_context.linear_in.denseT ||
          (await H({ aT: L, q: Ce, M: g, inFeatures: c })),
        t.ff.linear_in.denseT?.denseLayout === `k-out-vec4n` &&
          ee.dtype === `float16` &&
          me.dtype === `float16`)
      )
        await On(a, {
          aT: ee,
          wT: t.ff.linear_in.denseT,
          outT: me,
          M: h,
          inFeatures: c,
          innerFeatures: d,
        });
      else {
        let e = fe();
        (await be({
          aT: ee,
          q: Se,
          bitsT: t.ff.linear_in.bitsT,
          sbT: t.ff.linear_in.sbT,
          denseT: t.ff.linear_in.denseT,
          outT: e,
          M: h,
          inFeatures: c,
          outFeatures: d * 2,
        }),
          await In(a, { xT: e, yT: me, rows: h, mlpInner: d }));
      }
      if (
        t.ff_context.linear_in.denseT?.denseLayout === `k-out-vec4n` &&
        L.dtype === `float16` &&
        he.dtype === `float16`
      )
        await On(a, {
          aT: L,
          wT: t.ff_context.linear_in.denseT,
          outT: he,
          M: g,
          inFeatures: c,
          innerFeatures: d,
        });
      else {
        let e = pe();
        (await be({
          aT: L,
          q: Ce,
          bitsT: t.ff_context.linear_in.bitsT,
          sbT: t.ff_context.linear_in.sbT,
          denseT: t.ff_context.linear_in.denseT,
          outT: e,
          M: g,
          inFeatures: c,
          outFeatures: d * 2,
        }),
          await In(a, { xT: e, yT: he, rows: g, mlpInner: d }));
      }
      (t.ff.linear_out.denseT ||
        (await H({ aT: me, q: Ee, M: h, inFeatures: d })),
        t.ff_context.linear_out.denseT ||
          (await H({ aT: he, q: De, M: g, inFeatures: d })),
        await be({
          aT: me,
          q: Ee,
          bitsT: t.ff.linear_out.bitsT,
          sbT: t.ff.linear_out.sbT,
          denseT: t.ff.linear_out.denseT,
          outT: z,
          M: h,
          inFeatures: d,
          outFeatures: c,
        }),
        await be({
          aT: he,
          q: De,
          bitsT: t.ff_context.linear_out.bitsT,
          sbT: t.ff_context.linear_out.sbT,
          denseT: t.ff_context.linear_out.denseT,
          outT: ge,
          M: g,
          inFeatures: d,
          outFeatures: c,
        }),
        await Vr(a, {
          xT: N,
          addT: z,
          count: h * c,
          ...V(_e, `mlp`),
          inner: c,
        }),
        await Vr(a, {
          xT: P,
          addT: ge,
          count: g * c,
          ...V(ve, `mlp`),
          inner: c,
        }));
    }
    let Oe = a.empty(S, [_, c], `combined`);
    await Br(a, { aT: P, bT: N, outT: Oe, aElems: g * c, totalElems: _ * c });
    let ke = a.empty(S, [w, c], `s-norm`),
      Ae = a.empty(S, [w, o.fusedSingleOutRows], `s-fused`),
      je = a.empty(S, [_, c], `s-v`),
      Me = a.empty(S, [_, c], `s-q-norm`),
      Ne = a.empty(S, [_, c], `s-k-norm`),
      Pe = a.empty(S, [_, c], `s-attn`),
      Fe = a.empty(S, [_, d], `s-mlp-post`),
      Ie = a.empty(S, [w, c + d], `s-cat`),
      Le = a.empty(S, [w, c], `s-out`),
      Re = xe ? Kn(a, { M: w, inFeatures: c, groupSize: m }) : null,
      ze = xe ? Kn(a, { M: w, inFeatures: c + d, groupSize: m }) : null;
    for (let e = 0; e < o.num_single_layers; ++e) {
      let t = s.single[e];
      (await zr(a, {
        xT: Oe,
        yT: ke,
        rows: _,
        dim: c,
        eps: f,
        ...ye(B, `msa`),
      }),
        t.attn.to_qkv_mlp_proj.denseT ||
          (await H({ aT: ke, q: Re, M: w, inFeatures: c })),
        await be({
          aT: ke,
          q: Re,
          bitsT: t.attn.to_qkv_mlp_proj.bitsT,
          sbT: t.attn.to_qkv_mlp_proj.sbT,
          denseT: t.attn.to_qkv_mlp_proj.denseT,
          outT: Ae,
          M: w,
          inFeatures: c,
          outFeatures: o.fusedSingleOutRows,
        }),
        await Hr(a, {
          srcT: Ae,
          dstT: je,
          rows: _,
          srcStride: o.fusedSingleOutRows,
          srcStart: 2 * c,
          dstStride: c,
          copyCols: c,
        }),
        await Wr(a, {
          srcT: Ae,
          wT: t.attn.norm_q,
          cosT: F,
          sinT: I,
          yT: Me,
          seq: _,
          heads: l,
          headDim: u,
          srcStride: o.fusedSingleOutRows,
          srcStart: 0,
          eps: f,
        }),
        await Wr(a, {
          srcT: Ae,
          wT: t.attn.norm_k,
          cosT: F,
          sinT: I,
          yT: Ne,
          seq: _,
          heads: l,
          headDim: u,
          srcStride: o.fusedSingleOutRows,
          srcStart: c,
          eps: f,
        }),
        await Wn(a, {
          qT: Me,
          kT: Ne,
          vT: je,
          outT: Pe,
          seq: _,
          qHeads: l,
          kvHeads: l,
          headDim: u,
          causal: !1,
        }),
        await Ur(a, {
          srcT: Ae,
          yT: Fe,
          rows: _,
          srcStride: o.fusedSingleOutRows,
          srcStart: 3 * c,
          mlpInner: d,
        }),
        await Gr(a, { aT: Pe, bT: Fe, outT: Ie, rows: _, aCols: c, bCols: d }),
        t.attn.to_out.denseT ||
          (await H({ aT: Ie, q: ze, M: w, inFeatures: c + d })),
        await be({
          aT: Ie,
          q: ze,
          bitsT: t.attn.to_out.bitsT,
          sbT: t.attn.to_out.sbT,
          denseT: t.attn.to_out.denseT,
          outT: Le,
          M: w,
          inFeatures: c + d,
          outFeatures: c,
        }),
        await Vr(a, {
          xT: Oe,
          addT: Le,
          count: _ * c,
          ...V(B, `msa`),
          inner: c,
        }));
    }
    let Be = a.empty(S, [h, c], `img-only`);
    await a.copyBufferToBuffer({
      src: Oe.buffer,
      dst: Be.buffer,
      srcOffset: g * c * C,
      byteLength: h * c * C,
    });
    let Ve = a.empty(`float32`, [h, c], `final-normed`);
    await Kr(a, {
      xT: Be,
      yT: Ve,
      rows: h,
      dim: c,
      eps: f,
      tembT: O,
      normLinear: s.norm_out,
    });
    let He = a.empty(`float32`, [h, o.out_channels], `noise-pred`);
    return (
      await or(a, {
        aT: Ve,
        wT: s.proj_out.weight,
        bT: s.proj_out.bias,
        outT: He,
        M: h,
        inFeatures: c,
        outFeatures: o.out_channels,
      }),
      He
    );
  }
};
function Pr(e, t, n = 1e4) {
  let r = t / 2,
    i = new Float32Array(t);
  for (let t = 0; t < r; ++t) {
    let a = e * Math.exp((-Math.log(n) * t) / r);
    ((i[t] = Math.cos(a)), (i[r + t] = Math.sin(a)));
  }
  return i;
}
function Fr(e, t, n, r) {
  let i = n.length,
    a = 0;
  for (let e of n) a += e;
  let o = new Float32Array(t * a),
    s = new Float32Array(t * a),
    c = n.map((e) => {
      let t = e / 2,
        n = new Float64Array(t);
      for (let i = 0; i < t; ++i) n[i] = 1 / r ** ((2 * i) / e);
      return n;
    });
  for (let r = 0; r < t; ++r) {
    let t = r * a,
      l = 0;
    for (let a = 0; a < i; ++a) {
      let u = e[r * i + a],
        d = c[a],
        f = n[a],
        p = f / 2;
      for (let e = 0; e < p; ++e) {
        let n = u * d[e],
          r = Math.fround(Math.cos(n)),
          i = Math.fround(Math.sin(n));
        ((o[t + l + 2 * e] = r),
          (o[t + l + 2 * e + 1] = r),
          (s[t + l + 2 * e] = i),
          (s[t + l + 2 * e + 1] = i));
      }
      l += f;
    }
  }
  return { cos: o, sin: s };
}
function Ir(e) {
  let t = 2166136261;
  for (let n = 0; n < e.length; ++n)
    ((t ^= (Math.trunc(e[n]) + 2654435769 + (n << 6) + (n >>> 2)) >>> 0),
      (t = Math.imul(t, 16777619) >>> 0));
  return t.toString(16);
}
function Lr(e, t) {
  return {
    t: e,
    msaShiftOffset: 0,
    msaScaleOffset: t,
    msaGateOffset: 2 * t,
    mlpShiftOffset: 3 * t,
    mlpScaleOffset: 4 * t,
    mlpGateOffset: 5 * t,
  };
}
function Rr(e, t) {
  return { t: e, msaShiftOffset: 0, msaScaleOffset: t, msaGateOffset: 2 * t };
}
async function zr(
  e,
  {
    xT: t,
    yT: n,
    rows: r,
    dim: i,
    eps: a,
    shift: o,
    scale: s,
    shiftT: c = null,
    scaleT: l = null,
    shiftOffset: u = 0,
    scaleOffset: d = 0,
  },
) {
  let f = t.dtype,
    p = n.dtype,
    m = kr(f),
    h = kr(p),
    g = `lnmod_d${i}_e${a}_${f}_${p}`,
    _ = `${Ar(f, p)}struct Params { rows: u32, shiftOffset: u32, scaleOffset: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${m}>;
@group(0) @binding(1) var<storage, read>       shiftBuf: array<f32>;
@group(0) @binding(2) var<storage, read>       scaleBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<${h}>;
@group(0) @binding(4) var<uniform>             params: Params;
const DIM: u32 = ${i}u;
const EPS: f32 = ${a};
const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;
var<workgroup> sum_acc: f32;
fn red(tid: u32) -> f32 {
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; if (r >= params.rows) { return; }
  let tid = lid.x;
  let base = r * DIM;
  var s: f32 = 0.0;
  for (var i: u32 = tid; i < DIM; i = i + WG) { s = s + ${jr(`x[base + i]`, f)}; }
  partial[tid] = s;
  let mean = red(tid) / f32(DIM);
  var sq: f32 = 0.0;
  for (var i: u32 = tid; i < DIM; i = i + WG) {
    let d = ${jr(`x[base + i]`, f)} - mean;
    sq = sq + d * d;
  }
  partial[tid] = sq;
  let invStd = inverseSqrt(red(tid) / f32(DIM) + EPS);
  for (var i: u32 = tid; i < DIM; i = i + WG) {
    let normed_v = (${jr(`x[base + i]`, f)} - mean) * invStd;
    y[base + i] = ${Mr(`normed_v * (1.0 + scaleBuf[params.scaleOffset + i]) + shiftBuf[params.shiftOffset + i]`, p)};
  }
}
`,
    v = c ?? e.tensorFromTypedArray(`float32`, [i], o),
    y = l ?? e.tensorFromTypedArray(`float32`, [i], s),
    b = e.createUniformU32([r, u, d, 0], `lnmod-params`);
  await e.runProgram({
    name: `lnmod`,
    source: _,
    cacheKey: g,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: v, type: `read-only-storage` },
      { tensor: y, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: b, type: `uniform` },
    ],
    workgroups: [r, 1, 1],
  });
}
async function Br(e, { aT: t, bT: n, outT: r, aElems: i, totalElems: a }) {
  await Gn(e, { aT: t, bT: n, outT: r, aElems: i, totalElems: a });
}
async function Vr(
  e,
  {
    xT: t,
    addT: n,
    count: r,
    gate: i,
    gateT: a = null,
    gateOffset: o = 0,
    inner: s,
  },
) {
  let c = t.dtype;
  if (
    c === `float16` &&
    n.dtype === `float16` &&
    r % 4 == 0 &&
    s % 4 == 0 &&
    o % 4 == 0
  ) {
    let l = `gatedres_d${s}_${c}_v4`,
      u = a ?? e.tensorFromTypedArray(`float32`, [s], i),
      d = r / 4,
      f = Math.ceil(d / 64),
      p = Math.min(f, 1024),
      m = Math.ceil(f / p),
      h = e.createUniformU32([d, s / 4, p, o / 4], `gatedres-v4-params`);
    await e.runProgram({
      name: `gatedres`,
      source: `enable f16;
struct Params { countV4: u32, periodV4: u32, wgY: u32, gateOffsetV4: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       add: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read>       g: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= params.countV4) { return; }
  let p = i % params.periodV4;
  let xv = vec4<f32>(x[i]);
  let av = vec4<f32>(add[i]);
  let gv = g[params.gateOffsetV4 + p];
  x[i] = vec4<f16>(xv + gv * av);
}
`,
      cacheKey: l,
      bindings: [
        { tensor: t, type: `storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: u, type: `read-only-storage` },
        { buffer: h, type: `uniform` },
      ],
      workgroups: [p, m, 1],
    });
    return;
  }
  let l = kr(c),
    u = `gatedres_d${s}_${c}`,
    d = `${Ar(c)}struct Params { count: u32, period: u32, wgY: u32, gateOffset: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<${l}>;
@group(0) @binding(1) var<storage, read>       add: array<${l}>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= params.count) { return; }
  let p = i % params.period;
  x[i] = ${Mr(`${jr(`x[i]`, c)} + g[params.gateOffset + p] * ${jr(`add[i]`, c)}`, c)};
}
`,
    f = a ?? e.tensorFromTypedArray(`float32`, [s], i),
    p = Math.ceil(r / 64),
    m = Math.min(p, 1024),
    h = Math.ceil(p / m),
    g = e.createUniformU32([r, s, m, o], `gatedres-params`);
  await e.runProgram({
    name: `gatedres`,
    source: d,
    cacheKey: u,
    bindings: [
      { tensor: t, type: `storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: f, type: `read-only-storage` },
      { buffer: g, type: `uniform` },
    ],
    workgroups: [m, h, 1],
  });
}
async function Hr(
  e,
  {
    srcT: t,
    dstT: n,
    rows: r,
    srcStride: i,
    srcStart: a,
    dstStride: o,
    copyCols: s,
  },
) {
  let c = n.dtype;
  if (
    c === `float16` &&
    t.dtype === `float16` &&
    i % 4 == 0 &&
    a % 4 == 0 &&
    o % 4 == 0 &&
    s % 4 == 0
  ) {
    let l = `slice_cols_${c}_v4`,
      u = new ArrayBuffer(32),
      d = new Uint32Array(u);
    ((d[0] = r),
      (d[1] = s / 4),
      (d[2] = i / 4),
      (d[3] = a / 4),
      (d[4] = o / 4),
      (d[5] = 0));
    let f = e.createUniformU32(new Uint32Array(u), `slice-v4-params`);
    await e.runProgram({
      name: `slice_cols`,
      source: `enable f16;
struct Params { rows: u32, copyColsV4: u32, srcStrideV4: u32, srcStartV4: u32, dstStrideV4: u32, dstStartV4: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       s: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> d: array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; if (r >= p.rows) { return; }
  var i: u32 = lid.x;
  loop {
    if (i >= p.copyColsV4) { break; }
    d[r * p.dstStrideV4 + p.dstStartV4 + i] = s[r * p.srcStrideV4 + p.srcStartV4 + i];
    i = i + 64u;
  }
}
`,
      cacheKey: l,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `storage` },
        { buffer: f, type: `uniform` },
      ],
      workgroups: [r, 1, 1],
    });
    return;
  }
  let l = kr(c),
    u = `slice_cols_${c}`,
    d = `${Ar(c)}struct Params { rows: u32, copyCols: u32, srcStride: u32, srcStart: u32, dstStride: u32, dstStart: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       s: array<${l}>;
@group(0) @binding(1) var<storage, read_write> d: array<${l}>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; if (r >= p.rows) { return; }
  var i: u32 = lid.x;
  loop {
    if (i >= p.copyCols) { break; }
    d[r * p.dstStride + p.dstStart + i] = s[r * p.srcStride + p.srcStart + i];
    i = i + 64u;
  }
}
`,
    f = new ArrayBuffer(32),
    p = new Uint32Array(f);
  ((p[0] = r), (p[1] = s), (p[2] = i), (p[3] = a), (p[4] = o), (p[5] = 0));
  let m = e.createUniformU32(new Uint32Array(f), `slice-params`);
  await e.runProgram({
    name: `slice_cols`,
    source: d,
    cacheKey: u,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: m, type: `uniform` },
    ],
    workgroups: [r, 1, 1],
  });
}
async function Ur(
  e,
  { srcT: t, yT: n, rows: r, srcStride: i, srcStart: a, mlpInner: o },
) {
  let s = t.dtype,
    c = n.dtype;
  if (
    s === `float16` &&
    c === `float16` &&
    i % 4 == 0 &&
    a % 4 == 0 &&
    o % 4 == 0
  ) {
    let l = `swiglu_cols_${s}_${c}_v4`,
      u = e.createUniformU32([r, i / 4, a / 4, o / 4], `swiglu-cols-v4-params`);
    await e.runProgram({
      name: `swiglu`,
      source: `enable f16;
struct Params { rows: u32, srcStrideV4: u32, srcStartV4: u32, mlpInnerV4: u32 };
@group(0) @binding(0) var<storage, read>       x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> y: array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= p.rows) { return; }
  let i = wg.y * 64u + lid.x;
  if (i >= p.mlpInnerV4) { return; }
  let base = r * p.srcStrideV4 + p.srcStartV4 + i;
  let x1 = vec4<f32>(x[base]);
  let x2 = vec4<f32>(x[base + p.mlpInnerV4]);
  y[r * p.mlpInnerV4 + i] = vec4<f16>((x1 / (vec4<f32>(1.0) + exp(-x1))) * x2);
}
`,
      cacheKey: l,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `storage` },
        { buffer: u, type: `uniform` },
      ],
      workgroups: [r, Math.ceil(o / 4 / 64), 1],
    });
    return;
  }
  let l = kr(s),
    u = kr(c),
    d = `swiglu_cols_${s}_${c}`,
    f = `${Ar(s, c)}struct Params { rows: u32, srcStride: u32, srcStart: u32, mlpInner: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${l}>;
@group(0) @binding(1) var<storage, read_write> y: array<${u}>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= p.rows) { return; }
  let i = wg.y * 64u + lid.x;
  if (i >= p.mlpInner) { return; }
  let base = r * p.srcStride + p.srcStart + i;
  let x1 = ${jr(`x[base]`, s)};
  let x2 = ${jr(`x[base + p.mlpInner]`, s)};
  y[r * p.mlpInner + i] = ${Mr(`(x1 / (1.0 + exp(-x1))) * x2`, c)};
}
`,
    p = e.createUniformU32([r, i, a, o], `swiglu-cols-params`);
  await e.runProgram({
    name: `swiglu`,
    source: f,
    cacheKey: d,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `storage` },
      { buffer: p, type: `uniform` },
    ],
    workgroups: [r, Math.ceil(o / 64), 1],
  });
}
async function Wr(
  e,
  {
    srcT: t,
    wT: n,
    cosT: r,
    sinT: i,
    yT: a,
    seq: o,
    heads: s,
    headDim: c,
    srcStride: l,
    srcStart: u,
    dstRowOffset: d = 0,
    eps: f,
  },
) {
  if (c % 2 != 0) throw Error(`rmsNormRopeFromColumns requires even headDim`);
  let p = c / 2;
  if (p > 256)
    throw Error(`rmsNormRopeFromColumns headDim=${c} exceeds workgroup limit`);
  let m = t.dtype,
    h = n.dtype,
    g = a.dtype,
    _ = kr(m),
    v = kr(h),
    y = kr(g),
    b = `rmsrope_cols_hd${c}_${m}_${h}_${g}`,
    x = `${Ar(m, h, g)}struct Params { seq: u32, heads: u32, srcStride: u32, srcStart: u32, dstRowOffset: u32, eps: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       x: array<${_}>;
@group(0) @binding(1) var<storage, read>       w: array<${v}>;
@group(0) @binding(2) var<storage, read>       cosTbl: array<f32>;
@group(0) @binding(3) var<storage, read>       sinTbl: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<${y}>;
@group(0) @binding(5) var<uniform>             p: Params;

const HEAD_DIM: u32 = ${c}u;
const HALF_DIM: u32 = ${p}u;
const WG:       u32 = ${p}u;

var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  if (t >= p.seq || h >= p.heads) { return; }
  let tid = lid.x;
  let srcBase = t * p.srcStride + p.srcStart + h * HEAD_DIM;

  let a0 = ${jr(`x[srcBase + tid]`, m)};
  let a1 = ${jr(`x[srcBase + tid + HALF_DIM]`, m)};
  partial[tid] = a0 * a0 + a1 * a1;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partial[tid] = partial[tid] + partial[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let normScale = inverseSqrt(partial[0] / f32(HEAD_DIM) + p.eps);

  let idx = tid * 2u;
  let xe = ${jr(`x[srcBase + idx]`, m)} * normScale * ${jr(`w[idx]`, h)};
  let xo = ${jr(`x[srcBase + idx + 1u]`, m)} * normScale * ${jr(`w[idx + 1u]`, h)};
  let dstT = p.dstRowOffset + t;
  let tableBase = dstT * HEAD_DIM + idx;
  let c = cosTbl[tableBase];
  let s = sinTbl[tableBase];
  let outBase = (dstT * p.heads + h) * HEAD_DIM;
  y[outBase + idx] = ${Mr(`xe * c - xo * s`, g)};
  y[outBase + idx + 1u] = ${Mr(`xo * c + xe * s`, g)};
}
`,
    S = new ArrayBuffer(32),
    C = new Uint32Array(S),
    w = new Float32Array(S);
  ((C[0] = o), (C[1] = s), (C[2] = l), (C[3] = u), (C[4] = d), (w[5] = f));
  let T = e.createUniformU32(new Uint32Array(S), `rmsrope-cols-params`);
  await e.runProgram({
    name: `rmsnorm_rope`,
    source: x,
    cacheKey: b,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `read-only-storage` },
      { tensor: i, type: `read-only-storage` },
      { tensor: a, type: `storage` },
      { buffer: T, type: `uniform` },
    ],
    workgroups: [o, s, 1],
  });
}
async function Gr(e, { aT: t, bT: n, outT: r, rows: i, aCols: a, bCols: o }) {
  let s = r.dtype;
  if (
    s === `float16` &&
    t.dtype === `float16` &&
    n.dtype === `float16` &&
    a % 4 == 0 &&
    o % 4 == 0
  ) {
    let c = `row_concat_${s}_v4`,
      l = e.createUniformU32([i, a / 4, o / 4, 0], `rowconcat-v4-params`);
    await e.runProgram({
      name: `row_concat`,
      source: `enable f16;
struct Params { rows: u32, aColsV4: u32, bColsV4: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       b: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> o: array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; if (r >= p.rows) { return; }
  let outStride = p.aColsV4 + p.bColsV4;
  var i: u32 = lid.x;
  loop {
    if (i >= p.aColsV4) { break; }
    o[r * outStride + i] = a[r * p.aColsV4 + i];
    i = i + 64u;
  }
  i = lid.x;
  loop {
    if (i >= p.bColsV4) { break; }
    o[r * outStride + p.aColsV4 + i] = b[r * p.bColsV4 + i];
    i = i + 64u;
  }
}
`,
      cacheKey: c,
      bindings: [
        { tensor: t, type: `read-only-storage` },
        { tensor: n, type: `read-only-storage` },
        { tensor: r, type: `storage` },
        { buffer: l, type: `uniform` },
      ],
      workgroups: [i, 1, 1],
    });
    return;
  }
  let c = kr(s),
    l = `row_concat_${s}`,
    u = `${Ar(s)}struct Params { rows: u32, aCols: u32, bCols: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read>       a: array<${c}>;
@group(0) @binding(1) var<storage, read>       b: array<${c}>;
@group(0) @binding(2) var<storage, read_write> o: array<${c}>;
@group(0) @binding(3) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x; if (r >= p.rows) { return; }
  let outStride = p.aCols + p.bCols;
  var i: u32 = lid.x;
  loop {
    if (i >= p.aCols) { break; }
    o[r * outStride + i] = a[r * p.aCols + i];
    i = i + 64u;
  }
  i = lid.x;
  loop {
    if (i >= p.bCols) { break; }
    o[r * outStride + p.aCols + i] = b[r * p.bCols + i];
    i = i + 64u;
  }
}
`,
    d = e.createUniformU32([i, a, o, 0], `rowconcat-params`);
  await e.runProgram({
    name: `row_concat`,
    source: u,
    cacheKey: l,
    bindings: [
      { tensor: t, type: `read-only-storage` },
      { tensor: n, type: `read-only-storage` },
      { tensor: r, type: `storage` },
      { buffer: d, type: `uniform` },
    ],
    workgroups: [i, 1, 1],
  });
}
async function Kr(
  e,
  { xT: t, yT: n, rows: r, dim: i, eps: a, tembT: o, normLinear: s },
) {
  let c = e.empty(`float32`, [1, i], `ada-silu`);
  (await e.copyBufferToBuffer({
    src: o.buffer,
    dst: c.buffer,
    byteLength: i * 4,
  }),
    await Fn(e, { xT: c, count: i }));
  let l = e.empty(`float32`, [1, 2 * i], `ada-proj`);
  await or(e, {
    aT: c,
    wT: s.weight,
    bT: s.bias,
    outT: l,
    M: 1,
    inFeatures: i,
    outFeatures: 2 * i,
  });
  let u = await e.readTensor(l),
    d = u.slice(0, i);
  await zr(e, {
    xT: t,
    yT: n,
    rows: r,
    dim: i,
    eps: a,
    shift: u.slice(i, 2 * i),
    scale: d,
  });
}
function U(e) {
  return e === `float16` ? `f16` : `f32`;
}
function qr(...e) {
  return e.includes(`float16`)
    ? `enable f16;
`
    : ``;
}
function W(e, t) {
  return t === `float16` ? `f32(${e})` : e;
}
function Jr(e, t) {
  return t === `float16` ? `f16(${e})` : e;
}
function Yr({
  inC: e,
  outC: t,
  H: n,
  W: r,
  kH: i,
  kW: a,
  pad: o,
  hasBias: s,
  inputDtype: c = `float32`,
  weightDtype: l = `float32`,
  biasDtype: u = `float32`,
  outputDtype: d = `float32`,
}) {
  let f = U(c),
    p = U(l),
    m = U(u),
    h = U(d);
  return `${qr(c, l, u, d)}struct Params { _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${f}>;
@group(0) @binding(1) var<storage, read>       weight: array<${p}>;
${
  s
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${m}>;
`
    : ``
}@group(0) @binding(${s ? 3 : 2}) var<storage, read_write> output: array<${h}>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const KH: i32 = ${i};
const KW: i32 = ${a};
const PAD: i32 = ${o};

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  let x = gid.y;
  let oc = gid.z;
  if (y >= H || x >= W || oc >= OUT_C) { return; }

  var acc: f32 = ${s ? W(`bias[oc]`, u) : `0.0`};
  let oc_base = oc * IN_C * u32(KH * KW);
  for (var ic: u32 = 0u; ic < IN_C; ic = ic + 1u) {
    let ic_w_base = oc_base + ic * u32(KH * KW);
    let ic_in_base = ic * H * W;
    for (var kh: i32 = 0; kh < KH; kh = kh + 1) {
      let iy: i32 = i32(y) + kh - PAD;
      if (iy < 0 || iy >= i32(H)) { continue; }
      for (var kw: i32 = 0; kw < KW; kw = kw + 1) {
        let ix: i32 = i32(x) + kw - PAD;
        if (ix < 0 || ix >= i32(W)) { continue; }
        let w_idx = ic_w_base + u32(kh) * u32(KW) + u32(kw);
        let in_idx = ic_in_base + u32(iy) * W + u32(ix);
        acc = acc + ${W(`input[in_idx]`, c)} * ${W(`weight[w_idx]`, l)};
      }
    }
  }
  output[oc * H * W + y * W + x] = ${Jr(`acc`, d)};
}
`;
}
function Xr({
  inC: e,
  outC: t,
  H: n,
  W: r,
  kH: i,
  kW: a,
  pad: o,
  hasBias: s,
  outTile: c = 4,
  inputDtype: l = `float32`,
  weightDtype: u = `float32`,
  biasDtype: d = `float32`,
  outputDtype: f = `float32`,
}) {
  if (t % c !== 0) throw Error(`outC must be divisible by outTile`);
  let p = U(l),
    m = U(u),
    h = U(d),
    g = U(f),
    _ = [],
    v = [],
    y = [];
  for (let e = 0; e < c; ++e)
    (_.push(
      `  var acc${e}: f32 = ${s ? W(`bias[oc_base + ${e}u]`, d) : `0.0`};`,
    ),
      v.push(
        `        acc${e} = acc${e} + in_val * ${W(`weight[(oc_base + ${e}u) * OC_STRIDE + k_offset]`, u)};`,
      ),
      y.push(
        `  output[(oc_base + ${e}u) * H * W + y * W + x] = ${Jr(`acc${e}`, f)};`,
      ));
  return `${qr(l, u, d, f)}struct Params { _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${p}>;
@group(0) @binding(1) var<storage, read>       weight: array<${m}>;
${
  s
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${h}>;
`
    : ``
}@group(0) @binding(${s ? 3 : 2}) var<storage, read_write> output: array<${g}>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const KH: i32 = ${i};
const KW: i32 = ${a};
const PAD: i32 = ${o};
const OUT_TILE: u32 = ${c}u;
const K_PER_IC: u32 = ${i * a}u;
const OC_STRIDE: u32 = ${e * i * a}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  let x = gid.y;
  let oc_base = gid.z * OUT_TILE;
  if (y >= H || x >= W || oc_base >= OUT_C) { return; }

${_.join(`
`)}
  for (var ic: u32 = 0u; ic < IN_C; ic = ic + 1u) {
    let ic_in_base = ic * H * W;
    for (var kh: i32 = 0; kh < KH; kh = kh + 1) {
      let iy: i32 = i32(y) + kh - PAD;
      if (iy < 0 || iy >= i32(H)) { continue; }
      for (var kw: i32 = 0; kw < KW; kw = kw + 1) {
        let ix: i32 = i32(x) + kw - PAD;
        if (ix < 0 || ix >= i32(W)) { continue; }
        let k_offset = ic * K_PER_IC + u32(kh) * u32(KW) + u32(kw);
        let in_val = ${W(`input[ic_in_base + u32(iy) * W + u32(ix)]`, l)};
${v.join(`
`)}
      }
    }
  }
${y.join(`
`)}
}
`;
}
function Zr({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  outTile: a = 16,
  icTile: o = 16,
  inputDtype: s = `float32`,
  weightDtype: c = `float32`,
  biasDtype: l = `float32`,
  outputDtype: u = `float32`,
}) {
  if (t % a !== 0) throw Error(`outC must be divisible by outTile`);
  let d = c === `float16` ? `float16` : `float32`,
    f = d === `float16` ? 2 : 4;
  if (a * o * 9 * f > 16 * 1024)
    throw Error(`conv shared-weight tile exceeds 16KB workgroup storage`);
  let p = U(s),
    m = U(c),
    h = U(d),
    g = U(l),
    _ = U(u),
    v = a * o * 9,
    y = [],
    b = [],
    x = [];
  for (let e = 0; e < a; ++e) {
    y.push(
      `  var acc${e}: f32 = ${i ? W(`bias[oc_base + ${e}u]`, l) : `0.0`};`,
    );
    for (let t = 0; t < 9; ++t)
      b.push(
        `      acc${e} = acc${e} + v${t} * ${W(`wTile[${e * o * 9}u + li * 9u + ${t}u]`, d)};`,
      );
    x.push(
      `    output[(oc_base + ${e}u) * H * W + y * W + x] = ${Jr(`acc${e}`, u)};`,
    );
  }
  return `${qr(s, c, l, u)}struct Params { _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${p}>;
@group(0) @binding(1) var<storage, read>       weight: array<${m}>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${g}>;
`
    : ``
}@group(0) @binding(${i ? 3 : 2}) var<storage, read_write> output: array<${_}>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const OUT_TILE: u32 = ${a}u;
const IC_TILE: u32 = ${o}u;
const OC_STRIDE: u32 = ${e * 9}u;
const WTILE_ELEMS: u32 = ${v}u;

var<workgroup> wTile: array<${h}, ${v}>;

fn loadInput(ic: u32, iy: i32, ix: i32) -> f32 {
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {
    return 0.0;
  }
  return ${W(`input[ic * H * W + u32(iy) * W + u32(ix)]`, s)};
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32
) {
  let y = gid.x;
  let x = gid.y;
  let oc_base = gid.z * OUT_TILE;
  let isActive = y < H && x < W && oc_base < OUT_C;

${y.join(`
`)}
  for (var ic_base: u32 = 0u; ic_base < IN_C; ic_base = ic_base + IC_TILE) {
    let tile_count = min(IC_TILE, IN_C - ic_base);
    for (var wi: u32 = local_idx; wi < WTILE_ELEMS; wi = wi + 64u) {
      let tile_oc = wi / (IC_TILE * 9u);
      let rem0 = wi - tile_oc * IC_TILE * 9u;
      let tile_ic = rem0 / 9u;
      let tap = rem0 - tile_ic * 9u;
      if (tile_ic < tile_count) {
        wTile[wi] = weight[(oc_base + tile_oc) * OC_STRIDE + (ic_base + tile_ic) * 9u + tap];
      } else {
        wTile[wi] = ${d === `float16` ? `0.0h` : `0.0`};
      }
    }
    workgroupBarrier();

    for (var li: u32 = 0u; li < tile_count; li = li + 1u) {
      let ic = ic_base + li;
      var v0: f32;
      var v1: f32;
      var v2: f32;
      var v3: f32;
      var v4: f32;
      var v5: f32;
      var v6: f32;
      var v7: f32;
      var v8: f32;
      if (y > 0u && y + 1u < H && x > 0u && x + 1u < W) {
        let base0 = ic * H * W + (y - 1u) * W + x - 1u;
        let base1 = base0 + W;
        let base2 = base1 + W;
        v0 = ${W(`input[base0]`, s)};
        v1 = ${W(`input[base0 + 1u]`, s)};
        v2 = ${W(`input[base0 + 2u]`, s)};
        v3 = ${W(`input[base1]`, s)};
        v4 = ${W(`input[base1 + 1u]`, s)};
        v5 = ${W(`input[base1 + 2u]`, s)};
        v6 = ${W(`input[base2]`, s)};
        v7 = ${W(`input[base2 + 1u]`, s)};
        v8 = ${W(`input[base2 + 2u]`, s)};
      } else {
        let iy = i32(y);
        let ix = i32(x);
        v0 = loadInput(ic, iy - 1, ix - 1);
        v1 = loadInput(ic, iy - 1, ix);
        v2 = loadInput(ic, iy - 1, ix + 1);
        v3 = loadInput(ic, iy, ix - 1);
        v4 = loadInput(ic, iy, ix);
        v5 = loadInput(ic, iy, ix + 1);
        v6 = loadInput(ic, iy + 1, ix - 1);
        v7 = loadInput(ic, iy + 1, ix);
        v8 = loadInput(ic, iy + 1, ix + 1);
      }
${b.join(`
`)}
    }
    workgroupBarrier();
  }
  if (isActive) {
${x.join(`
`)}
  }
}
`;
}
function Qr({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  hasAdd: a = !1,
  inputDtype: o = `float16`,
  weightDtype: s = `float16`,
  biasDtype: c = `float16`,
  outputDtype: l = `float16`,
  residualDtype: u = `float16`,
  mTile: d = 32,
  nTile: f = 16,
  rowPerThread: p = 4,
  kTile: m = 64,
  accumDtype: h = `float16`,
}) {
  if (o !== `float16` || s !== `float16` || l !== `float16`)
    throw Error(`packed 3x3 conv requires f16 input/weight/output`);
  if (a && u !== `float16`)
    throw Error(`packed 3x3 conv fused add requires f16 residual`);
  if (t % 4 != 0 || m % 4 != 0 || d % p !== 0)
    throw Error(`invalid packed 3x3 conv tile`);
  if (h !== `float16` && h !== `float32`)
    throw Error(`invalid packed 3x3 conv accum dtype`);
  let g = f,
    _ = d / p,
    v = g * _;
  if (v > 256) throw Error(`packed 3x3 conv exceeds max workgroup invocations`);
  let y = e * 9,
    b = m / 4;
  if ((d * b + m * g) * 8 > 16 * 1024)
    throw Error(`packed 3x3 conv exceeds 16KB workgroup storage`);
  let x = t % (f * 4) == 0,
    S = y % m === 0,
    C = h === `float32`,
    w = C ? `f32` : `f16`,
    T = U(c),
    E = i ? 3 : 2,
    D = E + +!!a,
    O = (e) =>
      i
        ? C
          ? `f32(bias[n_group * 4u + ${e}u])`
          : `bias[n_group * 4u + ${e}u]`
        : C
          ? `0.0`
          : `0.0h`,
    k = Array.from(
      { length: p },
      (e, t) =>
        `  var acc${t}: vec4<${w}> = vec4<${w}>(${O(0)}, ${O(1)}, ${O(2)}, ${O(3)});`,
    ).join(`
`),
    A = Array.from({ length: p }, (e, t) =>
      C
        ? `        let a${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = acc${t} + vec4<f32>(b0) * f32(a${t}.x) + vec4<f32>(b1) * f32(a${t}.y) + vec4<f32>(b2) * f32(a${t}.z) + vec4<f32>(b3) * f32(a${t}.w);`
        : `        let a${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = acc${t} + b0 * a${t}.x + b1 * a${t}.y + b2 * a${t}.z + b3 * a${t}.w;`,
    ).join(`
`),
    j = Array.from({ length: p }, (e, t) => {
      let n = C ? `vec4<f16>(acc${t})` : `acc${t}`;
      return `  if (${x ? `m_base + ${t}u < HW` : `n_group < OUT_C_V4 && m_base + ${t}u < HW`}) {
    let m${t} = m_base + ${t}u;
    let oc${t} = n_group * 4u;
    let v${t} = ${n};
    let idx${t}0 = (oc${t} + 0u) * HW + m${t};
    let idx${t}1 = (oc${t} + 1u) * HW + m${t};
    let idx${t}2 = (oc${t} + 2u) * HW + m${t};
    let idx${t}3 = (oc${t} + 3u) * HW + m${t};
    output[idx${t}0] = v${t}.x${a ? ` + residual[idx${t}0]` : ``};
    output[idx${t}1] = v${t}.y${a ? ` + residual[idx${t}1]` : ``};
    output[idx${t}2] = v${t}.z${a ? ` + residual[idx${t}2]` : ``};
    output[idx${t}3] = v${t}.w${a ? ` + residual[idx${t}3]` : ``};
  }`;
    }).join(`
`),
    M =
      S && x
        ? `      bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg];`
        : S
          ? `      if (bg < OUT_C_V4) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`
          : x
            ? `      if (k_base + kk < K_TOTAL) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`
            : `      if (k_base + kk < K_TOTAL && bg < OUT_C_V4) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`,
    N = S
      ? `fn loadIm2col(m: u32, k: u32) -> f16 {
  if (m >= HW) { return 0.0h; }
  let ic = k / 9u;
  let tap = k - ic * 9u;
  let kh = tap / 3u;
  let kw = tap - kh * 3u;
  let y = m / W;
  let x = m - y * W;
  if (y > 0u && y + 1u < H && x > 0u && x + 1u < W) {
    return input[ic * HW + (y + kh - 1u) * W + (x + kw - 1u)];
  }
  let iy = i32(y) + i32(kh) - 1;
  let ix = i32(x) + i32(kw) - 1;
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) { return 0.0h; }
  return input[ic * HW + u32(iy) * W + u32(ix)];
}`
      : `fn loadIm2col(m: u32, k: u32) -> f16 {
  if (m >= HW || k >= K_TOTAL) { return 0.0h; }
  let ic = k / 9u;
  let tap = k - ic * 9u;
  let kh = tap / 3u;
  let kw = tap - kh * 3u;
  let y = m / W;
  let x = m - y * W;
  if (y > 0u && y + 1u < H && x > 0u && x + 1u < W) {
    return input[ic * HW + (y + kh - 1u) * W + (x + kw - 1u)];
  }
  let iy = i32(y) + i32(kh) - 1;
  let ix = i32(x) + i32(kw) - 1;
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) { return 0.0h; }
  return input[ic * HW + u32(iy) * W + u32(ix)];
}`;
  return `enable f16;
@group(0) @binding(0) var<storage, read>       input: array<f16>;
@group(0) @binding(1) var<storage, read>       weight: array<vec4<f16>>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${T}>;
`
    : ``
}${
    a
      ? `@group(0) @binding(${E}) var<storage, read>       residual: array<f16>;
`
      : ``
  }@group(0) @binding(${D}) var<storage, read_write> output: array<f16>;

const OUT_C_V4: u32 = ${t / 4}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const HW: u32 = ${n * r}u;
const K_TOTAL: u32 = ${y}u;
const M_TILE: u32 = ${d}u;
const WG_X: u32 = ${g}u;
const ROW_PER_THREAD: u32 = ${p}u;
const K_TILE: u32 = ${m}u;
const K_TILE_V4: u32 = ${b}u;
const WG: u32 = ${v}u;

var<workgroup> aTile: array<vec4<f16>, ${d * b}>;
var<workgroup> bTile: array<vec4<f16>, ${m * g}>;

${N}

@compute @workgroup_size(${g}, ${_}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * M_TILE + row_base;
  let n_group = wg.x * WG_X + lx;
${k}
  for (var k_base: u32 = 0u; k_base < K_TOTAL; k_base = k_base + K_TILE) {
    for (var i: u32 = tid; i < ${d * b}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let m = wg.y * M_TILE + tm;
      let k0 = k_base + kv * 4u;
      aTile[i] = vec4<f16>(loadIm2col(m, k0 + 0u), loadIm2col(m, k0 + 1u), loadIm2col(m, k0 + 2u), loadIm2col(m, k0 + 3u));
    }
    for (var i: u32 = tid; i < ${m * g}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let bg = wg.x * WG_X + nx;
${M}
    }
    workgroupBarrier();
    if (m_base < HW) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let b0 = bTile[(kv * 4u + 0u) * WG_X + lx];
        let b1 = bTile[(kv * 4u + 1u) * WG_X + lx];
        let b2 = bTile[(kv * 4u + 2u) * WG_X + lx];
        let b3 = bTile[(kv * 4u + 3u) * WG_X + lx];
${A}
      }
    }
    workgroupBarrier();
  }
${j}
}
`;
}
function $r({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  hasAdd: a = !1,
  inputDtype: o = `float16`,
  weightDtype: s = `float16`,
  biasDtype: c = `float16`,
  outputDtype: l = `float16`,
  residualDtype: u = `float16`,
  mTile: d = 32,
  nTile: f = 8,
  kTile: p = 8,
}) {
  if (o !== `float16` || s !== `float16` || l !== `float16`)
    throw Error(`winograd 3x3 conv requires f16 input/weight/output`);
  if (a && u !== `float16`)
    throw Error(`winograd 3x3 conv fused add requires f16 residual`);
  if (t % 4 != 0 || e % p !== 0) throw Error(`invalid winograd 3x3 conv shape`);
  let m = f,
    h = d,
    g = m * h;
  if (g > 256)
    throw Error(`winograd 3x3 conv exceeds max workgroup invocations`);
  let _ = d * p * 16,
    v = p * 16 * f;
  if (_ * 2 + v * 8 > 16 * 1024)
    throw Error(`winograd 3x3 conv exceeds 16KB workgroup storage`);
  let y = Math.ceil(r / 2),
    b = y * Math.ceil(n / 2),
    x = t % (f * 4) == 0,
    S = U(c),
    C = i ? 3 : 2,
    w = C + +!!a,
    T = i
      ? `vec4<f16>(
    ${c === `float16` ? `bias[n_group * 4u + 0u]` : `f16(bias[n_group * 4u + 0u])`},
    ${c === `float16` ? `bias[n_group * 4u + 1u]` : `f16(bias[n_group * 4u + 1u])`},
    ${c === `float16` ? `bias[n_group * 4u + 2u]` : `f16(bias[n_group * 4u + 2u])`},
    ${c === `float16` ? `bias[n_group * 4u + 3u]` : `f16(bias[n_group * 4u + 3u])`})`
      : `vec4<f16>(0.0h)`,
    E = Array.from(
      { length: 16 },
      (e, t) => `  var acc${t}: vec4<f16> = vec4<f16>(0.0h);`,
    ).join(`
`),
    D = Array.from(
      { length: 16 },
      (e, t) =>
        `        acc${t} = fma(bTile[(kk * 16u + ${t}u) * WG_X + lx], vec4<f16>(aTile[(tile_local * K_TILE + kk) * 16u + ${t}u]), acc${t});`,
    ).join(`
`),
    O = x
      ? `if (tile_global >= TILES_TOTAL) { return; }`
      : `if (tile_global >= TILES_TOTAL || n_group >= OUT_C_V4) { return; }`,
    k = x
      ? `      bTile[i] = weight[(ic * 16u + alpha) * OUT_C_V4 + bg];`
      : `      if (bg < OUT_C_V4) { bTile[i] = weight[(ic * 16u + alpha) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`;
  return `enable f16;
@group(0) @binding(0) var<storage, read>       input: array<f16>;
@group(0) @binding(1) var<storage, read>       weight: array<vec4<f16>>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${S}>;
`
    : ``
}${
    a
      ? `@group(0) @binding(${C}) var<storage, read>       residual: array<f16>;
`
      : ``
  }@group(0) @binding(${w}) var<storage, read_write> output: array<f16>;

const IN_C: u32 = ${e}u;
const OUT_C_V4: u32 = ${t / 4}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const HW: u32 = ${n * r}u;
const TILES_X: u32 = ${y}u;
const TILES_TOTAL: u32 = ${b}u;
const M_TILE: u32 = ${d}u;
const WG_X: u32 = ${m}u;
const WG_Y: u32 = ${h}u;
const WG: u32 = ${g}u;
const K_TILE: u32 = ${p}u;

var<workgroup> aTile: array<f16, ${_}>;
var<workgroup> bTile: array<vec4<f16>, ${v}>;

fn loadOne(ic: u32, y: i32, x: i32) -> f16 {
  if (y < 0 || y >= i32(H) || x < 0 || x >= i32(W)) {
    return 0.0h;
  }
  return input[ic * HW + u32(y) * W + u32(x)];
}

fn writeInputTransform(tile_global: u32, ic: u32, dst: u32) {
  if (tile_global >= TILES_TOTAL || ic >= IN_C) {
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      aTile[dst + i] = 0.0h;
    }
    return;
  }

  let ty = tile_global / TILES_X;
  let tx = tile_global - ty * TILES_X;
  let oy = i32(ty * 2u);
  let ox = i32(tx * 2u);

  let d00 = loadOne(ic, oy - 1, ox - 1);
  let d01 = loadOne(ic, oy - 1, ox + 0);
  let d02 = loadOne(ic, oy - 1, ox + 1);
  let d03 = loadOne(ic, oy - 1, ox + 2);
  let d10 = loadOne(ic, oy + 0, ox - 1);
  let d11 = loadOne(ic, oy + 0, ox + 0);
  let d12 = loadOne(ic, oy + 0, ox + 1);
  let d13 = loadOne(ic, oy + 0, ox + 2);
  let d20 = loadOne(ic, oy + 1, ox - 1);
  let d21 = loadOne(ic, oy + 1, ox + 0);
  let d22 = loadOne(ic, oy + 1, ox + 1);
  let d23 = loadOne(ic, oy + 1, ox + 2);
  let d30 = loadOne(ic, oy + 2, ox - 1);
  let d31 = loadOne(ic, oy + 2, ox + 0);
  let d32 = loadOne(ic, oy + 2, ox + 1);
  let d33 = loadOne(ic, oy + 2, ox + 2);

  let t00 = d00 - d20;
  let t01 = d01 - d21;
  let t02 = d02 - d22;
  let t03 = d03 - d23;
  let t10 = d10 + d20;
  let t11 = d11 + d21;
  let t12 = d12 + d22;
  let t13 = d13 + d23;
  let t20 = d20 - d10;
  let t21 = d21 - d11;
  let t22 = d22 - d12;
  let t23 = d23 - d13;
  let t30 = d10 - d30;
  let t31 = d11 - d31;
  let t32 = d12 - d32;
  let t33 = d13 - d33;

  aTile[dst + 0u] = t00 - t02;
  aTile[dst + 1u] = t01 + t02;
  aTile[dst + 2u] = t02 - t01;
  aTile[dst + 3u] = t01 - t03;
  aTile[dst + 4u] = t10 - t12;
  aTile[dst + 5u] = t11 + t12;
  aTile[dst + 6u] = t12 - t11;
  aTile[dst + 7u] = t11 - t13;
  aTile[dst + 8u] = t20 - t22;
  aTile[dst + 9u] = t21 + t22;
  aTile[dst + 10u] = t22 - t21;
  aTile[dst + 11u] = t21 - t23;
  aTile[dst + 12u] = t30 - t32;
  aTile[dst + 13u] = t31 + t32;
  aTile[dst + 14u] = t32 - t31;
  aTile[dst + 15u] = t31 - t33;
}

fn storeOutput(tile_global: u32, n_group: u32,
  y00: vec4<f16>, y01: vec4<f16>, y10: vec4<f16>, y11: vec4<f16>) {
  ${O}
  let ty = tile_global / TILES_X;
  let tx = tile_global - ty * TILES_X;
  let oy = ty * 2u;
  let ox = tx * 2u;
  let oc = n_group * 4u;

  if (oy < H && ox < W) {
    let idx = oy * W + ox;
    output[(oc + 0u) * HW + idx] = y00.x${a ? ` + residual[(oc + 0u) * HW + idx]` : ``};
    output[(oc + 1u) * HW + idx] = y00.y${a ? ` + residual[(oc + 1u) * HW + idx]` : ``};
    output[(oc + 2u) * HW + idx] = y00.z${a ? ` + residual[(oc + 2u) * HW + idx]` : ``};
    output[(oc + 3u) * HW + idx] = y00.w${a ? ` + residual[(oc + 3u) * HW + idx]` : ``};
  }
  if (oy < H && ox + 1u < W) {
    let idx = oy * W + ox + 1u;
    output[(oc + 0u) * HW + idx] = y01.x${a ? ` + residual[(oc + 0u) * HW + idx]` : ``};
    output[(oc + 1u) * HW + idx] = y01.y${a ? ` + residual[(oc + 1u) * HW + idx]` : ``};
    output[(oc + 2u) * HW + idx] = y01.z${a ? ` + residual[(oc + 2u) * HW + idx]` : ``};
    output[(oc + 3u) * HW + idx] = y01.w${a ? ` + residual[(oc + 3u) * HW + idx]` : ``};
  }
  if (oy + 1u < H && ox < W) {
    let idx = (oy + 1u) * W + ox;
    output[(oc + 0u) * HW + idx] = y10.x${a ? ` + residual[(oc + 0u) * HW + idx]` : ``};
    output[(oc + 1u) * HW + idx] = y10.y${a ? ` + residual[(oc + 1u) * HW + idx]` : ``};
    output[(oc + 2u) * HW + idx] = y10.z${a ? ` + residual[(oc + 2u) * HW + idx]` : ``};
    output[(oc + 3u) * HW + idx] = y10.w${a ? ` + residual[(oc + 3u) * HW + idx]` : ``};
  }
  if (oy + 1u < H && ox + 1u < W) {
    let idx = (oy + 1u) * W + ox + 1u;
    output[(oc + 0u) * HW + idx] = y11.x${a ? ` + residual[(oc + 0u) * HW + idx]` : ``};
    output[(oc + 1u) * HW + idx] = y11.y${a ? ` + residual[(oc + 1u) * HW + idx]` : ``};
    output[(oc + 2u) * HW + idx] = y11.z${a ? ` + residual[(oc + 2u) * HW + idx]` : ``};
    output[(oc + 3u) * HW + idx] = y11.w${a ? ` + residual[(oc + 3u) * HW + idx]` : ``};
  }
}

@compute @workgroup_size(${m}, ${h}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let tile_local = ly;
  let tile_global = wg.y * M_TILE + tile_local;
  let n_group = wg.x * WG_X + lx;
${E}

  for (var k_base: u32 = 0u; k_base < IN_C; k_base = k_base + K_TILE) {
    for (var i: u32 = tid; i < ${d * p}u; i = i + WG) {
      let tm = i / K_TILE;
      let kk = i - tm * K_TILE;
      writeInputTransform(wg.y * M_TILE + tm, k_base + kk, i * 16u);
    }
    for (var i: u32 = tid; i < ${v}u; i = i + WG) {
      let ka = i / WG_X;
      let nx = i - ka * WG_X;
      let kk = ka / 16u;
      let alpha = ka - kk * 16u;
      let ic = k_base + kk;
      let bg = wg.x * WG_X + nx;
${k}
    }
    workgroupBarrier();

    if (${x ? `tile_global < TILES_TOTAL` : `tile_global < TILES_TOTAL && n_group < OUT_C_V4`}) {
      for (var kk: u32 = 0u; kk < K_TILE; kk = kk + 1u) {
${D}
      }
    }
    workgroupBarrier();
  }

  let b = ${T};
  let s00 = acc0 + acc4 + acc8;
  let s01 = acc1 + acc5 + acc9;
  let s02 = acc2 + acc6 + acc10;
  let s03 = acc3 + acc7 + acc11;
  let s10 = acc4 - acc8 - acc12;
  let s11 = acc5 - acc9 - acc13;
  let s12 = acc6 - acc10 - acc14;
  let s13 = acc7 - acc11 - acc15;
  storeOutput(tile_global, n_group, s00 + s01 + s02 + b, s01 - s02 - s03 + b, s10 + s11 + s12 + b, s11 - s12 - s13 + b);
}
`;
}
function ei({ inC: e, outC: t }) {
  if (t % 4 != 0)
    throw Error(`winograd weight transform requires outC divisible by 4`);
  return `enable f16;
@group(0) @binding(0) var<storage, read>       weight: array<f16>;
@group(0) @binding(1) var<storage, read_write> packed: array<f16>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;

fn store(ic: u32, oc: u32, alpha: u32, value: f32) {
  packed[(ic * 16u + alpha) * OUT_C + oc] = f16(value);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x;
  let ic = gid.y;
  if (oc >= OUT_C || ic >= IN_C) { return; }

  let base = oc * IN_C * 9u + ic * 9u;
  let g00 = f32(weight[base + 0u]);
  let g01 = f32(weight[base + 1u]);
  let g02 = f32(weight[base + 2u]);
  let g10 = f32(weight[base + 3u]);
  let g11 = f32(weight[base + 4u]);
  let g12 = f32(weight[base + 5u]);
  let g20 = f32(weight[base + 6u]);
  let g21 = f32(weight[base + 7u]);
  let g22 = f32(weight[base + 8u]);

  let t00 = g00;
  let t01 = g01;
  let t02 = g02;
  let t10 = 0.5 * (g00 + g10 + g20);
  let t11 = 0.5 * (g01 + g11 + g21);
  let t12 = 0.5 * (g02 + g12 + g22);
  let t20 = 0.5 * (g00 - g10 + g20);
  let t21 = 0.5 * (g01 - g11 + g21);
  let t22 = 0.5 * (g02 - g12 + g22);
  let t30 = g20;
  let t31 = g21;
  let t32 = g22;

  store(ic, oc, 0u, t00);
  store(ic, oc, 1u, 0.5 * (t00 + t01 + t02));
  store(ic, oc, 2u, 0.5 * (t00 - t01 + t02));
  store(ic, oc, 3u, t02);
  store(ic, oc, 4u, t10);
  store(ic, oc, 5u, 0.5 * (t10 + t11 + t12));
  store(ic, oc, 6u, 0.5 * (t10 - t11 + t12));
  store(ic, oc, 7u, t12);
  store(ic, oc, 8u, t20);
  store(ic, oc, 9u, 0.5 * (t20 + t21 + t22));
  store(ic, oc, 10u, 0.5 * (t20 - t21 + t22));
  store(ic, oc, 11u, t22);
  store(ic, oc, 12u, t30);
  store(ic, oc, 13u, 0.5 * (t30 + t31 + t32));
  store(ic, oc, 14u, 0.5 * (t30 - t31 + t32));
  store(ic, oc, 15u, t32);
}
`;
}
function ti({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  hasAdd: a = !1,
  inputDtype: o = `float16`,
  weightDtype: s = `float16`,
  biasDtype: c = `float16`,
  outputDtype: l = `float16`,
  residualDtype: u = `float16`,
  mTile: d = 112,
  nTile: f = 16,
  rowPerThread: p = 14,
  kTile: m = 32,
  accumDtype: h = `float16`,
}) {
  if (o !== `float16` || s !== `float16` || l !== `float16`)
    throw Error(`packed 1x1 conv requires f16 input/weight/output`);
  if (a && u !== `float16`)
    throw Error(`packed 1x1 conv fused add requires f16 residual`);
  if (t % 4 != 0 || e % 4 != 0 || m % 4 != 0 || d % p !== 0)
    throw Error(`invalid packed 1x1 conv tile`);
  if (h !== `float16` && h !== `float32`)
    throw Error(`invalid packed 1x1 conv accum dtype`);
  let g = f,
    _ = d / p,
    v = g * _;
  if (v > 256) throw Error(`packed 1x1 conv exceeds max workgroup invocations`);
  let y = m / 4;
  if ((d * y + m * g) * 8 > 16 * 1024)
    throw Error(`packed 1x1 conv exceeds 16KB workgroup storage`);
  let b = t % (f * 4) == 0,
    x = e % m === 0,
    S = h === `float32`,
    C = S ? `f32` : `f16`,
    w = U(c),
    T = i ? 3 : 2,
    E = T + +!!a,
    D = (e) =>
      i
        ? S
          ? `f32(bias[n_group * 4u + ${e}u])`
          : `bias[n_group * 4u + ${e}u]`
        : S
          ? `0.0`
          : `0.0h`,
    O = Array.from(
      { length: p },
      (e, t) =>
        `  var acc${t}: vec4<${C}> = vec4<${C}>(${D(0)}, ${D(1)}, ${D(2)}, ${D(3)});`,
    ).join(`
`),
    k = Array.from({ length: p }, (e, t) =>
      S
        ? `        let a${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(vec4<f32>(b3), vec4<f32>(f32(a${t}.w)), fma(vec4<f32>(b2), vec4<f32>(f32(a${t}.z)), fma(vec4<f32>(b1), vec4<f32>(f32(a${t}.y)), fma(vec4<f32>(b0), vec4<f32>(f32(a${t}.x)), acc${t}))));`
        : `        let a${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(b3, vec4<f16>(a${t}.w), fma(b2, vec4<f16>(a${t}.z), fma(b1, vec4<f16>(a${t}.y), fma(b0, vec4<f16>(a${t}.x), acc${t}))));`,
    ).join(`
`),
    A = Array.from({ length: p }, (e, t) => {
      let n = S ? `vec4<f16>(acc${t})` : `acc${t}`;
      return `  if (${b ? `m_base + ${t}u < HW` : `n_group < OUT_C_V4 && m_base + ${t}u < HW`}) {
    let m${t} = m_base + ${t}u;
    let oc${t} = n_group * 4u;
    let v${t} = ${n};
    let idx${t}0 = (oc${t} + 0u) * HW + m${t};
    let idx${t}1 = (oc${t} + 1u) * HW + m${t};
    let idx${t}2 = (oc${t} + 2u) * HW + m${t};
    let idx${t}3 = (oc${t} + 3u) * HW + m${t};
    output[idx${t}0] = v${t}.x${a ? ` + residual[idx${t}0]` : ``};
    output[idx${t}1] = v${t}.y${a ? ` + residual[idx${t}1]` : ``};
    output[idx${t}2] = v${t}.z${a ? ` + residual[idx${t}2]` : ``};
    output[idx${t}3] = v${t}.w${a ? ` + residual[idx${t}3]` : ``};
  }`;
    }).join(`
`),
    j =
      x && b
        ? `      bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg];`
        : x
          ? `      if (bg < OUT_C_V4) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`
          : b
            ? `      if (k_base + kk < IN_C) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`
            : `      if (k_base + kk < IN_C && bg < OUT_C_V4) { bTile[i] = weight[(k_base + kk) * OUT_C_V4 + bg]; }
      else { bTile[i] = vec4<f16>(0.0h); }`;
  return `enable f16;
@group(0) @binding(0) var<storage, read>       input: array<f16>;
@group(0) @binding(1) var<storage, read>       weight: array<vec4<f16>>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${w}>;
`
    : ``
}${
    a
      ? `@group(0) @binding(${T}) var<storage, read>       residual: array<f16>;
`
      : ``
  }@group(0) @binding(${E}) var<storage, read_write> output: array<f16>;

const IN_C: u32 = ${e}u;
const OUT_C_V4: u32 = ${t / 4}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const HW: u32 = ${n * r}u;
const K_TILE: u32 = ${m}u;
const K_TILE_V4: u32 = ${y}u;
const WG_X: u32 = ${g}u;
const ROW_PER_THREAD: u32 = ${p}u;
const WG: u32 = ${v}u;

var<workgroup> aTile: array<vec4<f16>, ${d * y}>;
var<workgroup> bTile: array<vec4<f16>, ${m * g}>;

fn loadInput(m: u32, ic: u32) -> f16 {
  if (m >= HW || ic >= IN_C) { return 0.0h; }
  return input[ic * HW + m];
}

@compute @workgroup_size(${g}, ${_}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * ${d}u + row_base;
  let n_group = wg.x * WG_X + lx;
${O}
  for (var k_base: u32 = 0u; k_base < IN_C; k_base = k_base + K_TILE) {
    for (var i: u32 = tid; i < ${d * y}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let m = wg.y * ${d}u + tm;
      let k0 = k_base + kv * 4u;
      aTile[i] = vec4<f16>(loadInput(m, k0 + 0u), loadInput(m, k0 + 1u), loadInput(m, k0 + 2u), loadInput(m, k0 + 3u));
    }
    for (var i: u32 = tid; i < ${m * g}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let bg = wg.x * WG_X + nx;
${j}
    }
    workgroupBarrier();
    if (m_base < HW) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let b0 = bTile[(kv * 4u + 0u) * WG_X + lx];
        let b1 = bTile[(kv * 4u + 1u) * WG_X + lx];
        let b2 = bTile[(kv * 4u + 2u) * WG_X + lx];
        let b3 = bTile[(kv * 4u + 3u) * WG_X + lx];
${k}
      }
    }
    workgroupBarrier();
  }
${A}
}
`;
}
function ni({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  outTile: a = 16,
  icTile: o = 16,
  inputDtype: s = `float16`,
  weightDtype: c = `float16`,
  biasDtype: l = `float16`,
  outputDtype: u = `float16`,
  phaseY: d = 0,
  phaseX: f = 0,
}) {
  if (t % a !== 0) throw Error(`outC must be divisible by outTile`);
  if ((d !== 0 && d !== 1) || (f !== 0 && f !== 1))
    throw Error(`upsample conv phase must be 0 or 1`);
  let p = c === `float16` ? `float16` : `float32`,
    m = p === `float16` ? 2 : 4;
  if (a * o * 4 * m > 16 * 1024)
    throw Error(
      `upsample conv shared-weight tile exceeds 16KB workgroup storage`,
    );
  let h = U(s),
    g = U(c),
    _ = U(p),
    v = U(l),
    y = U(u),
    b = n * 2,
    x = r * 2,
    S = a * o * 4,
    C = d === 0 ? -1 : 0,
    w = d === 0 ? 0 : 1,
    T = f === 0 ? -1 : 0,
    E = f === 0 ? 0 : 1,
    D = [],
    O = [],
    k = [];
  for (let e = 0; e < a; ++e)
    (D.push(
      `  var acc${e}: f32 = ${i ? W(`bias[oc_base + ${e}u]`, l) : `0.0`};`,
    ),
      O.push(`      acc${e} = acc${e} + v0 * ${W(`wTile[${e * o * 4}u + li * 4u + 0u]`, p)}
          + v1 * ${W(`wTile[${e * o * 4}u + li * 4u + 1u]`, p)}
          + v2 * ${W(`wTile[${e * o * 4}u + li * 4u + 2u]`, p)}
          + v3 * ${W(`wTile[${e * o * 4}u + li * 4u + 3u]`, p)};`),
      k.push(
        `    output[(oc_base + ${e}u) * OUT_HW + out_y * OUT_W + out_x] = ${Jr(`acc${e}`, u)};`,
      ));
  return `${qr(s, c, l, u)}struct Params { _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${h}>;
@group(0) @binding(1) var<storage, read>       weight: array<${g}>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${v}>;
`
    : ``
}@group(0) @binding(${i ? 3 : 2}) var<storage, read_write> output: array<${y}>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const OUT_H: u32 = ${b}u;
const OUT_W: u32 = ${x}u;
const LOW_HW: u32 = ${n * r}u;
const OUT_HW: u32 = ${b * x}u;
const OUT_TILE: u32 = ${a}u;
const IC_TILE: u32 = ${o}u;
const WTILE_ELEMS: u32 = ${S}u;
const PHASE_Y: u32 = ${d}u;
const PHASE_X: u32 = ${f}u;

var<workgroup> wTile: array<${_}, ${S}>;

fn loadInput(ic: u32, iy: i32, ix: i32) -> f32 {
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {
    return 0.0;
  }
  return ${W(`input[ic * LOW_HW + u32(iy) * W + u32(ix)]`, s)};
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32
) {
  let y = gid.x;
  let x = gid.y;
  let oc_base = gid.z * OUT_TILE;
  let isActive = y < H && x < W && oc_base < OUT_C;
  let out_y = y * 2u + PHASE_Y;
  let out_x = x * 2u + PHASE_X;

${D.join(`
`)}
  for (var ic_base: u32 = 0u; ic_base < IN_C; ic_base = ic_base + IC_TILE) {
    let tile_count = min(IC_TILE, IN_C - ic_base);
    for (var wi: u32 = local_idx; wi < WTILE_ELEMS; wi = wi + 64u) {
      let tile_oc = wi / (IC_TILE * 4u);
      let rem0 = wi - tile_oc * IC_TILE * 4u;
      let tile_ic = rem0 / 4u;
      let tap = rem0 - tile_ic * 4u;
      if (tile_ic < tile_count) {
        wTile[wi] = weight[(oc_base + tile_oc) * IN_C * 4u + (ic_base + tile_ic) * 4u + tap];
      } else {
        wTile[wi] = ${p === `float16` ? `0.0h` : `0.0`};
      }
    }
    workgroupBarrier();

    for (var li: u32 = 0u; li < tile_count; li = li + 1u) {
      let ic = ic_base + li;
      let v0 = select(0.0, loadInput(ic, i32(y) + ${C}, i32(x) + ${T}), isActive);
      let v1 = select(0.0, loadInput(ic, i32(y) + ${C}, i32(x) + ${E}), isActive);
      let v2 = select(0.0, loadInput(ic, i32(y) + ${w}, i32(x) + ${T}), isActive);
      let v3 = select(0.0, loadInput(ic, i32(y) + ${w}, i32(x) + ${E}), isActive);
${O.join(`
`)}
    }
    workgroupBarrier();
  }

  if (isActive) {
${k.join(`
`)}
  }
}
`;
}
function ri({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  inputDtype: a = `float16`,
  weightDtype: o = `float16`,
  biasDtype: s = `float16`,
  outputDtype: c = `float16`,
  phaseY: l = 0,
  phaseX: u = 0,
  mTile: d = 88,
  nTile: f = 16,
  rowPerThread: p = 11,
  kTile: m = 32,
  accumDtype: h = `float16`,
}) {
  if (a !== `float16` || o !== `float16` || c !== `float16`)
    throw Error(
      `packed upsample conv currently requires f16 input, weights, and output`,
    );
  if (t % 4 != 0 || m % 4 != 0 || (e * 4) % m != 0)
    throw Error(
      `packed upsample conv requires outC divisible by 4 and K divisible by kTile`,
    );
  if (d % p !== 0)
    throw Error(
      `packed upsample conv requires mTile divisible by rowPerThread`,
    );
  if ((l !== 0 && l !== 1) || (u !== 0 && u !== 1))
    throw Error(`upsample conv phase must be 0 or 1`);
  let g = f,
    _ = d / p,
    v = g * _;
  if (v > 256)
    throw Error(`packed upsample conv exceeds max workgroup invocations`);
  let y = e * 4,
    b = m / 4;
  if ((d * b + m * g) * 8 > 16 * 1024)
    throw Error(`packed upsample conv exceeds 16KB workgroup storage`);
  let x = t % (f * 4) == 0;
  if (h !== `float16` && h !== `float32`)
    throw Error(`packed upsample conv accumulation must be f16 or f32`);
  let S = n * 2,
    C = r * 2,
    w = S * C,
    T = g * 4,
    E = h === `float32`,
    D = E ? `f32` : `f16`,
    O = U(s),
    k = l === 0 ? -1 : 0,
    A = l === 0 ? 0 : 1,
    j = u === 0 ? -1 : 0,
    M = u === 0 ? 0 : 1,
    N = i
      ? E
        ? `vec4<f32>(
    ${W(`bias[n_group * 4u + 0u]`, s)},
    ${W(`bias[n_group * 4u + 1u]`, s)},
    ${W(`bias[n_group * 4u + 2u]`, s)},
    ${W(`bias[n_group * 4u + 3u]`, s)})`
        : `vec4<f16>(
    ${s === `float16` ? `bias[n_group * 4u + 0u]` : `f16(bias[n_group * 4u + 0u])`},
    ${s === `float16` ? `bias[n_group * 4u + 1u]` : `f16(bias[n_group * 4u + 1u])`},
    ${s === `float16` ? `bias[n_group * 4u + 2u]` : `f16(bias[n_group * 4u + 2u])`},
    ${s === `float16` ? `bias[n_group * 4u + 3u]` : `f16(bias[n_group * 4u + 3u])`})`
      : E
        ? `vec4<f32>(0.0)`
        : `vec4<f16>(0.0h)`,
    P = Array.from({ length: p }, (e, t) => `  var acc${t}: vec4<${D}> = ${N};`)
      .join(`
`),
    F = Array.from({ length: p }, (e, t) =>
      E
        ? `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(vec4<f32>(b3), vec4<f32>(f32(a_vec${t}.w)), fma(vec4<f32>(b2), vec4<f32>(f32(a_vec${t}.z)), fma(vec4<f32>(b1), vec4<f32>(f32(a_vec${t}.y)), fma(vec4<f32>(b0), vec4<f32>(f32(a_vec${t}.x)), acc${t}))));`
        : `        let a_vec${t} = aTile[(row_base + ${t}u) * K_TILE_V4 + kv];
        acc${t} = fma(b3, vec4<f16>(a_vec${t}.w), fma(b2, vec4<f16>(a_vec${t}.z), fma(b1, vec4<f16>(a_vec${t}.y), fma(b0, vec4<f16>(a_vec${t}.x), acc${t}))));`,
    ).join(`
`),
    I = Array.from(
      { length: p },
      (e, t) =>
        `  storeVec4(m_base + ${t}u, n_group, ${E ? `vec4<f16>(acc${t})` : `acc${t}`});`,
    ).join(`
`),
    ee = x
      ? `if (m_global >= LOW_HW) {`
      : `if (m_global >= LOW_HW || n_group >= N_V4) {`,
    L = x
      ? `      bTile[i] = weight[(k_base + kk) * N_V4 + b_group];`
      : `      if (b_group < N_V4) {
        bTile[i] = weight[(k_base + kk) * N_V4 + b_group];
      } else {
        bTile[i] = vec4<f16>(0.0h);
      }`,
    te = i ? 3 : 2;
  return `enable f16;
@group(0) @binding(0) var<storage, read>       input: array<f16>;
@group(0) @binding(1) var<storage, read>       weight: array<vec4<f16>>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${O}>;
`
    : ``
}@group(0) @binding(${te}) var<storage, read_write> output: array<f16>;

const IN_C: u32 = ${e}u;
const OUT_C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const LOW_HW: u32 = ${n * r}u;
const OUT_H: u32 = ${S}u;
const OUT_W: u32 = ${C}u;
const OUT_HW: u32 = ${w}u;
const K_TOTAL: u32 = ${y}u;
const K_V4: u32 = ${e}u;
const N_V4: u32 = ${t / 4}u;
const M_TILE: u32 = ${d}u;
const WG_X: u32 = ${g}u;
const WG_Y: u32 = ${_}u;
const ROW_PER_THREAD: u32 = ${p}u;
const OUT_TILE: u32 = ${T}u;
const K_TILE: u32 = ${m}u;
const K_TILE_V4: u32 = ${b}u;
const WG: u32 = ${v}u;
const PHASE_Y: u32 = ${l}u;
const PHASE_X: u32 = ${u}u;

var<workgroup> aTile: array<vec4<f16>, ${d * b}>;
var<workgroup> bTile: array<vec4<f16>, ${m * g}>;

fn loadOne(ic: u32, iy: i32, ix: i32) -> f16 {
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {
    return 0.0h;
  }
  return input[ic * LOW_HW + u32(iy) * W + u32(ix)];
}

fn loadInputVec(m_global: u32, ic: u32) -> vec4<f16> {
  if (m_global >= LOW_HW || ic >= IN_C) {
    return vec4<f16>(0.0h);
  }
  let y = m_global / W;
  let x = m_global - y * W;
  return vec4<f16>(
    loadOne(ic, i32(y) + ${k}, i32(x) + ${j}),
    loadOne(ic, i32(y) + ${k}, i32(x) + ${M}),
    loadOne(ic, i32(y) + ${A}, i32(x) + ${j}),
    loadOne(ic, i32(y) + ${A}, i32(x) + ${M})
  );
}

fn storeVec4(m_global: u32, n_group: u32, v: vec4<f16>) {
  ${ee}
    return;
  }
  let y = m_global / W;
  let x = m_global - y * W;
  let out_y = y * 2u + PHASE_Y;
  let out_x = x * 2u + PHASE_X;
  let out_pos = out_y * OUT_W + out_x;
  let oc = n_group * 4u;
  output[(oc + 0u) * OUT_HW + out_pos] = v.x;
  output[(oc + 1u) * OUT_HW + out_pos] = v.y;
  output[(oc + 2u) * OUT_HW + out_pos] = v.z;
  output[(oc + 3u) * OUT_HW + out_pos] = v.w;
}

@compute @workgroup_size(${g}, ${_}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let tid = ly * WG_X + lx;
  let row_base = ly * ROW_PER_THREAD;
  let m_base = wg.y * M_TILE + row_base;
  let n_group = wg.x * WG_X + lx;
${P}

  for (var k_base: u32 = 0u; k_base < K_TOTAL; k_base = k_base + K_TILE) {
    let k_base_v4 = k_base / 4u;
    for (var i: u32 = tid; i < ${d * b}u; i = i + WG) {
      let tm = i / K_TILE_V4;
      let kv = i - tm * K_TILE_V4;
      let gm = wg.y * M_TILE + tm;
      aTile[i] = loadInputVec(gm, k_base_v4 + kv);
    }
    for (var i: u32 = tid; i < ${m * g}u; i = i + WG) {
      let kk = i / WG_X;
      let nx = i - kk * WG_X;
      let b_group = wg.x * WG_X + nx;
${L}
    }
    workgroupBarrier();

    if (m_base < LOW_HW) {
      for (var kv: u32 = 0u; kv < K_TILE_V4; kv = kv + 1u) {
        let b0 = bTile[(kv * 4u + 0u) * WG_X + lx];
        let b1 = bTile[(kv * 4u + 1u) * WG_X + lx];
        let b2 = bTile[(kv * 4u + 2u) * WG_X + lx];
        let b3 = bTile[(kv * 4u + 3u) * WG_X + lx];
${F}
      }
    }
    workgroupBarrier();
  }

${I}
}
`;
}
function ii({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  hasAdd: a = !1,
  inputDtype: o = `float32`,
  weightDtype: s = `float32`,
  biasDtype: c = `float32`,
  outputDtype: l = `float32`,
  residualDtype: u = `float32`,
  tileM: d = 32,
  tileN: f = 64,
  k: p = 3,
  pad: m = 1,
  weightLayout: h = `ic-tap`,
}) {
  if (e % 32 != 0)
    throw Error(`conv subgroup matrix requires inC divisible by 32`);
  if (t % 64 != 0)
    throw Error(`conv subgroup matrix requires outC divisible by 64`);
  if (d !== 32 && d !== 64)
    throw Error(`conv subgroup matrix supports tileM=32 or tileM=64`);
  if (f !== 64 && f !== 128)
    throw Error(`conv subgroup matrix supports tileN=64 or tileN=128`);
  if (d === 64 && f !== 64)
    throw Error(`conv subgroup matrix tileM=64 requires tileN=64`);
  if (t % f !== 0) throw Error(`outC must be divisible by tileN=${f}`);
  if (!((p === 3 && m === 1) || (p === 1 && m === 0)))
    throw Error(`conv subgroup matrix supports only 3x3 pad1 or 1x1 pad0`);
  if (h !== `ic-tap` && h !== `tap-ic`)
    throw Error(`unsupported conv subgroup weightLayout: ${h}`);
  if (p !== 3 && h !== `ic-tap`)
    throw Error(`tap-ic conv subgroup weight layout is only valid for 3x3`);
  let g = n * r,
    _ = e * p * p,
    v = U(o),
    y = U(s),
    b = U(c),
    x = U(l),
    S = U(u),
    C = d === 64 ? f : f / 2,
    w = C / 8,
    T = i ? 3 : 2,
    E = T + +!!a,
    D = (e) => (o === `float16` ? e : `f16(${e})`),
    O =
      s === `float16`
        ? `weight[oc * K_TOTAL + kk]`
        : `f16(weight[oc * K_TOTAL + kk])`,
    k = p === 3 && h === `tap-ic`,
    A = k
      ? ``
      : p === 1
        ? `fn loadInputValue(m_global: u32, kk: u32) -> f16 {
  if (m_global >= HW || kk >= K_TOTAL) {
    return 0.0h;
  }
  return ${D(`input[kk * HW + m_global]`)};
}
`
        : `fn loadInputValue(m_global: u32, kk: u32) -> f16 {
  if (m_global >= HW || kk >= K_TOTAL) {
    return 0.0h;
  }
  let y = m_global / W;
  let x = m_global - y * W;
  let ic = kk / 9u;
  let tap = kk - ic * 9u;
  let kh = tap / 3u;
  let kw = tap - kh * 3u;
  let iy = i32(y) + i32(kh) - 1;
  let ix = i32(x) + i32(kw) - 1;
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {
    return 0.0h;
  }
  return ${D(`input[ic * HW + u32(iy) * W + u32(ix)]`)};
}
`,
    j = k
      ? `fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let m_global = tile_base + row;
  let col = c_idx * 8u;
  if (m_global >= HW) {
    for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
      tile_A[row * TILE_K + col + col_offset] = 0.0h;
    }
    return;
  }

  let y = m_global / W;
  let x = m_global - y * W;
  let tap = k_idx / IN_C;
  let kh = tap / 3u;
  let kw = tap - kh * 3u;
  let iy = i32(y) + i32(kh) - 1;
  let ix = i32(x) + i32(kw) - 1;
  let valid_pos = iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W);
  let ic_base = k_idx - tap * IN_C + col;
  let in_base = u32(iy) * W + u32(ix);
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    if (valid_pos) {
      let ic = ic_base + col_offset;
      tile_A[row * TILE_K + col + col_offset] = ${D(`input[ic * HW + in_base]`)};
    } else {
      tile_A[row * TILE_K + col + col_offset] = 0.0h;
    }
  }
}
`
      : `fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let m_global = tile_base + row;
  let col = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let kk = k_idx + col + col_offset;
    tile_A[row * TILE_K + col + col_offset] = loadInputValue(m_global, kk);
  }
}
`,
    M =
      d === 64
        ? `  let subtile_idx = 0u;
  let subtile_idy = subtile_id;`
        : `  let subtile_idx = subtile_id / 2u;
  let subtile_idy = subtile_id % 2u;`,
    N =
      d === 64
        ? `    loadSHMA(m_global_base, kidx, local_idx / 4u, local_idx % 4u);
    loadSHMA(m_global_base, kidx, local_idx / 4u + 32u, local_idx % 4u);`
        : `    loadSHMA(m_global_base, kidx, local_idx / 4u, local_idx % 4u);`,
    P = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: w },
        (e, n) => `  var matC${t}${n}: subgroup_matrix_result<f16, 8, 8>;`,
      ).join(`
`),
    ).join(`
`),
    F = Array.from(
      { length: w },
      (e, t) =>
        `      var matB${t}: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset + ${t * 8}u * TILE_K, true, TILE_K);`,
    ).join(`
`),
    I = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: w },
        (e, n) =>
          `      matC${t}${n} = subgroupMatrixMultiplyAccumulate(matA${t}, matB${n}, matC${t}${n});`,
      ).join(`
`),
    ).join(`
`),
    ee = (e) =>
      Array.from(
        { length: w },
        (t, n) =>
          `  subgroupMatrixStore(&scratch[subtile_id][${n}], 0u, matC${e}${n}, false, 8u);`,
      ).join(`
`),
    L = Array.from({ length: w }, (e, t) => {
      let n = t * 8;
      return `  storeOne(m, oc_base + col + ${n}u, scratch[src_slot][${t}][row * 8u + col]);
  storeOne(m, oc_base + col2 + ${n}u, scratch[src_slot][${t}][row * 8u + col2]);`;
    }).join(`
`);
  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

@group(0) @binding(0) var<storage, read>       input: array<${v}>;
@group(0) @binding(1) var<storage, read>       weight: array<${y}>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${b}>;
`
    : ``
}${
    a
      ? `@group(0) @binding(${T}) var<storage, read>       residual: array<${S}>;
`
      : ``
  }@group(0) @binding(${E}) var<storage, read_write> output: array<${x}>;

const IN_C:       u32 = ${e}u;
const OUT_C:      u32 = ${t}u;
const H:          u32 = ${n}u;
const W:          u32 = ${r}u;
const HW:         u32 = ${g}u;
const K_TOTAL:    u32 = ${_}u;
const TILE_COLS:  u32 = ${f}u;
const TILE_ROWS:  u32 = ${d}u;
const TILE_K:     u32 = 32u;
const SUB_COLS:   u32 = ${C}u;
const SUB_ROWS:   u32 = 16u;

var<workgroup> tile_A: array<f16, ${d} * 32>;
var<workgroup> tile_B: array<f16, ${f} * 32>;
var<workgroup> scratch: array<array<array<f16, 64>, ${w}>, 4>;

${A}
${j}

fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let col = c_idx * 16u;
  for (var row_offset: u32 = 0u; row_offset < TILE_COLS; row_offset = row_offset + 64u) {
    let b_row = row + row_offset;
    let oc = tile_base + b_row;
    for (var i: u32 = 0u; i < 16u; i++) {
      let kk = k_idx + col + i;
      if (oc < OUT_C && kk < K_TOTAL) {
        tile_B[b_row * TILE_K + col + i] = ${O};
      } else {
        tile_B[b_row * TILE_K + col + i] = 0.0h;
      }
    }
  }
}

fn storeOne(m_global: u32, oc: u32, value: f16) {
  if (m_global < HW && oc < OUT_C) {
    let idx = oc * HW + m_global;
    let out_value = f32(value)${i ? ` + ${W(`bias[oc]`, c)}` : ``}${a ? ` + ${W(`residual[idx]`, u)}` : ``};
    output[idx] = ${Jr(`out_value`, l)};
  }
}

fn storeOutput(m_base: u32, oc_base: u32, row: u32, col: u32, src_slot: u32) {
  let m = m_base + row;
  let col2 = col + 1u;
${L}
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let m_global_base = workgroup_id.y * TILE_ROWS;
  let oc_global_base = workgroup_id.x * TILE_COLS;

  let subtile_id = local_idx / sg_size;
${M}
  let base_A = subtile_idy * SUB_ROWS;
  let base_B = subtile_idx * SUB_COLS;

${P}

  for (var kidx: u32 = 0u; kidx < K_TOTAL; kidx = kidx + TILE_K) {
${N}
    loadSHMB(oc_global_base, kidx, local_idx / 2u, local_idx % 2u);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step = step + 8u) {
      let matrix_a_offset = subtile_idy * SUB_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset = subtile_idx * SUB_COLS * TILE_K + step;
${F}

${I}
    }
    workgroupBarrier();
  }

${ee(0)}
  let row = sg_id / 4u;
  let col = (sg_id % 4u) * 2u;
  storeOutput(m_global_base + base_A, oc_global_base + base_B, row, col, subtile_id);

${ee(1)}
  storeOutput(m_global_base + base_A + 8u, oc_global_base + base_B, row, col, subtile_id);
}
`;
}
function ai({
  inC: e,
  outC: t,
  H: n,
  W: r,
  hasBias: i,
  inputDtype: a = `float16`,
  weightDtype: o = `float16`,
  biasDtype: s = `float16`,
  outputDtype: c = `float16`,
  tileN: l = 64,
  phaseY: u = 0,
  phaseX: d = 0,
  fusedWeights: f = !1,
}) {
  if (e % 32 != 0)
    throw Error(`upsample conv subgroup matrix requires inC divisible by 32`);
  if (t % 64 != 0)
    throw Error(`upsample conv subgroup matrix requires outC divisible by 64`);
  if (l !== 64 && l !== 128)
    throw Error(`upsample conv subgroup matrix supports tileN=64 or tileN=128`);
  if (t % l !== 0) throw Error(`outC must be divisible by tileN=${l}`);
  if ((u !== 0 && u !== 1) || (d !== 0 && d !== 1))
    throw Error(`upsample conv phase must be 0 or 1`);
  let p = n * r,
    m = n * 2,
    h = r * 2,
    g = m * h,
    _ = e * 4,
    v = U(a),
    y = U(o),
    b = U(s),
    x = U(c),
    S = l / 2,
    C = S / 8,
    w = i ? 3 : 2,
    T = u === 0 ? -1 : 0,
    E = u === 0 ? 0 : 1,
    D = d === 0 ? -1 : 0,
    O = d === 0 ? 0 : 1,
    k = u === 0 ? [[0], [1, 2]] : [[0, 1], [2]],
    A = d === 0 ? [[0], [1, 2]] : [[0, 1], [2]],
    j = f
      ? ``
      : [0, 1, 2, 3].map((e) => {
          let t = Math.floor(e / 2),
            n = e % 2,
            r = [];
          for (let e of k[t])
            for (let t of A[n])
              r.push(W(`weight[oc * OC_STRIDE + ic * 9u + ${e * 3 + t}u]`, o));
          return `${e === 0 ? `  if` : `  else if`} (tap == ${e}u) {
    return f16(${r.join(` + `)});
  }`;
        }).join(`
`),
    M = f
      ? `fn loadWeightValue(oc: u32, kk: u32) -> f16 {
  if (oc >= OUT_C || kk >= K_TOTAL) {
    return 0.0h;
  }
  return ${o === `float16` ? `weight[oc * K_TOTAL + kk]` : `f16(weight[oc * K_TOTAL + kk])`};
}
`
      : `fn loadWeightValue(oc: u32, kk: u32) -> f16 {
  if (oc >= OUT_C || kk >= K_TOTAL) {
    return 0.0h;
  }
  let ic = kk / 4u;
  let tap = kk - ic * 4u;
${j}
  return 0.0h;
}
`,
    N = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: C },
        (e, n) => `  var matC${t}${n}: subgroup_matrix_result<f16, 8, 8>;`,
      ).join(`
`),
    ).join(`
`),
    P = Array.from(
      { length: C },
      (e, t) =>
        `      var matB${t}: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset + ${t * 8}u * TILE_K, true, TILE_K);`,
    ).join(`
`),
    F = Array.from({ length: 2 }, (e, t) =>
      Array.from(
        { length: C },
        (e, n) =>
          `      matC${t}${n} = subgroupMatrixMultiplyAccumulate(matA${t}, matB${n}, matC${t}${n});`,
      ).join(`
`),
    ).join(`
`),
    I = (e) =>
      Array.from(
        { length: C },
        (t, n) =>
          `  subgroupMatrixStore(&scratch[subtile_id][${n}], 0u, matC${e}${n}, false, 8u);`,
      ).join(`
`),
    ee = Array.from({ length: C }, (e, t) => {
      let n = t * 8;
      return `  storeOne(m, oc_base + col + ${n}u, scratch[src_slot][${t}][row * 8u + col]);
  storeOne(m, oc_base + col2 + ${n}u, scratch[src_slot][${t}][row * 8u + col2]);`;
    }).join(`
`);
  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

@group(0) @binding(0) var<storage, read>       input: array<${v}>;
@group(0) @binding(1) var<storage, read>       weight: array<${y}>;
${
  i
    ? `@group(0) @binding(2) var<storage, read>       bias: array<${b}>;
`
    : ``
}@group(0) @binding(${w}) var<storage, read_write> output: array<${x}>;

const IN_C:       u32 = ${e}u;
const OUT_C:      u32 = ${t}u;
const H:          u32 = ${n}u;
const W:          u32 = ${r}u;
const LOW_HW:     u32 = ${p}u;
const OUT_H:      u32 = ${m}u;
const OUT_W:      u32 = ${h}u;
const OUT_HW:     u32 = ${g}u;
const K_TOTAL:    u32 = ${_}u;
const OC_STRIDE:  u32 = ${e * 9}u;
const TILE_COLS:  u32 = ${l}u;
const TILE_ROWS:  u32 = 32u;
const TILE_K:     u32 = 32u;
const SUB_COLS:   u32 = ${S}u;
const SUB_ROWS:   u32 = 16u;
const PHASE_Y:    u32 = ${u}u;
const PHASE_X:    u32 = ${d}u;

var<workgroup> tile_A: array<f16, 32 * 32>;
var<workgroup> tile_B: array<f16, ${l} * 32>;
var<workgroup> scratch: array<array<array<f16, 64>, ${C}>, 4>;

fn loadInputValue(m_global: u32, kk: u32) -> f16 {
  if (m_global >= LOW_HW || kk >= K_TOTAL) {
    return 0.0h;
  }
  let y = m_global / W;
  let x = m_global - y * W;
  let ic = kk / 4u;
  let tap = kk - ic * 4u;
  let tap_y = tap / 2u;
  let tap_x = tap - tap_y * 2u;
  let iy = i32(y) + select(${T}, ${E}, tap_y == 1u);
  let ix = i32(x) + select(${D}, ${O}, tap_x == 1u);
  if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {
    return 0.0h;
  }
  return ${a === `float16` ? `input[ic * LOW_HW + u32(iy) * W + u32(ix)]` : `f16(input[ic * LOW_HW + u32(iy) * W + u32(ix)])`};
}

${M}

fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let m_global = tile_base + row;
  let col = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let kk = k_idx + col + col_offset;
    tile_A[row * TILE_K + col + col_offset] = loadInputValue(m_global, kk);
  }
}

fn loadSHMB(tile_base: u32, k_idx: u32, row: u32, c_idx: u32) {
  let col = c_idx * 16u;
  for (var row_offset: u32 = 0u; row_offset < TILE_COLS; row_offset = row_offset + 64u) {
    let b_row = row + row_offset;
    let oc = tile_base + b_row;
    for (var i: u32 = 0u; i < 16u; i++) {
      let kk = k_idx + col + i;
      tile_B[b_row * TILE_K + col + i] = loadWeightValue(oc, kk);
    }
  }
}

fn storeOne(m_global: u32, oc: u32, value: f16) {
  if (m_global < LOW_HW && oc < OUT_C) {
    let low_y = m_global / W;
    let low_x = m_global - low_y * W;
    let out_y = low_y * 2u + PHASE_Y;
    let out_x = low_x * 2u + PHASE_X;
    let idx = oc * OUT_HW + out_y * OUT_W + out_x;
    let out_value = f32(value)${i ? ` + ${W(`bias[oc]`, s)}` : ``};
    output[idx] = ${Jr(`out_value`, c)};
  }
}

fn storeOutput(m_base: u32, oc_base: u32, row: u32, col: u32, src_slot: u32) {
  let m = m_base + row;
  let col2 = col + 1u;
${ee}
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let m_global_base = workgroup_id.y * TILE_ROWS;
  let oc_global_base = workgroup_id.x * TILE_COLS;

  let subtile_id = local_idx / sg_size;
  let subtile_idx = subtile_id / 2u;
  let subtile_idy = subtile_id % 2u;
  let base_A = subtile_idy * SUB_ROWS;
  let base_B = subtile_idx * SUB_COLS;

${N}

  for (var kidx: u32 = 0u; kidx < K_TOTAL; kidx = kidx + TILE_K) {
    loadSHMA(m_global_base, kidx, local_idx / 4u, local_idx % 4u);
    loadSHMB(oc_global_base, kidx, local_idx / 2u, local_idx % 2u);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step = step + 8u) {
      let matrix_a_offset = subtile_idy * SUB_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset = subtile_idx * SUB_COLS * TILE_K + step;
${P}

${F}
    }
    workgroupBarrier();
  }

${I(0)}
  let row = sg_id / 4u;
  let col = (sg_id % 4u) * 2u;
  storeOutput(m_global_base + base_A, oc_global_base + base_B, row, col, subtile_id);

${I(1)}
  storeOutput(m_global_base + base_A + 8u, oc_global_base + base_B, row, col, subtile_id);
}
`;
}
function oi({
  C: e,
  H: t,
  W: n,
  groups: r,
  eps: i,
  applySilu: a = !1,
  inputDtype: o = `float32`,
  weightDtype: s = `float32`,
  biasDtype: c = `float32`,
  outputDtype: l = `float32`,
}) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  let u = e / r,
    d = U(o),
    f = U(s),
    p = U(c),
    m = U(l);
  return `${qr(o, s, c, l)}struct Params { _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${d}>;
@group(0) @binding(1) var<storage, read>       gWeight: array<${f}>;
@group(0) @binding(2) var<storage, read>       gBias: array<${p}>;
@group(0) @binding(3) var<storage, read_write> output: array<${m}>;

const C: u32 = ${e}u;
const H: u32 = ${t}u;
const W: u32 = ${n}u;
const GROUPS: u32 = ${r}u;
const CPG: u32 = ${u}u;
const HW: u32 = ${t * n}u;
const GROUP_SIZE: u32 = ${u * t * n}u;
const EPS: f32 = ${i};
const WG: u32 = 256u;

var<workgroup> partial: array<f32, 256>;
fn reduce_sum(tid: u32) -> f32 {
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;
  if (g >= GROUPS) { return; }
  let tid = lid.x;
  let g_offset = g * GROUP_SIZE;

  // Sum to compute mean.
  var s: f32 = 0.0;
  for (var i: u32 = tid; i < GROUP_SIZE; i = i + WG) {
    s = s + ${W(`input[g_offset + i]`, o)};
  }
  partial[tid] = s;
  let mean = reduce_sum(tid) / f32(GROUP_SIZE);

  // Sum of squared deviations.
  var sq: f32 = 0.0;
  for (var i: u32 = tid; i < GROUP_SIZE; i = i + WG) {
    let d = ${W(`input[g_offset + i]`, o)} - mean;
    sq = sq + d * d;
  }
  partial[tid] = sq;
  let invStd = inverseSqrt(reduce_sum(tid) / f32(GROUP_SIZE) + EPS);

  // Apply per-channel weight/bias.
  for (var i: u32 = tid; i < GROUP_SIZE; i = i + WG) {
    let local_c = i / HW;
    let c_idx = g * CPG + local_c;
    var v = (${W(`input[g_offset + i]`, o)} - mean) * invStd * ${W(`gWeight[c_idx]`, s)} + ${W(`gBias[c_idx]`, c)};
${
  a
    ? `    v = v / (1.0 + exp(-v));
`
    : ``
}    output[g_offset + i] = ${Jr(`v`, l)};
  }
}
`;
}
function si({
  C: e,
  H: t,
  W: n,
  groups: r,
  elementsPerWorkgroup: i = 1024,
  inputDtype: a = `float32`,
}) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  let o = (e / r) * t * n,
    s = Math.ceil(o / i),
    c = U(a);
  return `${qr(a)}@group(0) @binding(0) var<storage, read>       input: array<${c}>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;

const GROUP_SIZE: u32 = ${o}u;
const CHUNKS: u32 = ${s}u;
const ELEMENTS_PER_WG: u32 = ${i}u;
const WG: u32 = 256u;

var<workgroup> partialSum: array<f32, 256>;
var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let chunk = wg.x;
  let g = wg.y;
  let tid = lid.x;
  let group_start = g * GROUP_SIZE;
  let start = group_start + chunk * ELEMENTS_PER_WG;
  let end = min(start + ELEMENTS_PER_WG, group_start + GROUP_SIZE);

  var s = 0.0;
  var ss = 0.0;
  for (var i = start + tid; i < end; i = i + WG) {
    let v = ${W(`input[i]`, a)};
    s = s + v;
    ss = ss + v * v;
  }
  partialSum[tid] = s;
  partialSq[tid] = ss;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partialSum[tid] = partialSum[tid] + partialSum[tid + stride];
      partialSq[tid] = partialSq[tid] + partialSq[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    let out_idx = (g * CHUNKS + chunk) * 2u;
    partial[out_idx] = partialSum[0];
    partial[out_idx + 1u] = partialSq[0];
  }
}
`;
}
function ci({ C: e, H: t, W: n, groups: r, elementsPerWorkgroup: i = 4096 }) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  if (i % 4 != 0)
    throw Error(
      `vec4 groupnorm reduce requires elementsPerWorkgroup divisible by 4`,
    );
  let a = (e / r) * t * n;
  if (a % 4 != 0)
    throw Error(`vec4 groupnorm reduce requires group size divisible by 4`);
  let o = a / 4,
    s = i / 4;
  return `enable f16;
@group(0) @binding(0) var<storage, read>       input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;

const GROUP_SIZE: u32 = ${a}u;
const VEC_GROUP_SIZE: u32 = ${o}u;
const CHUNKS: u32 = ${Math.ceil(o / s)}u;
const VECS_PER_WG: u32 = ${s}u;
const WG: u32 = 256u;

var<workgroup> partialSum: array<f32, 256>;
var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let chunk = wg.x;
  let g = wg.y;
  let tid = lid.x;
  let group_start = g * VEC_GROUP_SIZE;
  let start = group_start + chunk * VECS_PER_WG;
  let end = min(start + VECS_PER_WG, group_start + VEC_GROUP_SIZE);

  var s = 0.0;
  var ss = 0.0;
  for (var i = start + tid; i < end; i = i + WG) {
    let v = vec4<f32>(input[i]);
    s = s + v.x + v.y + v.z + v.w;
    ss = ss + dot(v, v);
  }
  partialSum[tid] = s;
  partialSq[tid] = ss;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partialSum[tid] = partialSum[tid] + partialSum[tid + stride];
      partialSq[tid] = partialSq[tid] + partialSq[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    let out_idx = (g * CHUNKS + chunk) * 2u;
    partial[out_idx] = partialSum[0];
    partial[out_idx + 1u] = partialSq[0];
  }
}
`;
}
function li({
  C: e,
  H: t,
  W: n,
  groups: r,
  eps: i,
  elementsPerWorkgroup: a = 1024,
}) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  let o = (e / r) * t * n;
  return `@group(0) @binding(0) var<storage, read>       partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;

const GROUP_SIZE: u32 = ${o}u;
const CHUNKS: u32 = ${Math.ceil(o / a)}u;
const EPS: f32 = ${i};
const WG: u32 = 256u;

var<workgroup> partialSum: array<f32, 256>;
var<workgroup> partialSq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;
  let tid = lid.x;
  var s = 0.0;
  var ss = 0.0;
  for (var chunk = tid; chunk < CHUNKS; chunk = chunk + WG) {
    let idx = (g * CHUNKS + chunk) * 2u;
    s = s + partial[idx];
    ss = ss + partial[idx + 1u];
  }
  partialSum[tid] = s;
  partialSq[tid] = ss;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partialSum[tid] = partialSum[tid] + partialSum[tid + stride];
      partialSq[tid] = partialSq[tid] + partialSq[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  if (tid == 0u) {
    let mean = partialSum[0] / f32(GROUP_SIZE);
    let ex2 = partialSq[0] / f32(GROUP_SIZE);
    let variance = max(ex2 - mean * mean, 0.0);
    stats[g * 2u] = mean;
    stats[g * 2u + 1u] = inverseSqrt(variance + EPS);
  }
}
`;
}
function ui({
  C: e,
  H: t,
  W: n,
  groups: r,
  applySilu: i = !1,
  inputDtype: a = `float32`,
  weightDtype: o = `float32`,
  biasDtype: s = `float32`,
  outputDtype: c = `float32`,
}) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  let l = e / r;
  e * t * n;
  let u = U(a),
    d = U(o),
    f = U(s),
    p = U(c);
  return `${qr(a, o, s, c)}struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       input: array<${u}>;
@group(0) @binding(1) var<storage, read>       stats: array<f32>;
@group(0) @binding(2) var<storage, read>       gWeight: array<${d}>;
@group(0) @binding(3) var<storage, read>       gBias: array<${f}>;
@group(0) @binding(4) var<storage, read_write> output: array<${p}>;
@group(0) @binding(5) var<uniform>             params: Params;

const HW: u32 = ${t * n}u;
const CPG: u32 = ${l}u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= params.count) { return; }
  let c = i / HW;
  let g = c / CPG;
  let mean = stats[g * 2u];
  let invStd = stats[g * 2u + 1u];
  var v = (${W(`input[i]`, a)} - mean) * invStd * ${W(`gWeight[c]`, o)} + ${W(`gBias[c]`, s)};
${
  i
    ? `  v = v / (1.0 + exp(-v));
`
    : ``
}  output[i] = ${Jr(`v`, c)};
}
`;
}
function di({ C: e, H: t, W: n, groups: r, applySilu: i = !1 }) {
  if (e % r !== 0) throw Error(`C must be divisible by groups`);
  if ((t * n) % 4 != 0)
    throw Error(`vec4 groupnorm apply requires HW divisible by 4`);
  let a = e / r;
  return (
    (e * t * n) / 4,
    `enable f16;
struct Params { count4: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       stats: array<f32>;
@group(0) @binding(2) var<storage, read>       gWeight: array<f16>;
@group(0) @binding(3) var<storage, read>       gBias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(5) var<uniform>             params: Params;

const HW: u32 = ${t * n}u;
const CPG: u32 = ${a}u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i4 = wg_idx * 64u + lid.x;
  if (i4 >= params.count4) { return; }
  let elem = i4 * 4u;
  let c = elem / HW;
  let g = c / CPG;
  let mean = stats[g * 2u];
  let invStd = stats[g * 2u + 1u];
  var v = (vec4<f32>(input[i4]) - vec4<f32>(mean)) * vec4<f32>(invStd * f32(gWeight[c])) + vec4<f32>(f32(gBias[c]));
${
  i
    ? `  v = v / (vec4<f32>(1.0) + exp(-v));
`
    : ``
}  output[i4] = vec4<f16>(v);
}
`
  );
}
function fi({ C: e, H: t, W: n, dtype: r = `float32` }) {
  let i = U(r);
  return `${qr(r)}@group(0) @binding(0) var<storage, read>       input: array<${i}>;
@group(0) @binding(1) var<storage, read_write> output: array<${i}>;
const C: u32 = ${e}u;
const H: u32 = ${t}u;
const W: u32 = ${n}u;
const H2: u32 = ${t * 2}u;
const W2: u32 = ${n * 2}u;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  let x = gid.y;
  let c = gid.z;
  if (y >= H2 || x >= W2 || c >= C) { return; }
  output[c * H2 * W2 + y * W2 + x] = input[c * H * W + (y / 2u) * W + (x / 2u)];
}
`;
}
var pi = class e {
  constructor({ rt: e, config: t, w: n }) {
    ((this.rt = e), (this.config = t), (this.w = n));
  }
  destroy() {
    (Bt(this.w), (this.w = null));
  }
  static async fromBf16SafeTensors({
    rt: t,
    config: n,
    safeTensors: r,
    onProgress: i = null,
    concurrency: a,
    chunkMaxBytes: o,
    signal: s,
  }) {
    let c = !!t.caps().f16,
      l = t.caps().adapter?.vendor === `apple`,
      u = n.latent_channels ?? 32,
      d = Zt(),
      f = (e, n) => {
        d.tensor(e, async (e) => {
          let r = jt(e);
          n(t.tensorFromTypedArray(`float32`, [r.length], r));
        });
      },
      p = (e, t) => {
        r.has(e) ? f(e, t) : t(null);
      },
      m = (e, n) => {
        d.tensor(e, async (e) => {
          let r = mi(Mt(e));
          n(t.tensorFromTypedArray(`float16`, [r.length], r));
        });
      },
      h = (e, t) => {
        r.has(e) ? m(e, t) : t(null);
      },
      g = (e, { inC: n = 0, outC: r = 0, k: i = 0 } = {}) => {
        let a = {};
        return c
          ? (d.tensor(`${e}.weight`, async (e) => {
              let o = l && i === 3 && r % 4 == 0 && n >= 128 && r >= 128,
                s = l && i === 3 && r % 4 == 0 && n >= 128 && r >= 128,
                c = l && i === 1 && r % 4 == 0 && n >= 128 && r >= 64;
              if (o) {
                a.weightWinogradF2x2 = await vi({
                  rt: t,
                  weightF16: Mt(e),
                  inC: n,
                  outC: r,
                });
                return;
              }
              if (s) {
                let i = Pt(e, r, n * 9);
                a.weightPackedKOut = t.tensorFromTypedArray(
                  `float16`,
                  [i.length],
                  i,
                );
                return;
              }
              if (c) {
                let i = Pt(e, r, n);
                a.weightPackedKOut1x1 = t.tensorFromTypedArray(
                  `float16`,
                  [i.length],
                  i,
                );
                return;
              }
              let u = Mt(e);
              if (!s && !c) {
                let e = mi(u);
                a.weight = t.tensorFromTypedArray(`float16`, [e.length], e);
              }
              !s &&
                t.caps().subgroupMatrix &&
                i === 3 &&
                n % 32 == 0 &&
                r % 64 == 0 &&
                (a.weightTapIC = _i({ rt: t, weightF16: u, inC: n, outC: r }));
            }),
            h(`${e}.bias`, (e) => {
              a.bias = e;
            }),
            a)
          : (f(`${e}.weight`, (e) => {
              a.weight = e;
            }),
            p(`${e}.bias`, (e) => {
              a.bias = e;
            }),
            a);
      },
      _ = (e, n) => {
        let r = {};
        return c
          ? (d.tensor(`${e}.weight`, async (e) => {
              let i = Mt(e),
                a = l && n >= 128 && n % 4 == 0;
              if (!a) {
                let e = mi(i);
                r.weight = t.tensorFromTypedArray(`float16`, [e.length], e);
              }
              (a ||
                (r.upsampleFusedWeights = hi({
                  rt: t,
                  weightF16: i,
                  inC: n,
                  outC: n,
                })),
                (r.upsamplePackedKOutWeights = gi({
                  rt: t,
                  weightF16: i,
                  inC: n,
                  outC: n,
                })));
            }),
            h(`${e}.bias`, (e) => {
              r.bias = e;
            }),
            r)
          : (f(`${e}.weight`, (e) => {
              r.weight = e;
            }),
            p(`${e}.bias`, (e) => {
              r.bias = e;
            }),
            r);
      },
      v = (e) => {
        let t = {};
        return (
          c
            ? (m(`${e}.weight`, (e) => {
                t.weight = e;
              }),
              h(`${e}.bias`, (e) => {
                t.bias = e;
              }))
            : (f(`${e}.weight`, (e) => {
                t.weight = e;
              }),
              p(`${e}.bias`, (e) => {
                t.bias = e;
              })),
          t
        );
      },
      y = (e, t, n) => ({
        norm1: v(`${e}.norm1`),
        conv1: g(`${e}.conv1`, { inC: t, outC: n, k: 3 }),
        norm2: v(`${e}.norm2`),
        conv2: g(`${e}.conv2`, { inC: n, outC: n, k: 3 }),
        conv_shortcut: r.has(`${e}.conv_shortcut.weight`)
          ? g(`${e}.conv_shortcut`, { inC: t, outC: n, k: 1 })
          : null,
        inC: t,
        outC: n,
      }),
      b = (e, t) => ({
        group_norm: v(`${e}.group_norm`),
        to_q: g(`${e}.to_q`, { inC: t, outC: t, k: 1 }),
        to_k: g(`${e}.to_k`, { inC: t, outC: t, k: 1 }),
        to_v: g(`${e}.to_v`, { inC: t, outC: t, k: 1 }),
        to_out: g(`${e}.to_out.0`, { inC: t, outC: t, k: 1 }),
        channels: t,
      }),
      x = {},
      S = n.batch_norm_eps ?? 1e-4;
    ((x.bn = { eps: S }),
      d.tensor(`bn.running_mean`, async (e) => {
        let n = jt(e);
        ((x.bn.running_mean = n),
          (x.bn.running_meanT = t.tensorFromTypedArray(
            `float32`,
            [n.length],
            n,
          )),
          x.bn.running_var &&
            ((x.bn.running_std = Float32Array.from(x.bn.running_var, (e) =>
              Math.sqrt(e + S),
            )),
            (x.bn.running_stdT = t.tensorFromTypedArray(
              `float32`,
              [x.bn.running_std.length],
              x.bn.running_std,
            ))));
      }),
      d.tensor(`bn.running_var`, async (e) => {
        let n = jt(e);
        ((x.bn.running_var = n),
          x.bn.running_mean &&
            ((x.bn.running_std = Float32Array.from(n, (e) => Math.sqrt(e + S))),
            (x.bn.running_stdT = t.tensorFromTypedArray(
              `float32`,
              [x.bn.running_std.length],
              x.bn.running_std,
            ))));
      }),
      (x.post_quant_conv = g(`post_quant_conv`, { inC: u, outC: u, k: 1 })),
      (x.decoder = {
        conv_in: g(`decoder.conv_in`, { inC: u, outC: 512, k: 3 }),
        mid_block: {
          resnets: [
            y(`decoder.mid_block.resnets.0`, 512, 512),
            y(`decoder.mid_block.resnets.1`, 512, 512),
          ],
          attentions: [b(`decoder.mid_block.attentions.0`, 512)],
        },
        up_blocks: [],
        conv_norm_out: v(`decoder.conv_norm_out`),
        conv_out: g(`decoder.conv_out`),
      }));
    let C = (n.block_out_channels ?? [128, 256, 512, 512]).slice().reverse(),
      w = (n.layers_per_block ?? 2) + 1,
      T = C[0];
    for (let e = 0; e < C.length; ++e) {
      let t = C[e],
        n = [];
      for (let r = 0; r < w; ++r) {
        let i = r === 0 ? T : t;
        n.push(y(`decoder.up_blocks.${e}.resnets.${r}`, i, t));
      }
      let i = r.has(`decoder.up_blocks.${e}.upsamplers.0.conv.weight`)
        ? _(`decoder.up_blocks.${e}.upsamplers.0.conv`, t)
        : null;
      (x.decoder.up_blocks.push({ resnets: n, upsampler: i, inC: T, outC: t }),
        (T = t));
    }
    return (
      await r.streamAll(d.onChunk, {
        concurrency: a,
        chunkMaxBytes: o,
        names: d.names(),
        onProgress: i,
        signal: s,
      }),
      d.assertComplete(),
      new e({ rt: t, config: { ...n, blockOut: C, layersPerBlock: w }, w: x })
    );
  }
  async decode(e, t, n, { scope: r = null } = {}) {
    let i = !r,
      a = r ?? Vt(),
      o = this.rt;
    this.rt = Ht(o, a);
    try {
      let r = await this._decodeWithRuntime(e, t, n);
      return (i && a.keep(r.image), r);
    } finally {
      ((this.rt = o), i && a.destroy());
    }
  }
  async _decodeWithRuntime(e, t, n) {
    let r = this.rt,
      i = this.w,
      a = this.config.latent_channels ?? 32,
      o = r.caps().f16 ? `float16` : `float32`,
      s = await this._conv2d({
        inT: e,
        conv: i.post_quant_conv,
        inC: a,
        outC: a,
        H: t,
        W: n,
        k: 1,
        pad: 0,
        outputDtype: o,
      });
    s = await this._conv2d({
      inT: s,
      conv: i.decoder.conv_in,
      inC: a,
      outC: 512,
      H: t,
      W: n,
      k: 3,
      pad: 1,
      outputDtype: o,
    });
    let c = 512,
      l = t,
      u = n;
    ((s = await this._resnet({
      inT: s,
      rn: i.decoder.mid_block.resnets[0],
      inC: c,
      H: l,
      W: u,
    })),
      (s = await this._attentionMid({
        inT: s,
        attn: i.decoder.mid_block.attentions[0],
        C: c,
        H: l,
        W: u,
      })),
      (s = await this._resnet({
        inT: s,
        rn: i.decoder.mid_block.resnets[1],
        inC: c,
        H: l,
        W: u,
      })));
    for (let e of i.decoder.up_blocks) {
      for (let t of e.resnets)
        ((s = await this._resnet({ inT: s, rn: t, inC: t.inC, H: l, W: u })),
          (c = t.outC));
      if (e.upsampler)
        if (
          this._canFuseUpsampleConv({
            inT: s,
            conv: e.upsampler,
            inC: c,
            outC: c,
          })
        )
          ((s = await this._upsampleConv2d({
            inT: s,
            conv: e.upsampler,
            inC: c,
            outC: c,
            H: l,
            W: u,
            outputDtype: o,
          })),
            (l *= 2),
            (u *= 2));
        else {
          let t = s.dtype,
            n = r.empty(t, [c, l * 2, u * 2], `ups${l}`),
            i = fi({ C: c, H: l, W: u, dtype: t });
          (await r.runProgram({
            name: `upsample`,
            source: i,
            cacheKey: `ups_c${c}_h${l}_w${u}_${t}`,
            bindings: [
              { tensor: s, type: `read-only-storage` },
              { tensor: n, type: `storage` },
            ],
            workgroups: [Math.ceil((l * 2) / 8), Math.ceil((u * 2) / 8), c],
          }),
            (l *= 2),
            (u *= 2),
            (s = await this._conv2d({
              inT: n,
              conv: e.upsampler,
              inC: c,
              outC: c,
              H: l,
              W: u,
              k: 3,
              pad: 1,
              outputDtype: o,
            })));
        }
    }
    return (
      (s = await this._groupnorm({
        inT: s,
        gn: i.decoder.conv_norm_out,
        C: c,
        H: l,
        W: u,
        groups: 32,
        eps: 1e-6,
        applySilu: !0,
        outputDtype: o,
      })),
      {
        image: await this._conv2d({
          inT: s,
          conv: i.decoder.conv_out,
          inC: c,
          outC: 3,
          H: l,
          W: u,
          k: 3,
          pad: 1,
          outputDtype: o,
        }),
        H: l,
        W: u,
        channels: 3,
      }
    );
  }
  async _conv2d({
    inT: e,
    conv: t,
    inC: n,
    outC: r,
    H: i,
    W: a,
    k: o,
    pad: s,
    addT: c = null,
    outputDtype: l = `float32`,
  }) {
    let u = this.rt,
      d = r % 16 == 0 ? 16 : r % 8 == 0 ? 8 : r % 4 == 0 ? 4 : 1,
      f = e.dtype,
      p =
        o === 3 &&
        s === 1 &&
        f === `float16` &&
        l === `float16` &&
        t.weightWinogradF2x2 &&
        n % 8 == 0 &&
        r % 4 == 0,
      m =
        o === 3 &&
        s === 1 &&
        f === `float16` &&
        l === `float16` &&
        t.weightPackedKOut,
      h =
        o === 1 &&
        s === 0 &&
        f === `float16` &&
        l === `float16` &&
        t.weightPackedKOut1x1 &&
        n >= 128 &&
        r >= 64,
      g = p && u.caps().adapter?.vendor === `apple` && n >= 128 && r >= 128,
      _ = m && u.caps().adapter?.vendor === `apple` && n >= 128 && r >= 128,
      v = h && u.caps().adapter?.vendor === `apple`,
      y =
        !g &&
        !_ &&
        !v &&
        u.caps().subgroupMatrix &&
        u.caps().f16 &&
        ((o === 3 && s === 1) || (o === 1 && s === 0)) &&
        n % 32 == 0 &&
        r % 64 == 0,
      b = y && o === 3 && t.weightTapIC ? t.weightTapIC : t.weight,
      x = b === t.weightTapIC ? `tap-ic` : `ic-tap`,
      S = t.bias?.dtype ?? `float32`,
      C = c?.dtype ?? `float32`,
      w = y && o === 3 && x === `tap-ic` ? 64 : 32,
      T = w === 64 ? 64 : y && r % 128 == 0 ? 128 : 64,
      E = !y && g,
      D = !y && !E && m,
      O = !y && v,
      k = !D && !y && o === 3 && s === 1 && d >= 16,
      A = E
        ? t.weightWinogradF2x2
        : D
          ? t.weightPackedKOut
          : O
            ? t.weightPackedKOut1x1
            : b,
      j = _ || v ? 112 : 32,
      M = _ || v ? 14 : 4,
      N = _ || v ? 32 : 64,
      P = y
        ? `sgmat_tm${w}_tn${T}${x === `tap-ic` ? `_wti` : ``}`
        : E
          ? `wgf2_tm32_tn8_tk8`
          : D
            ? `pkn4_tm${j}_tn16_rpt${M}_tk${N}`
            : O
              ? `pkn4_1x1_tm${j}_tn16_rpt${M}_tk${N}`
              : k
                ? `sw`
                : `ot${d}`,
      F = !!(c && (y || E || D || O)),
      I = A.dtype,
      ee = `conv2d_${n}_${r}_${i}_${a}_${o}_${s}_${t.bias ? `b` : `nb`}_${f}_${A.dtype}_${S}_${l}_${P}${F ? `_add_${C}` : ``}`,
      L = y
        ? ii({
            inC: n,
            outC: r,
            H: i,
            W: a,
            hasBias: !!t.bias,
            hasAdd: F,
            inputDtype: f,
            weightDtype: I,
            biasDtype: S,
            outputDtype: l,
            residualDtype: C,
            tileM: w,
            tileN: T,
            k: o,
            pad: s,
            weightLayout: x,
          })
        : E
          ? $r({
              inC: n,
              outC: r,
              H: i,
              W: a,
              hasBias: !!t.bias,
              hasAdd: F,
              inputDtype: f,
              weightDtype: I,
              biasDtype: S,
              outputDtype: l,
              residualDtype: C,
              mTile: 32,
              nTile: 8,
              kTile: 8,
            })
          : D
            ? Qr({
                inC: n,
                outC: r,
                H: i,
                W: a,
                hasBias: !!t.bias,
                hasAdd: F,
                inputDtype: f,
                weightDtype: I,
                biasDtype: S,
                outputDtype: l,
                residualDtype: C,
                mTile: j,
                nTile: 16,
                rowPerThread: M,
                kTile: N,
              })
            : O
              ? ti({
                  inC: n,
                  outC: r,
                  H: i,
                  W: a,
                  hasBias: !!t.bias,
                  hasAdd: F,
                  inputDtype: f,
                  weightDtype: I,
                  biasDtype: S,
                  outputDtype: l,
                  residualDtype: C,
                  mTile: j,
                  nTile: 16,
                  rowPerThread: M,
                  kTile: N,
                })
              : k
                ? Zr({
                    inC: n,
                    outC: r,
                    H: i,
                    W: a,
                    hasBias: !!t.bias,
                    outTile: d,
                    inputDtype: f,
                    weightDtype: I,
                    biasDtype: S,
                    outputDtype: l,
                  })
                : d > 1
                  ? Xr({
                      inC: n,
                      outC: r,
                      H: i,
                      W: a,
                      kH: o,
                      kW: o,
                      pad: s,
                      hasBias: !!t.bias,
                      outTile: d,
                      inputDtype: f,
                      weightDtype: I,
                      biasDtype: S,
                      outputDtype: l,
                    })
                  : Yr({
                      inC: n,
                      outC: r,
                      H: i,
                      W: a,
                      kH: o,
                      kW: o,
                      pad: s,
                      hasBias: !!t.bias,
                      inputDtype: f,
                      weightDtype: I,
                      biasDtype: S,
                      outputDtype: l,
                    }),
      te = u.empty(l, [r, i, a], `conv-out-${ee}`),
      R = [
        { tensor: e, type: `read-only-storage` },
        { tensor: A, type: `read-only-storage` },
      ];
    return (
      t.bias && R.push({ tensor: t.bias, type: `read-only-storage` }),
      F && R.push({ tensor: c, type: `read-only-storage` }),
      R.push({ tensor: te, type: `storage` }),
      await u.runProgram({
        name: `conv2d`,
        source: L,
        cacheKey: ee,
        bindings: R,
        workgroups: y
          ? [Math.ceil(r / T), Math.ceil((i * a) / w), 1]
          : E
            ? [
                Math.ceil(r / 32),
                Math.ceil((Math.ceil(i / 2) * Math.ceil(a / 2)) / 32),
                1,
              ]
            : D || O
              ? [Math.ceil(r / 64), Math.ceil((i * a) / j), 1]
              : [Math.ceil(i / 8), Math.ceil(a / 8), Math.ceil(r / d)],
      }),
      c && !F && (await this._addInplace({ yT: te, xT: c, count: r * i * a })),
      te
    );
  }
  _canFuseUpsampleConv({ inT: e, conv: t, inC: n, outC: r }) {
    let i = this.rt;
    return (i.caps().f16 &&
      i.caps().adapter?.vendor === `apple` &&
      e.dtype === `float16` &&
      t.upsamplePackedKOutWeights &&
      n >= 128 &&
      r >= 128 &&
      r % 4 == 0) ||
      (i.caps().subgroupMatrix &&
        i.caps().f16 &&
        e.dtype === `float16` &&
        t.weight?.dtype === `float16` &&
        n % 32 == 0 &&
        r % 64 == 0)
      ? !0
      : i.caps().f16 &&
          e.dtype === `float16` &&
          t.weight?.dtype === `float16` &&
          !!t.upsampleFusedWeights &&
          r % 16 == 0;
  }
  async _upsampleConv2d({
    inT: e,
    conv: t,
    inC: n,
    outC: r,
    H: i,
    W: a,
    outputDtype: o = `float16`,
  }) {
    let s = this.rt,
      c = e.dtype,
      l = t.bias?.dtype ?? `float32`,
      u =
        s.caps().f16 &&
        c === `float16` &&
        t.upsamplePackedKOutWeights &&
        o === `float16` &&
        r % 4 == 0 &&
        n >= 128 &&
        r >= 128,
      d = u && s.caps().adapter?.vendor === `apple`,
      f =
        !d &&
        s.caps().subgroupMatrix &&
        s.caps().f16 &&
        c === `float16` &&
        t.weight?.dtype === `float16` &&
        n % 32 == 0 &&
        r % 64 == 0,
      p = !f && u,
      m = r % 128 == 0 ? 128 : 64,
      h = r % 16 == 0 ? 16 : r % 8 == 0 ? 8 : r % 4 == 0 ? 4 : 1,
      g = d ? 96 : 32,
      _ = d ? 12 : 4,
      v = d ? 32 : 64,
      y = s.empty(o, [r, i * 2, a * 2], `ups-conv-out`);
    for (let u = 0; u < 2; ++u)
      for (let d = 0; d < 2; ++d) {
        let b = u * 2 + d,
          x = p
            ? t.upsamplePackedKOutWeights[b]
            : (t.upsampleFusedWeights?.[b] ?? t.weight),
          S = x !== t.weight,
          C = f
            ? `sg_tn${m}`
            : p
              ? `pkn4_tm${g}_tn16_rpt${_}_tk${v}`
              : `sw_ot${h}`,
          w = `ups_conv2d_${n}_${r}_${i}_${a}_${t.bias ? `b` : `nb`}_${c}_${x.dtype}_${l}_${o}_${C}_${S ? `fw` : `ow`}_py${u}_px${d}`,
          T = f
            ? ai({
                inC: n,
                outC: r,
                H: i,
                W: a,
                hasBias: !!t.bias,
                inputDtype: c,
                weightDtype: x.dtype,
                biasDtype: l,
                outputDtype: o,
                tileN: m,
                phaseY: u,
                phaseX: d,
                fusedWeights: S,
              })
            : p
              ? ri({
                  inC: n,
                  outC: r,
                  H: i,
                  W: a,
                  hasBias: !!t.bias,
                  inputDtype: c,
                  weightDtype: x.dtype,
                  biasDtype: l,
                  outputDtype: o,
                  phaseY: u,
                  phaseX: d,
                  mTile: g,
                  nTile: 16,
                  rowPerThread: _,
                  kTile: v,
                })
              : ni({
                  inC: n,
                  outC: r,
                  H: i,
                  W: a,
                  hasBias: !!t.bias,
                  inputDtype: c,
                  weightDtype: x.dtype,
                  biasDtype: l,
                  outputDtype: o,
                  outTile: h,
                  phaseY: u,
                  phaseX: d,
                }),
          E = [
            { tensor: e, type: `read-only-storage` },
            { tensor: x, type: `read-only-storage` },
          ];
        (t.bias && E.push({ tensor: t.bias, type: `read-only-storage` }),
          E.push({ tensor: y, type: `storage` }),
          await s.runProgram({
            name: `upsample_conv2d`,
            source: T,
            cacheKey: w,
            bindings: E,
            workgroups: f
              ? [Math.ceil(r / m), Math.ceil((i * a) / 32), 1]
              : p
                ? [Math.ceil(r / 64), Math.ceil((i * a) / g), 1]
                : [Math.ceil(i / 8), Math.ceil(a / 8), Math.ceil(r / h)],
          }));
      }
    return y;
  }
  async _groupnorm({
    inT: e,
    gn: t,
    C: n,
    H: r,
    W: i,
    groups: a,
    eps: o,
    applySilu: s = !1,
    outputDtype: c = `float32`,
  }) {
    let l = this.rt,
      u = e.dtype,
      d = t.weight.dtype,
      f = t.bias?.dtype ?? `float32`,
      p = l.empty(c, [n, r, i], `gn-out`),
      m = (n / a) * r * i;
    if (m >= 131072) {
      let h = u === `float16` && m % 4 == 0,
        g = h ? 8192 : 1024,
        _ = Math.ceil(m / g),
        v = l.empty(`float32`, [a, _, 2], `gn-partial`),
        y = l.empty(`float32`, [a, 2], `gn-stats`);
      (await l.runProgram({
        name: `groupnorm_reduce`,
        source: h
          ? ci({ C: n, H: r, W: i, groups: a, elementsPerWorkgroup: g })
          : si({
              C: n,
              H: r,
              W: i,
              groups: a,
              elementsPerWorkgroup: g,
              inputDtype: u,
            }),
        cacheKey: `gn_reduce_${n}_${r}_${i}_${a}_${g}_${u}${h ? `_v4` : ``}`,
        bindings: [
          { tensor: e, type: `read-only-storage` },
          { tensor: v, type: `storage` },
        ],
        workgroups: [_, a, 1],
      }),
        await l.runProgram({
          name: `groupnorm_stats`,
          source: li({
            C: n,
            H: r,
            W: i,
            groups: a,
            eps: o,
            elementsPerWorkgroup: g,
          }),
          cacheKey: `gn_stats_${n}_${r}_${i}_${a}_${o}_${g}`,
          bindings: [
            { tensor: v, type: `read-only-storage` },
            { tensor: y, type: `storage` },
          ],
          workgroups: [a, 1, 1],
        }));
      let b = n * r * i,
        x =
          u === `float16` &&
          d === `float16` &&
          f === `float16` &&
          c === `float16` &&
          (r * i) % 4 == 0,
        S = x ? b / 4 : b,
        C = Math.ceil(S / 64),
        w = Math.min(C, 1024),
        T = Math.ceil(C / w),
        E = l.createUniformU32([S, w, 0, 0], `gn-apply-params`);
      return (
        await l.runProgram({
          name: s ? `groupnorm_silu_apply` : `groupnorm_apply`,
          source: x
            ? di({ C: n, H: r, W: i, groups: a, applySilu: s })
            : ui({
                C: n,
                H: r,
                W: i,
                groups: a,
                applySilu: s,
                inputDtype: u,
                weightDtype: d,
                biasDtype: f,
                outputDtype: c,
              }),
          cacheKey: `gn_apply_${n}_${r}_${i}_${a}_${s ? `silu` : `linear`}_${u}_${d}_${f}_${c}${x ? `_v4` : ``}`,
          bindings: [
            { tensor: e, type: `read-only-storage` },
            { tensor: y, type: `read-only-storage` },
            { tensor: t.weight, type: `read-only-storage` },
            { tensor: t.bias, type: `read-only-storage` },
            { tensor: p, type: `storage` },
            { buffer: E, type: `uniform` },
          ],
          workgroups: [w, T, 1],
        }),
        p
      );
    }
    let h = `gn_${n}_${r}_${i}_${a}_${o}_${s ? `silu` : `linear`}_${u}_${d}_${f}_${c}`,
      g = oi({
        C: n,
        H: r,
        W: i,
        groups: a,
        eps: o,
        applySilu: s,
        inputDtype: u,
        weightDtype: d,
        biasDtype: f,
        outputDtype: c,
      });
    return (
      await l.runProgram({
        name: s ? `groupnorm_silu` : `groupnorm`,
        source: g,
        cacheKey: h,
        bindings: [
          { tensor: e, type: `read-only-storage` },
          { tensor: t.weight, type: `read-only-storage` },
          { tensor: t.bias, type: `read-only-storage` },
          { tensor: p, type: `storage` },
        ],
        workgroups: [a, 1, 1],
      }),
      p
    );
  }
  async _addInplace({ yT: e, xT: t, count: n }) {
    let r = this.rt;
    if (e.dtype !== t.dtype)
      throw Error(`_addInplace dtype mismatch: y=${e.dtype} x=${t.dtype}`);
    let i = e.dtype,
      a = i === `float16` ? `f16` : `f32`,
      o =
        i === `float16`
          ? `enable f16;
`
          : ``,
      s = `img_add_2d_${i}`,
      c = `${o}struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> y: array<${a}>;
@group(0) @binding(1) var<storage, read>       x: array<${a}>;
@group(0) @binding(2) var<uniform>             p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * p.wgY;
  let i = wg_idx * 64u + lid.x;
  if (i >= p.count) { return; }
  y[i] = y[i] + x[i];
}
`,
      l = Math.ceil(n / 64),
      u = Math.min(l, 1024),
      d = Math.ceil(l / u),
      f = r.createUniformU32([n, u, 0, 0], `img-add-params`);
    await r.runProgram({
      name: `img_add`,
      source: c,
      cacheKey: s,
      bindings: [
        { tensor: e, type: `storage` },
        { tensor: t, type: `read-only-storage` },
        { buffer: f, type: `uniform` },
      ],
      workgroups: [u, d, 1],
    });
  }
  async _resnet({ inT: e, rn: t, inC: n, H: r, W: i }) {
    let a = this.rt.caps().f16 ? `float16` : `float32`,
      o = await this._groupnorm({
        inT: e,
        gn: t.norm1,
        C: n,
        H: r,
        W: i,
        groups: 32,
        eps: 1e-6,
        applySilu: !0,
        outputDtype: a,
      });
    ((o = await this._conv2d({
      inT: o,
      conv: t.conv1,
      inC: n,
      outC: t.outC,
      H: r,
      W: i,
      k: 3,
      pad: 1,
      outputDtype: a,
    })),
      (o = await this._groupnorm({
        inT: o,
        gn: t.norm2,
        C: t.outC,
        H: r,
        W: i,
        groups: 32,
        eps: 1e-6,
        applySilu: !0,
        outputDtype: a,
      })));
    let s = e;
    return (
      t.conv_shortcut &&
        (s = await this._conv2d({
          inT: e,
          conv: t.conv_shortcut,
          inC: n,
          outC: t.outC,
          H: r,
          W: i,
          k: 1,
          pad: 0,
          outputDtype: a,
        })),
      (o = await this._conv2d({
        inT: o,
        conv: t.conv2,
        inC: t.outC,
        outC: t.outC,
        H: r,
        W: i,
        k: 3,
        pad: 1,
        addT: s,
        outputDtype: a,
      })),
      o
    );
  }
  async _attentionMid({ inT: e, attn: t, C: n, H: r, W: i }) {
    let a = this.rt,
      o = r * i,
      s = await this._groupnorm({
        inT: e,
        gn: t.group_norm,
        C: n,
        H: r,
        W: i,
        groups: 32,
        eps: 1e-6,
        outputDtype: a.caps().f16 ? `float16` : `float32`,
      }),
      c = a.caps().f16 ? `float16` : `float32`,
      l = await this._conv2d({
        inT: s,
        conv: t.to_q,
        inC: n,
        outC: n,
        H: r,
        W: i,
        k: 1,
        pad: 0,
        outputDtype: c,
      }),
      u = await this._conv2d({
        inT: s,
        conv: t.to_k,
        inC: n,
        outC: n,
        H: r,
        W: i,
        k: 1,
        pad: 0,
        outputDtype: c,
      }),
      d = await this._conv2d({
        inT: s,
        conv: t.to_v,
        inC: n,
        outC: n,
        H: r,
        W: i,
        k: 1,
        pad: 0,
        outputDtype: c,
      }),
      f = await this._transposeChwToNc({ inT: l, C: n, H: r, W: i }),
      p = await this._transposeChwToNc({ inT: u, C: n, H: r, W: i }),
      m = await this._transposeChwToNc({ inT: d, C: n, H: r, W: i }),
      h = f,
      g = p,
      _ = m;
    a.caps().f16 &&
      f.dtype !== `float16` &&
      ((h = a.empty(`float16`, [o, 1, n], `vae-attn-q-f16`)),
      (g = a.empty(`float16`, [o, 1, n], `vae-attn-k-f16`)),
      (_ = a.empty(`float16`, [o, 1, n], `vae-attn-v-f16`)),
      await Bn(a, { xT: f, yT: h, count: o * n }),
      await Bn(a, { xT: p, yT: g, count: o * n }),
      await Bn(a, { xT: m, yT: _, count: o * n }));
    let v = a.empty(c, [o, 1, n], `vae-attn-out`);
    if (
      a.caps().f16 &&
      h.dtype === `float16` &&
      g.dtype === `float16` &&
      _.dtype === `float16` &&
      c === `float16` &&
      n === 512 &&
      o <= 4096 &&
      o % 64 == 0
    ) {
      let e = a.empty(`float16`, [o, o], `vae-attn-logits`),
        t = a.caps().subgroupMatrix
          ? g
          : await this._transposeNcToCn({ inT: g, rows: o, cols: n });
      (a.caps().subgroupMatrix
        ? await or(a, {
            aT: h,
            wT: t,
            outT: e,
            M: o,
            inFeatures: n,
            outFeatures: o,
          })
        : await ir(a, {
            aT: h,
            wT: t,
            outT: e,
            M: o,
            inFeatures: n,
            outFeatures: o,
          }),
        await Hn(a, { xT: e, rows: o, cols: o, scale: 1 / Math.sqrt(n) }));
      let r = await this._transposeNcToCn({ inT: _, rows: o, cols: n });
      a.caps().subgroupMatrix
        ? await or(a, {
            aT: e,
            wT: r,
            outT: v,
            M: o,
            inFeatures: o,
            outFeatures: n,
          })
        : await ir(a, {
            aT: e,
            wT: r,
            outT: v,
            M: o,
            inFeatures: o,
            outFeatures: n,
          });
    } else
      await Wn(a, {
        qT: h,
        kT: g,
        vT: _,
        outT: v,
        seq: o,
        qHeads: 1,
        kvHeads: 1,
        headDim: n,
        causal: !1,
      });
    let y = await this._transposeNcToChw({ inT: v, C: n, H: r, W: i });
    return await this._conv2d({
      inT: y,
      conv: t.to_out,
      inC: n,
      outC: n,
      H: r,
      W: i,
      k: 1,
      pad: 0,
      addT: e,
      outputDtype: c,
    });
  }
  async _transposeChwToNc({ inT: e, C: t, H: n, W: r }) {
    let i = this.rt,
      a = n * r,
      o = e.dtype,
      s = o === `float16` ? `f16` : `f32`,
      c =
        o === `float16`
          ? `enable f16;
`
          : ``,
      l = `t_chw_nc_${t}_${n}_${r}_${o}`,
      u = `${c}@group(0) @binding(0) var<storage, read>       x: array<${s}>;
@group(0) @binding(1) var<storage, read_write> y: array<${s}>;
const C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const N: u32 = ${a}u;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x; let w = gid.y; let c = gid.z;
  if (h >= H || w >= W || c >= C) { return; }
  let n_idx = h * W + w;
  y[n_idx * C + c] = x[c * N + n_idx];
}
`,
      d = i.empty(o, [a, t], `t-chw-nc`);
    return (
      await i.runProgram({
        name: `t_chw_nc`,
        source: u,
        cacheKey: l,
        bindings: [
          { tensor: e, type: `read-only-storage` },
          { tensor: d, type: `storage` },
        ],
        workgroups: [Math.ceil(n / 8), Math.ceil(r / 8), t],
      }),
      d
    );
  }
  async _transposeNcToCn({ inT: e, rows: t, cols: n }) {
    let r = this.rt,
      i = e.dtype,
      a = i === `float16` ? `f16` : `f32`,
      o =
        i === `float16`
          ? `enable f16;
`
          : ``,
      s = `t_nc_cn_${t}_${n}_${i}`,
      c = `${o}@group(0) @binding(0) var<storage, read>       x: array<${a}>;
@group(0) @binding(1) var<storage, read_write> y: array<${a}>;
const ROWS: u32 = ${t}u;
const COLS: u32 = ${n}u;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let r = gid.y;
  if (r >= ROWS || c >= COLS) { return; }
  y[c * ROWS + r] = x[r * COLS + c];
}
`,
      l = r.empty(i, [n, t], `vae-attn-v-cn`);
    return (
      await r.runProgram({
        name: `t_nc_cn`,
        source: c,
        cacheKey: s,
        bindings: [
          { tensor: e, type: `read-only-storage` },
          { tensor: l, type: `storage` },
        ],
        workgroups: [Math.ceil(n / 16), Math.ceil(t / 16), 1],
      }),
      l
    );
  }
  async _transposeNcToChw({ inT: e, C: t, H: n, W: r }) {
    let i = this.rt,
      a = n * r,
      o = e.dtype,
      s = o === `float16` ? `f16` : `f32`,
      c =
        o === `float16`
          ? `enable f16;
`
          : ``,
      l = `t_nc_chw_${t}_${n}_${r}_${o}`,
      u = `${c}@group(0) @binding(0) var<storage, read>       x: array<${s}>;
@group(0) @binding(1) var<storage, read_write> y: array<${s}>;
const C: u32 = ${t}u;
const H: u32 = ${n}u;
const W: u32 = ${r}u;
const N: u32 = ${a}u;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x; let w = gid.y; let c = gid.z;
  if (h >= H || w >= W || c >= C) { return; }
  let n_idx = h * W + w;
  y[c * N + n_idx] = x[n_idx * C + c];
}
`,
      d = i.empty(o, [t, n, r], `t-nc-chw`);
    return (
      await i.runProgram({
        name: `t_nc_chw`,
        source: u,
        cacheKey: l,
        bindings: [
          { tensor: e, type: `read-only-storage` },
          { tensor: d, type: `storage` },
        ],
        workgroups: [Math.ceil(n / 8), Math.ceil(r / 8), t],
      }),
      d
    );
  }
};
function mi(e) {
  if (e.byteLength % 4 == 0) return e;
  let t = new Uint16Array(e.length + 1);
  return (t.set(e), t);
}
function hi({ rt: e, weightF16: t, inC: n, outC: r }) {
  let i = bi(),
    a = [];
  for (let o = 0; o < 2; ++o) {
    let s = o === 0 ? [[0], [1, 2]] : [[0, 1], [2]];
    for (let o = 0; o < 2; ++o) {
      let c = o === 0 ? [[0], [1, 2]] : [[0, 1], [2]],
        l = new Uint16Array(r * n * 4);
      for (let e = 0; e < r; ++e)
        for (let r = 0; r < n; ++r)
          for (let a = 0; a < 2; ++a)
            for (let o = 0; o < 2; ++o) {
              let u = 0;
              for (let l of s[a])
                for (let a of c[o]) u += i[t[e * n * 9 + r * 9 + l * 3 + a]];
              l[e * n * 4 + r * 4 + a * 2 + o] = It(u);
            }
      let u = mi(l);
      a.push(e.tensorFromTypedArray(`float16`, [u.length], u));
    }
  }
  return a;
}
function gi({ rt: e, weightF16: t, inC: n, outC: r }) {
  let i = bi(),
    a = [];
  for (let o = 0; o < 2; ++o) {
    let s = o === 0 ? [[0], [1, 2]] : [[0, 1], [2]];
    for (let o = 0; o < 2; ++o) {
      let c = o === 0 ? [[0], [1, 2]] : [[0, 1], [2]],
        l = new Uint16Array(n * 4 * r);
      for (let e = 0; e < n; ++e)
        for (let a = 0; a < 2; ++a)
          for (let o = 0; o < 2; ++o) {
            let u = a * 2 + o,
              d = (e * 4 + u) * r;
            for (let u = 0; u < r; ++u) {
              let r = 0;
              for (let l of s[a])
                for (let a of c[o]) r += i[t[u * n * 9 + e * 9 + l * 3 + a]];
              l[d + u] = It(r);
            }
          }
      let u = mi(l);
      a.push(e.tensorFromTypedArray(`float16`, [u.length], u));
    }
  }
  return a;
}
function _i({ rt: e, weightF16: t, inC: n, outC: r }) {
  let i = new Uint16Array(r * 9 * n);
  for (let e = 0; e < r; ++e) {
    let r = e * n * 9;
    for (let e = 0; e < 9; ++e) {
      let a = r + e * n;
      for (let o = 0; o < n; ++o) i[a + o] = t[r + o * 9 + e];
    }
  }
  return e.tensorFromTypedArray(`float16`, [i.length], i);
}
async function vi({ rt: e, weightF16: t, inC: n, outC: r }) {
  let i = e.tensorFromTypedArray(`float16`, [t.length], t),
    a = e.empty(`float16`, [n * 16, r], `conv-wino-f2x2-weight`);
  return (
    await e.runProgram({
      name: `conv2d_winograd_weight_transform`,
      source: ei({ inC: n, outC: r }),
      cacheKey: `conv2d_winograd_weight_transform_${n}_${r}`,
      bindings: [
        { tensor: i, type: `read-only-storage` },
        { tensor: a, type: `storage` },
      ],
      workgroups: [Math.ceil(r / 16), Math.ceil(n / 16), 1],
    }),
    e.clearBindGroupCache?.(),
    e.host.device.queue
      .onSubmittedWorkDone()
      .then(() => i.destroy?.())
      .catch(() => {}),
    a
  );
}
var yi = null;
function bi() {
  if (yi) return yi;
  let e = new Float32Array(65536);
  for (let t = 0; t < e.length; ++t) e[t] = Ft(t);
  return ((yi = e), e);
}
function xi(e) {
  let t = new Float32Array(e * 4);
  for (let n = 0; n < e; ++n) t[n * 4 + 3] = n;
  return t;
}
function Si(e, t) {
  let n = new Float32Array(e * t * 4);
  for (let r = 0; r < e; ++r)
    for (let e = 0; e < t; ++e) {
      let i = (r * t + e) * 4;
      ((n[i + 1] = r), (n[i + 2] = e));
    }
  return n;
}
var PipelineBase = class {
    constructor({
      rt: e,
      snapshotDir: t,
      tokenizer: n,
      textEncoder: r,
      transformer: i,
      vae: a,
      vaeConfig: o,
      schedulerConfig: s,
      bnStats: c,
    }) {
      ((this.rt = e),
        (this.snapshotDir = t),
        (this.tokenizer = n),
        (this.textEncoder = r),
        (this.transformer = i),
        (this.vae = a),
        (this.vaeConfig = o ?? a?.config ?? null),
        (this.schedulerConfig = s),
        (this.bnStats = c),
        (this.destroyed = !1));
    }
    async ensureVae() {
      if (this.vae && this.bnStats) return this.vae;
      throw Error(
        `VAE was not loaded; construct the pipeline without skipVae to decode images`,
      );
    }
    async generate(e) {
      let t = Vt();
      try {
        return await this._generate(e, t);
      } finally {
        (this.rt.clearTransientCaches?.(),
          this.rt.clearReadbackPool?.(),
          t.destroy(),
          this.rt.clearTransientCaches?.());
      }
    }
    async _generate(
      {
        prompt: e,
        height: t = 1024,
        width: n = 1024,
        numInferenceSteps: r = 4,
        seed: i = 0,
        log: a = null,
        callbackOnStepEnd: o = null,
        encoderHiddenStatesT: s = null,
        signal: c_signal = null,
      },
      c = null,
    ) {
      if (t % 16 != 0 || n % 16 != 0)
        throw Error(`height and width must be divisible by 16`);
      let l = this.rt,
        u = this.schedulerConfig,
        d = this.transformer.config;
      d.num_attention_heads * d.attention_head_dim;
      let f = s,
        p,
        m;
      if (f) {
        if (f.runtime !== l)
          throw Error(`encoderHiddenStatesT belongs to a different runtime`);
        if (f.shape.length !== 2)
          throw Error(`encoderHiddenStatesT must have shape [seq, stackDim]`);
        (([p, m] = f.shape), a?.(`text encode cache`));
      } else
        ({
          hiddenStackT: f,
          seq: p,
          stackDim: m,
        } = await this.encodePrompt(e, { log: a, scope: c }));
      if (m !== d.joint_attention_dim)
        throw Error(
          `text stackDim ${m} != joint_attention_dim ${d.joint_attention_dim}`,
        );
      a?.(`scheduler`);
      let h = dr((t / 16) * (n / 16), r),
        g = new pr(u);
      g.setTimesteps({ numInferenceSteps: r, mu: h });
      let _ = t / 8,
        v = n / 8,
        y = this.vaeConfig?.latent_channels ?? 32,
        b = _ / 2,
        x = v / 2,
        S = y * 4,
        C = Ii(Fi(i, S * b * x), S, b, x),
        w =
          c?.track(l.tensorFromTypedArray(`float32`, [b * x, S], C)) ??
          l.tensorFromTypedArray(`float32`, [b * x, S], C),
        T = Si(b, x),
        E = xi(p);
      for (let e = 0; e < r; ++e) {
        if (c_signal && c_signal.aborted) {
          throw new DOMException("The operation was aborted.", "AbortError");
        }
        let t = Vt(),
          n = Ht(l, t);
        try {
          let i = g.timesteps[e] / 1e3;
          a?.(`step ${e}/${r} t=${i.toFixed(4)}`);
          let s = await this.transformer.forward({
              hiddenStatesT: w,
              encoderHiddenStatesT: f,
              timestep: i,
              imgIds: T,
              txtIds: E,
              scope: t,
            }),
            c = g.stepDelta(e);
          (await Ln(n, { xT: w, yT: s, count: b * x * 128, alpha: c }),
            o && (await o(this, e, g.timesteps[e], { latents: w })));
        } finally {
          (l.clearBindGroupCache?.(), t.destroy());
        }
      }
      (await this.ensureVae(), a?.(`unpack + BN-denorm`));
      let D = await Rn(c ? Ht(l, c) : l, {
        packedT: w,
        meanT: this.bnStats.running_meanT,
        stdT: this.bnStats.running_stdT,
        outputDtype: l.caps().f16 ? `float16` : `float32`,
        latentC: y,
        latentH: _,
        latentW: v,
      });
      a?.(`vae decode`);
      let {
          image: O,
          H: k,
          W: A,
        } = await this.vae.decode(D, _, v, { scope: c }),
        j = await Ei(l, O);
      a?.(`to RGB`);
      let M = new Uint8Array(k * A * 3);
      for (let e = 0; e < 3; e++)
        for (let t = 0; t < k; t++)
          for (let n = 0; n < A; n++) {
            let r = (j[e * k * A + t * A + n] + 1) * 127.5;
            M[(t * A + n) * 3 + e] = Math.min(255, Math.max(0, Math.round(r)));
          }
      return (a?.(`png encode`), br(A, k, M));
    }
    destroy() {
      this.destroyed ||
        ((this.destroyed = !0),
        this.rt.clearTransientCaches?.(),
        this.textEncoder?.destroy?.(),
        this.transformer?.destroy?.(),
        this.vae?.destroy?.(),
        (this.tokenizer = null),
        (this.textEncoder = null),
        (this.transformer = null),
        (this.vae = null),
        (this.bnStats = null),
        this.rt.clearTransientCaches?.());
    }
    async encodePrompt(e, { log: t = null, scope: n = null } = {}) {
      if (!this.tokenizer || !this.textEncoder)
        throw Error(
          `Text encoder was not loaded; provide encoderHiddenStatesT to generate()`,
        );
      t?.(`tokenize`);
      let r = Pi(e),
        i = (await this.tokenizer.encode(r)).ids.slice(0, 512),
        a = new Uint32Array(Math.max(1, i.length));
      for (let e = 0; e < i.length; ++e) a[e] = i[e];
      return (
        t?.(`tokens: ${i.length}`),
        t?.(`text encode`),
        this.textEncoder.encode(a, { scope: n })
      );
    }
  },
  HttpPipeline = class e extends PipelineBase {
    static async fromSnapshot(t, n, r = {}) {
      return loadPipelineFromSnapshot(e, t, n, r, {
        readJsonResource: Di,
        readJsonResourceOptional: Oi,
        openSafeTensorsResource: ki,
      });
    }
  };
async function loadPipelineFromSnapshot(
  e,
  t,
  n,
  {
    onProgress: r = null,
    fetch: i = null,
    cacheStorage: a = null,
    cacheName: o = null,
    cache: s = void 0,
    force: c = !1,
    signal: l = null,
    skipTextEncoder: u = !1,
    skipVae: d = !1,
    requireRangeRequests: f = !0,
  } = {},
  p,
) {
  let m = {
      fetch: i,
      cacheStorage: a,
      cacheName: o,
      cache: s,
      force: c,
      signal: l,
      requireRangeRequests: f,
    },
    h = null,
    g = null,
    _ = null,
    v = (e) => {
      r && r({ component: e, phase: `open` });
    },
    y = (e) => (t) => {
      r && r({ component: e, phase: `download`, ...t });
    };
  try {
    r && r({ phase: `init` });
    let i = p.readJsonResourceOptional
      ? await p.readJsonResourceOptional(n, `scheduler/scheduler_config.json`, m)
      : null;
    i = i ?? {
      base_image_seq_len: 256,
      base_shift: 0.5,
      max_image_seq_len: 4096,
      max_shift: 1.15,
      num_train_timesteps: 1e3,
      shift: 3,
      use_dynamic_shifting: !0,
    };
    let a = new At(
      await p.readJsonResource(n, `tokenizer/tokenizer.json`, m),
      await p.readJsonResource(n, `tokenizer/tokenizer_config.json`, m),
    );
    if (!u) {
      v(`text_encoder`);
      let e = await p.readJsonResource(
          n,
          `text_encoder-mlx-4bit/config.json`,
          m,
        ),
        r = await p.openSafeTensorsResource(
          n,
          `text_encoder-mlx-4bit/model.safetensors`,
          m,
        );
      try {
        h = await cr.fromMlxSafeTensors({
          rt: t,
          config: e,
          safeTensors: r,
          onProgress: y(`text_encoder`),
          signal: l,
        });
      } finally {
        await r.close();
      }
    }
    v(`transformer`);
    let o = await p.readJsonResource(
        n,
        `transformer-packed-mflux/config.json`,
        m,
      ),
      s = p.readJsonResourceOptional
        ? await p.readJsonResourceOptional(
            n,
            `transformer-packed-mflux/quantization_config.json`,
            m,
          )
        : null,
      c = s ? { ...o, quantization_config: s } : o,
      f = await p.openSafeTensorsResource(
        n,
        `transformer-packed-mflux/diffusion_pytorch_model.safetensors`,
        m,
      );
    try {
      g = await Nr.fromMlxSafeTensors({
        rt: t,
        config: c,
        safeTensors: f,
        onProgress: y(`transformer`),
        signal: l,
      });
    } finally {
      await f.close();
    }
    let b = await p.readJsonResource(n, `vae/config.json`, m),
      x = null;
    if (!d) {
      v(`vae`);
      let e = await p.openSafeTensorsResource(
        n,
        `vae/diffusion_pytorch_model.safetensors`,
        m,
      );
      try {
        ((_ = await pi.fromBf16SafeTensors({
          rt: t,
          config: b,
          safeTensors: e,
          onProgress: y(`vae`),
          signal: l,
        })),
          (x = _.w.bn));
      } finally {
        await e.close();
      }
    }
    return new e({
      rt: t,
      snapshotDir: n,
      tokenizer: a,
      textEncoder: h,
      transformer: g,
      vae: _,
      vaeConfig: b,
      schedulerConfig: i,
      bnStats: x,
    });
  } catch (e) {
    throw (
      h?.destroy?.(),
      g?.destroy?.(),
      _?.destroy?.(),
      t.clearTransientCaches?.(),
      e
    );
  }
}
async function Ei(e, t) {
  let n = await e.readTensor(t);
  return t.dtype === `float16` ? Float32Array.from(n, Ft) : n;
}
async function Di(e, t, n) {
  return JSON.parse(await ji(Ai(e, t), n));
}
async function Oi(e, t, n) {
  try {
    return await Di(e, t, n);
  } catch {
    return null;
  }
}
async function ki(e, t, n = {}) {
  return A(Ai(e, t), n);
}
function Ai(e, t) {
  let n = e instanceof URL ? e.toString() : String(e);
  return new URL(t, Ni(n), globalThis.location?.href).toString();
}
async function ji(e, t = {}) {
  let n = t.fetch ?? globalThis.fetch;
  if (typeof n != `function`) throw Error(`No fetch implementation available`);
  let r = await Mi(t);
  if (r && !t.force) {
    let t = await r.match(e);
    if (t) return t.text();
  }
  let i = await n(e, { signal: t.signal });
  if (!i.ok) throw Error(`GET ${e} failed: ${i.status} ${i.statusText}`);
  if (r)
    try {
      await r.put(e, i.clone());
    } catch (e) {
      typeof console < `u` &&
        console.warn(`resource cache write failed: ${e.message}`);
    }
  return i.text();
}
async function Mi(e = {}) {
  if (e.cache === !1) return null;
  let t = e.cacheStorage ?? globalThis.caches;
  return t?.open ? t.open(e.cacheName ?? `bonsai-pipeline-v1`) : null;
}
function Ni(e) {
  return e.endsWith(`/`) ? e : `${e}/`;
}
function Pi(e) {
  return `<|im_start|>user
${e}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;
}
function Fi(e, t) {
  let n = (e >>> 0) ^ 3735928559;
  function r() {
    n = (n + 1831565813) >>> 0;
    let e = n;
    return (
      (e = Math.imul(e ^ (e >>> 15), e | 1)),
      (e ^= e + Math.imul(e ^ (e >>> 7), e | 61)),
      (e ^ (e >>> 14)) >>> 0
    );
  }
  function i() {
    return (r() + 1) / 4294967297;
  }
  let a = new Float32Array(t);
  for (let e = 0; e < t; e += 2) {
    let n = i(),
      r = i(),
      o = Math.sqrt(-2 * Math.log(n)),
      s = 2 * Math.PI * r;
    ((a[e] = o * Math.cos(s)), e + 1 < t && (a[e + 1] = o * Math.sin(s)));
  }
  return a;
}
function Ii(e, t, n, r) {
  let i = n * r,
    a = new Float32Array(i * t);
  for (let n = 0; n < t; ++n)
    for (let r = 0; r < i; ++r) a[r * t + n] = e[n * i + r];
  return a;
}
var HF_BASE_URL = `https://huggingface.co`,
  DEFAULT_MODEL_ID = `prism-ml/bonsai-image-ternary-4B-mlx-2bit`,
  DEFAULT_CACHE_NAME = `bonsai-image-v1`,
  BonsaiImagePipeline = class e {
    constructor({ runtime: e, pipeline: t, modelRoot: n, ownsRuntime: r, modelName: s }) {
      ((this.runtime = e),
        (this.gpuPipeline = t),
        (this.modelRoot = n),
        (this.ownsRuntime = r),
        (this.modelName = s),
        (this.destroyed = !1));
    }
    getModelName() {
      return this.modelName;
    }
    static async clearCache(cacheName = null) {
      let idbName = cacheName ?? DEFAULT_CACHE_NAME;
      if (I.has(idbName)) {
        try {
          let db = await I.get(idbName);
          db.close();
        } catch (e) {}
        I.delete(idbName);
      }
      if (typeof indexedDB !== "undefined") {
        await new Promise((resolve, reject) => {
          let req = indexedDB.deleteDatabase(idbName);
          req.onsuccess = () => resolve();
          req.onerror = () => reject(req.error);
          req.onblocked = () => {
            console.warn(`indexedDB.deleteDatabase(${idbName}) was blocked. It will be deleted once all connections are closed.`);
            resolve();
          };
        });
      }
      if (typeof globalThis.caches !== "undefined") {
        try {
          await globalThis.caches.delete(cacheName ?? `bonsai-pipeline-v1`);
        } catch (e) {
          console.warn(`Cache API deletion failed:`, e);
        }
      }
    }
    static async from_pretrained(t = null, n = {}) {
      let r = resolveModelUrl(t, n),
        i = n.runtime ?? n.rt ?? (await a(n.runtimeOptions ?? {})),
        o = !(n.runtime ?? n.rt),
        s,
        c = t ?? n.repoId ?? DEFAULT_MODEL_ID;
      try {
        s = await HttpPipeline.fromSnapshot(i, r, {
          onProgress: n.onProgress,
          fetch: n.fetch,
          cacheStorage: n.cacheStorage,
          cacheName: n.cacheName ?? DEFAULT_CACHE_NAME,
          cache: n.cache,
          force: n.force,
          signal: n.signal,
          requireRangeRequests:
            n.requireRangeRequests ?? n.require_range_requests ?? !0,
        });
      } catch (e) {
        if (o)
          try {
            await i.destroy();
          } catch {}
        throw e;
      }
      return new e({ runtime: i, pipeline: s, modelRoot: r, ownsRuntime: o, modelName: c });
    }
    async generate(e = {}) {
      let t = validateGenerateOptions(e);
      return new BonsaiImageResult({
        bytes: await this.gpuPipeline.generate(t),
        width: t.width,
        height: t.height,
        prompt: t.prompt,
        seed: t.seed,
      });
    }
    async destroy() {
      if (this.destroyed) return;
      this.destroyed = !0;
      let e = this.gpuPipeline;
      ((this.gpuPipeline = null),
        e?.destroy?.(),
        this.ownsRuntime && (await this.runtime.destroy()));
    }
  },
  BonsaiImageResult = class {
    constructor({
      bytes: e,
      width: t,
      height: n,
      prompt: r = ``,
      seed: i = 0,
    }) {
      ((this.bytes = e),
        (this.width = t),
        (this.height = n),
        (this.prompt = r),
        (this.seed = i));
    }
    toBlob() {
      return new Blob([this.bytes], { type: `image/png` });
    }
  };
function validateGenerateOptions(e) {
  let t = e.prompt;
  if (typeof t != `string` || t.length === 0)
    throw Error(`Flux2KleinPipeline requires a non-empty prompt string`);
  if ((e.guidanceScale ?? e.guidance_scale ?? 1) !== 1)
    throw Error(
      `Flux2-Klein Bonsai currently supports guidance_scale/guidanceScale = 1.0 only`,
    );
  return {
    prompt: t,
    height: Number(e.height ?? 1024),
    width: Number(e.width ?? 1024),
    numInferenceSteps: Number(
      e.numInferenceSteps ?? e.num_inference_steps ?? 4,
    ),
    seed: Number(e.seed ?? 0),
    log: e.log ?? null,
    callbackOnStepEnd: e.callbackOnStepEnd ?? e.callback_on_step_end ?? null,
    signal: e.signal ?? null,
  };
}
function resolveModelUrl(e, t) {
  let n = e ?? t.repoId ?? DEFAULT_MODEL_ID;
  return isHttpUrl(n) || n.startsWith(`/`) || n.startsWith(`.`)
    ? n
    : `${HF_BASE_URL}/${n}/resolve/${t.revision ?? `main`}`;
}
function isHttpUrl(e) {
  return typeof e == `string` && /^https?:/i.test(e);
}

// ======================================================================
// Public API Extensions
// ======================================================================

/**
 * WebGPU がサポートされているかチェックする
 * @returns {Promise<boolean>}
 */
BonsaiImagePipeline.isSupported = async function () {
  if (!globalThis.navigator?.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
};

/**
 * Data URL を返す（便利メソッド）
 * @returns {Promise<string>}
 */
BonsaiImageResult.prototype.toDataURL = async function () {
  const blob = this.toBlob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

/**
 * ImageBitmap を返す（Canvas描画用）
 * @returns {Promise<ImageBitmap>}
 */
BonsaiImageResult.prototype.toImageBitmap = async function () {
  return createImageBitmap(this.toBlob());
};

// ======================================================================
// ESM Exports
// ======================================================================
export { BonsaiImagePipeline, BonsaiImageResult };