/**
 * Source of the Web Worker that hosts the Bergamot (Marian) WASM runtime.
 *
 * Kept as a string so the SDK needs no bundler configuration: the main thread turns it into a
 * Blob URL. The worker pulls the emscripten glue and the `.wasm` from `runtimePath` (jsDelivr by
 * default, or a self-hosted copy), receives model files as ArrayBuffers and answers RPC calls.
 *
 * Adapted from `translator-worker.js` in bergamot-translator (MPL-2.0).
 */
export const WORKER_SOURCE = String.raw`
'use strict'
var Module = {}
self.Module = Module

var GEMM_FALLBACKS = {
  int8_prepare_a: 'int8PrepareAFallback',
  int8_prepare_b: 'int8PrepareBFallback',
  int8_prepare_b_from_transposed: 'int8PrepareBFromTransposedFallback',
  int8_prepare_b_from_quantized_transposed: 'int8PrepareBFromQuantizedTransposedFallback',
  int8_prepare_bias: 'int8PrepareBiasFallback',
  int8_multiply_and_add_bias: 'int8MultiplyAndAddBiasFallback',
  int8_select_columns_of_b: 'int8SelectColumnsOfBFallback',
}

function fallbackGemm() {
  var out = {}
  Object.keys(GEMM_FALLBACKS).forEach(function (name) {
    var target = GEMM_FALLBACKS[name]
    out[name] = function () {
      return Module.asm[target].apply(null, arguments)
    }
  })
  return out
}

var service = null
var models = new Map()

async function init(args) {
  var runtime = args.runtimePath
  var response = await fetch(runtime + 'bergamot-translator-worker.wasm', { credentials: 'omit' })
  if (!response.ok) throw new Error('Bergamot runtime: HTTP ' + response.status + ' for ' + runtime)

  await new Promise(function (resolve, reject) {
    Module.instantiateWasm = function (info, accept) {
      var imports = Object.assign({}, info, { wasm_gemm: fallbackGemm() })
      ;(async function () {
        try {
          return (await WebAssembly.instantiateStreaming(response.clone(), imports)).instance
        } catch (_) {
          return (await WebAssembly.instantiate(await response.arrayBuffer(), imports)).instance
        }
      })()
        .then(accept)
        .catch(reject)
      return {}
    }
    Module.onRuntimeInitialized = function () {
      resolve()
    }
    Module.onAbort = function (what) {
      reject(new Error('Bergamot runtime aborted: ' + what))
    }
    try {
      importScripts(runtime + 'bergamot-translator-worker.js')
    } catch (err) {
      reject(err)
    }
  })

  service = new Module.BlockingService({ cacheSize: 0 })
  return true
}

function alignedMemory(buffer, alignment) {
  var bytes = new Int8Array(buffer)
  var memory = new Module.AlignedMemory(bytes.byteLength, alignment)
  memory.getByteArrayView().set(bytes)
  return memory
}

function yaml(config) {
  return Object.keys(config)
    .map(function (key) {
      return key + ': ' + config[key]
    })
    .join('\n') + '\n'
}

function loadModel(args) {
  var key = args.key
  if (models.has(key)) return true

  var vocabs = new Module.AlignedMemoryList()
  args.vocabs.forEach(function (buffer) {
    vocabs.push_back(alignedMemory(buffer, 64))
  })

  var config = {
    'beam-size': 1,
    normalize: 1.0,
    'word-penalty': 0,
    'cpu-threads': 0,
    'gemm-precision': args.modelName.endsWith('intgemm8.bin') ? 'int8shiftAll' : 'int8shiftAlphaAll',
    'skip-cost': true,
    alignment: 'soft',
    quiet: true,
    'quiet-translation': true,
    'max-length-break': 128,
    'mini-batch-words': 1024,
    workspace: 128,
    'max-length-factor': 2.0,
  }

  models.set(
    key,
    new Module.TranslationModel(yaml(config), alignedMemory(args.model, 256), alignedMemory(args.lex, 64), vocabs, null)
  )
  return true
}

function hasModel(args) {
  return models.has(args.key)
}

function freeModel(args) {
  var model = models.get(args.key)
  if (!model) return false
  models.delete(args.key)
  model.delete()
  return true
}

function translate(args) {
  var input = new Module.VectorString()
  args.texts.forEach(function (text) {
    input.push_back(text)
  })
  var options = new Module.VectorResponseOptions()
  args.texts.forEach(function () {
    options.push_back({ alignment: false, html: !!args.html, qualityScores: false })
  })

  var route = args.route.map(function (key) {
    var model = models.get(key)
    if (!model) throw new Error('Model not loaded: ' + key)
    return model
  })

  var responses =
    route.length > 1
      ? service.translateViaPivoting(route[0], route[1], input, options)
      : service.translate(route[0], input, options)

  var out = []
  for (var i = 0; i < args.texts.length; i++) out.push(responses.get(i).getTranslatedText())

  input.delete()
  options.delete()
  responses.delete()
  return out
}

var handlers = { init: init, loadModel: loadModel, hasModel: hasModel, freeModel: freeModel, translate: translate }

self.addEventListener('message', async function (event) {
  var id = event.data.id
  var name = event.data.name
  try {
    var handler = handlers[name]
    if (!handler) throw new TypeError('Unknown worker call: ' + name)
    var result = await handler(event.data.args || {})
    self.postMessage({ id: id, result: result })
  } catch (error) {
    self.postMessage({
      id: id,
      error: { name: error && error.name, message: (error && error.message) || String(error), stack: error && error.stack },
    })
  }
})
`
