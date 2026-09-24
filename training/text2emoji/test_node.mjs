// Verify the exported folder loads with Transformers.js (same runtime the site uses).
import { pipeline, env } from '@huggingface/transformers'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
env.allowRemoteModels = false
env.allowLocalModels = true
env.localModelPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'out')
const name = process.argv[2] ?? 'tiny'
const dtype = process.argv[3] ?? 'q8'
const pipe = await pipeline('text2text-generation', name, { dtype, device: 'cpu' })
const tests = ['I love pizza and my dog', "Happy birthday! Let's party tonight",
  'It is raining and I forgot my umbrella', 'Going to the gym then a coffee with friends',
  'I am so tired of work, need a vacation on the beach', 'Learning to code in JavaScript is fun',
  'Good morning! Coffee, sunshine and a long walk on the beach']
const t0 = performance.now()
for (const t of tests) {
  const out = await pipe(t, { max_new_tokens: 20 })
  console.log(t.padEnd(60), '->', out[0].generated_text.replace(/\s+/g, ''))
}
console.log(`${((performance.now() - t0) / tests.length).toFixed(0)} ms/sentence (${dtype})`)
