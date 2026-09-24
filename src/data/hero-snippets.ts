import type { Model } from './models'

/**
 * Short, equal-weight usage snippets shown in the home hero, one per model slug.
 * Same API as `usageSnippet`, trimmed to the happy path so they rotate cleanly.
 */
export const HERO_SNIPPETS: Record<Model['slug'], string> = {
  stt: `import { SpeechToText } from 'runonweb/stt'

const stt = new SpeechToText()
await stt.load()

const { text } = await stt.transcribe(audioBlob)
console.log(text) // "Hello from the browser"`,

  'remove-bg': `import { RemoveBackground } from 'runonweb/remove-bg'

const remover = new RemoveBackground()
await remover.load()

const png = await remover.remove(imageFile)
img.src = URL.createObjectURL(png) // transparent PNG`,

  caption: `import { ImageCaptioner } from 'runonweb/caption'

const captioner = new ImageCaptioner({ language: 'es' })
await captioner.load()

const { text } = await captioner.caption(imageFile)
// "Un gato sentado sobre una mesa de madera."`,

  depth: `import { DepthEstimator } from 'runonweb/depth'

const estimator = new DepthEstimator()
await estimator.load()

const { depth } = await estimator.estimate(imageFile)
img.src = URL.createObjectURL(depth) // grayscale PNG`,

  detect: `import { ObjectDetector } from 'runonweb/detect'

const detector = new ObjectDetector({ threshold: 0.5 })
await detector.load()

const objects = await detector.detect(imageFile)
// [{ label: 'cat', score: 0.98, box: { … } }]`,

  ocr: `import { OCR } from 'runonweb/ocr'

const ocr = new OCR({ size: 'small' })
await ocr.load()

const { text, lines } = await ocr.read(imageFile)
console.log(text)`,

  embed: `import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'

const embedder = new TextEmbedder()
await embedder.load()

const { embeddings } = await embedder.embed(texts)
cosineSimilarity(embeddings[0], embeddings[1]) // ~0.7`,

  translate: `import { Translator } from 'runonweb/translate'

const translator = new Translator({ from: 'en', to: 'es' })
await translator.load()

const { text } = await translator.translate('Hello world')
// "Hola mundo"`,

  clean: `import { TranscriptCleaner } from 'runonweb/clean'

const cleaner = new TranscriptCleaner()
await cleaner.load()

const { text } = await cleaner.clean(rawTranscript)
// "I need to send the report by Thursday."`,

  classify: `import { Classifier } from 'runonweb/classify'

const classifier = new Classifier()
const { answers } = await classifier.classify({
  state: 'I was charged twice. Please fix this ASAP.',
  questions: { billing: { type: 'noul', instructions: 'Is this about billing?' } },
})
answers.billing.noul // 0.91 = p(yes)`,

  emoji: `import { Emojifier } from 'runonweb/emoji'

const emojifier = new Emojifier()
await emojifier.load()

const { text } = await emojifier.emojify('I love pizza and my dog')
// "🍕❤️🐶"`,

  tts: `import { TextToSpeech } from 'runonweb/tts'

const tts = new TextToSpeech({ voice: 'af_heart' })
await tts.load()

for await (const { audio } of tts.speakStream('Hello from the browser')) {
  play(audio) // 24 kHz PCM, as it arrives
}`,

  image: `import { ImageGenerator } from 'runonweb/image'

const gen = new ImageGenerator()
await gen.load()

const { image } = await gen.generate('A bonsai tree in a ceramic studio')
img.src = URL.createObjectURL(image)`,
}
