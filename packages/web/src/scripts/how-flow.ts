import { onPage } from './lifecycle'

type Example = 'audio' | 'image' | 'text'
type Route = 'webgpu' | 'wasm'

const examples: Record<Example, {
  modulePath: string
  modelName: string
  baseModel: string
  task: string
  output: Example
  gpuDtype: string
  wasmDtype: string
}> = {
  audio: {
    modulePath: 'runonweb/stt',
    modelName: 'Scribe',
    baseModel: 'Whisper tiny.en',
    task: 'Transcribes speech into text.',
    output: 'text',
    gpuDtype: 'fp32',
    wasmDtype: 'q8',
  },
  image: {
    modulePath: 'runonweb/remove-bg',
    modelName: 'Remover',
    baseModel: 'BEN2',
    task: 'Removes the background from a photo.',
    output: 'image',
    gpuDtype: 'fp16',
    wasmDtype: 'fp16',
  },
  text: {
    modulePath: 'runonweb/tts',
    modelName: 'Voice',
    baseModel: 'Kokoro 82M',
    task: 'Synthesizes spoken audio from text.',
    output: 'audio',
    gpuDtype: 'fp32',
    wasmDtype: 'q8',
  },
}

onPage(() => {
  const figure = document.querySelector<HTMLElement>('[data-how-flow]')
  if (!figure) return

  const inputChoices = [...figure.querySelectorAll<HTMLButtonElement>('[data-how-example-choice]')]
  const routeChoices = [...figure.querySelectorAll<HTMLButtonElement>('[data-how-choice]')]
  const outputs = [...figure.querySelectorAll<HTMLElement>('[data-how-output]')]
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches
  let timer: number | undefined
  let userSelected = false

  const setText = (selector: string, value: string) => {
    const element = figure.querySelector<HTMLElement>(selector)
    if (element) element.textContent = value
  }

  const selectExample = (example: Example) => {
    const data = examples[example]
    figure.dataset.example = example
    setText('[data-module-path]', data.modulePath)
    setText('[data-model-name]', data.modelName)
    setText('[data-base-model]', data.baseModel)
    setText('[data-model-task]', data.task)
    setText('[data-gpu-dtype]', data.gpuDtype)
    setText('[data-wasm-dtype]', data.wasmDtype)
    for (const choice of inputChoices) {
      choice.setAttribute('aria-pressed', String(choice.dataset.howExampleChoice === example))
    }
    for (const output of outputs) {
      output.toggleAttribute('data-selected', output.dataset.howOutput === data.output)
    }
  }

  const selectRoute = (route: Route) => {
    figure.dataset.route = route
    for (const choice of routeChoices) {
      choice.setAttribute('aria-pressed', String(choice.dataset.howChoice === route))
    }
  }

  const stopAutoplay = () => {
    userSelected = true
    window.clearInterval(timer)
    timer = undefined
  }

  for (const choice of inputChoices) {
    choice.addEventListener('click', () => {
      stopAutoplay()
      selectExample(choice.dataset.howExampleChoice as Example)
    })
  }
  for (const choice of routeChoices) {
    choice.addEventListener('click', () => {
      stopAutoplay()
      selectRoute(choice.dataset.howChoice === 'wasm' ? 'wasm' : 'webgpu')
    })
  }

  const cycle: Example[] = ['audio', 'image', 'text']
  const observer = new IntersectionObserver(([entry]) => {
    if (entry?.isIntersecting) {
      figure.dataset.inView = ''
      if (!reducedMotion && !userSelected && timer === undefined) {
        timer = window.setInterval(() => {
          const current = cycle.indexOf(figure.dataset.example as Example)
          selectExample(cycle[(current + 1) % cycle.length]!)
        }, 5500)
      }
    } else {
      delete figure.dataset.inView
      window.clearInterval(timer)
      timer = undefined
    }
  }, { threshold: 0.2 })

  observer.observe(figure)
  return () => {
    observer.disconnect()
    window.clearInterval(timer)
  }
})
