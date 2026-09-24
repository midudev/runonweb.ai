/** Run on first load and after every client-side navigation. */
export function onPage(init: () => void | (() => void)) {
  let dispose: (() => void) | void
  let started = false
  const run = () => {
    started = true
    try {
      dispose?.()
    } catch {
      /* previous page already torn down */
    }
    dispose = init()
  }
  document.addEventListener('astro:page-load', run)
  // Late-evaluated modules miss the first `astro:page-load`. Only catch up if
  // that event already happened — otherwise page-load will start us.
  if (document.readyState === 'complete') {
    queueMicrotask(() => {
      if (!started) run()
    })
  }
}
