/** Loading LED + keep persisted header in sync with the current route. */

document.addEventListener('astro:before-preparation', () => {
  document.documentElement.dataset.nav = 'loading'
})

document.addEventListener('astro:after-swap', () => {
  delete document.documentElement.dataset.nav
})

document.addEventListener('astro:page-load', () => {
  const path = location.pathname.replace(/\/+$/, '') || '/'
  const models = path === '/models' || path.startsWith('/models/')
  const docs = path === '/docs' || path.startsWith('/docs/')

  const modelsLink = document.querySelector<HTMLAnchorElement>('[data-nav="models"]')
  const docsLink = document.querySelector<HTMLAnchorElement>('[data-nav="docs"]')

  modelsLink?.classList.toggle('bg-bg-3', models)
  docsLink?.classList.toggle('bg-bg-3', docs)

  if (modelsLink) {
    if (path === '/models') modelsLink.setAttribute('aria-current', 'page')
    else modelsLink.removeAttribute('aria-current')
  }
  if (docsLink) {
    if (docs) docsLink.setAttribute('aria-current', 'page')
    else docsLink.removeAttribute('aria-current')
  }

  document.querySelectorAll<HTMLAnchorElement>('[data-nav-model]').forEach((a) => {
    if (a.getAttribute('href') === path) a.setAttribute('aria-current', 'page')
    else a.removeAttribute('aria-current')
  })
})
