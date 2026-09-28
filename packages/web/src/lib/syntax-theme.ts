import { createHighlighter, type Highlighter, type ThemeRegistration } from 'shiki'

/**
 * Vercel dark syntax theme as used by ray.so.
 * Token scopes match ray.so's CSS-variables Shiki theme so highlighting
 * splits keywords, functions, constants and strings the same way.
 */
const c = {
  fg: 'hsla(0, 0%, 93%, 1)',
  constant: 'oklch(71.7% 0.1648 250.79360374054167)',
  string: 'oklch(73.1% 0.2158 148.29)',
  comment: 'hsla(0, 0%, 63%, 1)',
  keyword: 'oklch(69.36% 0.2223 3.91)',
  parameter: 'oklch(77.21% 0.1991 64.28)',
  fn: 'oklch(69.87% 0.2037 309.51)',
  punctuation: 'hsla(0, 0%, 93%, 1)',
  number: '#ffffff',
  property: 'oklch(71.7% 0.1648 250.79360374054167)',
}

export const vercelDark: ThemeRegistration = {
  name: 'vercel-dark',
  type: 'dark',
  colors: {
    'editor.foreground': c.fg,
    'editor.background': 'transparent',
  },
  tokenColors: [
    {
      scope: [
        'keyword.operator.accessor',
        'meta.group.braces.round.function.arguments',
        'meta.template.expression',
        'markup.fenced_code meta.embedded.block',
      ],
      settings: { foreground: c.fg },
    },
    {
      scope: [
        'string',
        'markup.fenced_code',
        'markup.inline',
        'string.quoted.docstring.multi.python',
      ],
      settings: { foreground: c.string },
    },
    {
      scope: ['comment', 'string.quoted.docstring.multi', 'meta.diff.header.from-file', 'meta.diff.header.to-file'],
      settings: { foreground: c.comment },
    },
    {
      scope: [
        'constant.numeric',
        'constant.language',
        'constant.other.placeholder',
        'constant.character.format.placeholder',
        'variable.language.this',
        'variable.other.object',
        'variable.other.class',
        'variable.other.constant',
        'meta.property-name',
        'meta.property-value',
        'support',
      ],
      settings: { foreground: c.constant },
    },
    {
      scope: [
        'keyword',
        'storage.modifier',
        'storage.type',
        'storage.control.clojure',
        'entity.name.function.clojure',
        'entity.name.tag.yaml',
        'support.function.node',
        'support.type.property-name.json',
        'punctuation.separator.key-value',
        'punctuation.definition.template-expression',
      ],
      settings: { foreground: c.keyword },
    },
    {
      scope: 'variable.parameter.function',
      settings: { foreground: c.parameter },
    },
    {
      scope: [
        'support.function',
        'entity.name.type',
        'entity.other.inherited-class',
        'meta.function-call',
        'meta.instance.constructor',
        'entity.other.attribute-name',
        'entity.name.function',
        'constant.keyword.clojure',
      ],
      settings: { foreground: c.fn },
    },
    {
      scope: [
        'entity.name.tag',
        'string.quoted',
        'string.regexp',
        'string.interpolated',
        'string.template',
        'string.unquoted.plain.out.yaml',
        'keyword.other.template',
      ],
      settings: { foreground: c.string },
    },
    {
      scope: [
        'punctuation.definition.arguments',
        'punctuation.definition.dict',
        'punctuation.separator',
        'meta.function-call.arguments',
      ],
      settings: { foreground: c.punctuation },
    },
    {
      scope: ['constant.numeric.decimal', 'constant.language.boolean', 'meta.var.exp.ts'],
      settings: { foreground: c.number },
    },
    {
      scope: ['support.variable.property'],
      settings: { foreground: c.property },
    },
  ],
}

const ALIAS: Record<string, string> = {
  ts: 'typescript',
  js: 'javascript',
  sh: 'bash',
  shell: 'bash',
}

const LANGS = ['typescript', 'javascript', 'bash', 'json'] as const

let highlighterPromise: Promise<Highlighter> | null = null

function getHighlighter() {
  highlighterPromise ??= createHighlighter({
    themes: [vercelDark],
    langs: [...LANGS],
  })
  return highlighterPromise
}

export async function highlightCode(code: string, language: string) {
  const highlighter = await getHighlighter()
  const lang = ALIAS[language] ?? language
  const loaded = highlighter.getLoadedLanguages()
  if (!loaded.includes(lang)) {
    try {
      await highlighter.loadLanguage(lang as (typeof LANGS)[number])
    } catch {
      return highlighter.codeToHtml(code, { lang: 'text', theme: 'vercel-dark' })
    }
  }
  return highlighter.codeToHtml(code, { lang, theme: 'vercel-dark' })
}
