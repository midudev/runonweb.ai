/**
 * Request / answer shapes for `runonweb/classify`. They follow TypeSafe's System One contract
 * (`POST /v1/systemone`), which is also what Kev serves, so a request written for either runs here unchanged.
 *
 * Ported from `kev/api.py` (github.com/jaredpalmer/kev, Apache-2.0): same rendering of objects to text,
 * same option strings, same answer math. Changing any string here changes what the model reads.
 */

export type JSONContent = string | number | boolean | null | undefined | JSONContent[] | { [key: string]: JSONContent }

/** Yes/no question. The answer is the probability of yes. */
export type NoulQuestion = {
  type: 'noul'
  instructions?: JSONContent
  /** Optional descriptions of what counts as yes (`true`) and no (`false`). */
  criteria?: { true?: JSONContent; false?: JSONContent }
}

/** Pick one option. Keys are the option names; values describe them (or `null`). */
export type ChoiceQuestion = {
  type: 'choice'
  instructions?: JSONContent
  criteria: Record<string, JSONContent>
}

/** Rate on an ordered scale. Levels go from lowest to highest. */
export type ScoreQuestion = {
  type: 'score'
  instructions?: JSONContent
  criteria: JSONContent[]
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion

export type ClassifyRequest<Q extends Record<string, Question> = Record<string, Question>> = {
  /** The text to evaluate. Objects and arrays are rendered as labeled text. */
  state: JSONContent
  /** You choose the ids; the model never sees them. Each question only sees the state and itself. */
  questions: Q
}

export type NoulAnswer = { type: 'noul'; noul: number }

export type ChoiceAnswer<K extends string = string> = {
  type: 'choice'
  choice: K
  /** `(p_max − 1/K) / (1 − 1/K)`. Not a measured accuracy rate. */
  confidence: number
  probabilities: Record<K, number>
}

export type ScoreAnswer = {
  type: 'score'
  /** Expected level index, starting at 0. */
  score: number
  /** How concentrated the distribution is around its most likely level. */
  confidence: number
  legend: Record<string, string>
  probabilities: Record<string, number>
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer

export type Answers<Q extends Record<string, Question>> = {
  [K in keyof Q]: Q[K] extends NoulQuestion
    ? NoulAnswer
    : Q[K] extends ChoiceQuestion
      ? ChoiceAnswer<Extract<keyof Q[K]['criteria'], string>>
      : ScoreAnswer
}

export const MAX_OPTIONS = 255

/** Build a yes/no question (same as TypeSafe's `Noul(...)`). */
export const noul = (instructions: JSONContent, criteria?: NoulQuestion['criteria']): NoulQuestion => ({
  type: 'noul',
  instructions,
  ...(criteria ? { criteria } : {}),
})

/** Build a multiple-choice question (same as TypeSafe's `Choice(...)`). */
export const choice = <const C extends Record<string, JSONContent>>(instructions: JSONContent, criteria: C) => ({
  type: 'choice' as const,
  instructions,
  criteria,
})

/** Build a rating question (same as TypeSafe's `Score(...)`). */
export const score = (instructions: JSONContent, criteria: JSONContent[]): ScoreQuestion => ({
  type: 'score',
  instructions,
  criteria,
})

/** A record the encoder reads: state text plus, per question, instruction text and option strings. */
export type DecisionRecord = {
  state: string
  questions: { instr: string; options: string[] }[]
}

export type QuestionMeta = {
  id: string
  type: Question['type']
  /** Keys probabilities are reported under, in option order. */
  keys: string[]
  legend?: Record<string, string>
}

const isScalar = (v: JSONContent): v is string | number | boolean =>
  typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'

/** Python's `str()` for the scalars JSON can carry (Kev renders booleans as `True` / `False`). */
function pyStr(v: string | number | boolean): string {
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  return String(v)
}

/** Flatten str | object | array into the text the model sees. Field names are kept as labels. */
export function render(v: JSONContent, indent = 0): string {
  const pad = '  '.repeat(indent)
  if (v === null || v === undefined) return ''
  if (isScalar(v)) return pyStr(v)
  if (Array.isArray(v)) return v.map((x) => `${pad}- ${render(x, indent + 1).trimStart()}`).join('\n')
  return Object.entries(v)
    .map(([k, x]) =>
      x !== null && typeof x === 'object' ? `${pad}${k}:\n${render(x, indent + 1)}` : `${pad}${k}: ${render(x)}`
    )
    .join('\n')
}

export function optionText(name: string, desc: JSONContent): string {
  return desc === null || desc === undefined || desc === '' ? name : `${name}: ${render(desc)}`
}

const MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'
const DATE_RE = new RegExp(`\\b(?:${MONTHS}) \\d{1,2}, \\d{4}\\b|\\b\\d{4}-\\d{2}-\\d{2}\\b`, 'g')

function parseDate(raw: string): number | null {
  let y: number, m: number, d: number
  if (raw.includes(',')) {
    const [month, day, year] = raw.replace(',', '').split(' ')
    m = MONTHS.split('|').indexOf(month!) + 1
    d = Number(day)
    y = Number(year)
  } else {
    ;[y, m, d] = raw.split('-').map(Number) as [number, number, number]
  }
  const t = Date.UTC(y, m - 1, d)
  const back = new Date(t)
  if (back.getUTCFullYear() !== y || back.getUTCMonth() !== m - 1 || back.getUTCDate() !== d) return null
  return t / 86_400_000
}

/**
 * Day counts between every pair of absolute dates in `text` ("August 3, 2026 is 12 days after July 22, 2026.").
 * Kev can't subtract dates reliably but uses a stated day count. Empty when fewer than two dates are found.
 */
export function dateFacts(text: string): string {
  const found: [string, number][] = []
  for (const m of text.matchAll(DATE_RE)) {
    const raw = m[0]
    const day = parseDate(raw)
    if (day === null) continue
    if (!found.some(([r]) => r === raw)) found.push([raw, day])
  }
  const facts: string[] = []
  for (let i = 0; i < found.length; i++) {
    for (let j = i + 1; j < found.length; j++) {
      const n = found[j]![1] - found[i]![1]
      const a = Math.abs(n)
      facts.push(
        n
          ? `${found[j]![0]} is ${a} day${a !== 1 ? 's' : ''} ${n > 0 ? 'after' : 'before'} ${found[i]![0]}.`
          : `${found[j]![0]} is the same day as ${found[i]![0]}.`
      )
    }
  }
  return facts.join(' ')
}

/** State with a `date_facts` field (objects), entry (arrays) or paragraph (text) when two or more dates appear. */
export function withDateFacts(state: JSONContent): JSONContent {
  const facts = dateFacts(render(state))
  if (!facts) return state
  if (Array.isArray(state)) return [...state, { date_facts: facts }]
  if (state !== null && typeof state === 'object') return { ...state, date_facts: facts }
  return `${render(state)}\n\ndate_facts: ${facts}`
}

function questionKeys(q: Question): string[] {
  if (q.type === 'choice') return Object.keys(q.criteria)
  if (q.type === 'noul') return ['false', 'true']
  return q.criteria.map((_, i) => String(i))
}

/** Validate a request the way the Kev server does (it answers 422); throws a readable error instead. */
export function validateRequest(req: ClassifyRequest): void {
  if (!req || typeof req !== 'object') throw new TypeError('Request must be an object with `state` and `questions`')
  const entries = Object.entries(req.questions ?? {})
  if (entries.length === 0) throw new Error('`questions` needs at least one question')
  for (const [id, q] of entries) {
    if (!q || typeof q !== 'object') throw new Error(`Question "${id}" must be an object`)
    if (q.type === 'choice') {
      const n = Object.keys(q.criteria ?? {}).length
      if (n < 1 || n > MAX_OPTIONS) throw new Error(`Choice "${id}" needs 1–${MAX_OPTIONS} options in \`criteria\``)
    } else if (q.type === 'score') {
      const n = Array.isArray(q.criteria) ? q.criteria.length : 0
      if (n < 1 || n > MAX_OPTIONS) throw new Error(`Score "${id}" needs 1–${MAX_OPTIONS} levels in \`criteria\``)
    } else if (q.type !== 'noul') {
      throw new Error(`Question "${id}" has unknown type "${(q as { type?: string }).type}" (noul | choice | score)`)
    }
  }
}

/** Request → encoder record plus per-question metadata to map probabilities back. */
export function toRecord(req: ClassifyRequest): { record: DecisionRecord; meta: QuestionMeta[] } {
  const questions: DecisionRecord['questions'] = []
  const meta: QuestionMeta[] = []
  for (const [id, q] of Object.entries(req.questions)) {
    const m: QuestionMeta = { id, type: q.type, keys: questionKeys(q) }
    let options: string[]
    if (q.type === 'noul') {
      const c = q.criteria ?? {}
      options = [optionText('no', c.false), optionText('yes', c.true)]
    } else if (q.type === 'choice') {
      options = Object.entries(q.criteria).map(([k, v]) => optionText(k, v))
    } else {
      options = q.criteria.map((x) => render(x))
      m.legend = Object.fromEntries(m.keys.map((k, i) => [k, options[i]!]))
    }
    questions.push({ instr: render(q.instructions), options })
    meta.push(m)
  }
  return { record: { state: render(req.state), questions }, meta }
}

const round = (x: number) => Math.round(x * 10_000) / 10_000

export function choiceConfidence(p: ArrayLike<number>): number {
  const K = p.length
  return K === 1 ? 1 : (Math.max(...Array.from(p)) - 1 / K) / (1 - 1 / K)
}

export function scoreConfidence(p: ArrayLike<number>): number {
  const L = p.length
  if (L === 1) return 1
  let mode = 0
  for (let i = 1; i < L; i++) if (p[i]! > p[mode]!) mode = i
  let spread = 0
  for (let i = 0; i < L; i++) spread += p[i]! * Math.abs(i - mode)
  return 1 - spread / (L - 1)
}

/** Probabilities per question → System One answers (4-decimal rounding, as served by Kev). */
export function toAnswers(probs: Float64Array[], meta: QuestionMeta[]): Record<string, Answer> {
  const out: Record<string, Answer> = {}
  probs.forEach((p, qi) => {
    const m = meta[qi]!
    if (m.type === 'noul') {
      out[m.id] = { type: 'noul', noul: round(p[1]!) }
    } else if (m.type === 'choice') {
      let best = 0
      for (let i = 1; i < p.length; i++) if (p[i]! > p[best]!) best = i
      out[m.id] = {
        type: 'choice',
        choice: m.keys[best]!,
        confidence: round(choiceConfidence(p)),
        probabilities: Object.fromEntries(m.keys.map((k, i) => [k, round(p[i]!)])),
      }
    } else {
      let s = 0
      for (let i = 0; i < p.length; i++) s += i * p[i]!
      out[m.id] = {
        type: 'score',
        score: round(s),
        confidence: round(scoreConfidence(p)),
        legend: m.legend ?? {},
        probabilities: Object.fromEntries(m.keys.map((k, i) => [k, round(p[i]!)])),
      }
    }
  })
  return out
}
