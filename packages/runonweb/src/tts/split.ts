const MAX_CHARS = 240

/** Split text into sentence-sized chunks for streaming synthesis. */
export function splitUtterances(text: string): string[] {
  const trimmed = text.trim()
  if (!trimmed) return []

  const sentences = trimmed.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean)
  const out: string[] = []

  for (const sentence of sentences) {
    if (sentence.length <= MAX_CHARS) {
      out.push(sentence)
      continue
    }
    const words = sentence.split(/\s+/)
    let buf = ''
    for (const word of words) {
      const next = buf ? `${buf} ${word}` : word
      if (next.length > MAX_CHARS && buf) {
        out.push(buf)
        buf = word
      } else {
        buf = next
      }
    }
    if (buf) out.push(buf)
  }

  return out
}
