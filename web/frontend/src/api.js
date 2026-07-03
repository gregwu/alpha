const cache = {}

export async function api(path) {
  if (cache[path]) return cache[path]
  const res = await fetch(`/api/${path}`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} on /api/${path}`)
  }
  const data = await res.json()
  cache[path] = data
  return data
}

export const fmtPct = (v, d = 2) =>
  v == null ? '—' : `${(v * 100).toFixed(d)}%`
export const fmtNum = (v, d = 2) => (v == null ? '—' : v.toFixed(d))
export const fmtCompact = (v) =>
  v == null ? '—' : Intl.NumberFormat('en', { notation: 'compact' }).format(v)
