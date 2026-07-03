import { useEffect, useState } from 'react'
import { api } from '../api'
import { REGIME_COLORS } from '../theme'

export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null })
  useEffect(() => {
    let live = true
    api(path)
      .then((data) => live && setState({ data, error: null }))
      .catch((error) => live && setState({ data: null, error }))
    return () => { live = false }
  }, [path])
  return state
}

export function Panel({ title, children }) {
  return (
    <section className="card">
      {title && <h2>{title}</h2>}
      {children}
    </section>
  )
}

export function Loading({ error }) {
  if (error) return <div className="err">{String(error.message || error)}</div>
  return <div className="loading">loading…</div>
}

export function RegimeChip({ regime }) {
  if (!regime) return null
  return (
    <span className="chip" style={{ background: REGIME_COLORS[regime] || '#898781' }}>
      {regime}
    </span>
  )
}

export function StatTile({ label, value, delta, deltaGoodWhenUp = true, tone }) {
  let cls = ''
  if (tone) {
    cls = tone
  } else if (delta != null) {
    const up = !String(delta).startsWith('-')
    cls = up === deltaGoodWhenUp ? 'up' : 'down'
  }
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta != null && <div className={`delta ${cls}`}>{delta}</div>}
    </div>
  )
}
