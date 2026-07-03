import { useState } from 'react'
import { useApi, RegimeChip } from './components/util'
import Dashboard from './pages/Dashboard'
import Signals from './pages/Signals'
import Rankings from './pages/Rankings'
import Config from './pages/Config'

const TABS = {
  Dashboard: Dashboard,
  Signals: Signals,
  Rankings: Rankings,
  Config: Config,
}

export default function App() {
  const initial = window.location.hash.slice(1)
  const [tab, setTabState] = useState(TABS[initial] ? initial : 'Dashboard')
  const setTab = (t) => {
    window.location.hash = t
    setTabState(t)
  }
  const { data: status } = useApi('status')
  const Page = TABS[tab]

  return (
    <div className="shell">
      <header className="top">
        <h1>alpha</h1>
        <span className="sub">cross-sectional quant equity system</span>
        {status?.regime && (
          <span className="sub" style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            data through {status.features?.max_date} · regime{' '}
            <RegimeChip regime={status.regime.current} />
          </span>
        )}
      </header>
      <nav className="tabs">
        {Object.keys(TABS).map((t) => (
          <button key={t} className={t === tab ? 'active' : ''} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>
      <Page />
    </div>
  )
}
