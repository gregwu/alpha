import { useApi, Panel, Loading } from '../components/util'

const SECTION_TITLES = {
  universe: 'Universe filters',
  features: 'Feature engineering',
  model: 'ML model (walk-forward LightGBM)',
  portfolio: 'Portfolio construction & risk',
  backtest: 'Backtest window',
  composite: 'Composite factor weights',
}

export default function Config() {
  const { data, error } = useApi('config')
  if (!data) return <Loading error={error} />

  return (
    <>
      <p style={{ color: 'var(--ink-2)', marginTop: 0 }}>
        Read-only view of <code>alpha/config.py</code>. Edit that file and re-run
        the pipeline scripts to change behavior.
      </p>
      <div className="grid-2">
        {Object.entries(data).map(([section, values]) => (
          <Panel key={section} title={SECTION_TITLES[section] || section}>
            <pre className="cfg">{JSON.stringify(values, null, 2)}</pre>
          </Panel>
        ))}
      </div>
    </>
  )
}
