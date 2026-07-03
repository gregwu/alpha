import { fmtPct, fmtNum } from '../api'
import { useApi, Panel, Loading, RegimeChip } from '../components/util'
import { DecileChart, ImportanceChart } from '../components/charts'

const GROUP_LABELS = {
  score_relative_strength: 'Rel. strength',
  score_trend: 'Trend',
  score_volume: 'Volume',
  score_volatility: 'Volatility',
  score_structure: 'Structure',
  score_graph: 'Graph',
}

export default function Signals() {
  const { data: report, error: e1 } = useApi('report')
  const { data: importance, error: e2 } = useApi('importance?top=20')

  if (!report || !importance) return <Loading error={e1 || e2} />
  const d = report.data

  return (
    <>
      {d?.ic && (
        <Panel title="Information coefficient — daily Spearman rank corr. vs forward 20d return">
          <div className="scroll-x">
            <table className="data">
              <thead>
                <tr>
                  <th>Score</th>
                  <th className="num">Mean IC</th>
                  <th className="num">IC IR</th>
                  <th className="num">% days positive</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(d.ic).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num">{v.mean >= 0 ? '+' : ''}{v.mean.toFixed(4)}</td>
                    <td className="num">{fmtNum(v.ir)}</td>
                    <td className="num">{fmtPct(v.pct_positive, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {d?.deciles && (
        <Panel title="Mean forward 20d return by final-score decile (D10 = highest score)">
          <DecileChart data={d.deciles} />
        </Panel>
      )}

      {d?.attribution && (
        <Panel title="Factor-group IC by market regime — which factors work in which regime">
          <div className="scroll-x">
            <table className="data">
              <thead>
                <tr>
                  <th>Regime</th>
                  {Object.keys(GROUP_LABELS).map((g) => (
                    <th className="num" key={g}>{GROUP_LABELS[g]}</th>
                  ))}
                  <th className="num">Days</th>
                </tr>
              </thead>
              <tbody>
                {d.attribution.map((row) => (
                  <tr key={row.regime}>
                    <td><RegimeChip regime={row.regime} /></td>
                    {Object.keys(GROUP_LABELS).map((g) => (
                      <td className="num" key={g}
                        style={{ color: (row[g] ?? 0) < 0 ? '#d03b3b' : '#006300' }}>
                        {row[g] == null ? '—' : (row[g] >= 0 ? '+' : '') + row[g].toFixed(4)}
                      </td>
                    ))}
                    <td className="num">{row.n_days}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <Panel title="Feature importance — top 20 by mean LightGBM gain across folds">
        <ImportanceChart data={importance} />
      </Panel>
    </>
  )
}
