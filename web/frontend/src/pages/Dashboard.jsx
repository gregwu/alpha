import { fmtPct, fmtNum } from '../api'
import { useApi } from '../components/util'
import { Panel, Loading, StatTile } from '../components/util'
import { EquityChart, DrawdownChart, ExposureChart } from '../components/charts'

export default function Dashboard() {
  const { data: summary, error: e1 } = useApi('backtest/summary')
  const { data: series, error: e2 } = useApi('backtest/series')

  if (!summary || !series) return <Loading error={e1 || e2} />
  const s = summary.strategy
  const b = summary.spy

  return (
    <>
      <div className="tile-row">
        <StatTile label="CAGR" value={fmtPct(s.CAGR)}
          delta={`SPY ${fmtPct(b.CAGR)}`} />
        <StatTile label="Sharpe" value={fmtNum(s.Sharpe)}
          delta={`SPY ${fmtNum(b.Sharpe)}`} />
        <StatTile label="Max drawdown" value={fmtPct(s.MaxDD)}
          delta={`SPY ${fmtPct(b.MaxDD)}`} />
        <StatTile label="Ann. volatility" value={fmtPct(s.AnnVol)}
          delta={`SPY ${fmtPct(b.AnnVol)}`} />
        <StatTile label="Avg gross exposure" value={fmtPct(summary.avg_gross_exposure, 0)} />
        <StatTile label="Turnover (1-way, ann.)" value={`${fmtNum(summary.annual_turnover, 1)}x`} />
      </div>

      <Panel title={`Equity curve — ${summary.period.start} … ${summary.period.end} (log scale)`}>
        <EquityChart data={series} />
      </Panel>
      <Panel title="Drawdown">
        <DrawdownChart data={series} />
      </Panel>
      <Panel title="Gross exposure (scaled by market regime)">
        <ExposureChart data={series} />
      </Panel>
    </>
  )
}
