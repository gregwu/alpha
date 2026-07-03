import { fmtPct, fmtNum, fmtCompact } from '../api'
import { useApi, Panel, Loading } from '../components/util'

export default function Rankings() {
  const { data: rank, error: e1 } = useApi('rankings?limit=100')
  const { data: holdings, error: e2 } = useApi('backtest/holdings')

  if (!rank) return <Loading error={e1} />
  const rows = rank.rows || []

  return (
    <>
      {holdings && !e2 && (
        <Panel title={`Latest backtest target portfolio — ${holdings[0]?.date} (${holdings.length} names, regime: ${holdings[0]?.regime})`}>
          <div className="scroll-x">
            <table className="data">
              <thead>
                <tr><th>Ticker</th><th>Sector</th><th className="num">Weight</th></tr>
              </thead>
              <tbody>
                {holdings.map((h) => (
                  <tr key={h.ticker}>
                    <td>{h.ticker?.replace('.US', '')}</td>
                    <td>{h.sector}</td>
                    <td className="num">{fmtPct(h.weight, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <Panel title={`Live ranking — top 100 as of ${rank.as_of}`}>
        <div className="scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th className="num">#</th>
                <th>Ticker</th>
                <th>Sector</th>
                <th className="num">Close</th>
                <th className="num">Final score</th>
                <th className="num">ML z</th>
                <th className="num">Comp z</th>
                <th className="num">20d ret</th>
                <th className="num">RS 20d</th>
                <th className="num">HV 20</th>
                <th className="num">Mkt cap</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.ticker}>
                  <td className="num">{r.rank}</td>
                  <td>{String(r.ticker).replace('.US', '')}</td>
                  <td>{r.sector}</td>
                  <td className="num">{fmtNum(r.close)}</td>
                  <td className="num">{fmtNum(r.final_score, 3)}</td>
                  <td className="num">{fmtNum(r.ml_z, 2)}</td>
                  <td className="num">{fmtNum(r.comp_z, 2)}</td>
                  <td className="num">{fmtPct(r.ret_20, 1)}</td>
                  <td className="num">{fmtPct(r.rs_20, 1)}</td>
                  <td className="num">{fmtPct(r.hv_20, 0)}</td>
                  <td className="num">{fmtCompact(r.market_cap)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  )
}
