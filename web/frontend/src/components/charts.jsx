import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts'
import { T } from '../theme'

const axisProps = {
  stroke: T.axis,
  tick: { fill: T.inkMuted, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: T.axis, strokeWidth: 1 },
}

function VizTooltip({ active, payload, label, fmt }) {
  if (!active || !payload?.length) return null
  return (
    <div className="viz-tooltip">
      <div className="t-date">{label}</div>
      {payload.map((p) => (
        <div className="t-row" key={p.dataKey}>
          <span className="swatch" style={{ background: p.stroke || p.fill }} />
          <span className="t-val">{fmt(p.value)}</span>
          <span className="t-name">{p.name}</span>
        </div>
      ))}
    </div>
  )
}

const yearTicks = (data) => {
  const seen = new Set()
  return data
    .filter((d) => {
      const y = d.date.slice(0, 4)
      if (seen.has(y)) return false
      seen.add(y)
      return true
    })
    .map((d) => d.date)
}

export function EquityChart({ data }) {
  const fmt = (v) => `${v?.toFixed(2)}x`
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={T.grid} strokeWidth={1} vertical={false} />
        <XAxis dataKey="date" {...axisProps} ticks={yearTicks(data)}
          tickFormatter={(d) => d.slice(0, 4)} />
        <YAxis {...axisProps} scale="log" domain={['auto', 'auto']}
          tickFormatter={fmt} width={52} />
        <Tooltip content={<VizTooltip fmt={fmt} />}
          cursor={{ stroke: T.axis, strokeWidth: 1 }} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: T.inkSecondary }} />
        <Line isAnimationActive={false} type="monotone" dataKey="equity" name="Strategy" stroke={T.strategy}
          strokeWidth={2} dot={false} strokeLinejoin="round" strokeLinecap="round" />
        <Line isAnimationActive={false} type="monotone" dataKey="spy_equity" name="SPY" stroke={T.benchmark}
          strokeWidth={2} dot={false} strokeLinejoin="round" strokeLinecap="round" />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function DrawdownChart({ data }) {
  const fmt = (v) => `${(v * 100)?.toFixed(1)}%`
  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={T.grid} strokeWidth={1} vertical={false} />
        <XAxis dataKey="date" {...axisProps} ticks={yearTicks(data)}
          tickFormatter={(d) => d.slice(0, 4)} />
        <YAxis {...axisProps} tickFormatter={fmt} width={52} />
        <Tooltip content={<VizTooltip fmt={fmt} />}
          cursor={{ stroke: T.axis, strokeWidth: 1 }} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: T.inkSecondary }} />
        <Area isAnimationActive={false} type="monotone" dataKey="drawdown" name="Strategy DD" stroke={T.negative}
          strokeWidth={2} fill={T.negative} fillOpacity={0.1} />
        <Area isAnimationActive={false} type="monotone" dataKey="spy_drawdown" name="SPY DD" stroke={T.benchmark}
          strokeWidth={2} fill={T.benchmark} fillOpacity={0.08} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function ExposureChart({ data }) {
  const fmt = (v) => `${(v * 100)?.toFixed(0)}%`
  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={T.grid} strokeWidth={1} vertical={false} />
        <XAxis dataKey="date" {...axisProps} ticks={yearTicks(data)}
          tickFormatter={(d) => d.slice(0, 4)} />
        <YAxis {...axisProps} tickFormatter={fmt} width={52} domain={[0, 1.05]} />
        <Tooltip content={<VizTooltip fmt={fmt} />}
          cursor={{ stroke: T.axis, strokeWidth: 1 }} />
        <Area isAnimationActive={false} type="stepAfter" dataKey="gross_exposure" name="Gross exposure"
          stroke={T.violet} strokeWidth={2} fill={T.violet} fillOpacity={0.1} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function DecileChart({ data }) {
  const fmt = (v) => `${(v * 100)?.toFixed(2)}%`
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barCategoryGap="25%">
        <CartesianGrid stroke={T.grid} strokeWidth={1} vertical={false} />
        <XAxis dataKey="decile" {...axisProps} tickFormatter={(d) => `D${d}`} />
        <YAxis {...axisProps} tickFormatter={fmt} width={56} />
        <Tooltip content={<VizTooltip fmt={fmt} />} cursor={{ fill: 'rgba(11,11,11,0.04)' }} />
        <Bar isAnimationActive={false} dataKey="fwd_ret_20" name="Mean fwd 20d return" radius={[4, 4, 0, 0]} maxBarSize={24}>
          {data.map((d, i) => (
            <Cell key={d.decile} fill={T.blueRamp[i] || T.blueRamp.at(-1)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function ImportanceChart({ data }) {
  const fmt = (v) => `${(v * 100)?.toFixed(1)}%`
  const h = Math.max(200, data.length * 22)
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical"
        margin={{ top: 4, right: 40, bottom: 0, left: 30 }} barCategoryGap="30%">
        <CartesianGrid stroke={T.grid} strokeWidth={1} horizontal={false} />
        <XAxis type="number" {...axisProps} tickFormatter={fmt} />
        <YAxis type="category" dataKey="feature" {...axisProps} width={150}
          interval={0} tick={{ fill: T.inkSecondary, fontSize: 11.5 }} />
        <Tooltip content={<VizTooltip fmt={fmt} />} cursor={{ fill: 'rgba(11,11,11,0.04)' }} />
        <Bar isAnimationActive={false} dataKey="gain_share" name="Share of model gain" fill={T.strategy}
          radius={[0, 4, 4, 0]} maxBarSize={14} />
      </BarChart>
    </ResponsiveContainer>
  )
}
