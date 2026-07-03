// Chart palette — reference dataviz palette (light mode).
// Series hues are assigned to entities and never re-ordered.
export const T = {
  surface: '#fcfcfb',
  page: '#f9f9f7',
  inkPrimary: '#0b0b0b',
  inkSecondary: '#52514e',
  inkMuted: '#898781',
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  border: 'rgba(11,11,11,0.10)',

  strategy: '#2a78d6',   // categorical slot 1 (blue)
  benchmark: '#1baf7a',  // categorical slot 2 (aqua)
  negative: '#e34948',   // categorical red — drawdown series
  violet: '#4a3aa7',     // categorical slot 5 — exposure series

  good: '#0ca30c',
  critical: '#d03b3b',

  // Ordinal blue ramp (steps 250→700) for ordered categories like deciles
  blueRamp: ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
             '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'],
}

export const REGIME_COLORS = {
  bull: '#0ca30c',
  neutral: '#898781',
  high_vol: '#fab219',
  bear: '#ec835a',
  panic: '#d03b3b',
}
