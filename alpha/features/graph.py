"""Graph-derived cross-sectional features (the spec's differentiator).

Each month a stock-relationship graph is built from the trailing 126 days
of returns of the then-current universe: edges = each stock's top-K most
positively correlated peers. The graph built from data through month-end
m-1 is used for every day of month m — point-in-time safe.

Daily features:
  graph_nbr_ret_20 / graph_nbr_rs_20 / graph_nbr_ret_60
      correlation-weighted mean of the neighbors' momentum — "neighbor
      momentum" that leads laggards (lead-lag propagation)
  graph_mom_gap_20
      own ret_20 minus neighbor ret_20 (negative = laggard vs peers)
  graph_centrality
      mean correlation to the top-K neighbors (cluster tightness)
  graph_avg_corr
      mean correlation to the whole universe (systematic-ness)
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TOP_K = 10
CORR_WINDOW = 126
MIN_OBS = 100

NBR_FEATURES = ["ret_20", "rs_20", "ret_60"]


def add_graph_features(feat: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Append graph feature columns to the tall feature frame."""
    px = prices.pivot_table(index="date", columns="ticker", values="close")
    rets = px.pct_change(fill_method=None)
    all_dates = px.index

    # Wide views of the inputs we propagate over the graph
    wide = {
        f: feat.pivot_table(index="date", columns="ticker", values=f)
        for f in NBR_FEATURES
    }
    uni = feat.pivot_table(index="date", columns="ticker", values="in_universe",
                           aggfunc="last").fillna(False).astype(bool)

    months = pd.PeriodIndex(feat["date"].unique(), freq="M").unique().sort_values()
    out_wide = {f"graph_nbr_{f}": [] for f in NBR_FEATURES}
    out_wide["graph_centrality"] = []
    out_wide["graph_avg_corr"] = []

    for m in months:
        m_start = m.to_timestamp()
        hist = all_dates[all_dates < m_start]
        if len(hist) < CORR_WINDOW:
            continue
        window = hist[-CORR_WINDOW:]
        graph_date = window[-1]

        # Universe as known at the end of the prior month
        uni_dates = uni.index[uni.index <= graph_date]
        if not len(uni_dates):
            continue
        members = uni.columns[uni.loc[uni_dates[-1]]]
        r = rets.loc[window, members]
        r = r.dropna(axis=1, thresh=MIN_OBS)
        if r.shape[1] < 50:
            continue
        tickers = r.columns.to_numpy()

        c = np.corrcoef(r.fillna(0.0).to_numpy(), rowvar=False)
        np.fill_diagonal(c, -np.inf)

        # top-K positively-correlated neighbors, weighted by correlation
        idx = np.argpartition(-c, TOP_K, axis=1)[:, :TOP_K]
        w = np.take_along_axis(c, idx, axis=1).astype("float64")
        w = np.clip(w, 0.0, None)
        centrality = w.mean(axis=1)
        row_sum = w.sum(axis=1)
        w_norm = np.divide(w, row_sum[:, None], out=np.zeros_like(w),
                           where=row_sum[:, None] > 0)
        # Dense adjacency: A[i, j] = normalized weight of neighbor j for stock i
        n = len(tickers)
        A = np.zeros((n, n), dtype="float64")
        np.put_along_axis(A, idx, w_norm, axis=1)

        c_finite = np.where(np.isfinite(c), c, 0.0)
        avg_corr = c_finite.sum(axis=1) / (len(tickers) - 1)

        month_days = all_dates[(all_dates >= m_start)
                               & (all_dates < (m + 1).to_timestamp())]
        if not len(month_days):
            continue

        for f in NBR_FEATURES:
            vals = wide[f].reindex(index=month_days, columns=tickers).to_numpy()
            filled = np.nan_to_num(vals, nan=0.0)
            valid = (~np.isnan(vals)).astype("float64")
            nbr_sum = filled @ A.T                       # (days, tickers)
            nbr_cov = valid @ A.T                        # coverage of neighbor data
            nbr = np.divide(nbr_sum, nbr_cov, out=np.full_like(nbr_sum, np.nan),
                            where=nbr_cov > 0.5)
            out_wide[f"graph_nbr_{f}"].append(
                pd.DataFrame(nbr, index=month_days, columns=tickers))

        out_wide["graph_centrality"].append(
            pd.DataFrame(np.tile(centrality, (len(month_days), 1)),
                         index=month_days, columns=tickers))
        out_wide["graph_avg_corr"].append(
            pd.DataFrame(np.tile(avg_corr, (len(month_days), 1)),
                         index=month_days, columns=tickers))

    for name, frames in out_wide.items():
        if not frames:
            feat[name] = np.nan
            continue
        tall = (pd.concat(frames).rename_axis("date")
                .melt(ignore_index=False, var_name="ticker", value_name=name)
                .reset_index())
        feat = feat.merge(tall, on=["ticker", "date"], how="left")

    feat["graph_mom_gap_20"] = feat["ret_20"] - feat["graph_nbr_ret_20"]
    n_cov = feat["graph_nbr_ret_20"].notna().mean()
    log.info("graph features done (coverage %.1f%% of rows)", 100 * n_cov)
    return feat
