"""Performance and research reporting: headline metrics, information
coefficients, decile spreads, and factor attribution by market regime.
"""

import numpy as np
import pandas as pd

from .config import REPORTS_DIR


def perf_metrics(daily: pd.DataFrame, ret_col: str = "ret") -> dict:
    r = daily[ret_col].dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    downside = r[r < 0].std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": (r.mean() * 252) / vol if vol > 0 else np.nan,
        "Sortino": (r.mean() * 252) / downside if downside > 0 else np.nan,
        "MaxDD": dd,
        "HitRate": (r > 0).mean(),
        "Best day": r.max(),
        "Worst day": r.min(),
    }


def information_coefficient(scores: pd.DataFrame, feat: pd.DataFrame,
                            horizon: int = 20) -> pd.DataFrame:
    """Daily Spearman IC of ml/composite/final scores vs forward returns."""
    lab = f"fwd_ret_{horizon}"
    df = scores.merge(feat[["ticker", "date", lab]], on=["ticker", "date"], how="left")
    df = df.dropna(subset=[lab])

    def daily_ic(col):
        return (df.dropna(subset=[col])
                  .groupby("date")
                  .apply(lambda g: g[col].corr(g[lab], method="spearman"),
                         include_groups=False))

    out = pd.DataFrame({c: daily_ic(c) for c in ("ml_score", "composite", "final_score")
                        if c in df.columns})
    return out


def decile_returns(scores: pd.DataFrame, feat: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Mean forward return by final-score decile (1 = worst, 10 = best)."""
    lab = f"fwd_ret_{horizon}"
    df = scores.merge(feat[["ticker", "date", lab]], on=["ticker", "date"], how="left")
    df = df.dropna(subset=[lab, "final_score"])
    df["decile"] = df.groupby("date")["final_score"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop") + 1)
    return df.groupby("decile")[lab].mean()


def factor_ic_by_regime(feat: pd.DataFrame, group_scores: pd.DataFrame,
                        regime: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Rolling factor attribution: IC of each factor-group score vs forward
    returns, split by market regime."""
    lab = f"fwd_ret_{horizon}"
    cols = [c for c in group_scores.columns if c.startswith("score_")]
    df = pd.concat([feat[["ticker", "date", lab, "in_universe"]], group_scores], axis=1)
    df = df[df["in_universe"] & df[lab].notna()]
    df = df.merge(regime[["date", "regime"]], on="date", how="left")

    rows = []
    for (reg), sub in df.groupby("regime"):
        ics = sub.groupby("date").apply(
            lambda g: pd.Series({c: g[c].corr(g[lab], method="spearman") for c in cols}),
            include_groups=False)
        row = ics.mean()
        row["n_days"] = len(ics)
        row.name = reg
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(daily: pd.DataFrame, ic: pd.DataFrame, deciles: pd.Series,
                 attribution: pd.DataFrame, holdings: pd.DataFrame) -> str:
    """Text report + charts saved under reports/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_strat = perf_metrics(daily, "ret")
    m_spy = perf_metrics(daily, "spy_ret")
    lines = ["=" * 64, "ALPHA SYSTEM — WALK-FORWARD BACKTEST", "=" * 64, ""]
    lines.append(f"Period: {daily.index.min().date()} .. {daily.index.max().date()}")
    lines.append(f"Avg gross exposure: {daily['gross_exposure'].mean():.2f}")
    ann_turnover = daily["cost"].sum() / (10 / 1e4) / (len(daily) / 252)
    lines.append(f"Approx. annual turnover (one-way): {ann_turnover:.1f}x")
    lines.append("")
    lines.append(f"{'metric':<12}{'strategy':>12}{'SPY':>12}")
    for k in m_strat:
        pct = "{:>11.2%}" if k not in ("Sharpe", "Sortino") else "{:>11.2f}"
        lines.append(f"{k:<12}" + pct.format(m_strat[k]) + pct.format(m_spy[k]))
    lines.append("")

    if not ic.empty:
        lines.append("Information coefficient (daily Spearman vs fwd 20d return):")
        for c in ic.columns:
            s = ic[c].dropna()
            if len(s):
                lines.append(f"  {c:<14} mean={s.mean():+.4f}  IR={s.mean()/s.std():+.2f}  "
                             f"%positive={(s > 0).mean():.1%}")
        lines.append("")

    lines.append("Mean fwd 20d return by final-score decile:")
    for d, v in deciles.items():
        lines.append(f"  D{int(d):<3}{v:+.4f}")
    lines.append("")
    lines.append("Factor-group IC by market regime (vs fwd 20d return):")
    lines.append(attribution.round(4).to_string())
    text = "\n".join(lines)

    (REPORTS_DIR / "backtest_report.txt").write_text(text)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    ax = axes[0]
    ax.plot(daily.index, daily["equity"], label="Strategy", lw=1.4)
    ax.plot(daily.index, daily["spy_equity"], label="SPY", lw=1.2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Equity curve (log scale)")
    ax.legend()
    dd = daily["equity"] / daily["equity"].cummax() - 1
    axes[1].fill_between(daily.index, dd, 0, alpha=0.6)
    axes[1].set_title("Drawdown")
    axes[2].plot(daily.index, daily["gross_exposure"], lw=0.8)
    axes[2].set_title("Gross exposure (regime-scaled)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "backtest_charts.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    deciles.plot(kind="bar", ax=ax)
    ax.set_title("Mean forward 20d return by score decile")
    ax.set_xlabel("decile (10 = highest score)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "decile_returns.png", dpi=120)
    plt.close(fig)
    return text
