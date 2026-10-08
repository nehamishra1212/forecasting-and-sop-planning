"""Figures for the Forecasting & S&OP Planning project. Every function saves a PNG and returns the path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import CORE_COLS, EXOG_COLS, EXOG_GROUPS, Data, rolling_mean
from models import demand_class

COL = dict(consensus="#0b4f8a", gbm="#1b7f5c", sales="#e08a1e", naive="#8a8f98", stat="#3a6ea5", actual="#222222", core="#7ab8a0", bad="#c0392b")
LABEL = dict(naive="Naive (last month)", snaive="Seasonal naive", ma3="3-month avg", stat="Auto-ETS / Croston-SBA", sales_fc="Sales-submitted forecast",
             gbm_core="GBM (history + calendar + lifecycle)", gbm_full="Statistical GBM (all inputs)", blend="Blend", consensus="Consensus (GBM + sales input)  <- proposed")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ EDA ---------------------------------------------------
def fig_history(D: Data, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    for ax, (li, line) in zip(axes.ravel(), enumerate(D.lines)):
        idx = np.flatnonzero(D.meta["line_idx"] == li)
        y = D.Y[:, idx].sum(1)
        ax.plot(D.dates, y, color="#9bb8d3", lw=1.2, label="Monthly orders")
        ax.plot(D.dates, pd.Series(y).rolling(12, min_periods=6).mean(), color="#1f4e79", lw=2, label="12-mo average")
        ax.axvspan(D.dates[16], D.dates[34], color="#e0a030", alpha=0.15, label="Component shortage")
        ax.set_title(line)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Orders (unconstrained demand) by product line, 2020-2025 - synthetic data", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def seasonal_index(D: Data) -> pd.DataFrame:
    out = {}
    for li, line in enumerate(D.lines):
        idx = np.flatnonzero(D.meta["line_idx"] == li)
        y = pd.Series(D.Y[:, idx].sum(1), index=D.dates)
        ratio = y / y.rolling(12, center=True, min_periods=12).mean()
        out[line] = ratio.groupby(ratio.index.month).mean()
    return pd.DataFrame(out)


def fig_seasonality(D: Data, path):
    si = seasonal_index(D)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(si.T.values, cmap="RdYlGn", vmin=0.75, vmax=1.35, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(si.columns)))
    ax.set_yticklabels(si.columns)
    ax.grid(False)
    for i in range(si.shape[1]):
        for j in range(12):
            ax.text(j, i, f"{si.iloc[j, i]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Seasonal index by month (1.00 = average month): enterprise lines peak at quarter-end, consumer lines at back-to-school & holidays")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, path)


def fig_constraint(D: Data, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, line in zip(axes, ["Client PCs", "Servers"]):
        li = D.lines.index(line)
        idx = np.flatnonzero(D.meta["line_idx"] == li)
        ax.plot(D.dates, D.Y[:, idx].sum(1), color=COL["actual"], label="Orders (demand)")
        ax.plot(D.dates, D.SHIP[:, idx].sum(1), color=COL["stat"], label="Shipments (supply-constrained)")
        ax2 = ax.twinx()
        ax2.fill_between(D.dates, D.BACKLOG[:, idx].sum(1), color="#e0a030", alpha=0.25, label="Backlog (right axis)")
        ax2.grid(False)
        ax.axvspan(D.dates[16], D.dates[34], color="#e0a030", alpha=0.08)
        ax.set_title(f"{line}: orders vs shipments")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Why we forecast ORDERS, not shipments: shipments were capped by supply in 2021-22 and would understate demand", fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, path)


def fig_drivers(D: Data, path):
    li = D.lines.index("Servers")
    L = D.line_arr
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.8), sharex=True)
    series = [("Sales pipeline index (leading indicator)", L["li"][:, li]), ("Our price / competitor price", L["rel_px"][:, li]),
              ("Competitor share % (quarterly, 1-qtr lag)", L["comp_share"][:, li]), ("IT capex index (macro)", L["macro"][:, li]),
              ("Memory / commodity price index", L["commodity"][:, li]), ("Supplier lead time (weeks)", L["lead"][:, li])]
    for ax, (t, y) in zip(axes.ravel(), series):
        ax.plot(D.dates, y, color="#1f4e79")
        ax.set_title(t, fontsize=9)
    fig.suptitle("Example external & operational inputs for the Servers line (all synthetic)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def abc_xyz(D: Data, o: int = 59) -> pd.DataFrame:
    """ABC by 2024 revenue share; XYZ by coefficient of variation of the last 12 months (post-launch)."""
    rows = []
    for j, s in enumerate(D.skus):
        y = D.Y[o - 11: o + 1, j]
        rows.append(dict(sku=s, line=D.lines[D.meta["line_idx"][j]], revenue=(y * D.meta["asp"][j]).sum(),
                         cv=(y.std() / y.mean()) if y.mean() > 0 else np.nan, cls=demand_class(D.Y[: o + 1, j])))
    t = pd.DataFrame(rows).sort_values("revenue", ascending=False)
    t["cum_share"] = t["revenue"].cumsum() / t["revenue"].sum()
    t["abc"] = np.where(t["cum_share"] <= 0.80, "A", np.where(t["cum_share"] <= 0.95, "B", "C"))
    t["xyz"] = np.where(t["cv"] < 0.30, "X", np.where(t["cv"] < 0.60, "Y", "Z"))
    return t


def fig_abc_xyz(D: Data, path):
    t = abc_xyz(D)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    ct = pd.crosstab(t["abc"], t["xyz"]).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    ax = axes[0]
    ax.imshow(ct.values, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels(["X (stable)", "Y (variable)", "Z (volatile)"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["A (80% revenue)", "B (next 15%)", "C (last 5%)"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("ABC / XYZ segmentation of the 36 SKUs (as of Dec-2024)")
    ax = axes[1]
    tt = t.sort_values("revenue", ascending=False)
    ax.bar(range(len(tt)), tt["revenue"] / 1e6, color=[{"A": "#1b7f5c", "B": "#e0a030", "C": "#9aa3ad"}[a] for a in tt["abc"]])
    ax.set_xticks(range(len(tt))); ax.set_xticklabels(tt["sku"], rotation=90, fontsize=6)
    ax.set_ylabel("Trailing-12m revenue ($M)")
    ax.set_title("Revenue is highly concentrated: few SKUs drive most of the plan")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------- model results ---------------------------------------------
def fig_total_backtest(fc, D, path, h=3):
    t = fc[(fc["h"] == h) & (fc["T"] >= 60)].copy()
    t["A"], t["F"] = t["actual"] * t["asp"], t["forecast"] * t["asp"]
    piv = t.groupby(["model", "T"])["F"].sum().unstack(0) / 1e6
    act = t[t["model"] == "gbm_full"].groupby("T")["A"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x = [D.dates[i] for i in act.index]
    ax.plot(x, act.values, color=COL["actual"], lw=2.5, marker="o", label="Actual orders ($M)")
    for m, c in [("consensus", COL["consensus"]), ("gbm_full", COL["gbm"]), ("sales_fc", COL["sales"]), ("naive", COL["naive"])]:
        ax.plot(x, piv[m].values, color=c, lw=2.2 if m == "consensus" else 1.6, ls="-" if m in ("consensus", "gbm_full") else "--", label=LABEL[m].replace("  <- proposed", ""))
    ax.set_ylabel("Total orders, $M"); ax.set_title(f"2025 backtest: total revenue forecast made {h} months ahead (each month forecast with data available then)")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, path)


def fig_line_backtest(fc, D, path, h=3):
    t = fc[(fc["h"] == h) & (fc["T"] >= 60)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    for ax, line in zip(axes.ravel(), D.lines):
        g = t[t["line"] == line]
        x = [D.dates[i] for i in sorted(g["T"].unique())]
        ax.plot(x, g[g["model"] == "gbm_full"].groupby("T")["actual"].sum().values, color=COL["actual"], lw=2.4, marker="o", ms=3, label="Actual")
        ax.plot(x, g[g["model"] == "gbm_full"].groupby("T")["forecast"].sum().values, color=COL["gbm"], lw=1.8, label="GBM")
        ax.plot(x, g[g["model"] == "sales_fc"].groupby("T")["forecast"].sum().values, color=COL["sales"], lw=1.5, ls="--", label="Sales forecast")
        ax.set_title(line)
        ax.tick_params(axis="x", rotation=45)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("2025 backtest by product line (units, forecast made 3 months ahead)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def fig_model_comparison(acc_sku, acc_line, acc_total, path):
    order = ["snaive", "naive", "ma3", "stat", "sales_fc", "gbm_core", "gbm_full", "consensus"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    for ax, (name, tab) in zip(axes, [("SKU-month level (where supply plans are made)", acc_sku), ("Product-line level", acc_line), ("Total company", acc_total)]):
        t = tab.set_index("model").reindex(order)
        colors = [COL["consensus"] if m == "consensus" else COL["gbm"] if m == "gbm_full" else COL["core"] if m == "gbm_core" else COL["sales"] if m == "sales_fc" else COL["stat"] if m == "stat" else COL["naive"] for m in order]
        ax.barh([LABEL[m].replace("  <- proposed", "") for m in order], t["wmape_rev"] * 100, color=colors)
        for i, (v, b) in enumerate(zip(t["wmape_rev"], t["bias_rev"])):
            ax.text(v * 100 + 0.5, i, f"{v * 100:.1f}%  (bias {b * 100:+.1f}%)", va="center", fontsize=7.5)
        ax.set_xlim(0, 75)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("Revenue-weighted WMAPE (lower is better)")
    fig.suptitle("Forecast accuracy, 2025 test year, 3-month horizon", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def fig_horizon(acc_h, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m, c in [("naive", COL["naive"]), ("stat", COL["stat"]), ("gbm_core", COL["core"]), ("gbm_full", COL["gbm"])]:
        t = acc_h[acc_h["model"] == m].sort_values("h")
        ax.plot(t["h"], t["wmape_rev"] * 100, marker="o", color=c, lw=2, label=LABEL[m].replace("  <- proposed", ""))
    ax.set_xticks([1, 3, 6]); ax.set_xlabel("Forecast horizon (months ahead)"); ax.set_ylabel("WMAPE %, revenue-weighted")
    ax.set_title("Accuracy decays with horizon - and external inputs help most at short horizons")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_segments(acc_line, acc_life, acc_abc, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (key, tab, title) in zip(axes, [("line", acc_line, "By product line"), ("life", acc_life, "By lifecycle stage"), ("abc", acc_abc, "By ABC class")]):
        p = tab[tab["model"].isin(["naive", "sales_fc", "gbm_full"])].pivot(index=key, columns="model", values="wmape_rev") * 100
        p = p[["naive", "sales_fc", "gbm_full"]]
        p.plot.bar(ax=ax, color=[COL["naive"], COL["sales"], COL["gbm"]], width=0.8)
        ax.set_title(title); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
        ax.legend(["Naive", "Sales forecast", "GBM"], fontsize=7)
    axes[0].set_ylabel("WMAPE %")
    fig.suptitle("Where does the model win - and where does it not? (2025, h=3)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def fig_importance(imp: pd.Series, path):
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values / imp.sum() * 100, color=[COL["gbm"] if i in EXOG_COLS else COL["core"] for i in top.index])
    ax.set_xlabel("% of total split gain"); ax.set_title("Top drivers in the GBM (dark = external input, light = history/calendar/lifecycle)")
    return _save(fig, path)


def fig_ablation(ab, path):
    t = ab[ab["variant"].str.startswith("All inputs minus")].copy()
    t["variant"] = t["variant"].str.replace("All inputs minus: ", "")
    t = t.sort_values("delta_vs_all_pp")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(t["variant"], t["delta_vs_all_pp"], color=[COL["bad"] if v > 0 else COL["naive"] for v in t["delta_vs_all_pp"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Change in WMAPE (pp) when the input group is REMOVED (positive = the input was helping)")
    ax.set_title("Which inputs earn their keep? (2025, h=3)")
    return _save(fig, path)


def fig_interval(q, fc, D, sku, path):
    j = D.skus.index(sku)
    t = q[(q["j"] == j) & (q["T"] >= 60)].sort_values("T")
    f = fc[(fc["model"] == "gbm_full") & (fc["h"] == 3) & (fc["j"] == j) & (fc["T"] >= 60)].sort_values("T")
    x = [D.dates[i] for i in t["T"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(x, t["lo"], t["hi"], color=COL["gbm"], alpha=0.18, label="80% prediction interval (conformalised)")
    ax.plot(x, f["forecast"].values, color=COL["gbm"], lw=2, label="GBM forecast (3 mo ahead)")
    ax.plot(x, t["actual"].values, color=COL["actual"], lw=2, marker="o", ms=4, label="Actual orders")
    ax.set_title(f"{sku}: forecast with uncertainty band - use the P90 for supply risk, the P50 for the consensus plan")
    ax.legend(fontsize=8)
    return _save(fig, path)


def fig_inventory(frontier, inv95, path):
    show = ["naive", "stat", "sales_fc", "gbm_full", "consensus"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    cmap = dict(naive=COL["naive"], stat=COL["stat"], sales_fc=COL["sales"], gbm_core=COL["core"], gbm_full=COL["gbm"], consensus=COL["consensus"])
    for m in show:
        g = frontier[frontier["model"] == m].sort_values("avg_inventory_usd_m")
        ax.plot(g["avg_inventory_usd_m"], g["fill_rate_rev"] * 100, color=cmap[m], lw=2.4 if m == "consensus" else 1.5, label=LABEL[m].replace("  <- proposed", ""))
    ax.axhline(95, color="k", ls=":", lw=1)
    ax.set_ylim(80, 100.5)
    ax.set_xlabel("Average inventory on hand ($M at cost)"); ax.set_ylabel("Revenue fill rate %")
    ax.set_title("Service-vs-inventory frontier (up-and-left is better)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    t = inv95[inv95["model"].isin(show)].sort_values("avg_inventory_usd_m")
    ax.barh([LABEL[m].replace("  <- proposed", "") for m in t["model"]], t["avg_inventory_usd_m"], color=[cmap[m] for m in t["model"]])
    for i, (v, w) in enumerate(zip(t["avg_inventory_usd_m"], t["weeks_of_supply"])):
        ax.text(v + 0.5, i, f"${v:.0f}M  ({w:.1f} wks)", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, t["avg_inventory_usd_m"].max() * 1.35)
    ax.set_xlabel("Average inventory ($M) needed to reach a 95% revenue fill rate")
    ax.set_title("Same service, different inventory: what forecast quality is worth")
    fig.tight_layout()
    return _save(fig, path)


def fig_tracking(ts, path):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for line, g in ts.groupby("line"):
        ax.plot(g["T"], g["tracking_signal"], marker="o", ms=3, label=line)
    ax.axhline(4, color=COL["bad"], ls="--", lw=1); ax.axhline(-4, color=COL["bad"], ls="--", lw=1)
    ax.set_xticks(sorted(ts["T"].unique())); ax.set_xticklabels([f"{i - 59:02d}" for i in sorted(ts["T"].unique())])
    ax.set_xlabel("Month of 2025"); ax.set_ylabel("Tracking signal")
    ax.set_title("Tracking signal by line (cumulative error / MAD): |TS| > 4 triggers an S&OP review")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


def fig_fva_waterfall(acc_sku, path):
    """Forecast Value Added: does each step of the process make the forecast better or worse?"""
    t = acc_sku.set_index("model")["wmape_rev"] * 100
    steps = [("Naive\n(last month)", t["naive"]), ("Auto-ETS /\nCroston", t["stat"]), ("GBM: history,\ncalendar, lifecycle", t["gbm_core"]),
             ("GBM + external\ninputs", t["gbm_full"]), ("Consensus:\n+ sales input", t["consensus"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    # ETS and GBM-core are compared with naive; the two later steps with the step before them
    ref = [None, 0, 0, 2, 3]
    for i, (name, v) in enumerate(steps):
        ax.bar(i, v, color=COL["consensus"] if i == len(steps) - 1 else COL["gbm"] if i >= 2 else COL["naive"])
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontweight="bold")
        if ref[i] is not None:
            d = steps[ref[i]][1] - v
            ax.text(i, v / 2, f"{d:+.1f} pp\nvs {'naive' if ref[i] == 0 else 'prior step'}", ha="center", color="white", fontsize=8.5, fontweight="bold")
    ax.axhline(t["sales_fc"], color=COL["sales"], ls="--"); ax.text(len(steps) - 0.5, t["sales_fc"] + 0.5, f"Sales-submitted forecast: {t['sales_fc']:.1f}%", color=COL["sales"], ha="right", fontsize=8)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([n for n, _ in steps], fontsize=8.5)
    ax.set_ylabel("WMAPE %, revenue-weighted (lower is better)")
    ax.set_title("Forecast Value Added (SKU level, 2025, h=3): every step measured, including the ones that did not help", fontsize=9.5)
    return _save(fig, path)
