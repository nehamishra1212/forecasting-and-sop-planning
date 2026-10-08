"""
Rolling-origin backtest, accuracy metrics, forecast-value-added, prediction intervals and an
inventory / service-level simulation that translates forecast error into business impact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import CORE_COLS, EXOG_COLS, EXOG_GROUPS, HORIZONS, Data, build_feature_table
from models import fit_predict_gbm, stat_models_at_origin

VAL_START, TEST_START = 48, 60          # target month indices: 2024-01 (validation year), 2025-01 (test year)
LAST = 71                               # 2025-12


# ================================================================================================
# 1. Rolling-origin backtest
# ================================================================================================
def run_backtest(D: Data, F: pd.DataFrame, horizons=HORIZONS, verbose: bool = True) -> dict:
    """
    For every forecast origin O, using ONLY data available at O, forecast month O+h for h in horizons.
    Targets cover 2024 (validation year) and 2025 (test year).
    """
    S = len(D.skus)
    origins = sorted({T - h for h in horizons for T in range(VAL_START, LAST + 1)})
    recs, quant, imps, classes_by_origin = [], [], [], {}
    full_cols = CORE_COLS + EXOG_COLS
    for n, O in enumerate(origins):
        if verbose and n % 5 == 0:
            print(f"  origin {n + 1}/{len(origins)}  ({D.dates[O].strftime('%Y-%m')})", flush=True)
        st = stat_models_at_origin(D, O, H=max(horizons))
        classes_by_origin[O] = st["classes"]
        for h in horizons:
            T = O + h
            if not (VAL_START <= T <= LAST):
                continue
            act = D.Y[T]
            scale = F[(F["o"] == O) & (F["h"] == h)].sort_values("j")["scale"].to_numpy()
            for name in ("naive", "snaive", "ma3", "stat"):
                recs.append(pd.DataFrame(dict(model=name, h=h, O=O, T=T, j=np.arange(S), forecast=st[name][:, h - 1], actual=act, scale=scale)))
            if h == 3:
                recs.append(pd.DataFrame(dict(model="sales_fc", h=h, O=O, T=T, j=np.arange(S), forecast=D.SALES_FC[T], actual=act, scale=scale)))
            for name, cols in (("gbm_core", CORE_COLS), ("gbm_full", full_cols)):
                r = fit_predict_gbm(F, O, h, cols, D.meta["asp"])
                recs.append(pd.DataFrame(dict(model=name, h=h, O=O, T=T, j=r["j"].to_numpy(), forecast=r["pred_ratio"].to_numpy() * r["scale"].to_numpy(),
                                              actual=r["actual"].to_numpy(), scale=r["scale"].to_numpy())))
                if name == "gbm_full" and O == LAST - 3 and h == 3:
                    imps.append(pd.Series(r.attrs["model"].booster_.feature_importance("gain"), index=cols))
            if h == 3:                                     # prediction intervals (P10 / P90) for the S&OP horizon
                lo = fit_predict_gbm(F, O, h, full_cols, D.meta["asp"], objective="quantile", alpha=0.10)
                hi = fit_predict_gbm(F, O, h, full_cols, D.meta["asp"], objective="quantile", alpha=0.90)
                quant.append(pd.DataFrame(dict(O=O, T=T, j=lo["j"].to_numpy(), scale=lo["scale"].to_numpy(), actual=lo["actual"].to_numpy(),
                                               lo_ratio=lo["pred_ratio"].to_numpy(), hi_ratio=hi["pred_ratio"].to_numpy())))
    fc = pd.concat(recs, ignore_index=True)
    fc["sku"] = [D.skus[j] for j in fc["j"]]
    fc["line"] = [D.lines[D.meta["line_idx"][j]] for j in fc["j"]]
    fc["asp"] = D.meta["asp"][fc["j"].to_numpy()]
    fc["cost"] = D.meta["cost"][fc["j"].to_numpy()]
    cls = pd.DataFrame([(O, j, c) for O, arr in classes_by_origin.items() for j, c in enumerate(arr)], columns=["O", "j", "demand_class"])
    fc = fc.merge(cls, on=["O", "j"], how="left")
    q = pd.concat(quant, ignore_index=True) if quant else pd.DataFrame()
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else pd.Series(dtype=float)
    return dict(fc=fc, quant=q, importance=imp)


def add_blend(fc: pd.DataFrame, w: float) -> pd.DataFrame:
    a = fc[fc["model"] == "gbm_full"].set_index(["h", "O", "T", "j"])
    b = fc[fc["model"] == "stat"].set_index(["h", "O", "T", "j"])
    out = a.copy()
    out["forecast"] = w * a["forecast"] + (1 - w) * b["forecast"]
    out["model"] = "blend"
    return pd.concat([fc, out.reset_index()], ignore_index=True)


def select_blend_weight(fc: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Choose gbm-vs-stat blend weight on the 2024 VALIDATION year only (h = 3). The 2025 test year is untouched."""
    a = fc[(fc["model"] == "gbm_full") & (fc["h"] == 3) & (fc["T"] < TEST_START)].set_index(["O", "T", "j"])
    b = fc[(fc["model"] == "stat") & (fc["h"] == 3) & (fc["T"] < TEST_START)].set_index(["O", "T", "j"])
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        f = w * a["forecast"] + (1 - w) * b["forecast"]
        rows.append(dict(weight_gbm=w, val_wmape_rev=wmape(a["actual"], f, a["asp"])))
    grid = pd.DataFrame(rows)
    return float(grid.loc[grid["val_wmape_rev"].idxmin(), "weight_gbm"]), grid


def select_consensus_weight(fc: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Weight on the statistical GBM in the consensus plan (rest = commercial / sales input). Chosen on 2024 validation only."""
    a = fc[(fc["model"] == "gbm_full") & (fc["h"] == 3) & (fc["T"] < TEST_START)].set_index(["O", "T", "j"])
    b = fc[(fc["model"] == "sales_fc") & (fc["h"] == 3) & (fc["T"] < TEST_START)].set_index(["O", "T", "j"]).reindex(a.index)
    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        f = w * a["forecast"] + (1 - w) * b["forecast"]
        rows.append(dict(weight_stat_model=w, val_wmape_rev=wmape(a["actual"], f, a["asp"])))
    grid = pd.DataFrame(rows)
    return float(grid.loc[grid["val_wmape_rev"].idxmin(), "weight_stat_model"]), grid


def add_consensus(fc: pd.DataFrame, w: float) -> pd.DataFrame:
    """Consensus demand plan (h = 3): statistical GBM forecast blended with the sales-submitted forecast."""
    a = fc[(fc["model"] == "gbm_full") & (fc["h"] == 3)].set_index(["h", "O", "T", "j"])
    b = fc[(fc["model"] == "sales_fc") & (fc["h"] == 3)].set_index(["h", "O", "T", "j"]).reindex(a.index)
    out = a.copy()
    out["forecast"] = w * a["forecast"] + (1 - w) * b["forecast"]
    out["model"] = "consensus"
    return pd.concat([fc, out.reset_index()], ignore_index=True)


# ================================================================================================
# 2. Metrics
# ================================================================================================
def wmape(actual, fcst, w=None) -> float:
    a, f = np.asarray(actual, float), np.asarray(fcst, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    return float((w * np.abs(f - a)).sum() / max((w * a).sum(), 1e-9))


def bias(actual, fcst, w=None) -> float:
    a, f = np.asarray(actual, float), np.asarray(fcst, float)
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    return float((w * (f - a)).sum() / max((w * a).sum(), 1e-9))


def mape(actual, fcst) -> float:
    a, f = np.asarray(actual, float), np.asarray(fcst, float)
    m = a > 0
    return float(np.mean(np.abs(f[m] - a[m]) / a[m])) if m.any() else np.nan


def accuracy_table(df: pd.DataFrame, by: list[str] | None = None, level: str = "sku") -> pd.DataFrame:
    """
    WMAPE / bias at SKU-month, line-month or total-month level. All headline numbers are REVENUE-weighted
    (units x ASP) so a lost server order counts for more than a lost cable.
    """
    by = by or []
    d = df.copy()
    d["A_rev"], d["F_rev"] = d["actual"] * d["asp"], d["forecast"] * d["asp"]
    out = []
    for key, g in d.groupby(by + ["model"] if by else ["model"]):
        key = key if isinstance(key, tuple) else (key,)
        if level == "line":
            g = g.groupby(["T", "line"]).agg(A_rev=("A_rev", "sum"), F_rev=("F_rev", "sum")).reset_index()
        elif level == "total":
            g = g.groupby(["T"]).agg(A_rev=("A_rev", "sum"), F_rev=("F_rev", "sum")).reset_index()
        row = dict(zip(by + ["model"], key))
        row.update(wmape_rev=wmape(g["A_rev"], g["F_rev"]), bias_rev=bias(g["A_rev"], g["F_rev"]), n=len(g))
        if level == "sku":
            row.update(wmape_units=wmape(g["actual"], g["forecast"]), mape_nonzero=mape(g["actual"], g["forecast"]))
        out.append(row)
    return pd.DataFrame(out)


def add_fva(tab: pd.DataFrame, baseline: str, by: list[str] | None = None) -> pd.DataFrame:
    """Forecast Value Added = baseline WMAPE - model WMAPE (positive = model beats the baseline)."""
    by = by or []
    t = tab.copy()
    base = t[t["model"] == baseline].set_index(by)["wmape_rev"] if by else pd.Series({0: t.loc[t["model"] == baseline, "wmape_rev"].iloc[0]})
    if by:
        t["fva_vs_" + baseline + "_pp"] = [100 * (base.loc[tuple(r[b] for b in by) if len(by) > 1 else r[by[0]]] - r["wmape_rev"]) for _, r in t.iterrows()]
    else:
        t["fva_vs_" + baseline + "_pp"] = 100 * (base.iloc[0] - t["wmape_rev"])
    return t


def tracking_signal(df: pd.DataFrame, model: str, h: int = 3) -> pd.DataFrame:
    """Cumulative forecast error / mean absolute deviation by product line (S&OP exception trigger: |TS| > 4)."""
    d = df[(df["model"] == model) & (df["h"] == h) & (df["T"] >= TEST_START)].copy()
    d["err"] = (d["forecast"] - d["actual"]) * d["asp"]
    g = d.groupby(["line", "T"])["err"].sum().reset_index().sort_values(["line", "T"])
    g["cum_err"] = g.groupby("line")["err"].cumsum()
    g["mad"] = g.groupby("line")["err"].transform(lambda s: s.abs().expanding().mean())
    g["tracking_signal"] = g["cum_err"] / g["mad"].replace(0, np.nan)
    return g


def exception_report(df: pd.DataFrame, model: str, h: int = 3, top: int = 10) -> pd.DataFrame:
    """Exception-based review list: SKUs that contribute most to total absolute revenue error in the test year."""
    d = df[(df["model"] == model) & (df["h"] == h) & (df["T"] >= TEST_START)].copy()
    d["abs_err_rev"] = np.abs(d["forecast"] - d["actual"]) * d["asp"]
    d["err_rev"] = (d["forecast"] - d["actual"]) * d["asp"]
    d["act_rev"] = d["actual"] * d["asp"]
    g = d.groupby(["sku", "line"]).agg(abs_err_rev=("abs_err_rev", "sum"), err_rev=("err_rev", "sum"), act_rev=("act_rev", "sum")).reset_index()
    g["wmape"] = g["abs_err_rev"] / g["act_rev"]
    g["bias"] = g["err_rev"] / g["act_rev"]
    g["share_of_total_error"] = g["abs_err_rev"] / g["abs_err_rev"].sum()
    return g.sort_values("abs_err_rev", ascending=False).head(top).reset_index(drop=True)


# ================================================================================================
# 3. Prediction intervals (conformalised quantile regression) -> uncertainty for S&OP buffers
# ================================================================================================
def intervals(q: pd.DataFrame, nominal: float = 0.80) -> pd.DataFrame:
    """CQR: widen raw P10/P90 by the (1-alpha) quantile of validation conformity scores realised at each origin."""
    q = q.copy()
    q["y"] = q["actual"] / q["scale"]
    q["score"] = np.maximum(q["lo_ratio"] - q["y"], q["y"] - q["hi_ratio"])
    qh = []
    for _, r in q.iterrows():
        cal = q[(q["T"] >= VAL_START) & (q["T"] <= r["O"])]            # only errors already realised at the origin
        qh.append(np.quantile(cal["score"], min(1, nominal * (1 + 1 / max(len(cal), 1)))) if len(cal) >= 20 else 0.0)
    q["qhat"] = qh
    q["lo"] = np.clip(q["lo_ratio"] - q["qhat"], 0, None) * q["scale"]
    q["hi"] = (q["hi_ratio"] + q["qhat"]) * q["scale"]
    q["lo_raw"], q["hi_raw"] = q["lo_ratio"] * q["scale"], q["hi_ratio"] * q["scale"]
    return q


def coverage_table(q: pd.DataFrame) -> pd.DataFrame:
    t = q[q["T"] >= TEST_START]
    return pd.DataFrame([
        dict(interval="Raw quantile model (nominal 80%)", coverage=((t["actual"] >= t["lo_raw"]) & (t["actual"] <= t["hi_raw"])).mean(),
             avg_width_ratio=((t["hi_raw"] - t["lo_raw"]) / t["scale"]).mean()),
        dict(interval="Conformalised (nominal 80%)", coverage=((t["actual"] >= t["lo"]) & (t["actual"] <= t["hi"])).mean(),
             avg_width_ratio=((t["hi"] - t["lo"]) / t["scale"]).mean()),
    ])


# ================================================================================================
# 4. Inventory / service simulation: what does forecast error cost the business?
# ================================================================================================
def inventory_simulation(fc: pd.DataFrame, models: list[str], z: float = 1.28, carry_rate: float = 0.25,
                         margin_by_sku: bool = True) -> pd.DataFrame:
    """
    For each month m in the test year, supply is committed 3 months earlier using that month's h=3 forecast plus
    safety stock z * sigma (sigma = trailing RMSE of the same model's already-realised errors).
    Leftover inventory carries forward and offsets the next commitment; unmet demand is lost.

    ASSUMPTIONS (deliberately simple, identical for every model):
      * 90% cycle-service-level target (z = 1.28), 25%/yr cost of carrying inventory, lost sales cost the gross margin.
    """
    res = []
    for model in models:
        d = fc[(fc["model"] == model) & (fc["h"] == 3)].sort_values(["j", "T"])
        tot = dict(model=model, demand_rev=0.0, fulfilled_rev=0.0, lost_margin=0.0, inv_cost_sum=0.0, inv_units_sum=0.0, demand_units=0.0, n_months=0)
        for j, g in d.groupby("j"):
            g = g.set_index("T")
            left = 0.0
            asp, cost = g["asp"].iloc[0], g["cost"].iloc[0]
            for T in range(TEST_START, LAST + 1):
                O = T - 3
                hist = g[(g.index <= O)]                              # errors realised by planning time
                hist = hist.tail(12)
                sigma = np.sqrt(((hist["forecast"] - hist["actual"]) ** 2).mean()) if len(hist) >= 3 else 0.35 * g.loc[T, "forecast"]
                f = g.loc[T, "forecast"]
                receipts = max(0.0, f + z * sigma - left)
                avail = left + receipts
                a = g.loc[T, "actual"]
                ful = min(a, avail)
                left = avail - ful
                tot["demand_rev"] += a * asp
                tot["fulfilled_rev"] += ful * asp
                tot["lost_margin"] += (a - ful) * (asp - cost)
                tot["inv_cost_sum"] += left * cost
                tot["inv_units_sum"] += left
                tot["demand_units"] += a
        n_sku = d["j"].nunique()
        months = LAST - TEST_START + 1
        tot["fill_rate_rev"] = tot["fulfilled_rev"] / tot["demand_rev"]
        tot["avg_inventory_usd_m"] = tot["inv_cost_sum"] / months / 1e6
        tot["lost_sales_usd_m"] = (tot["demand_rev"] - tot["fulfilled_rev"]) / 1e6
        tot["carrying_cost_usd_m"] = tot["avg_inventory_usd_m"] * carry_rate
        tot["lost_margin_usd_m"] = tot["lost_margin"] / 1e6
        tot["total_cost_usd_m"] = tot["carrying_cost_usd_m"] + tot["lost_margin_usd_m"]
        tot["weeks_of_supply"] = (tot["inv_units_sum"] / months) / ((tot["demand_units"] / months) / 4.33)
        res.append(tot)
    out = pd.DataFrame(res)
    out["z"] = z
    return out[["model", "z", "fill_rate_rev", "avg_inventory_usd_m", "weeks_of_supply", "lost_sales_usd_m", "carrying_cost_usd_m", "lost_margin_usd_m", "total_cost_usd_m"]]


def inventory_frontier(fc: pd.DataFrame, models: list[str], zs=tuple(np.round(np.arange(0.0, 3.51, 0.25), 2)), **kw) -> pd.DataFrame:
    """Service-vs-inventory curve per model: sweep the safety-stock multiplier z so models can be compared at EQUAL service."""
    return pd.concat([inventory_simulation(fc, models, z=z, **kw) for z in zs], ignore_index=True)


def inventory_at_service(frontier: pd.DataFrame, target: float = 0.95, carry_rate: float = 0.25) -> pd.DataFrame:
    """Average inventory each model needs to reach the target revenue fill rate (linear interpolation along its frontier)."""
    rows = []
    for m, g in frontier.groupby("model"):
        g = g.sort_values("fill_rate_rev")
        fr = g["fill_rate_rev"].to_numpy()
        ok = fr.max() >= target >= fr.min()
        inv = float(np.interp(target, fr, g["avg_inventory_usd_m"])) if ok else np.nan
        wos = float(np.interp(target, fr, g["weeks_of_supply"])) if ok else np.nan
        rows.append(dict(model=m, target_fill_rate=target, avg_inventory_usd_m=inv, weeks_of_supply=wos, carrying_cost_usd_m=inv * carry_rate))
    t = pd.DataFrame(rows)
    if "sales_fc" in set(t["model"]):
        base = t.set_index("model").loc["sales_fc", "avg_inventory_usd_m"]
        t["inventory_vs_sales_fc_pct"] = 100 * (t["avg_inventory_usd_m"] / base - 1)
    return t.sort_values("avg_inventory_usd_m").reset_index(drop=True)


def paired_bootstrap(fc: pd.DataFrame, model: str, rival: str, h: int = 3, n: int = 5000, seed: int = 1) -> dict:
    """
    Is the accuracy gap real or luck? Resample the 12 TEST MONTHS with replacement (months, not SKUs, because errors within
    a month are correlated) and recompute the revenue-weighted WMAPE difference. Returns mean gap (pp) and a 95% interval.
    """
    d = fc[(fc["h"] == h) & (fc["T"] >= TEST_START) & fc["model"].isin([model, rival])].copy()
    d["ae"] = np.abs(d["forecast"] - d["actual"]) * d["asp"]
    d["a"] = d["actual"] * d["asp"]
    piv = d.pivot_table(index="T", columns="model", values="ae", aggfunc="sum")
    den = d[d["model"] == model].groupby("T")["a"].sum()
    rng = np.random.default_rng(seed)
    months = piv.index.to_numpy()
    gaps = []
    for _ in range(n):
        pick = rng.choice(months, len(months), replace=True)
        gaps.append(100 * (piv.loc[pick, rival].sum() - piv.loc[pick, model].sum()) / den.loc[pick].sum())
    gaps = np.array(gaps)
    return dict(model=model, rival=rival, gap_pp_mean=float(gaps.mean()), ci95_low=float(np.quantile(gaps, 0.025)),
                ci95_high=float(np.quantile(gaps, 0.975)), prob_model_better=float((gaps > 0).mean()))


# ================================================================================================
# 5. Ablation: which input groups add accuracy? (h = 3, 2025 test year, GBM)
# ================================================================================================
def ablation(D: Data, F: pd.DataFrame, verbose: bool = True, seeds=(7, 11, 23)) -> pd.DataFrame:
    full = CORE_COLS + EXOG_COLS
    variants = {"History + calendar + lifecycle only (no external inputs)": CORE_COLS, "All inputs": full}
    for g, cols in EXOG_GROUPS.items():
        variants[f"All inputs minus: {g}"] = [c for c in full if c not in cols]
    rows = []
    for name, cols in variants.items():
        if verbose:
            print(f"  ablation: {name}", flush=True)
        recs = []
        for T in range(TEST_START, LAST + 1):
            preds = [fit_predict_gbm(F, T - 3, 3, cols, D.meta["asp"], seed=sd) for sd in seeds]       # seed-averaged to reduce noise
            r = preds[0]
            fcst = np.mean([p["pred_ratio"].to_numpy() for p in preds], axis=0) * r["scale"].to_numpy()
            recs.append(pd.DataFrame(dict(actual=r["actual"].to_numpy(), forecast=fcst, asp=D.meta["asp"][r["j"].to_numpy()])))
        d = pd.concat(recs)
        rows.append(dict(variant=name, wmape_rev=wmape(d["actual"] * d["asp"], d["forecast"] * d["asp"]), bias_rev=bias(d["actual"] * d["asp"], d["forecast"] * d["asp"])))
    t = pd.DataFrame(rows)
    ref = t.loc[t["variant"] == "All inputs", "wmape_rev"].iloc[0]
    t["delta_vs_all_pp"] = 100 * (t["wmape_rev"] - ref)
    return t
