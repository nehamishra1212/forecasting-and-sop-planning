"""
Data loading and feature engineering.

Design principle: every feature is tagged with WHEN it would be known in a real S&OP cycle.
  * "origin" features  : known at forecast origin `o` (history, supplier signals, competitor data, ...)
  * "plan"   features  : known at `o` for the TARGET month `T = o + h` because they are plans/calendars
                         (promo calendar, marketing budget, product roadmap, announced events)
Nothing observed after `o` is ever used. That is what makes the backtest honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONSUMER_LINES = {"Client PCs", "Accessories & Peripherals"}
HORIZONS = (1, 3, 6)          # months ahead. h=3 is the "S&OP cycle" horizon (build decisions)
MIN_ORIGIN = 12               # need a full year of history for lag-12 / rolling-12 features

CORE_COLS = [
    "h", "lag0", "lag1", "lag2", "lag5", "seas12", "seas24", "m3", "m6", "m12", "sd6", "mom", "yoy",
    "grp_mom", "line_mom", "month_sin", "month_cos", "is_qtr_end", "age_T", "to_eol_T", "gen", "tier", "line_code", "hist_len",
]
EXOG_COLS = [
    "promo_T", "depth_T", "promo_o", "mkt_ratio", "refresh_T",
    "li_o", "li_chg3", "rel_px_o", "rel_px_chg3", "comp_share_o", "comp_launch_3m",
    "macro_o", "macro_chg3", "commodity_o", "lead_o", "alloc_o", "otd_o", "yield_o", "backlog_ratio",
]
EXOG_GROUPS = {
    "Promotions & pricing plan": ["promo_T", "depth_T", "promo_o"],
    "Marketing plan": ["mkt_ratio"],
    "Known events": ["refresh_T"],
    "Leading indicator (pipeline / traffic)": ["li_o", "li_chg3"],
    "Competitor": ["rel_px_o", "rel_px_chg3", "comp_share_o", "comp_launch_3m"],
    "Macro & commodity": ["macro_o", "macro_chg3", "commodity_o"],
    "Supply chain & operations": ["lead_o", "alloc_o", "otd_o", "yield_o", "backlog_ratio"],
}


@dataclass
class Data:
    dates: pd.DatetimeIndex
    skus: list
    sku_master: pd.DataFrame
    lines: list
    Y: np.ndarray                  # orders (T x S)  -> forecast target (unconstrained demand)
    SHIP: np.ndarray
    BACKLOG: np.ndarray
    SALES_FC: np.ndarray           # legacy sales-submitted forecast made 3 months ahead
    PROMO: np.ndarray
    DEPTH: np.ndarray
    line_arr: dict                 # name -> (T x L) arrays
    meta: dict                     # per-SKU arrays (line idx, tier idx, gen, launch, eol, asp, cost)


def load_data(data_dir: str | Path = "data") -> Data:
    d = Path(data_dir)
    panel = pd.read_csv(d / "processed" / "panel.csv", parse_dates=["date"])
    sku_master = pd.read_csv(d / "raw" / "sku_master.csv")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    skus = sku_master["sku"].tolist()
    lines = list(dict.fromkeys(sku_master["product_line"]))
    T, S, L = len(dates), len(skus), len(lines)

    def pivot(col):
        return panel.pivot(index="date", columns="sku", values=col).reindex(index=dates, columns=skus).to_numpy(float)

    def pivot_line(col):
        sub = panel.drop_duplicates(["date", "product_line"])
        return sub.pivot(index="date", columns="product_line", values=col).reindex(index=dates, columns=lines).to_numpy(float)

    macro = np.where(np.array([l in CONSUMER_LINES for l in lines])[None, :], pivot_line("consumer_demand_index"), pivot_line("it_capex_index"))
    line_arr = dict(
        li=pivot_line("leading_indicator_index"), refresh=pivot_line("refresh_event_index"),
        mkt_plan=pivot_line("marketing_spend_plan_usd_k"), rel_px=pivot_line("relative_price"),
        comp_share=pivot_line("competitor_share_pct_observed"), comp_launch=pivot_line("competitor_launch_flag"),
        macro=macro, commodity=pivot_line("commodity_price_index"), lead=pivot_line("supplier_lead_time_weeks"),
        alloc=pivot_line("component_allocation_pct"), otd=pivot_line("supplier_otd_pct"), yld=pivot_line("factory_yield_pct"),
    )
    meta = dict(
        line_idx=np.array([lines.index(x) for x in sku_master["product_line"]]),
        tier_idx=np.array([["Entry", "Mainstream", "Premium"].index(x) for x in sku_master["tier"]]),
        gen=sku_master["generation"].to_numpy(),
        launch=sku_master["launch_idx"].to_numpy(),
        eol=sku_master["eol_idx"].to_numpy(),
        asp=sku_master["asp_usd"].to_numpy(float),
        cost=sku_master["unit_cost_usd"].to_numpy(float),
    )
    return Data(dates, skus, sku_master, lines, pivot("orders"), pivot("shipments"), pivot("backlog_units"),
                pivot("sales_forecast_3m_ahead"), pivot("promo_flag"), pivot("promo_depth_pct") / 100.0, line_arr, meta)


# ----------------------------------------------------------------------------------------------
def group_ids(D: Data) -> np.ndarray:
    """SKU family = (product line, tier). New-generation SKUs borrow their family's scale."""
    return D.meta["line_idx"] * 3 + D.meta["tier_idx"]


def rolling_mean(Y: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(Y).rolling(w, min_periods=1).mean().to_numpy()


def family_scale(D: Data, o: int) -> np.ndarray:
    """Per-SKU scale at origin o: trailing-12m mean of the whole SKU family (so new SKUs have a scale)."""
    m12 = rolling_mean(D.Y, 12)[o]
    g = group_ids(D)
    fam = pd.Series(m12).groupby(g).transform("sum").to_numpy()
    return np.maximum(fam, 1.0)


def hist_len(D: Data, o: int) -> np.ndarray:
    out = np.zeros(len(D.skus), int)
    for j in range(len(D.skus)):
        nz = np.flatnonzero(D.Y[: o + 1, j] > 0)
        out[j] = 0 if len(nz) == 0 else o - nz[0] + 1
    return out


def build_feature_table(D: Data, horizons=HORIZONS) -> pd.DataFrame:
    """One row per (origin o, horizon h, SKU j). Target y = orders at T=o+h (NaN if unobserved)."""
    T, S = D.Y.shape
    Y = D.Y
    M3, M6, M12 = rolling_mean(Y, 3), rolling_mean(Y, 6), rolling_mean(Y, 12)
    SD6 = pd.DataFrame(Y).rolling(6, min_periods=2).std().to_numpy()
    g = group_ids(D)
    line_idx = D.meta["line_idx"]
    L = D.line_arr
    rows = []
    for o in range(MIN_ORIGIN, T):
        s = family_scale(D, o)
        fam_m3 = pd.Series(M3[o]).groupby(g).transform("sum").to_numpy()
        fam_m12 = pd.Series(M12[o]).groupby(g).transform("sum").to_numpy()
        line_m3 = pd.Series(M3[o]).groupby(line_idx).transform("sum").to_numpy()
        line_m12 = pd.Series(M12[o]).groupby(line_idx).transform("sum").to_numpy()
        hl = hist_len(D, o)
        for h in horizons:
            Tt = o + h
            if Tt >= T + 6:     # never needed
                continue
            for j in range(S):
                l = line_idx[j]
                lag = lambda k: Y[o - k, j] / s[j] if o - k >= 0 else np.nan
                has_T = Tt < T
                r = dict(o=o, h=h, T=Tt, j=j, sku=D.skus[j], line=l, scale=s[j], hist_len=hl[j],
                         y=(Y[Tt, j] / s[j]) if has_T else np.nan, actual=(Y[Tt, j] if has_T else np.nan),
                         lag0=lag(0), lag1=lag(1), lag2=lag(2), lag5=lag(5),
                         seas12=(Y[Tt - 12, j] / s[j]) if Tt - 12 >= 0 else np.nan,
                         seas24=(Y[Tt - 24, j] / s[j]) if Tt - 24 >= 0 else np.nan,
                         m3=M3[o, j] / s[j], m6=M6[o, j] / s[j], m12=M12[o, j] / s[j], sd6=SD6[o, j] / s[j] if not np.isnan(SD6[o, j]) else np.nan,
                         mom=min(M3[o, j] / (M12[o, j] + 1e-6), 5.0),
                         yoy=(min(M3[o, j] / (M3[o - 12, j] + 1e-6), 5.0) if o - 12 >= 0 and M3[o - 12, j] > 0 else np.nan),
                         grp_mom=fam_m3[j] / (fam_m12[j] + 1e-6), line_mom=line_m3[j] / (line_m12[j] + 1e-6),
                         month_sin=np.sin(2 * np.pi * (Tt % 12) / 12), month_cos=np.cos(2 * np.pi * (Tt % 12) / 12),
                         is_qtr_end=int((Tt % 12) in (2, 5, 8, 11)),
                         age_T=float(np.clip(Tt - D.meta["launch"][j], -12, 60)),
                         to_eol_T=float(np.clip(D.meta["eol"][j] - Tt, -12, 36)),
                         gen=D.meta["gen"][j], tier=D.meta["tier_idx"][j], line_code=l, h_=h)
                if has_T:
                    r.update(promo_T=D.PROMO[Tt, j], depth_T=D.DEPTH[Tt, j], mkt_ratio=L["mkt_plan"][Tt, l] / (L["mkt_plan"][o - 11: o + 1, l].mean() + 1e-6),
                             refresh_T=L["refresh"][Tt, l])
                else:
                    r.update(promo_T=np.nan, depth_T=np.nan, mkt_ratio=np.nan, refresh_T=np.nan)
                r.update(
                    promo_o=D.PROMO[o, j], li_o=L["li"][o, l], li_chg3=L["li"][o, l] - L["li"][o - 3, l],
                    rel_px_o=L["rel_px"][o, l], rel_px_chg3=L["rel_px"][o, l] - L["rel_px"][o - 3, l],
                    comp_share_o=L["comp_share"][o, l], comp_launch_3m=L["comp_launch"][o - 2: o + 1, l].sum(),
                    macro_o=L["macro"][o, l], macro_chg3=L["macro"][o, l] - L["macro"][o - 3, l], commodity_o=L["commodity"][o, l],
                    lead_o=L["lead"][o, l], alloc_o=L["alloc"][o, l], otd_o=L["otd"][o, l], yield_o=L["yld"][o, l],
                    backlog_ratio=D.BACKLOG[o, j] / (M3[o, j] + 1.0),
                )
                rows.append(r)
    F = pd.DataFrame(rows).drop(columns=["h_"])
    return F
