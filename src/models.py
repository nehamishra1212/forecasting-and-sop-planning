"""
Forecasting models.

  naive / seasonal naive / moving average : simple benchmarks every model must beat (Forecast Value Added)
  stat   : Holt-Winters ETS (smooth/erratic demand) or Croston-SBA (intermittent/lumpy demand)
  gbm    : ONE global LightGBM model across all SKUs ("cross-learning") with calendar, lifecycle,
           promo, marketing, competitor, macro, supply-chain and leading-indicator features
  blend  : weighted average of gbm and stat (weight picked on the 2024 validation year only)

New products with no history use a family-analog fallback in the statistical/benchmark models.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from features import Data, family_scale, hist_len

warnings.filterwarnings("ignore")

ANALOG_SHARE = 0.30            # new SKU with <6 months of history: 30% of its family's trailing volume


# ------------------------------------------------------------------------------------------------
# Demand classification (Syntetos-Boylan): ADI = avg demand interval, CV2 = squared coeff. of variation
# ------------------------------------------------------------------------------------------------
def demand_class(y: np.ndarray) -> str:
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) < 6:
        return "new"
    y = y[nz[0]:][-24:]
    pos = y[y > 0]
    adi = len(y) / max(len(pos), 1)
    cv2 = (pos.std() / pos.mean()) ** 2 if len(pos) > 1 else 0.0
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def croston_sba(y: np.ndarray, alpha: float = 0.15) -> float:
    """Croston with Syntetos-Boylan bias correction. Returns a flat forecast (units / period)."""
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z, p, q = y[nz[0]], 1.0, 1
    first = True
    for v in y[nz[0] + 1:]:
        if v > 0:
            z = alpha * v + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p if not first else q
            first = False
            q = 1
        else:
            q += 1
    return float((1 - alpha / 2) * z / p)


def ets_multi(y: np.ndarray, H: int = 6) -> np.ndarray | None:
    """
    Auto-ETS: fit several exponential-smoothing configurations and keep the one with the lowest AICc
    (level only / damped trend, each with and without multiplicative yearly seasonality when >= 24 months exist).
    """
    nz = np.flatnonzero(y > 0)
    if len(nz) < 6:
        return None
    y = np.asarray(y[nz[0]:], float) + 1.0
    configs = [dict(trend=None, seasonal=None), dict(trend="add", damped_trend=True, seasonal=None)]
    if len(y) >= 24:
        configs += [dict(trend=None, seasonal="mul", seasonal_periods=12), dict(trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12)]
    best, best_aicc = None, np.inf
    for cfg in configs:
        try:
            fit = ExponentialSmoothing(y, initialization_method="estimated", **cfg).fit(optimized=True)
            f = fit.forecast(H)
            if np.isfinite(fit.aicc) and np.all(np.isfinite(f)) and fit.aicc < best_aicc:
                best, best_aicc = f, fit.aicc
        except Exception:
            continue
    if best is None:
        return np.full(H, max(y[-6:].mean() - 1.0, 0.0))
    return np.clip(best - 1.0, 0, None)


# ------------------------------------------------------------------------------------------------
# Benchmarks + statistical model for one origin (returns dict model -> array [S, H])
# ------------------------------------------------------------------------------------------------
def stat_models_at_origin(D: Data, o: int, H: int = 6) -> dict[str, np.ndarray]:
    S = len(D.skus)
    fam = family_scale(D, o)
    hl = hist_len(D, o)
    out = {k: np.zeros((S, H)) for k in ("naive", "snaive", "ma3", "stat")}
    classes = []
    for j in range(S):
        y = D.Y[: o + 1, j]
        analog = ANALOG_SHARE * fam[j]
        if hl[j] < 6:                                            # new product: no usable history
            for k in out:
                out[k][j] = analog
            classes.append("new")
            continue
        out["naive"][j] = y[-1]
        out["ma3"][j] = y[-3:].mean()
        for hh in range(1, H + 1):
            T = o + hh
            out["snaive"][j, hh - 1] = D.Y[T - 12, j] if (T - 12 >= 0 and D.Y[T - 12, j] > 0) else y[-3:].mean()
        cls = demand_class(y)
        classes.append(cls)
        if cls in ("intermittent", "lumpy"):
            out["stat"][j] = croston_sba(y)
        else:
            f = ets_multi(y, H)
            out["stat"][j] = f if f is not None else y[-3:].mean()
    out["classes"] = np.array(classes)
    return out


# ------------------------------------------------------------------------------------------------
# Global gradient-boosting model
# ------------------------------------------------------------------------------------------------
def _weights(F: pd.DataFrame, asp: np.ndarray) -> np.ndarray:
    w = np.sqrt(F["scale"].to_numpy() * asp[F["j"].to_numpy()])
    return w / w.mean()


def fit_predict_gbm(F: pd.DataFrame, o: int, h: int, cols: list[str], asp: np.ndarray, objective: str = "tweedie",
                    alpha: float | None = None, seed: int = 7) -> pd.DataFrame:
    """Train on every (origin', SKU) pair whose target was already observed at origin o; predict target o+h."""
    sub = F[F["h"] == h]
    train = sub[(sub["o"] + h <= o) & sub["y"].notna()]
    test = sub[sub["o"] == o]
    params = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=15, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=1)
    if objective == "tweedie":
        params.update(objective="tweedie", tweedie_variance_power=1.3)
    else:
        params.update(objective="quantile", alpha=alpha)
    m = lgb.LGBMRegressor(**params)
    m.fit(train[cols], train["y"], sample_weight=_weights(train, asp))
    pred = np.clip(m.predict(test[cols]), 0, None)
    res = test[["o", "h", "T", "j", "scale", "actual"]].copy()
    res["pred_ratio"] = pred
    res.attrs["model"] = m
    return res
