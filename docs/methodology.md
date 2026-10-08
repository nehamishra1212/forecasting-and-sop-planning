# Methodology: why these models, how they were judged, and where they fall short

## 1. Framing the problem

| Decision | Choice | Why |
|---|---|---|
| What is forecast | **Orders** (unconstrained demand), not shipments | In the 2021-22 shortage shipments were capped by supply; training on them teaches the model that demand fell (fig 03) |
| Planning horizon | **3 months** (also 1 and 6) | Contract-manufacturer builds and component buys are committed ~3 months ahead; that is the decision the forecast supports |
| Grain | SKU x month, rolled up to line and company | Supply is planned by SKU; executives review by line and total. Accuracy is reported at all three levels because it differs a lot (see results) |
| Headline metric | **Revenue-weighted WMAPE** (units x ASP) and **bias** | A missed storage array matters more than a missed cable. WMAPE is robust to zero-demand months, unlike MAPE |

## 2. Honest evaluation design

* **Rolling-origin backtest.** For every forecast origin from Jul-2023 to Sep-2025 every model is re-fit using only data available then, and forecasts months *o+1*, *o+3*, *o+6*. No random train/test splits.
* **Validation year = 2024, test year = 2025.** Every choice that was tuned (the consensus weight, the GBM-vs-ETS blend weight, the conformal calibration) used 2024 only. The 2025 test year was used to *report*, not to tune. A feature idea (interaction terms for the leading indicator) was tried and **rejected** because it did not help on 2024.
* **Benchmarks every model must beat:** naive, seasonal naive, 3-month average, auto-ETS / Croston-SBA, and the (simulated) **sales-submitted forecast**.
* **Forecast Value Added (FVA)** = benchmark WMAPE - model WMAPE. Each process step is judged on whether it improved the forecast.
* **Uncertainty on the result.** Paired bootstrap over the 12 test months (months, not SKUs, because errors within a month are correlated).

## 3. Models and the reasoning behind them

| Model | What it is | Why it is in the project |
|---|---|---|
| Naive / seasonal naive / 3-month average | Last value, same month last year, recent average | The floor. If a model cannot beat these it adds no value |
| **Auto-ETS + Croston-SBA** | Exponential smoothing (level / damped trend, with / without multiplicative seasonality, chosen by AICc) for smooth and erratic SKUs; Croston with Syntetos-Boylan correction for intermittent and lumpy SKUs, chosen by ADI / CV² classification | The standard statistical engine in most planning systems. Transparent, needs no external data, but sees each SKU in isolation |
| **Global LightGBM (Tweedie loss)** | One model across all 36 SKUs. Target = orders / family scale, so SKUs of very different size share what they learn. Features: lags, rolling stats, seasonality, lifecycle stage, promo and marketing plan, competitor, macro, supply-chain signals, leading indicator | **Cross-learning** lets new and short-history SKUs borrow behaviour from the family; handles interactions (promo x quarter-end x lifecycle) that single-series models cannot; Tweedie loss suits non-negative, zero-heavy counts. External inputs are only useful if a model can use them |
| **Consensus** | 0.6 x statistical GBM + 0.4 x sales-submitted forecast (weight chosen on 2024) | Mirrors the real S&OP step: statistical baseline + commercial overlay. Forecast combination usually beats either input when their errors are not perfectly correlated |
| GBM quantile models + conformal calibration | P10 / P90 forecasts, widened so that empirical coverage matches the nominal 80% | Supply decisions need a range, not a point; raw quantile models were over-confident (62% coverage) |

**New products** have no history. Family scaling (the SKU's family = line + tier, so generation 2 shares the scale of generation 1) gives the GBM a usable target; the statistical models fall back to an analog (30% of the family's volume).

## 4. Metric definitions

* `WMAPE_rev = sum(ASP x |F - A|) / sum(ASP x A)`.
* `Bias_rev = sum(ASP x (F - A)) / sum(ASP x A)`; negative = under-forecast. *Always read next to WMAPE.*
* `MAPE` is reported in the tables but is **not** used for conclusions: it blows up on small or intermittent volumes.
* `Tracking signal = cumulative error / mean absolute deviation`, by line; |TS| > 4 flags persistent bias for review.
* **Business translation:** a simple supply policy (commit forecast + z x trailing RMSE three months ahead, carry leftover, lose unmet demand) is run for every model; `z` is swept so models are compared at **equal fill rate**. Assumptions: 25% annual carrying cost, lost sales cost gross margin.

## 5. Results (synthetic data, 2025 test year, 3-month horizon)

All values come from `outputs/tables/`. SKU-level, revenue-weighted.

| Model | WMAPE | Bias | FVA vs naive |
|---|---|---|---|
| Seasonal naive | 54.5% | -3.2% | -17.1 pp |
| Naive (last month) | 37.4% | -2.2% | 0 |
| Auto-ETS / Croston-SBA | 39.5% | -1.5% | -2.1 pp |
| Sales-submitted forecast (simulated) | 30.9% | -3.8% | +6.4 pp |
| GBM: history + calendar + lifecycle | 30.7% | -1.4% | +6.7 pp |
| GBM: all inputs | 29.0% | -2.6% | +8.4 pp |
| **Consensus (GBM + sales input)** | **24.8%** | -3.1% | **+12.6 pp** |

Paired-bootstrap gaps (WMAPE points, 95% interval over the 12 test months): consensus vs naive **+12.5 [6.3, 19.5]**; consensus vs sales forecast **+6.1 [2.2, 10.8]**; GBM vs sales forecast **+1.8 [-4.9, 9.1]** (not distinguishable from zero); GBM with all inputs vs history-only GBM **+1.7 [0.3, 3.5]**.

Inventory needed to reach a 95% revenue fill rate: consensus **$44M** (1.2 weeks of supply), sales forecast $48M, statistical GBM $62M, naive $71M, auto-ETS $90M.

## 6. What did NOT work (kept in on purpose)

* **Auto-ETS / Croston was worse than naive** (-2.1 pp) on this volatile, trending, step-changing data. A single-series model chasing a structural break (the AI-infrastructure ramp) is a realistic failure; the GBM-vs-ETS blend weight chosen on validation was 1.0 (ETS got zero weight).
* **The GBM alone is not significantly better than the sales forecast** (+1.8 pp, wide interval) and needs 28% *more* inventory than it for the same service. Its value shows up in the **consensus**.
* **At total-company level, seasonal naive (10.1%) is as good as anything** (consensus 10.6%). Model gains are concentrated at SKU and line level, where supply is planned. Errors that move all SKUs together (the growth ramp) do not cancel.
* **External inputs help less than expected and unevenly** (seed-averaged ablation): removing promotions/pricing, marketing, or known events costs 0.3-0.8 pp; removing the leading indicator, competitor or macro/commodity inputs did not hurt (slightly helped) - the model could not exploit them. At a 6-month horizon the all-inputs GBM (42.1%) is *worse* than the history-only GBM (38.6%).
* **Volatile "Y" SKUs** (CV 0.3-0.6) are where the GBM is weakest (42.5% vs 30.9% for the sales forecast).
* Persistent under-forecast of Client PCs and Accessories (tracking signal -6.8 and -5.9 by Dec-2025; both lines carry the 2025 OS-refresh surge) is exactly the kind of exception a planner should review: 16 line-months breached |TS| > 4.

## 7. Limitations (read before quoting any number)

1. **Synthetic data, one random seed, one 12-month test year.** Effect sizes (promo elasticity, competitor impact, leading-indicator noise) are assumptions. Which inputs "matter" reflects how the data was built.
2. **The sales forecast is simulated**, and its errors are independent of the model's errors apart from a shared line-level term. Forecast-combination gains therefore **may be overstated** here; with real data the consensus gain must be measured, not assumed.
3. GBM results move with the random seed: the all-inputs GBM scores 29.0% in the single-seed backtest and 29.7% when averaged over three seeds in the ablation (about 0.7 pp). The seed-averaged ablation is therefore more reliable than a single run for small effects; it puts the value of external inputs at about 1.0 pp (history-only 30.7% vs all inputs 29.7%).
4. Statistical baselines were run with sensible defaults, not exhaustively tuned. A tuned engine could close some of the gap.
5. The inventory simulation is deliberately simple (one policy, no MOQ / lead-time variability / expediting) so models can be compared like for like. It ranks forecasts; it does not predict real cash impact.
6. No SAP IBP, ERP, or real company data was used.
7. **Development history, disclosed.** (a) After the first run, the simulated sales forecast had independent errors across SKUs, which cancel unrealistically when summed; I added a persistent line-level error component (SKU-level error size unchanged) and regenerated the data. (b) The first statistical baseline used a single fixed ETS configuration and was weaker than naive; I replaced it with auto-ETS (AICc selection), the standard practice, so the benchmark is fair. (c) The comparison of safety-stock policies was changed to an equal-fill-rate comparison after noticing that a fixed safety-stock multiplier lands each forecast at a different service level. (d) One GBM feature idea was rejected on 2024 validation. The 2025 results were inspected during development, so they are not a pristine blind test.

## 8. What I would do next

* Re-run on a public retail or electronics demand data set to test whether the ranking of methods survives real noise.
* Add hierarchical reconciliation so SKU, line and total forecasts agree, which should reduce the total-level gap.
* Engineer the leading indicator properly (pipeline stage x historical conversion by line) instead of passing it as a raw index.
* Add a supply-constraint layer: constrained vs unconstrained plan with allocation rules.
* Put the consensus step behind a simple review UI with override tracking, so *Forecast Value Added of human overrides* can be measured.
