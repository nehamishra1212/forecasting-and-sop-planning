# Data dictionary and data-generating process

**All data is synthetic** (`src/generate_data.py`, seed 42). The fictional company, "Meridian Systems", sells six product lines
through 36 SKUs (3 tiers x 2 product generations per line) from Jan-2020 to Dec-2025 (72 months).
Raw tables live in `data/raw/`; `data/processed/panel.csv` is the SKU-month table the models use.

## Product lines

| Line | Units / month at start | Character |
|---|---|---|
| Client PCs | ~18,000 | Consumer seasonality, pandemic surge and hangover, OS end-of-support refresh in 2025 |
| Accessories & Peripherals | ~30,000 | Attach to PCs, heavy promotion, holiday peak |
| Network Switches | ~1,800 | Quarter-end peaks, AI-infrastructure growth |
| Servers | ~900 | Quarter-end peaks, strongest AI-infrastructure ramp, occasional large deals |
| Storage Arrays | ~260 | Low volume, very high value, lumpy; quarter-end peaks |
| Edge Sensors & Modules | ~1,200 | Step-ups when new customer fleets/cities come online; lumpy |

## Tables

| File | Grain | Key columns | Who owns this in a real company |
|---|---|---|---|
| `sales_history.csv` | SKU x month | `orders` (**forecast target**, unconstrained demand), `shipments` (supply-constrained), `backlog_units`, `sales_forecast_3m_ahead` (legacy sales-submitted forecast) | Sales ops / order management |
| `sku_master.csv` | SKU | line, tier, generation, `launch_idx`/`eol_idx`, `launch_date`, `eol_date`, `asp_usd`, `unit_cost_usd`, lead time, MOQ, supplier, CM site | Product / planning master data |
| `promo_calendar.csv` | SKU x month | `promo_flag`, `promo_depth_pct` (planned) | Marketing / pricing |
| `marketing.csv` | line x month | `marketing_spend_plan_usd_k` (known ahead), `marketing_spend_actual_usd_k`, `campaign_multiplier` | Marketing |
| `competitor.csv` | line x month | competitor and own price index, `relative_price`, `competitor_launch_flag`, `competitor_share_pct_observed` (quarterly, 1-quarter lag) | Competitive intelligence |
| `macro.csv` | month | IT-capex index, consumer-demand index, commodity (memory / silicon) price index | Finance / market research |
| `supply_chain.csv` | line x month | `supplier_lead_time_weeks`, `component_allocation_pct`, `supplier_otd_pct` | Procurement / supply chain |
| `operations.csv` | line x month | `cm_capacity_units`, `factory_yield_pct`, `line_backlog_units` | Manufacturing / operations |
| `leading_indicators.csv` | line x month | `leading_indicator_index` (sales-pipeline index for enterprise lines, web/search traffic for consumer lines), `refresh_event_index` (announced OS-refresh event) | Sales ops / digital marketing |
| `events.csv` | event | shortage start/end, competitor launches and price cuts, NPI launches, end-of-life starts, `known_in_advance` flag | Planning |

## How demand is generated (modelling assumptions, not measurements)

For SKU *i* in month *t*:

```
mean_demand = line_volume0 x tier_weight x generation_share(t) x line_trend_and_events(t) x seasonal_index(month)
              x NPI_channel_fill x last_time_buy x promo_uplift x post_promo_dip x competitor_launch_effect
              x price_effect(relative price) x macro_effect x marketing_uplift(lagged)
              x shortage_order_inflation x post_shortage_digestion
orders = Poisson( mean_demand x lognormal noise ) + occasional large deals (enterprise lines)
```

* **Generation share**: the new generation ramps in over ~5 months after launch while the old one decays after end-of-life;
  total tier demand is conserved (cannibalisation, not magic growth).
* **Component shortage (ramps up from Mar-2021, peaks Sep-2021 to May-2022, eases by Nov-2022)**: customers over-order (+12% x line sensitivity), supply is capped by allocation,
  so `shipments < orders` and `backlog` builds. A post-shortage digestion dip follows (bullwhip).
* **Leading indicator**: the line's true (noise-free) demand two months ahead, relative to its trailing average, times
  lognormal noise (sd 10-15%). *In real life the strength of such a signal depends entirely on data quality - treat it as an assumption.*
* **Legacy sales forecast**: made 3 months ahead = true expected demand x 1.08 optimism x a persistent line-level error
  (AR(1), sd 0.15 - planners are optimistic or pessimistic together) x independent SKU noise (sd 0.27).
  **This is a simulated stand-in for a real sales-submitted forecast.** Its error structure drives the size of any gain from
  the consensus step (see `docs/methodology.md`, limitations).

## What is known when? (leakage controls)

Features are split by availability at forecast origin *o* for target month *T = o + h*:

* **Origin features** - observed up to *o*: order history, supplier signals, competitor price and share, macro indices, leading indicator, backlog.
* **Plan features** - known in advance for month *T*: promo calendar, marketing budget plan, product-roadmap dates (launch / end-of-life), announced events, calendar month.

Nothing observed after *o* is used. Models are retrained at every origin on rows whose targets were already realised at *o*.
