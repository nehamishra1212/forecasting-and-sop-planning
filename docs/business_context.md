# What an organisation actually needs from demand forecasting

This project was designed backwards from two real-world planning roles (paraphrased from public job postings):

* **Everpure - SAP Plan-to-Forecast Lead (storage / data-platform company).** Owns the monthly consensus S&OP and
  demand-planning cycle for a global business; defines the statistical forecasting framework and the error-measurement
  standards (MAPE, WMAPE, bias); runs exception-based reviews; folds new product introductions, promotions, phase-outs
  and revenue targets from Sales / Marketing / Product / Finance into one demand plan; owns demand master data and KPIs and
  does root-cause analysis on misses; specifies system enhancements with IT.
* **Waymo - Supply & Demand Planning (autonomous-vehicle hardware).** Builds and runs the global demand/supply balance for
  vehicles and sensor suites; directs OEM and contract-manufacturer build execution; leads executive S&OP to reconcile
  variances and manage working capital; defines a planning "system of record" and uses AI-assisted tools and automated models
  to remove manual planning work; allocates sensors through forward and reverse distribution; aligns supply with city-level demand.

Both roles boil down to the same question: **"How many of what do we commit to build or buy, 3+ months before we know the
real demand - and how do we know whether our forecast process is getting better?"**

## From the job to the project

| What the organisation needs | Inputs it requires (and who owns them) | The hard part | How it is measured here | Where in this repo |
|---|---|---|---|---|
| **One consensus demand plan** every month | Statistical baseline (planning), sales commitments (Sales), revenue targets (Finance) | Optimism / sandbagging, several "versions of the truth" | Revenue-weighted WMAPE, **bias**, **Forecast Value Added** at every step | `consensus` model, fig 16, `accuracy_*` tables |
| **Supply commitments with 3-month lead time** (CM / OEM builds, component buys) | Forecast at h = 3, plus uncertainty | Accuracy decays with horizon; over-build = cash, under-build = lost sales | **Inventory needed for the same fill rate**, lost margin, weeks of supply | `inventory_*` tables, fig 14, fig 09 |
| **New products, phase-outs, last-time buys** | Product roadmap dates (Product) | No history for new SKUs; cannibalisation between generations | Accuracy split by lifecycle stage; family-analog fallback | `features.py` (family scale), fig 10 |
| **Promotions, marketing, pricing, competitor moves** | Promo calendar and marketing plan (Marketing); price and share intel (Competitive) | Planned events are known, competitor moves are not; effects overlap | **Ablation**: accuracy lost when each input group is removed | `ablation_input_groups.csv`, fig 12 |
| **Seasonality** (quarter-end for enterprise, back-to-school/holiday for consumer) | Calendar | Different shapes per line; shifting with launches | Seasonal index by line; seasonal naive as a benchmark | fig 02 |
| **Supply constraints** (allocation, long lead times, shortages) | Supplier lead time, allocation, OTD, yield, CM capacity (Supply chain / Ops) | **Shipments understate demand when supply is capped** - a model trained on shipments learns the wrong thing | Orders (unconstrained) used as the target; backlog tracked | fig 03, `generate_data.py` |
| **Lumpy, high-value items** (storage arrays, sensors, servers) | Deal pipeline (Sales) | Intermittent demand, a few big orders dominate error | Demand classification (ADI / CV²), Croston-SBA, **prediction intervals** | `models.py`, fig 13, `prediction_interval_coverage.csv` |
| **Working capital / excess & obsolete** | Unit cost, ASP, carrying cost | Bias is invisible in WMAPE alone | Bias % reported next to every accuracy number | `accuracy_*` tables |
| **Exception-based review & KPI governance** | Forecast vs actual by SKU / line | Too many SKUs to review; need to focus on what matters | **ABC/XYZ** segmentation, tracking signal, top-10 exception report | figs 05 and 15, `exception_report_top10.csv` |
| **A planning "system of record" and automation** | Clean, versioned inputs; repeatable pipeline | Manual spreadsheets, no audit trail | One command rebuilds data -> models -> metrics -> charts | `src/run_all.py`, `docs/data_dictionary.md` |

## Honest scope note

This is a **methods demonstration on synthetic data**. It does **not** use SAP IBP, Oracle or any company's real data, and
nothing here is a result from any employer. The statistical methods used (ETS, Croston-SBA, gradient boosting, WMAPE / bias,
ABC / XYZ segmentation, exception-based review) are the same families of methods that planning engines such as SAP IBP expose,
but the implementation here is Python.
