"""
End-to-end pipeline:  python src/run_all.py            (about 4-5 minutes on a laptop)
                      python src/run_all.py --cached   (re-use saved backtest, rebuild tables + figures)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_data
import plots
from evaluate import (TEST_START, ablation, accuracy_table, add_blend, coverage_table, exception_report, intervals, inventory_at_service,
                      inventory_frontier, inventory_simulation, paired_bootstrap, run_backtest, select_blend_weight, tracking_signal,
                      select_consensus_weight, add_consensus)
from features import build_feature_table, load_data

TAB, FIG, DATA = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "data"
TEST_MODELS = ["snaive", "naive", "ma3", "stat", "sales_fc", "gbm_core", "gbm_full", "consensus"]


def assemble(fc: pd.DataFrame, D, F):
    """Add blend + consensus models and segment labels (ABC/XYZ, lifecycle) to the raw backtest forecasts."""
    w, grid = select_blend_weight(fc)
    fc = add_blend(fc, w) if "blend" not in set(fc["model"]) else fc
    wc, cgrid = select_consensus_weight(fc)
    fc = add_consensus(fc, wc) if "consensus" not in set(fc["model"]) else fc
    seg = plots.abc_xyz(D)[["sku", "abc", "xyz"]]
    fc = fc.merge(seg, on="sku", how="left").merge(F[["o", "h", "j", "hist_len"]].rename(columns={"o": "O"}), on=["O", "h", "j"], how="left")
    fc["life"] = np.where(fc["hist_len"] < 12, "NPI (<12m history)", "Established")
    return fc, dict(w=w, grid=grid, wc=wc, cgrid=cgrid)


def main(cached: bool = False):
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if not (DATA / "processed" / "panel.csv").exists():
        print("[1/8] generating synthetic data"); generate_data.main(DATA)
    D = load_data(DATA)
    print("[2/8] building feature table"); F = build_feature_table(D)

    if cached and (TAB / "backtest_forecasts.csv").exists():
        print("[3/8] loading cached backtest")
        fc = pd.read_csv(TAB / "backtest_forecasts.csv")
        quant = pd.read_csv(TAB / "backtest_quantiles.csv")
        imp = pd.read_csv(TAB / "feature_importance.csv", index_col=0)["gain"]
    else:
        print("[3/8] rolling-origin backtest (origins 2023-07 .. 2025-09; models retrained at every origin)")
        bt = run_backtest(D, F)
        fc, quant, imp = bt["fc"], bt["quant"], bt["importance"]
        fc.to_csv(TAB / "backtest_forecasts.csv", index=False)
        quant.to_csv(TAB / "backtest_quantiles.csv", index=False)
        imp.rename("gain").to_csv(TAB / "feature_importance.csv")

    fc, info = assemble(fc, D, F)
    w, wc = info["w"], info["wc"]
    info["grid"].to_csv(TAB / "blend_weight_validation_grid.csv", index=False)
    info["cgrid"].to_csv(TAB / "consensus_weight_validation_grid.csv", index=False)
    print(f"[4/8] weights chosen on the 2024 validation year only: GBM-vs-ETS blend = {w}, consensus (share on statistical GBM) = {wc}")
    test = fc[fc["T"] >= TEST_START]
    t3 = test[(test["h"] == 3) & test["model"].isin(TEST_MODELS)]

    # ---- accuracy tables ---------------------------------------------------------------------------------
    print("[5/8] accuracy, FVA, segments")
    acc = {lv: accuracy_table(t3, level=lv) for lv in ("sku", "line", "total")}
    for lv, t in acc.items():
        base = t.set_index("model")["wmape_rev"]
        t["fva_vs_naive_pp"] = 100 * (base["naive"] - t["wmape_rev"])
        t["fva_vs_sales_fc_pp"] = 100 * (base["sales_fc"] - t["wmape_rev"])
        t.to_csv(TAB / f"accuracy_{lv}_level_h3.csv", index=False)
    acc_h = accuracy_table(test[test["model"].isin(TEST_MODELS)], by=["h"])
    acc_h.to_csv(TAB / "accuracy_by_horizon.csv", index=False)
    acc_line = accuracy_table(t3, by=["line"]); acc_line.to_csv(TAB / "accuracy_by_line_h3.csv", index=False)
    acc_life = accuracy_table(t3, by=["life"]); acc_life.to_csv(TAB / "accuracy_by_lifecycle_h3.csv", index=False)
    acc_abc = accuracy_table(t3, by=["abc"]); acc_abc.to_csv(TAB / "accuracy_by_abc_h3.csv", index=False)
    acc_xyz = accuracy_table(t3, by=["xyz"]); acc_xyz.to_csv(TAB / "accuracy_by_xyz_h3.csv", index=False)
    acc_cls = accuracy_table(t3, by=["demand_class"]); acc_cls.to_csv(TAB / "accuracy_by_demand_class_h3.csv", index=False)
    t3[t3["model"].isin(["gbm_full", "sales_fc", "naive"])].to_csv(TAB / "forecast_detail_2025_h3.csv", index=False)

    # ---- intervals, tracking signal, exceptions -----------------------------------------------------------
    q = intervals(quant)
    cov = coverage_table(q); cov.to_csv(TAB / "prediction_interval_coverage.csv", index=False)
    ts = tracking_signal(fc, "gbm_full"); ts.to_csv(TAB / "tracking_signal_gbm.csv", index=False)
    exc = exception_report(fc, "gbm_full"); exc.to_csv(TAB / "exception_report_top10.csv", index=False)

    # ---- inventory / service simulation ----------------------------------------------------------------------
    print("[6/8] inventory & service simulation")
    frontier = inventory_frontier(fc, TEST_MODELS)
    frontier.to_csv(TAB / "inventory_service_frontier.csv", index=False)
    inv95 = inventory_at_service(frontier, 0.95)
    inv95.to_csv(TAB / "inventory_needed_for_95pct_fill.csv", index=False)
    inv90 = inventory_at_service(frontier, 0.90)
    inv90.to_csv(TAB / "inventory_needed_for_90pct_fill.csv", index=False)
    boot = pd.DataFrame([paired_bootstrap(fc, "gbm_full", r) for r in ("naive", "sales_fc", "gbm_core", "stat")] + [paired_bootstrap(fc, "consensus", r) for r in ("sales_fc", "gbm_full", "naive")])
    boot.to_csv(TAB / "bootstrap_accuracy_gap.csv", index=False)

    # ---- ablation -----------------------------------------------------------------------------------------------
    print("[7/8] input-group ablation")
    ab_path = TAB / "ablation_input_groups.csv"
    if cached and ab_path.exists():
        ab = pd.read_csv(ab_path)
    else:
        ab = ablation(D, F); ab.to_csv(ab_path, index=False)

    # ---- figures ----------------------------------------------------------------------------------------------------
    print("[8/8] figures")
    plots.fig_history(D, FIG / "01_demand_history.png")
    plots.fig_seasonality(D, FIG / "02_seasonality_heatmap.png")
    plots.fig_constraint(D, FIG / "03_orders_vs_shipments.png")
    plots.fig_drivers(D, FIG / "04_external_inputs_example.png")
    plots.fig_abc_xyz(D, FIG / "05_abc_xyz.png")
    plots.fig_total_backtest(fc[fc["model"].isin(["gbm_full", "sales_fc", "naive", "consensus"])], D, FIG / "06_backtest_total_revenue.png")
    plots.fig_fva_waterfall(acc["sku"], FIG / "16_fva_waterfall.png")
    plots.fig_line_backtest(fc[fc["model"].isin(["gbm_full", "sales_fc"])], D, FIG / "07_backtest_by_line.png")
    plots.fig_model_comparison(acc["sku"], acc["line"], acc["total"], FIG / "08_model_comparison.png")
    plots.fig_horizon(acc_h, FIG / "09_accuracy_by_horizon.png")
    plots.fig_segments(acc_line, acc_life, acc_abc, FIG / "10_accuracy_by_segment.png")
    plots.fig_importance(imp, FIG / "11_feature_importance.png")
    plots.fig_ablation(ab, FIG / "12_ablation_input_groups.png")
    plots.fig_interval(q, fc, D, "ST-M2", FIG / "13_prediction_interval_example.png")
    plots.fig_inventory(frontier, inv95, FIG / "14_inventory_service_tradeoff.png")
    plots.fig_tracking(ts, FIG / "15_tracking_signal.png")

    summary = dict(
        blend_weight_gbm=w, consensus_weight_stat_model=wc, test_year="2025", horizon_months=3,
        wmape_sku=acc["sku"].set_index("model")["wmape_rev"].round(4).to_dict(),
        wmape_line=acc["line"].set_index("model")["wmape_rev"].round(4).to_dict(),
        wmape_total=acc["total"].set_index("model")["wmape_rev"].round(4).to_dict(),
        bias_sku=acc["sku"].set_index("model")["bias_rev"].round(4).to_dict(),
        interval_coverage=cov.round(4).to_dict("records"),
    )
    (ROOT / "outputs" / "results_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== SKU-level (h=3, 2025) ===\n", acc["sku"][["model", "wmape_rev", "bias_rev", "fva_vs_naive_pp", "fva_vs_sales_fc_pp"]].round(3).sort_values("wmape_rev").to_string(index=False))
    print("\n=== Inventory needed for 95% fill rate ===\n", inv95.round(3).to_string(index=False))
    print("\n=== Bootstrap (GBM vs rival, WMAPE gap in pp, 12 test months) ===\n", boot.round(3).to_string(index=False))
    print("\n=== Ablation ===\n", ab.round(3).to_string(index=False))
    print("\n=== Interval coverage ===\n", cov.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    main(ap.parse_args().cached)
