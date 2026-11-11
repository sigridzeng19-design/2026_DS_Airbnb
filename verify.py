"""Focused checks of temporal features, held-out metrics and scenario arithmetic."""
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import joblib
from analysis import ROOT, scenario_calculation, cluster_gain


def main():
    checks = []
    panel = pd.read_csv(ROOT / "data/review_month_panel.csv.gz")
    assert not panel.duplicated(["listing_id", "month"]).any()
    checks.append("Panel listing-month keys are unique.")
    # Check lag values independently against the previous row's target, including split boundaries.
    ordered = panel.sort_values(["listing_id", "month"])
    for k in [1, 2, 3, 6, 12]:
        prev = ordered.groupby("listing_id").target.shift(k)
        mask = prev.notna()
        assert np.allclose(ordered.loc[mask, f"lag_{k}"], prev[mask])
    checks.append("All five lag features match earlier targets, including validation/test boundaries.")
    monthly = pd.read_csv(ROOT / "results/revenue_scenarios_monthly.csv")
    horizon = pd.read_csv(ROOT / "results/revenue_scenarios_horizons.csv")
    for name, group in monthly.groupby("scenario", sort=False):
        for n in [1, 3, 12]:
            value = horizon[(horizon.scenario == name) & (horizon.horizon_months == n)].gross_revenue_eur.iloc[0]
            assert np.isclose(group.head(n).gross_revenue_eur.sum(), value)
    assert monthly.cash_flow_eur.isna().all()
    checks.append("Month, three-month and annual revenue sums reconcile; unknown costs leave cash flow unavailable.")
    factors = pd.Series(1., index=range(1, 13))
    cfg = json.loads((ROOT / "scenario_inputs.json").read_text())
    cfg.update(occupancy_scenarios={"test": .5}, variable_cost_rate=.2, fixed_cost_per_month_eur=100., financing_cash_per_month_eur=20.)
    test = scenario_calculation(100., factors, cfg)
    assert np.allclose(test.gross_revenue_eur, 500.)
    assert np.allclose(test.cash_flow_eur, 280.)
    cfg["annual_booked_night_budget"] = 7
    test = scenario_calculation(100., factors, cfg)
    assert np.isclose(test.scenario_booked_nights.sum(), 7.)
    cfg["open_nights_by_month"] = [0]*12
    test = scenario_calculation(100., factors, cfg)
    assert np.isclose(test.gross_revenue_eur.sum(), 0)
    assert np.allclose(test.cash_flow_eur, -120.)
    cfg["open_nights_by_month"][0] = -1
    try:
        scenario_calculation(100., factors, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid opening nights should be rejected.")
    checks.append("Independent arithmetic checks pass for costs, zero capacity, night budget and invalid inputs.")
    metrics = pd.read_csv(ROOT / "results/price_metrics.csv")
    for protocol in metrics.protocol.unique():
        pred = pd.read_csv(ROOT / f"results/price_predictions_{protocol}.csv")
        for model in ["median_baseline", "internal", "location", "transit"]:
            independent_mae = np.abs(pred.actual_price_eur - pred[model]).mean()
            stored = metrics[(metrics.protocol == protocol) & (metrics.model == model)].mae_eur.iloc[0]
            assert np.isclose(independent_mae, stored)
    checks.append("Price headline errors match independently recomputed held-out predictions.")
    saved = joblib.load(ROOT / "models/review_backtest_model.joblib")
    te = panel[panel.month.isin(["2026-06", "2026-07", "2026-08"])].copy()
    pred = saved["pipeline"].predict(te[saved["features"]])
    m = pd.read_csv(ROOT / "results/review_metrics.csv")
    expected = m[(m.evaluation == "rolling_one_month") & (m.model == "gradient_boosting")].mae_reviews.iloc[0]
    assert np.isclose(np.abs(te.target.to_numpy() - pred).mean(), expected)
    gain = cluster_gain(te.target.to_numpy(), te.mean_3.to_numpy(), pred, te.listing_id)
    # Rename units: the same paired-cluster bootstrap helper is reused for review counts.
    gain["mae_reduction_reviews"] = gain.pop("mae_reduction_eur")
    gain["listing_clusters"] = gain.pop("bootstrap_host_clusters")
    (ROOT / "results/review_gain.json").write_text(json.dumps(gain, indent=2))
    groups = []
    for label, mask in [("zero_observed_reviews", te.target.to_numpy() == 0), ("positive_observed_reviews", te.target.to_numpy() > 0)]:
        groups.append({"group": label, "rows": int(mask.sum()), "mae_reviews": float(np.abs(te.target.to_numpy()[mask] - pred[mask]).mean())})
    pd.DataFrame(groups).to_csv(ROOT / "results/review_error_subgroups.csv", index=False)
    checks.append("Saved review model reproduces test error; paired listing-cluster uncertainty computed.")
    temporal = pd.read_csv(ROOT / "data/insideairbnb/temporal_review_panel.csv.gz", parse_dates=["last_scraped"])
    assert not temporal.duplicated(["id", "target_month"]).any()
    assert temporal.loc[~temporal.followup_observed, "target"].isna().all()
    assert temporal.loc[temporal.followup_observed, "target"].notna().all()
    assert (temporal.last_scraped < pd.to_datetime(temporal.target_month + "-01")).all()
    coverage = pd.read_csv(ROOT / "results/temporal_coverage.csv")
    for row in coverage.itertuples():
        origin = pd.Period(row.target_month, freq="M").start_time
        end = (pd.Period(row.target_month, freq="M") + 1).start_time
        assert pd.Timestamp(row.source_capture_max) < origin
        assert end <= pd.Timestamp(row.followup_capture_min)
    predictions = pd.read_csv(ROOT / "results/temporal_review_predictions.csv")
    scores = pd.read_csv(ROOT / "results/temporal_review_metrics.csv")
    test = temporal[(temporal.split == "test") & temporal.followup_observed].set_index("id")
    assert len(test) == len(predictions)
    assert np.allclose(predictions.target, predictions.listing_id.map(test.target))
    for row in scores.itertuples():
        error = predictions[row.model] - predictions.target
        assert np.isclose(abs(error).mean(), row.mae_reviews)
        assert np.isclose(np.sqrt((error**2).mean()), row.rmse_reviews)
        assert np.isclose(100 * error.sum() / predictions.target.sum(), row.aggregate_bias_pct)
    checks.append("Historical predictors precede targets; missing follow-up stays missing; temporal MAE, RMSE and total bias match saved predictions.")
    (ROOT / "results/verification.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2))
    print("\n".join(checks))


if __name__ == "__main__":
    main()
