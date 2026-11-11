"""Quote-price benchmark and revenue scenarios for a user-supplied property."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from analysis import ROOT, add_features, scenario_calculation


def predict(config_path, scenario_path, out_dir):
    cfg = json.loads(Path(config_path).read_text())
    required = ["latitude", "longitude", "neighbourhood_cleansed", "property_type", "accommodates",
                "bedrooms", "beds", "bathrooms", "minimum_nights", "amenities", "quote_date",
                "checkin_date", "checkout_date"]
    missing = [x for x in required if x not in cfg]
    if missing:
        raise ValueError(f"Missing property inputs: {missing}")
    if not (48.81 <= cfg["latitude"] <= 48.91 and 2.22 <= cfg["longitude"] <= 2.48):
        raise ValueError("This prototype is limited to Paris coordinates covered by the sample.")
    if not 1 <= cfg["minimum_nights"] < 30:
        raise ValueError("The price model covers entire homes with minimum stays of 1–29 nights.")
    if pd.Timestamp(cfg["checkout_date"]) <= pd.Timestamp(cfg["checkin_date"]):
        raise ValueError("Checkout must be after check-in.")
    if pd.Timestamp(cfg["checkin_date"]) < pd.Timestamp(cfg["quote_date"]):
        raise ValueError("Check-in cannot precede the quote date.")
    p = dict(cfg, last_scraped=cfg["quote_date"], price_quote_checkin_date=cfg["checkin_date"],
             price_quote_checkout_date=cfg["checkout_date"], bathrooms_text=str(cfg["bathrooms"]))
    p["amenities"] = json.dumps(cfg["amenities"])
    frame = add_features(pd.DataFrame([p]))
    saved = joblib.load(ROOT / "models/price_model.joblib")
    if saved["feature_set"] != "location":
        raise ValueError("This input interface expects the location model.")
    rate = float(np.exp(saved["pipeline"].predict(frame)[0]))
    factors = pd.read_csv(ROOT / "results/seasonality.csv").set_index("month").daily_review_factor
    assumptions = json.loads(Path(scenario_path).read_text())
    monthly = scenario_calculation(rate, factors, assumptions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out_dir / "property_monthly_scenarios.csv", index=False)
    annual = monthly.groupby("scenario")["gross_revenue_eur"].sum().to_dict()
    result = {"quoted_nightly_price_benchmark_eur": rate, "annual_gross_scenarios_eur": annual,
              "cash_flow_available": bool(monthly.cash_flow_eur.notna().all()),
              "status": "Exploratory benchmark from September 2026 quotes. Revenue depends on user assumptions, not observed bookings."}
    if result["cash_flow_available"]:
        cash = monthly.groupby("scenario").cash_flow_eur.sum().to_dict()
        result["annual_cash_flow_eur"] = cash
        capital = assumptions.get("initial_cash_investment_eur")
        result["annual_cash_return_fraction"] = {k: v/capital for k,v in cash.items()} if capital and capital > 0 else None
    (out_dir / "property_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--property", type=Path, default=ROOT / "property_example.json")
    p.add_argument("--assumptions", type=Path, default=ROOT / "scenario_inputs.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/property_example")
    args = p.parse_args()
    predict(args.property, args.assumptions, args.out)
