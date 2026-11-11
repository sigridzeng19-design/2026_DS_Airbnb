"""Reproducible Paris Airbnb analysis. Run from this directory; see README.md."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/airbnb_project_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/raw"
SEED = 42
EARTH_KM = 6371.0088
INTERNAL = ["accommodates", "bedrooms", "beds", "bathrooms_n", "minimum_nights",
            "amenity_count", "wifi", "kitchen", "washer", "air_conditioning", "elevator",
            "quote_stay_nights", "quote_month_sin", "quote_month_cos", "quote_lead_days"]
LOCATION = ["latitude", "longitude"]
TRANSIT = ["nearest_metro_km", "metro_stations_500m", "metro_stations_1000m", "nearest_rer_km"]
PRICE_SETS = {"internal": (INTERNAL, ["property_type"]),
              "location": (INTERNAL + LOCATION, ["property_type", "neighbourhood_cleansed"]),
              "transit": (INTERNAL + LOCATION + TRANSIT, ["property_type", "neighbourhood_cleansed"])}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
COLORS = ["#9aa6b3", "#557893", "#0b6e75", "#cf8c36"]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_data(data_dir=DEFAULT_DATA):
    """Read only selected analytical fields; no names, comments or profile text."""
    fields = ["id", "host_id", "last_scraped", "room_type", "property_type", "price",
              "price_quote_raw", "price_quote_checkin_date", "price_quote_checkout_date",
              "latitude", "longitude", "neighbourhood_cleansed", "accommodates", "bedrooms",
              "beds", "bathrooms", "bathrooms_text", "minimum_nights", "amenities"]
    l = pd.read_csv(data_dir / "listings.csv", usecols=fields, low_memory=False)
    r = pd.read_csv(data_dir / "reviews.csv", usecols=["listing_id", "id", "date"],
                    dtype={"listing_id": "int64", "id": "int64"}, parse_dates=["date"])
    assert l.id.is_unique, "Duplicate listing IDs require resolution."
    assert r.date.notna().all(), "Missing review dates require resolution."
    audit = {"listing_rows": len(l), "listing_fields_in_source": len(pd.read_csv(data_dir / "listings.csv", nrows=0).columns),
             "review_rows_raw": len(r), "snapshot_start": l.last_scraped.min(),
             "snapshot_end": l.last_scraped.max(), "review_date_min": str(r.date.min().date()),
             "review_date_max": str(r.date.max().date()), "review_orphan_rows": int((~r.listing_id.isin(l.id)).sum())}
    # The same review ID can occur for several listings. Do not assign it arbitrarily.
    conflicts = r.groupby("id").agg(listings=("listing_id", "nunique"), dates=("date", "nunique"))
    ambiguous = conflicts.index[(conflicts.listings > 1) | (conflicts.dates > 1)]
    audit["ambiguous_review_ids"] = len(ambiguous)
    audit["ambiguous_review_rows_excluded"] = int(r.id.isin(ambiguous).sum())
    r = r[~r.id.isin(ambiguous)].copy()
    audit["same_listing_duplicate_rows_removed"] = int(r.duplicated(["id", "listing_id", "date"]).sum())
    r = r.drop_duplicates(["id", "listing_id", "date"])
    # Match each review against its own listing's scrape date, not the latest city scrape.
    scrape = pd.to_datetime(l.set_index("id").last_scraped)
    bad = r.date > r.listing_id.map(scrape)
    audit["reviews_after_own_scrape_excluded"] = int(bad.sum())
    r = r[~bad & r.listing_id.isin(l.id)].copy()
    audit["review_rows_clean"] = len(r)
    l["price_eur"] = pd.to_numeric(l.price.str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    def currency(s):
        try:
            return json.loads(s).get("quote", {}).get("currency")
        except (TypeError, ValueError):
            return None
    l["quote_currency"] = l.price_quote_raw.map(currency)
    audit["quote_currency_counts"] = l.quote_currency.fillna("missing").value_counts().to_dict()
    audit["price_missing_rows"] = int(l.price_eur.isna().sum())
    audit["price_missing_fraction"] = float(l.price_eur.isna().mean())
    audit["room_types"] = l.room_type.value_counts().to_dict()
    audit["quote_checkin_min"] = l.price_quote_checkin_date.min()
    audit["quote_checkin_max"] = l.price_quote_checkin_date.max()
    price_known = l.assign(price_missing=l.price_eur.isna()).groupby("room_type").agg(
        listings=("id", "size"), missing_price_fraction=("price_missing", "mean"), median_price_eur=("price_eur", "median"))
    price_known.to_csv(ROOT / "results/price_coverage.csv")
    l.drop(columns=["price_quote_raw"], inplace=True)
    return l, r, audit


def add_features(l):
    l = l.copy()
    l["bathrooms_n"] = pd.to_numeric(l.bathrooms, errors="coerce").fillna(
        pd.to_numeric(l.bathrooms_text.str.extract(r"([0-9.]+)")[0], errors="coerce"))
    def amenities(s):
        try:
            a = json.loads(s)
            return [str(x).lower() for x in a] if isinstance(a, list) else []
        except (TypeError, ValueError):
            return []
    a = l.amenities.map(amenities)
    l["amenity_count"] = a.map(len)
    for name, term in [("wifi", "wifi"), ("kitchen", "kitchen"), ("washer", "washer"),
                       ("air_conditioning", "air conditioning"), ("elevator", "elevator")]:
        l[name] = a.map(lambda xs: int(any(term in x for x in xs)))
    checkin = pd.to_datetime(l.price_quote_checkin_date, errors="coerce")
    checkout = pd.to_datetime(l.price_quote_checkout_date, errors="coerce")
    l["quote_stay_nights"] = (checkout - checkin).dt.days
    l["quote_month_sin"] = np.sin(2 * np.pi * checkin.dt.month / 12)
    l["quote_month_cos"] = np.cos(2 * np.pi * checkin.dt.month / 12)
    l["quote_lead_days"] = (checkin - pd.to_datetime(l.last_scraped)).dt.days
    for c in INTERNAL:
        l[c] = pd.to_numeric(l[c], errors="coerce")
    for c in ["property_type", "neighbourhood_cleansed"]:
        l[c] = l[c].fillna("Unknown").astype(str)
    return l


def add_transit(l):
    stations = pd.DataFrame(json.loads((ROOT / "data/idfm_stations.json").read_text()))
    stations["latitude"] = stations.geo_point_2d.map(lambda p: p["lat"])
    stations["longitude"] = stations.geo_point_2d.map(lambda p: p["lon"])
    # October 2026 additions postdate the Airbnb snapshot; exclude line 18 explicitly.
    excluded = stations.res_com.fillna("").str.contains(r"METRO\s*18", regex=True)
    stations = stations[~excluded].copy()
    stations = stations.drop_duplicates("codeunique")
    points = np.radians(l[["latitude", "longitude"]].to_numpy())
    counts = {"raw_station_records": 1002, "line18_records_removed": int(excluded.sum()),
              "unique_station_records": len(stations), "retrieved": "2026-11-10",
              "source": "https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/"}
    for mode in ["metro", "rer"]:
        s = stations[stations[mode] == 1]
        tree = BallTree(np.radians(s[["latitude", "longitude"]].to_numpy()), metric="haversine")
        l[f"nearest_{mode}_km"] = tree.query(points, k=1)[0][:, 0] * EARTH_KM
        counts[f"{mode}_stations"] = len(s)
        if mode == "metro":
            for metres in [500, 1000]:
                l[f"metro_stations_{metres}m"] = tree.query_radius(points, r=metres / 1000 / EARTH_KM, count_only=True)
    stations[["codeunique", "nom_long", "latitude", "longitude", "metro", "rer", "res_com"]].to_csv(
        ROOT / "data/stations_used.csv", index=False)
    dump(ROOT / "results/transit_provenance.json", counts)
    return l


def price_population(l, audit):
    funnel = {"all_listings": len(l)}
    d = l[l.room_type == "Entire home/apt"].copy()
    funnel["entire_homes"] = len(d)
    d = d[d.minimum_nights.between(1, 29)].copy()
    funnel["minimum_stay_1_to_29"] = len(d)
    d = d[d.price_eur.notna() & d.quote_currency.eq("EUR")].copy()
    funnel["observed_eur_price"] = len(d)
    d = d[d.price_eur.between(20, 1000)].copy()
    funnel["price_20_to_1000_eur"] = len(d)
    d = d[d.host_id.notna() & d.latitude.notna() & d.longitude.notna()].reset_index(drop=True)
    funnel["final_model_sample"] = len(d)
    audit["price_sample_funnel"] = funnel
    audit["price_sample_median_eur"] = float(d.price_eur.median())
    audit["price_sample_hosts"] = int(d.host_id.nunique())
    audit["price_sample_quote_month_counts"] = pd.to_datetime(d.price_quote_checkin_date).dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
    d.groupby("neighbourhood_cleansed").agg(listings=("id", "size"), median_price_eur=("price_eur", "median"),
        median_metro_distance_m=("nearest_metro_km", lambda x: x.median() * 1000)).to_csv(ROOT / "results/neighbourhood_summary.csv")
    return d


def price_model(name):
    nums, cats = PRICE_SETS[name]
    prep = ColumnTransformer([("numeric", "passthrough", nums),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                        encoded_missing_value=np.nan), cats)])
    model = HistGradientBoostingRegressor(max_iter=220, learning_rate=.06, max_leaf_nodes=20,
        l2_regularization=4, min_samples_leaf=30, early_stopping=False, random_state=SEED,
        categorical_features=[False] * len(nums) + [True] * len(cats))
    return Pipeline([("prepare", prep), ("model", model)])


def price_metrics(y, pred):
    return {"mae_eur": float(mean_absolute_error(y, pred)),
            "rmse_eur": float(np.sqrt(mean_squared_error(y, pred))),
            "r2_eur": float(r2_score(y, pred)),
            "r2_log_price": float(r2_score(np.log(y), np.log(pred)))}


def cluster_gain(y, old, new, groups):
    # Positive = smaller MAE with the new feature set. Host-clustered paired bootstrap.
    a = pd.DataFrame({"gain": np.abs(y - old) - np.abs(y - new), "host": np.asarray(groups)})
    g = a.groupby("host").gain.agg(["sum", "size"])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(400):
        picked = rng.integers(0, len(g), len(g))
        b = g.iloc[picked]
        samples.append(b["sum"].sum() / b["size"].sum())
    return {"mae_reduction_eur": float(a.gain.mean()), "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "bootstrap_host_clusters": len(g)}


def fit_prices(d):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    tr_host, te_host = next(splitter.split(d, groups=d.host_id))
    tiles = (np.floor((d.latitude - 48.80) * 111).astype(int).astype(str) + "_" +
             np.floor((d.longitude - 2.22) * 73).astype(int).astype(str))
    tr_space, te_space = next(splitter.split(d, groups=tiles))
    test_hosts = set(d.iloc[te_space].host_id)
    tr_space = np.array([i for i in tr_space if d.iloc[i].host_id not in test_hosts])
    rows, gains, splits = [], [], []
    for protocol, tr, te in [("host_holdout", tr_host, te_host), ("spatial_holdout", tr_space, te_space)]:
        assert set(d.iloc[tr].host_id).isdisjoint(set(d.iloc[te].host_id))
        if protocol == "spatial_holdout":
            assert set(tiles.iloc[tr]).isdisjoint(set(tiles.iloc[te]))
        y = d.iloc[te].price_eur.to_numpy()
        predictions = {"median_baseline": np.full(len(te), d.iloc[tr].price_eur.median())}
        for name in PRICE_SETS:
            print(f"Price model: {protocol} / {name}; train={len(tr)}, test={len(te)}", flush=True)
            model = price_model(name)
            model.fit(d.iloc[tr], np.log(d.iloc[tr].price_eur))
            predictions[name] = np.exp(model.predict(d.iloc[te]))
        for name, pred in predictions.items():
            rows.append({"protocol": protocol, "model": name, "train_rows": len(tr), "test_rows": len(te), **price_metrics(y, pred)})
        for old, new in [("internal", "location"), ("location", "transit")]:
            gains.append({"protocol": protocol, "comparison": f"{new}_vs_{old}", **cluster_gain(y, predictions[old], predictions[new], d.iloc[te].host_id)})
        p = pd.DataFrame({"listing_id": d.iloc[te].id.astype(str).to_numpy(), "actual_price_eur": y, **predictions})
        p.to_csv(ROOT / f"results/price_predictions_{protocol}.csv", index=False)
        splits.append({"protocol": protocol, "train_rows": len(tr), "test_rows": len(te),
                       "test_hosts": int(d.iloc[te].host_id.nunique()), "host_overlap": 0})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "results/price_metrics.csv", index=False)
    pd.DataFrame(gains).to_csv(ROOT / "results/price_feature_gains.csv", index=False)
    dump(ROOT / "results/splits.json", splits)
    # Keep the location specification for the prototype. The feature comparisons are
    # exploratory; a later independent sample is required after model selection.
    chosen = "location"
    final = price_model(chosen).fit(d, np.log(d.price_eur))
    joblib.dump({"pipeline": final, "feature_set": chosen, "target": "log quoted nightly price EUR",
                 "sklearn_version": sklearn.__version__}, ROOT / "models/price_model.joblib")
    # Calibration illustration: a held-out record, scored without fitting on that record.
    candidate = d.iloc[te_host].copy()
    candidate["distance_to_median"] = abs(candidate.price_eur - d.price_eur.median())
    example = candidate.sort_values(["distance_to_median", "id"]).iloc[0]
    p = pd.read_csv(ROOT / "results/price_predictions_host_holdout.csv", dtype={"listing_id": str})
    predicted = float(p.set_index("listing_id").loc[str(example.id), chosen])
    dump(ROOT / "results/example_property.json", {"source_listing_id": str(example.id),
         "neighbourhood": example.neighbourhood_cleansed, "accommodates": int(example.accommodates),
         "observed_quote_eur": float(example.price_eur), "heldout_predicted_quote_eur": predicted,
         "model": chosen, "note": "Illustration using an existing held-out listing, not the user's property."})
    return metrics, gains, predicted, chosen


def review_panel(l, r, audit, sample_size=12000):
    complete_end = pd.Timestamp(l.last_scraped.min()).to_period("M") - 1
    train_start = complete_end - 23
    first = r.groupby("listing_id").date.min()
    eligible = first[first < train_start.start_time].index.to_numpy()
    ids = np.sort(np.random.default_rng(SEED).choice(eligible, min(len(eligible), sample_size), replace=False))
    months = pd.period_range(train_start - 12, complete_end, freq="M")
    rr = r[r.listing_id.isin(ids)].copy()
    rr["month"] = rr.date.dt.to_period("M")
    matrix = rr.groupby(["listing_id", "month"]).size().unstack(fill_value=0).reindex(index=ids, columns=months, fill_value=0).astype(float)
    # Empty cells mean no observed reviews in this surviving cohort, never zero bookings.
    data = []
    first_month = first.reindex(ids).dt.to_period("M").astype("int64").to_numpy()
    for t in range(12, len(months)):
        x = review_features(matrix.to_numpy(), t, months[t], first_month)
        x["listing_id"] = ids
        x["month"] = str(months[t])
        x["target"] = matrix.iloc[:, t].to_numpy()
        data.append(x)
    panel = pd.concat(data, ignore_index=True)
    months_string = [str(m) for m in months[12:]]
    train_months, valid_months, test_months = months_string[:-6], months_string[-6:-3], months_string[-3:]
    audit["review_panel"] = {"population": "Current-snapshot listings with a review before training start; all room types",
        "eligible_listings": len(eligible), "sample_listings": len(ids), "rows": len(panel),
        "train_months": train_months, "validation_months": valid_months, "test_months": test_months,
        "last_complete_month": str(complete_end), "zero_target_share": float((panel.target == 0).mean())}
    panel.to_csv(ROOT / "data/review_month_panel.csv.gz", index=False, compression="gzip")
    return panel, matrix, months, first_month, train_months, valid_months, test_months


def review_features(arr, t, month, first_month):
    x = pd.DataFrame({f"lag_{k}": arr[:, t-k] for k in [1, 2, 3, 6, 12]})
    for k in [3, 6, 12]:
        x[f"mean_{k}"] = arr[:, t-k:t].mean(axis=1)
    x["active_months_12"] = (arr[:, t-12:t] > 0).sum(axis=1)
    x["months_since_first_review"] = month.ordinal - first_month
    x["month_sin"] = np.sin(2 * np.pi * month.month / 12)
    x["month_cos"] = np.cos(2 * np.pi * month.month / 12)
    return x


def review_metrics(y, pred):
    return {"mae_reviews": float(mean_absolute_error(y, pred)),
            "rmse_reviews": float(np.sqrt(mean_squared_error(y, pred))),
            "aggregate_bias_pct": float(100 * (pred.sum() - y.sum()) / y.sum()),
            "observed_reviews": float(y.sum()), "predicted_reviews": float(pred.sum())}


def fit_reviews(inputs):
    panel, matrix, months, first_month, train_months, valid_months, test_months = inputs
    features = [c for c in panel if c not in ["listing_id", "month", "target"]]
    tr = panel[panel.month.isin(train_months)]
    va = panel[panel.month.isin(valid_months)]
    te = panel[panel.month.isin(test_months)]
    choices = []
    for leaves in [8, 20]:
        print(f"Review model validation: {leaves} leaves", flush=True)
        m = HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=leaves, max_iter=150,
            learning_rate=.06, l2_regularization=5, min_samples_leaf=50, early_stopping=False, random_state=SEED)
        m.fit(tr[features], tr.target)
        choices.append((mean_absolute_error(va.target, m.predict(va[features])), leaves))
    leaves = min(choices)[1]
    model = HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=leaves, max_iter=150,
        learning_rate=.06, l2_regularization=5, min_samples_leaf=50, early_stopping=False, random_state=SEED)
    train_valid = panel[~panel.month.isin(test_months)]
    model.fit(train_valid[features], train_valid.target)
    predictions = {"last_month": te.lag_1.to_numpy(), "trailing_3_month_mean": te.mean_3.to_numpy(),
                   "same_month_last_year": te.lag_12.to_numpy(), "gradient_boosting": model.predict(te[features])}
    rows = []
    y = te.target.to_numpy()
    for name, pred in predictions.items():
        rows.append({"evaluation": "rolling_one_month", "model": name, "test_rows": len(y), **review_metrics(y, pred)})
    monthly_results = []
    for month in test_months:
        mask = (te.month == month).to_numpy()
        for name, pred in predictions.items():
            monthly_results.append({"month": month, "model": name, **review_metrics(y[mask], pred[mask])})
    pd.DataFrame(monthly_results).to_csv(ROOT / "results/review_metrics_by_month.csv", index=False)
    # Three-month forecasts from one origin: no actual future lags are allowed.
    first_t = months.get_loc(pd.Period(test_months[0], freq="M"))
    arr = matrix.to_numpy().copy()
    actual_quarter = arr[:, first_t:first_t+3].sum(axis=1)
    qpreds = {"last_3_month_total": arr[:, first_t-3:first_t].sum(axis=1),
              "same_quarter_last_year": arr[:, first_t-12:first_t-9].sum(axis=1)}
    arr[:, first_t:] = np.nan
    for t in range(first_t, first_t + 3):
        arr[:, t] = model.predict(review_features(arr, t, months[t], first_month)[features])
    qpreds["gradient_boosting_recursive"] = arr[:, first_t:first_t+3].sum(axis=1)
    for name, pred in qpreds.items():
        rows.append({"evaluation": "fixed_origin_three_months", "model": name, "test_rows": len(pred), **review_metrics(actual_quarter, pred)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "results/review_metrics.csv", index=False)
    dump(ROOT / "results/review_validation.json", {"validation_candidates": [{"max_leaf_nodes": l, "mae_reviews": a} for a, l in choices],
        "selected_max_leaf_nodes": leaves, "features": features, "quarter_origin": test_months[0]})
    joblib.dump({"pipeline": model, "features": features, "target": "monthly observed reviews", "trained_through": valid_months[-1]},
                ROOT / "models/review_backtest_model.joblib")
    return metrics


def seasonality(r, audit):
    end = pd.Period(audit["review_panel"]["last_complete_month"], freq="M")
    start = end - 11
    first = r.groupby("listing_id").date.min()
    cohort = first[first < start.start_time].index
    rr = r[r.listing_id.isin(cohort) & (r.date >= start.start_time) & (r.date < (end+1).start_time)]
    counts = rr.groupby(rr.date.dt.to_period("M")).size().reindex(pd.period_range(start, end, freq="M"), fill_value=0)
    per_day = counts.to_numpy() / np.array([m.days_in_month for m in counts.index])
    factor = per_day / (counts.sum() / sum(m.days_in_month for m in counts.index))
    s = pd.DataFrame({"period": counts.index.astype(str), "month": counts.index.month, "reviews": counts.to_numpy(),
                      "daily_review_factor": factor}).sort_values("month")
    s.to_csv(ROOT / "results/seasonality.csv", index=False)
    audit["seasonality"] = {"start": str(start), "end": str(end), "cohort_listings": len(cohort),
                            "interpretation": "One realized year of reviews among surviving established listings; scenario weight only."}
    # Plot uses full months only and labels snapshot-survivor population.
    recent = r[(r.date >= (end-47).start_time) & (r.date < (end+1).start_time)]
    monthly = recent.groupby(recent.date.dt.to_period("M")).size()
    monthly.to_csv(ROOT / "results/monthly_reviews.csv", header=["observed_reviews"])
    return s, monthly


def scenario_calculation(rate, factors, config):
    """Conditional scenario arithmetic; no estimation of actual occupancy or return."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Nightly rate must be finite and positive.")
    if len(config["open_nights_by_month"]) != 12:
        raise ValueError("Provide twelve monthly opening-night values.")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in config["occupancy_scenarios"].values()):
        raise ValueError("Occupancy assumptions must lie between zero and one.")
    if not np.isfinite(config["realised_price_multiplier"]) or config["realised_price_multiplier"] <= 0:
        raise ValueError("Realised price multiplier must be positive.")
    if config.get("variable_cost_rate") is not None and not 0 <= config["variable_cost_rate"] <= 1:
        raise ValueError("Variable-cost share must lie between zero and one.")
    for key in ["annual_booked_night_budget", "fixed_cost_per_month_eur", "financing_cash_per_month_eur", "initial_cash_investment_eur"]:
        if config.get(key) is not None and (not np.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f"{key} must be non-negative or null.")
    start = pd.Period(config["start_month"], freq="M")
    periods = pd.period_range(start, periods=12, freq="M")
    rows = []
    for name, base_occupancy in config["occupancy_scenarios"].items():
        annual_remaining = config.get("annual_booked_night_budget")
        for i, period in enumerate(periods):
            open_nights = float(config["open_nights_by_month"][i])
            if not np.isfinite(open_nights) or not 0 <= open_nights <= period.days_in_month:
                raise ValueError(f"Open nights must fit within {period}.")
            factor = float(factors.loc[period.month])
            occupied = min(open_nights, open_nights * base_occupancy * factor)
            if annual_remaining is not None:
                occupied = min(occupied, max(0, annual_remaining))
                annual_remaining -= occupied
            gross = occupied * rate * config["realised_price_multiplier"]
            inputs = [config.get(k) for k in ["variable_cost_rate", "fixed_cost_per_month_eur", "financing_cash_per_month_eur"]]
            cash = None if any(x is None for x in inputs) else gross*(1-inputs[0])-inputs[1]-inputs[2]
            rows.append({"scenario": name, "month": str(period), "calendar_quarter": str(period.asfreq("Q")),
                         "nightly_rate_eur": rate, "open_nights": open_nights,
                         "assumed_base_occupancy": base_occupancy, "review_season_factor": factor,
                         "scenario_booked_nights": occupied, "gross_revenue_eur": gross,
                         "cash_flow_eur": cash})
    return pd.DataFrame(rows)


def scenarios(rate, s):
    config = json.loads((ROOT / "scenario_inputs.json").read_text())
    result = scenario_calculation(rate, s.set_index("month").daily_review_factor, config)
    result.to_csv(ROOT / "results/revenue_scenarios_monthly.csv", index=False)
    quarterly = result.groupby(["scenario", "calendar_quarter"])[["open_nights", "scenario_booked_nights", "gross_revenue_eur"]].sum()
    quarterly["months_in_period"] = result.groupby(["scenario", "calendar_quarter"]).size()
    quarterly.to_csv(ROOT / "results/revenue_scenarios_quarterly.csv")
    horizon = []
    for name, g in result.groupby("scenario", sort=False):
        for n in [1, 3, 12]:
            a = g.head(n)
            cash = a.cash_flow_eur.sum() if a.cash_flow_eur.notna().all() else None
            investment = config.get("initial_cash_investment_eur")
            roi = cash/investment if n == 12 and cash is not None and investment and investment > 0 else None
            horizon.append({"scenario": name, "horizon_months": n, "gross_revenue_eur": a.gross_revenue_eur.sum(),
                            "booked_nights": a.scenario_booked_nights.sum(), "cash_flow_eur": cash,
                            "annual_cash_return_fraction": roi})
    h = pd.DataFrame(horizon)
    h.to_csv(ROOT / "results/revenue_scenarios_horizons.csv", index=False)
    assert np.allclose(result.groupby("scenario").gross_revenue_eur.sum().sort_index(),
                       h[h.horizon_months == 12].set_index("scenario").gross_revenue_eur.sort_index())
    assert (result.scenario_booked_nights <= result.open_nights + 1e-9).all()
    return result, h


def plots(d, prices, reviews, s, monthly, scenario):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].hist(d.price_eur, bins=np.arange(20, 1020, 20), color=COLORS[2])
    axes[0].set(title="Quoted nightly prices", xlabel="EUR per night", ylabel="Listings")
    g = d.groupby("neighbourhood_cleansed").price_eur.median().sort_values()
    axes[1].barh(g.index, g.values, color=COLORS[1])
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].set(title="Median quote by arrondissement", xlabel="EUR per night")
    fig.savefig(ROOT / "figures/prices.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = {"median_baseline": "Median baseline", "internal": "Property + quote", "location": "+ Location", "transit": "+ Transit"}
    for ax, protocol, title in zip(axes, ["host_holdout", "spatial_holdout"], ["Unseen hosts", "Unseen 1 km cells; shared hosts removed"]):
        p = prices[prices.protocol == protocol]
        ax.bar([labels[x] for x in p.model], p.mae_eur, color=COLORS)
        ax.tick_params(axis="x", rotation=20)
        ax.set(title=title, ylabel="Mean absolute error (EUR/night)")
    fig.savefig(ROOT / "figures/price_model_comparison.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(monthly.index.to_timestamp(), monthly.values, color=COLORS[2])
    axes[0].set(title="Review history of snapshot survivors", ylabel="Observed reviews per month")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar([calendar.month_abbr[m] for m in s.month], s.daily_review_factor, color=COLORS[1])
    axes[1].axhline(1, color="#555", lw=1, ls="--")
    axes[1].set(title="Latest complete year's review pattern", ylabel="Daily review rate / annual daily rate")
    fig.savefig(ROOT / "figures/review_history.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, eval_, title in zip(axes, ["rolling_one_month", "fixed_origin_three_months"], ["Rolling next-month review counts", "Three months from a single origin"]):
        p = reviews[reviews.evaluation == eval_]
        ax.barh([x.replace("_", " ") for x in p.model], p.mae_reviews, color=COLORS[:len(p)])
        ax.set(title=title, xlabel="Mean absolute error (reviews per listing)")
    fig.savefig(ROOT / "figures/review_model_comparison.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for (name, p), color in zip(scenario.groupby("scenario", sort=False), [COLORS[1], COLORS[2], COLORS[3]]):
        ax.plot(p.month, p.gross_revenue_eur, "o-", color=color, label=name)
    ax.set(title="Illustrative gross revenue scenarios", ylabel="EUR per month", xlabel="Scenario month")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.savefig(ROOT / "figures/revenue_scenarios.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--review-sample", type=int, default=12000)
    args = parser.parse_args()
    for p in ["data", "results", "figures", "models"]:
        (ROOT / p).mkdir(exist_ok=True)
    print("Auditing source files and review identifiers...", flush=True)
    l, r, audit = read_data(args.data_dir)
    audit["source_files"] = [{"path": str(args.data_dir / name), "bytes": (args.data_dir / name).stat().st_size,
                              "sha256": sha256(args.data_dir / name)} for name in ["listings.csv", "reviews.csv"]]
    dump(ROOT / "results/data_audit.json", audit)
    l = add_transit(add_features(l))
    d = price_population(l, audit)
    d[["id", "host_id", "price_eur", *list(dict.fromkeys(INTERNAL + LOCATION + TRANSIT)),
       "property_type", "neighbourhood_cleansed"]].to_csv(ROOT / "data/price_features.csv.gz", index=False, compression="gzip")
    prices, gains, rate, chosen = fit_prices(d)
    inputs = review_panel(l, r, audit, args.review_sample)
    reviews = fit_reviews(inputs)
    s, monthly = seasonality(r, audit)
    scenario, horizons = scenarios(rate, s)
    plots(d, prices, reviews, s, monthly, scenario)
    audit["price_deployment_feature_set"] = chosen
    audit["package_versions"] = {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    dump(ROOT / "results/data_audit.json", audit)
    print("Analysis complete. Outputs in", ROOT, flush=True)
    print(prices.to_string(index=False), flush=True)
    print(reviews.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
