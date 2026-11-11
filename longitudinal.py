"""Earlier snapshots predict a complete month of later observed reviews."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from analysis import ROOT, DEFAULT_DATA, SEED, dump, sha256, review_metrics

DATES = ["2025-12-13", "2026-03-21", "2026-06-16", "2026-09-16"]
HISTORY_COLS = ["reviews_30", "reviews_90", "reviews_365", "reviews_total",
                "days_since_last_review", "days_since_first_review", "capture_age_days"]
PROPERTY_COLS = ["accommodates", "bedrooms", "beds", "bathrooms_n", "minimum_nights", "superhost", "review_scores_rating"]
TEMPORAL_SETS = {
    "history": (HISTORY_COLS, []),
    "history_property": (HISTORY_COLS + PROPERTY_COLS, ["room_type"]),
    "history_property_location": (HISTORY_COLS + PROPERTY_COLS + ["latitude", "longitude"], ["room_type", "neighbourhood_cleansed"]),
}


def snapshot_path(date, kind):
    if date == DATES[-1]:
        return DEFAULT_DATA / f"{kind}.csv"
    return ROOT / "data/insideairbnb" / f"{kind}_{date}.csv.gz"


def read_historical_snapshot(date):
    columns = ["id", "host_id", "last_scraped", "room_type", "accommodates", "bedrooms", "beds",
               "bathrooms", "bathrooms_text", "minimum_nights", "host_is_superhost", "review_scores_rating",
               "latitude", "longitude", "neighbourhood_cleansed"]
    listings = pd.read_csv(snapshot_path(date, "listings"), usecols=columns, low_memory=False)
    listings["last_scraped"] = pd.to_datetime(listings.last_scraped)
    assert listings.id.is_unique
    reviews = pd.read_csv(snapshot_path(date, "reviews"), usecols=["id", "listing_id", "date"],
                          dtype={"id": "int64", "listing_id": "int64"}, parse_dates=["date"])
    audit = {"snapshot": date, "raw_reviews": len(reviews)}
    counts = reviews.groupby("id").agg(listing_count=("listing_id", "nunique"), date_count=("date", "nunique"))
    ambiguous = counts.index[(counts.listing_count > 1) | (counts.date_count > 1)]
    audit["ambiguous_review_ids"] = len(ambiguous)
    audit["ambiguous_review_rows_excluded"] = int(reviews.id.isin(ambiguous).sum())
    reviews = reviews[~reviews.id.isin(ambiguous)].drop_duplicates(["id", "listing_id", "date"]).copy()
    own_capture = reviews.listing_id.map(listings.set_index("id").last_scraped)
    valid = own_capture.notna() & reviews.date.notna() & (reviews.date <= own_capture)
    audit["unmatched_or_after_capture_rows_excluded"] = int((~valid).sum())
    reviews = reviews[valid].copy()
    audit["clean_reviews"] = len(reviews)
    return listings, reviews, audit


def historical_features(listings, reviews, origin):
    x = listings.copy().set_index("id", drop=False)
    own_capture = reviews.listing_id.map(x.last_scraped)
    for days in [30, 90, 365]:
        counts = reviews[reviews.date > own_capture - pd.Timedelta(days=days)].groupby("listing_id").size()
        x[f"reviews_{days}"] = counts.reindex(x.index, fill_value=0).astype(float)
    history = reviews.groupby("listing_id").agg(total=("id", "size"), first=("date", "min"), last=("date", "max"))
    x["reviews_total"] = history.total.reindex(x.index, fill_value=0).astype(float)
    x["days_since_last_review"] = (origin - history["last"].reindex(x.index)).dt.days
    x["days_since_first_review"] = (origin - history["first"].reindex(x.index)).dt.days
    x["capture_age_days"] = (origin - x.last_scraped).dt.days
    x["bathrooms_n"] = pd.to_numeric(x.bathrooms, errors="coerce").fillna(
        pd.to_numeric(x.bathrooms_text.str.extract(r"([0-9.]+)")[0], errors="coerce"))
    x["superhost"] = x.host_is_superhost.map({"t": 1., "f": 0.})
    for col in PROPERTY_COLS + ["latitude", "longitude"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in ["room_type", "neighbourhood_cleansed"]:
        x[col] = x[col].fillna("Unknown").astype(str)
    return x


def build_temporal_panel():
    output, coverage, review_audits, manifest = [], [], [], []
    source_l, source_r, source_audit = read_historical_snapshot(DATES[0])
    review_audits.append(source_audit)
    for index, (earlier, later) in enumerate(zip(DATES[:-1], DATES[1:])):
        future_l, future_r, future_audit = read_historical_snapshot(later)
        review_audits.append(future_audit)
        # Use the first complete calendar month after all source records were captured.
        origin = source_l.last_scraped.max().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
        end = origin + pd.offsets.MonthBegin(1)
        assert source_l.last_scraped.max() < origin
        assert end <= future_l.last_scraped.min()
        x = historical_features(source_l, source_r, origin)
        followup_observed = x.id.isin(future_l.id)
        targets = future_r[(future_r.date >= origin) & (future_r.date < end)].groupby("listing_id").size()
        x["target"] = targets.reindex(x.index, fill_value=0).astype(float)
        # Without a follow-up listing, a zero count is not a known zero target.
        x.loc[~followup_observed, "target"] = np.nan
        x["followup_observed"] = followup_observed
        x["source_snapshot"] = earlier
        x["followup_snapshot"] = later
        x["target_month"] = origin.strftime("%Y-%m")
        x["target_days"] = origin.days_in_month
        x["split"] = ["train", "validation", "test"][index]
        coverage.append({"source_snapshot": earlier, "followup_snapshot": later,
            "target_month": origin.strftime("%Y-%m"), "split": x.split.iloc[0],
            "source_listings": len(x), "observable_outcomes": int(followup_observed.sum()),
            "missing_followup_outcomes": int((~followup_observed).sum()),
            "mean_target_reviews_observed": float(x.target.mean()),
            "source_capture_max": str(source_l.last_scraped.max().date()),
            "followup_capture_min": str(future_l.last_scraped.min().date())})
        selected = ["id", "host_id", "last_scraped", "source_snapshot", "followup_snapshot", "target_month", "target_days", "split", "target", "followup_observed",
                    *HISTORY_COLS, *PROPERTY_COLS, "latitude", "longitude", "room_type", "neighbourhood_cleansed"]
        output.append(x[selected].reset_index(drop=True))
        print(f"{earlier} → {later}: predict {origin:%Y-%m}; observed {followup_observed.sum():,}/{len(x):,}", flush=True)
        source_l, source_r = future_l, future_r
    panel = pd.concat(output, ignore_index=True)
    assert not panel.duplicated(["id", "target_month"]).any()
    assert panel.loc[~panel.followup_observed, "target"].isna().all()
    assert (pd.to_datetime(panel.last_scraped) < pd.to_datetime(panel.target_month + "-01")).all()
    panel.to_csv(ROOT / "data/insideairbnb/temporal_review_panel.csv.gz", index=False, compression="gzip")
    pd.DataFrame(coverage).to_csv(ROOT / "results/temporal_coverage.csv", index=False)
    pd.DataFrame(review_audits).to_csv(ROOT / "results/historical_review_audit.csv", index=False)
    for date in DATES:
        p = snapshot_path(date, "reviews")
        manifest.append({"snapshot": date, "url": f"https://data.insideairbnb.com/france/ile-de-france/paris/{date}/data/reviews.csv.gz",
            "local_file": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
            "retrieval": "user-supplied" if date == DATES[-1] else "2026-11-11"})
    dump(ROOT / "results/historical_review_manifest.json", manifest)
    return panel


def temporal_model(name, leaves):
    nums, cats = TEMPORAL_SETS[name]
    transforms = [("numeric", "passthrough", nums)]
    if cats:
        transforms.append(("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), cats))
    prepare = ColumnTransformer(transforms)
    model = HistGradientBoostingRegressor(loss="poisson", learning_rate=.06, max_iter=140,
        max_leaf_nodes=leaves, min_samples_leaf=50, l2_regularization=5, early_stopping=False,
        categorical_features=[False]*len(nums)+[True]*len(cats), random_state=SEED)
    return Pipeline([("prepare", prepare), ("model", model)])


def fit_temporal_models(panel):
    observed = panel[panel.followup_observed].copy()
    train = observed[observed.split == "train"]
    valid = observed[observed.split == "validation"]
    test = observed[observed.split == "test"]
    candidates = []
    for leaves in [8, 20]:
        model = temporal_model("history_property", leaves).fit(train, train.target)
        score = float(np.abs(valid.target.to_numpy() - model.predict(valid)).mean())
        candidates.append({"leaves": leaves, "validation_mae_reviews": score})
    chosen = min(candidates, key=lambda x: x["validation_mae_reviews"])["leaves"]
    preds = {"zero_baseline": np.zeros(len(test)),
        "recent_30_day_rate": test.reviews_30.to_numpy() * test.target_days.to_numpy()/30,
        "recent_90_day_rate": test.reviews_90.to_numpy() * test.target_days.to_numpy()/90}
    refit = observed[observed.split.isin(["train", "validation"])]
    for name in TEMPORAL_SETS:
        model = temporal_model(name, chosen).fit(refit, refit.target)
        preds[name] = model.predict(test)
    metrics = pd.DataFrame([{"model": name, "target_month": test.target_month.iloc[0], "test_rows": len(test),
                             **review_metrics(test.target.to_numpy(), prediction)} for name, prediction in preds.items()])
    metrics.to_csv(ROOT / "results/temporal_review_metrics.csv", index=False)
    prediction_table = pd.DataFrame({"listing_id": test.id.astype(str).to_numpy(), "host_id": test.host_id.astype(str).to_numpy(),
                                     "target": test.target.to_numpy(), **preds})
    prediction_table.to_csv(ROOT / "results/temporal_review_predictions.csv", index=False)
    dump(ROOT / "results/temporal_model_config.json", {"validation_candidates": candidates, "selected_leaves": chosen,
          "train_month": train.target_month.iloc[0], "validation_month": valid.target_month.iloc[0],
          "test_month": test.target_month.iloc[0], "features": TEMPORAL_SETS,
          "scope": "All source listing types with observed follow-up. Missing follow-up excluded from scoring, not set to zero."})
    # Historical feature dates and input files are auditable; the test month never selects hyperparameters.
    assert np.allclose(metrics.set_index("model").loc["recent_90_day_rate", "mae_reviews"],
                       np.abs(test.target.to_numpy()-preds["recent_90_day_rate"]).mean())
    return metrics


def temporal_figure(metrics):
    import matplotlib.pyplot as plt
    labels = {"zero_baseline": "Always zero", "recent_30_day_rate": "Prior 30-day rate", "recent_90_day_rate": "Prior 90-day rate",
              "history": "Review history", "history_property": "History + property", "history_property_location": "History + property + location"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = [labels[n] for n in metrics.model]
    axes[0].barh(names, metrics.mae_reviews, color=["#a0a9b0"]*3 + ["#557893", "#0b6e75", "#cf8c36"])
    axes[0].set(xlabel="Mean absolute error (July reviews per listing)", title="Earlier snapshots predict July 2026")
    axes[1].barh(names, metrics.aggregate_bias_pct, color=["#a0a9b0"]*3 + ["#557893", "#0b6e75", "#cf8c36"])
    axes[1].axvline(0, color="#333", lw=1)
    axes[1].set(xlabel="Error in total predicted reviews (%)", title="Aggregate bias")
    fig.savefig(ROOT / "figures/temporal_review_models.png", dpi=170)
    plt.close(fig)


def main():
    panel = build_temporal_panel()
    metrics = fit_temporal_models(panel)
    temporal_figure(metrics)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
