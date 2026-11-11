"""Build the English report without modifying the executed Jupyter notebook."""
from pathlib import Path
import base64
import html
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent


def table(df, decimals=3):
    d = df.copy()
    for c in d:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "Unavailable" if pd.isna(x) else f"{x:,.{decimals}f}")
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "| " + " | ".join(["---"]*len(d.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join(rows)


def inline(text):
    t = html.escape(text)
    t = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def md_to_html(md):
    blocks = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
            i += 1
        elif line.startswith("!["):
            m = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            data = base64.b64encode((ROOT / m.group(2)).read_bytes()).decode()
            blocks.append(f'<figure><img alt="{html.escape(m.group(1))}" src="data:image/png;base64,{data}"><figcaption>{inline(m.group(1))}</figcaption></figure>')
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in row):
                    rows.append(row)
                i += 1
            blocks.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>" +
                "".join("<tr>"+"".join(f"<td>{inline(c)}</td>" for c in row)+"</tr>" for row in rows[1:]) + "</tbody></table></div>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("<li>"+inline(lines[i][2:])+"</li>"); i += 1
            blocks.append("<ul>"+"".join(items)+"</ul>")
        else:
            paragraph = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![", "- ")):
                paragraph.append(lines[i]); i += 1
            blocks.append("<p>"+inline(" ".join(paragraph))+"</p>")
    return "\n".join(blocks)


def report():
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    price = pd.read_csv(ROOT / "results/price_metrics.csv")
    gains = pd.read_csv(ROOT / "results/price_feature_gains.csv")
    review = pd.read_csv(ROOT / "results/review_metrics.csv")
    valid = json.loads((ROOT / "results/review_validation.json").read_text())
    gain = json.loads((ROOT / "results/review_gain.json").read_text())
    subgroup = pd.read_csv(ROOT / "results/review_error_subgroups.csv")
    example = json.loads((ROOT / "results/example_property.json").read_text())
    horizons = pd.read_csv(ROOT / "results/revenue_scenarios_horizons.csv")
    p = price[(price.protocol == "host_holdout") & (price.model == "location")].iloc[0]
    ri = review[(review.evaluation == "rolling_one_month") & (review.model == "gradient_boosting")].iloc[0]
    rb = review[(review.evaluation == "rolling_one_month") & (review.model == "trailing_3_month_mean")].iloc[0]
    q = review[(review.evaluation == "fixed_origin_three_months") & (review.model == "gradient_boosting_recursive")].iloc[0]
    h = horizons.pivot(index="scenario", columns="horizon_months", values="gross_revenue_eur").reindex(["Lower utilisation", "Central utilisation", "Higher utilisation"])
    h.columns = ["First month EUR", "First 3 months EUR", "Full 12 months EUR"]
    h = h.reset_index().rename(columns={"scenario": "Illustrative scenario"})
    funnel = pd.DataFrame(audit["price_sample_funnel"].items(), columns=["Selection step", "Listings"])
    funnel["Selection step"] = ["All supplied listings", "Entire homes", "Minimum stay of 1–29 nights", "Observed EUR quote", "Quote between EUR 20 and 1,000", "Final modelling sample"]
    overview = pd.DataFrame([
        ("Listings", f'{audit["listing_rows"]:,}'),
        ("Raw review records", f'{audit["review_rows_raw"]:,}'),
        ("Collection window", f'{audit["snapshot_start"]} to {audit["snapshot_end"]}'),
        ("Listings with missing price", f'{audit["price_missing_rows"]:,} ({audit["price_missing_fraction"]:.1%})'),
        ("Clean review records", f'{audit["review_rows_clean"]:,}'),
        ("Last complete review month", "August 2026"),
        ("Price modelling sample", f'{audit["price_sample_funnel"]["final_model_sample"]:,} entire homes'),
        ("Median price in modelling sample", f'EUR {audit["price_sample_median_eur"]:,.0f} per quoted night')], columns=["Measure", "Observed value"])
    price_display = price[["protocol", "model", "test_rows", "mae_eur", "rmse_eur", "r2_log_price"]].rename(columns={
        "protocol": "Test", "model": "Features/model", "test_rows": "Test rows", "mae_eur": "MAE EUR", "rmse_eur": "RMSE EUR", "r2_log_price": "R² log price"})
    review_display = review[["evaluation", "model", "mae_reviews", "rmse_reviews", "aggregate_bias_pct"]].rename(columns={
        "evaluation": "Horizon", "model": "Model", "mae_reviews": "MAE reviews", "rmse_reviews": "RMSE reviews", "aggregate_bias_pct": "Total bias %"})
    price_display["Test"] = price_display["Test"].map({"host_holdout": "Unseen hosts", "spatial_holdout": "Unseen map cells"})
    price_display["Features/model"] = price_display["Features/model"].map({"median_baseline": "Median baseline", "internal": "Property + quote", "location": "+ Location", "transit": "+ Transit"})
    review_display["Horizon"] = review_display["Horizon"].map({"rolling_one_month": "Next month", "fixed_origin_three_months": "Fixed 3 months"})
    review_display["Model"] = review_display["Model"].map({"last_month": "Previous month", "trailing_3_month_mean": "Recent 3-month average", "same_month_last_year": "Same month last year", "gradient_boosting": "Gradient boosting", "last_3_month_total": "Previous 3-month total", "same_quarter_last_year": "Same quarter last year", "gradient_boosting_recursive": "Recursive gradient boosting"})
    md = f"""# Paris Airbnb revenue potential

Research report · Updated 11 November 2026 · Preliminary empirical results

## Abstract

This project examines how far public listing, review and transport data can support a Paris host's earnings assessment. It uses {audit['listing_rows']:,} listings and {audit['review_rows_raw']:,} raw review records supplied in the Airbnb_2 folder. The study also adds three earlier listing/review snapshots and the September calendar. Three components keep the evidence distinct: quoted-price prediction, review-activity forecasting and conditional revenue scenarios.

The location price model achieves a held-out mean absolute error of **EUR {p.mae_eur:.2f} per night** and a log-price R² of **{p.r2_log_price:.3f}** on unseen hosts. Transport-station features do not improve materially on location alone. In the retrospective survivor cohort, a monthly review model reduces error by **{100*(1-ri.mae_reviews/rb.mae_reviews):.1f}%** relative to a trailing-three-month baseline, but overpredicts total reviews by **{ri.aggregate_bias_pct:.1f}%**. A separate earlier-snapshot experiment predicts July reviews: the simple 90-day review-rate baseline has MAE **0.5403**, compared with **0.5820** for boosting with history, property and location. The simpler baseline also has much smaller total bias (**+0.26%** versus **+18.51%**). The earlier improvement therefore does not generalize automatically to the stronger temporal design. Actual host revenue is unavailable, so the earnings component is a scenario calculator with visible assumptions. Its outputs have not been validated against actual income.

## 1. Research questions and scope

- Does adding location improve quoted-price predictions beyond property characteristics and quote context?
- Do official transport-access variables add predictive information beyond coordinates and arrondissement?
- Can earlier review counts predict next-month and next-quarter observed review activity?
- What monthly, quarterly and annual gross revenue follows from an explicit opening plan, price benchmark and occupancy assumption?

The price model covers short-stay entire homes with an observed EUR quote from 20 to 1,000. The original rolling review experiment covers established listings of all room types that survived into September 2026. The added earlier-snapshot experiment covers source listings of all room types observed again at follow-up, including listings with no earlier reviews. These are different populations and different prediction tasks. The review model is not directly converted into booking probabilities or revenue. Cold-start demand, exits, actual occupancy and optimal pricing remain outside the validated scope.

## 2. Data audit

{table(overview)}

All listing IDs are unique and every raw review record maps to a supplied listing. There are **{audit['ambiguous_review_ids']:,} review IDs associated with multiple listings**, affecting **{audit['ambiguous_review_rows_excluded']:,} rows ({audit['ambiguous_review_rows_excluded']/audit['review_rows_raw']:.2%})**. Their allocation is ambiguous; the primary analysis excludes all affected rows rather than assigning them to one listing. No additional exact within-listing duplicates or reviews after their own listing's scrape date remain. This policy may undercount activity for affected properties; a future sensitivity analysis should compare alternative defensible allocation rules.

The raw quotes explicitly identify EUR. The export's dollar symbol is formatting, consistent with the Inside Airbnb dictionary. The prices refer to stays with differing check-in dates and lengths, including dates after collection. The model therefore includes quote month, lead time and stay length. These are quotes observable at collection, not subsequently realized revenue.

Missing prices affect {audit['price_missing_fraction']:.1%} of listings. They are excluded from supervised price fitting, not filled with zero. This selection can favour properties with available quotes; no inference is made about all Paris homes. The EUR 20–1,000 band is an analyst-defined focus on ordinary quotes, not an assertion that excluded prices are errors. Results do not describe the luxury tail.

{table(funnel)}

![Price distribution and arrondissement medians for the selected sample.](figures/prices.png)

## 3. Quoted-price prediction

### Features and model

The internal specification uses capacity, bedrooms, beds, bathrooms, minimum stay, amenities count, five amenity indicators and quote context. The location specification adds latitude, longitude and arrondissement. The transit specification adds distance to the nearest metro and RER station plus metro-station counts within 500 and 1,000 metres. Listing/host IDs, review ratings, review totals, availability and estimated revenue are not predictors.

Official Île-de-France Mobilités data were retrieved on 10 November 2026. The generalized station file contains 1,002 records. Four Métro 18 records were excluded because that addition postdates the Airbnb snapshot, leaving 998 unique station records; 321 serve metro and 239 serve RER. Distances are straight-line haversine distances, not walking times. The contemporary station file is not a fully archived September vintage, so historical alignment remains approximate. Airbnb coordinate anonymization also limits fine distance precision. [Station source](https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/) and [coordinate limitations](https://insideairbnb.com/data-assumptions/).

All three specifications use histogram gradient boosting on log price: 220 iterations, learning rate 0.06, maximum 20 leaves, minimum 30 observations per leaf, L2 penalty 4 and seed 42. Categorical encoding is fitted inside each training split; missing numeric features are handled by the estimator. Exponentiating a log-price prediction gives a price benchmark, not an unbiased expected transaction price. A training-sample median is the simple baseline.

### Evaluation and findings

The first split holds out 20% of hosts. The second holds out approximately 1 km map cells and removes test hosts from training. The spatial split is a dispersed-cell holdout without a buffer; it does not establish performance in a completely separate city. Both protocols use one fixed seed and should be treated as exploratory comparisons. The supplied prototype uses the location specification; a new independent evaluation is needed after choosing a deployment model.

{table(price_display)}

Adding location reduces host-holdout MAE by EUR 12.28, or 13.6%, from EUR 90.00 to EUR 77.72. Its paired host-cluster bootstrap 95% interval for the MAE reduction is EUR 10.73–13.92. The spatial reduction is EUR 6.74 (interval EUR 4.84–8.65). These intervals use 400 resamples and hold the trained model fixed; they do not include retraining variability or all spatial dependence.

Transit additions change host-holdout MAE by a reduction of **EUR −0.15** (interval −0.39 to 0.09) and spatial-holdout MAE by **EUR −0.02** (interval −0.27 to 0.29). Negative reductions indicate slightly worse error. Both intervals include zero. This experiment provides no evidence of useful additional predictive gain from these particular station features. It does not establish that transport has no economic relationship with prices.

![Price prediction errors under the two holdout protocols. Lower is better.](figures/price_model_comparison.png)

## 4. Monthly and quarterly review activity

### Temporal construction

A seeded sample of {audit['review_panel']['sample_listings']:,} listings was drawn from {audit['review_panel']['eligible_listings']:,} current-snapshot listings with at least one review before September 2024. The resulting panel has {audit['review_panel']['rows']:,} listing-month rows. It uses only review-derived features: lags of 1, 2, 3, 6 and 12 months; trailing means; the number of recently active months; months since first review; and month-of-year sine/cosine. A zero denotes no recorded reviews in that cell, not no bookings, no availability or no revenue.

Training targets span September 2024–February 2026. March–May 2026 selects between 8 and 20 leaves; the selected model uses {valid['selected_max_leaf_nodes']} leaves, 150 iterations and Poisson loss. It is refitted through May and tested on June–August 2026. For rolling one-month forecasts, later origins may use reviews already observed in earlier test months. For the three-month experiment, the origin stays at 1 June: subsequent lag inputs are recursively predicted, never replaced with actual June or July counts.

This reconstructs historical features from one final snapshot. It excludes listings that disappeared before collection and assumes a dated review was observable at its posted date. It is therefore a retrospective backtest conditional on survival, not a fully prospective market experiment. Current prices, current ratings and current host status are deliberately absent from historical review features.

{table(review_display)}

The monthly model's MAE is **{ri.mae_reviews:.3f} reviews per listing-month**, compared with **{rb.mae_reviews:.3f}** for the recent-average baseline. The paired listing-cluster interval for the MAE reduction is {gain['ci95_low']:.3f}–{gain['ci95_high']:.3f} reviews, conditional on the fitted model. It still overpredicts the total by {ri.aggregate_bias_pct:.1f}%.

At the fixed three-month origin, model MAE is **{q.mae_reviews:.3f} reviews per listing**, compared with 1.229 for the preceding-quarter total. Aggregate bias is **{q.aggregate_bias_pct:.1f}%**, versus 1.1% for that baseline. Thus the model improves typical listing error while the baseline better matches the aggregate total in this quarter. The preferred metric depends on the intended use.

About {audit['review_panel']['zero_target_share']:.1%} of panel targets are zero. In the test set, error is {subgroup.iloc[0].mae_reviews:.3f} for zero-review rows and {subgroup.iloc[1].mae_reviews:.3f} for positive-review rows. The overall error should not be interpreted as equally strong performance for busy listings. There are only three held-out months, so year-round forecast stability remains untested.

![Monthly review history and one year's seasonal pattern. The population is conditioned on survival into the snapshot.](figures/review_history.png)

![Review forecasting errors for rolling monthly and fixed-origin quarterly evaluation.](figures/review_model_comparison.png)

## 5. Revenue scenarios and investment-return logic

The worked example uses an existing held-out listing in **{example['neighbourhood']}**, accommodating {example['accommodates']} people. Its observed quote is EUR {example['observed_quote_eur']:.2f}; its held-out predicted benchmark is **EUR {example['heldout_predicted_quote_eur']:.2f}**. It is an illustration, not the user's property or a recommendation.

The scenario period is December 2026–November 2027. Each month has 10 assumed open nights. Base utilization assumptions are 35%, 50% and 65%. They are weighted by the daily review pattern observed over September 2025–August 2026 in a fixed established cohort of {audit['seasonality']['cohort_listings']:,} surviving listings. This transfers a single year's review pattern into assumed occupancy seasonality; it is not an estimated occupancy relationship. The rate is held constant, the realized-price multiplier is 1, and no annual night-budget constraint is imposed.

- Scenario booked nights = the smaller of open nights and open nights × assumed base utilization × seasonal weight.
- Gross accommodation revenue = scenario booked nights × nightly benchmark × assumed realized-price multiplier.
- Monthly cash flow = gross revenue × (1 − variable-cost share) − fixed monthly costs − financing cash payments.
- Annual cash-on-cash return = annual cash flow ÷ initial cash invested, when both are known and the investment is positive. This is before income tax and excludes capital appreciation.

{table(h, decimals=0)}

These amounts are gross scenario outcomes. They are not prediction intervals, guaranteed earnings, estimates of rental legality or simulated labels used to score a revenue model. Price-demand interactions, season-specific prices and uncertainty in the nightly benchmark are not jointly estimated. Quarterly totals add the corresponding monthly values; the first three months are December–February, not a calendar quarter. The separate calendar-quarter export labels partial quarters with their month count.

Costs, financing and investment are missing. Cash flow and return therefore remain unavailable. The calculator supports those inputs without treating blanks as zero. For example, the verification uses a synthetic EUR 100 rate, 10 open nights, 50% utilization, 20% variable costs, EUR 100 fixed costs and EUR 20 financing to verify EUR 500 revenue and EUR 280 cash flow. That synthetic arithmetic check is not part of the empirical results.

![Illustrative revenue scenarios under the stated opening and utilization assumptions.](figures/revenue_scenarios.png)

## 6. What is needed to complete the original earnings objective

**The highest priority is anonymized booking or monthly revenue data**, including offered nights, occupied nights, actual accommodation revenue, refunds and period definitions. These labels are necessary to measure actual revenue error. For personal returns, supply the opening plan, operating costs, financing payments and initial investment.

Four dated listing/review snapshots have now been acquired and used for an initial temporal experiment described below. A longer history with consistent coverage is still needed. The downloaded September calendar contains availability and stay restrictions, with no price fields; calendar unavailability does not uniquely identify bookings. Weather, tourism amenities and socioeconomic data are extensions, not substitutes for observed earnings.

For a weather study, match forecasts to their original issue time and prediction horizon. The forecast archive must be known at the origin; a full future month's realized weather cannot be an input. For neighborhood income, the [INSEE 2021 IRIS release](https://www.insee.fr/fr/statistiques/8229323) is one documented option and requires matching boundary geography. Neither income nor weather was incorporated into the results above. Details and source links are in DATA_NEEDED.md.

## 7. Reproducibility and study limitations

Raw files were left unchanged. Their SHA-256 hashes, data-selection counts, package versions, saved predictions and metrics are retained. The notebook includes the implementation in its own cells and was executed in a Jupyter kernel. All 22 code cells passed. By default, it refits every model using the packaged analytical tables; its rebuild option reconstructs inputs from the raw public files. Verification independently recomputed headline price errors, checked temporal lags across split boundaries, reproduced the review score from the saved model and reconciled monthly-to-annual arithmetic, including zero-capacity and missing-cost cases.

The main limitations are one snapshot for price modeling, uneven coverage across the historical snapshots, missing follow-up outcomes, selection from missing prices and price trimming, review allocation ambiguity, survivorship, approximate station vintage and anonymized coordinates, short temporal testing, and no observed occupancy or revenue. Price modeling is cross-sectional; it has not been tested against later price realizations. The source quotes were collected in September, so they may be stale for a current November decision. No causal or investment recommendation follows from these associations.

The evidence supports a price benchmark, while gains in review forecasting depend on the evaluation design. The original rolling experiment improves on its recent-average baseline, but the added earlier-snapshot experiment favors a simple 90-day rate on MAE and total calibration. It does not support a claim that external data necessarily produces a large performance jump or that actual annual income has been predicted accurately. The project can advance to validated earnings forecasting once real revenue labels and time-appropriate inputs are available. No course rubric was supplied, so report length and required assessment methods still need alignment with the course.

## References

- Inside Airbnb. Paris detailed listings and reviews, collected 17–24 September 2026, supplied locally and analyzed 10 November 2026. [Dataset portal](https://insideairbnb.com/get-the-data/). Source-file hashes appear in results/data_audit.json.
- Inside Airbnb. [Data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit) and [data assumptions](https://insideairbnb.com/data-assumptions/). Consulted 10 November 2026.
- Île-de-France Mobilités. [Gares et stations du réseau ferré d'Île-de-France, donnée généralisée](https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/). Retrieved 10 November 2026; filtered as described.
- INSEE. [Revenus, pauvreté et niveau de vie en 2021, IRIS](https://www.insee.fr/fr/statistiques/8229323). Published 31 July 2024, geography 1 January 2022. Proposed extension, not used in fitted models.
- Open-Meteo. [Previous Model Runs API](https://open-meteo.com/en/docs/previous-runs-api). Proposed short-range weather source, not used in fitted models.
"""
    for filename in ["ARCHIVE_UPDATE.md", "TEMPORAL_UPDATE.md"]:
        update = ROOT / filename
        if update.exists():
            md = md.replace("## References", update.read_text(encoding="utf-8") + "\n\n## References")
    (ROOT / "REPORT.md").write_text(md, encoding="utf-8")
    style = """body{margin:0;background:#f4f2ed;color:#20323d;font:17px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1050px;margin:40px auto;padding:55px 65px;background:white;border-top:7px solid #0b6e75}h1{font-size:43px;line-height:1.16;margin:0 0 20px;color:#123945}h2{font-size:27px;margin-top:52px;border-bottom:1px solid #dae2e4;padding-bottom:12px}h3{font-size:20px;margin-top:28px}p{max-width:950px}a{color:#096977}strong{color:#163d49}.table-wrap{overflow-x:auto;margin:25px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}th{background:#e4eeef;text-align:left;color:#153e46}th,td{padding:11px 12px;border-bottom:1px solid #dae3e5;vertical-align:top}tr:nth-child(even) td{background:#f8fafb}td:not(:first-child){font-variant-numeric:tabular-nums}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#5d6c74;margin-top:10px}code{font-size:90%;background:#edf3f3;padding:1px 4px}li{margin:8px 0}footer{margin-top:50px;border-top:1px solid #ccd9dd;padding-top:15px;font-size:13px;color:#687a82}@media(max-width:800px){main{margin:0;padding:25px}h1{font-size:32px}h2{font-size:23px}}@media print{body{background:white;font-size:10pt}main{padding:0;margin:0;max-width:none;border:0}h1{font-size:25pt}h2{font-size:17pt;break-after:avoid}figure,table{break-inside:avoid}a{color:inherit}h2{margin-top:24pt}th,td{padding:6px}}"""
    doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paris Airbnb revenue potential</title><style>'+style+'</style></head><body><main>'+md_to_html(md)+'<footer>Paris Airbnb project · Reproducible preliminary analysis · 11 November 2026</footer></main></body></html>'
    (ROOT / "REPORT.html").write_text(doc, encoding="utf-8")
    return md


if __name__ == "__main__":
    report()
    print("English report updated; the notebook is preserved.")
