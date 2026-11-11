"""Assemble a notebook with the analysis implementation visible in code cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def functions_from(filename):
    source = (ROOT / filename).read_text()
    tree = ast.parse(source)
    return source, {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def build():
    source, fn = functions_from("analysis.py")
    temporal_source, tf = functions_from("longitudinal.py")
    archive_source, af = functions_from("audit_archives.py")
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True), "id": f"cell-{len(cells):03d}"})
    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None,
                      "outputs": [], "id": f"cell-{len(cells):03d}"})
    def defs(*names):
        code("\n\n\n".join(fn[n] for n in names))
    md("""# Paris Airbnb: prices, future review activity and revenue scenarios

**Jupyter research notebook · Updated 11 November 2026 · English**

This notebook contains the implementation, model fitting and outputs in its cells. It uses four Paris listing snapshots, four review snapshots and the September calendar, with official transport locations for an external-data comparison. No project Python module is imported: the processing functions are included below.

The study distinguishes quoted prices, observed reviews and assumed earnings. Actual host revenue is unavailable. Nothing in this notebook treats calendar unavailability as confirmed bookings.

**How to run:** extract the project package, open this notebook from its `airbnb_project` directory and choose **Restart Kernel and Run All**. The default reuses the supplied cleaned analytical tables, then refits every model. Set `REBUILD_INPUTS = True` to also reconstruct those tables from the raw source files. Missing public raw files can be downloaded when that option is enabled. The full raw inputs use several GB of disk space. The original user-supplied files are never overwritten.

Dependencies: Python 3, numpy, pandas, scikit-learn, matplotlib, joblib and IPython; see `requirements.txt`. Jupyter Notebook or JupyterLab is the interface used to open this file.
""")
    prefix = source[:source.index("def dump(")]
    prefix = prefix.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path.cwd().resolve()')
    prefix += "\nfrom IPython.display import display, Image\nimport gzip, shutil, urllib.request\nREBUILD_INPUTS = False\nDOWNLOAD_MISSING = True\nif not DEFAULT_DATA.exists():\n    DEFAULT_DATA = ROOT / 'data/raw'\nfor folder in ['data', 'data/insideairbnb', 'results', 'figures', 'models']:\n    (ROOT / folder).mkdir(parents=True, exist_ok=True)\nprint('Project directory:', ROOT)\nprint('Rebuild raw inputs:', REBUILD_INPUTS)\nprint('pandas:', pd.__version__, 'scikit-learn:', sklearn.__version__)\n"
    code(prefix)
    defs("dump", "sha256")
    md("""## 1. Source files and download controls

The public Paris snapshots are dated 13 December 2025, 21 March 2026, 16 June 2026 and 16 September 2026. Each listing also has its actual capture date; that date, rather than the batch label, determines whether an input precedes a forecast.

The official station file was acquired on 10 November 2026. The original September listings and reviews are reused if available at the configured local path. On a different computer, change `DEFAULT_DATA` above or enable rebuilding to download the public copies.
""")
    code("""SOURCE_ROOT = 'https://data.insideairbnb.com/france/ile-de-france/paris/'
DATES = ['2025-12-13', '2026-03-21', '2026-06-16', '2026-09-16']

def download_if_missing(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    if not DOWNLOAD_MISSING:
        raise FileNotFoundError(f'Missing source: {destination}. Enable DOWNLOAD_MISSING or supply the file.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    print('Downloading', url, flush=True)
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as out:
        shutil.copyfileobj(response, out, length=8*1024*1024)
    temporary.replace(destination)

if REBUILD_INPUTS:
    DEFAULT_DATA.mkdir(parents=True, exist_ok=True)
    for kind in ['listings', 'reviews']:
        dest = DEFAULT_DATA / f'{kind}.csv'
        if not dest.exists():
            compressed = ROOT / 'data/insideairbnb' / f'{kind}_{DATES[-1]}.csv.gz'
            download_if_missing(SOURCE_ROOT + DATES[-1] + f'/data/{kind}.csv.gz', compressed)
            temporary = dest.with_name(dest.name + '.part')
            with gzip.open(compressed, 'rb') as src, temporary.open('wb') as out:
                shutil.copyfileobj(src, out)
            temporary.replace(dest)
    for date in DATES[:-1]:
        for kind in ['listings', 'reviews']:
            download_if_missing(SOURCE_ROOT + date + f'/data/{kind}.csv.gz', ROOT / 'data/insideairbnb' / f'{kind}_{date}.csv.gz')
    download_if_missing(SOURCE_ROOT + DATES[-1] + '/data/calendar.csv.gz', ROOT / 'data/insideairbnb/calendar_2026-09-16.csv.gz')
    download_if_missing('https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/emplacement-des-gares-idf-data-generalisee/exports/json', ROOT / 'data/idfm_stations.json')
else:
    needed = ['data/price_features.csv.gz', 'data/review_month_panel.csv.gz', 'data/insideairbnb/temporal_review_panel.csv.gz', 'results/data_audit.json']
    missing = [p for p in needed if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f'Extract the complete project package, or set REBUILD_INPUTS=True. Missing: {missing}')
print('Source configuration is ready.')
""")
    md("""## 2. Clean the latest listings and reviews

Reviews are keyed by review ID and listing ID. All rows for an ID assigned to multiple listings or dates are excluded rather than allocated arbitrarily. Reviews after their own listing's capture date are also excluded. No review text, guest names or host names are read.

Prices are parsed as EUR after checking the quote's currency metadata. Missing prices are not zero. The model population is entire homes with a minimum stay of 1–29 nights and an observed quote between EUR 20 and EUR 1,000. This limits generalization to missing-price listings and the luxury tail.
""")
    defs("read_data", "add_features", "add_transit", "price_population")
    code("""if REBUILD_INPUTS:
    latest_listings, latest_reviews, audit = read_data(DEFAULT_DATA)
    audit['source_files'] = [{'path': str(DEFAULT_DATA/name), 'bytes': (DEFAULT_DATA/name).stat().st_size,
                             'sha256': sha256(DEFAULT_DATA/name)} for name in ['listings.csv', 'reviews.csv']]
    latest_listings = add_transit(add_features(latest_listings))
    price_data = price_population(latest_listings, audit)
    keep = ['id', 'host_id', 'price_eur', *list(dict.fromkeys(INTERNAL + LOCATION + TRANSIT)), 'property_type', 'neighbourhood_cleansed']
    price_data[keep].to_csv(ROOT/'data/price_features.csv.gz', index=False, compression='gzip')
else:
    audit = json.loads((ROOT/'results/data_audit.json').read_text())
    price_data = pd.read_csv(ROOT/'data/price_features.csv.gz')
display(pd.DataFrame(audit['price_sample_funnel'].items(), columns=['Selection step', 'Listings']))
print('Missing prices in original listings:', f"{audit['price_missing_fraction']:.1%}")
print('Ambiguous review rows excluded:', f"{audit['ambiguous_review_rows_excluded']:,}")
display(price_data.head())
""")
    md("""## 3. Price features, model and holdout design

The internal model uses physical attributes, amenities and quote context. Location adds coordinates and arrondissement. Transit adds distances and station densities. IDs, current review scores, availability and estimated revenue are excluded.

The estimator predicts log quoted price. Exponentiating yields a price benchmark, not an unbiased expected transaction price or an optimal rate. All encoders are fitted on the training split. One evaluation holds out hosts; the other holds out approximately 1 km cells and removes overlapping hosts from training. The bootstrap intervals are conditional on a fixed fitted model and do not capture all spatial dependence.
""")
    defs("price_model", "price_metrics", "cluster_gain", "fit_prices")
    code("""price_metrics_table, price_gains, example_rate, chosen_price_features = fit_prices(price_data)
display(price_metrics_table.round(3))
display(pd.DataFrame(price_gains).round(3))
""")
    md("""## 4. Reconstruct the rolling monthly review experiment

This first review experiment reconstructs history for 12,000 listings visible in September 2026. It is conditional on survival into that snapshot. It uses only past review counts; contemporary listing attributes are absent from historical predictors.

Training: September 2024–February 2026. Validation: March–May 2026. Test: June–August 2026. A missing listing-month in this observed cohort means no recorded reviews, never zero bookings. The latest September month is excluded as incomplete.
""")
    defs("review_features", "review_panel")
    code("""def restore_review_inputs(audit):
    panel = pd.read_csv(ROOT/'data/review_month_panel.csv.gz')
    cfg = audit['review_panel']
    months = pd.period_range(pd.Period(cfg['train_months'][0], freq='M')-12, cfg['last_complete_month'], freq='M')
    ids = np.sort(panel.listing_id.unique())
    matrix = pd.DataFrame(np.nan, index=ids, columns=months)
    first_month = None
    for key, group in panel.groupby('month', sort=True):
        month = pd.Period(key, freq='M')
        g = group.set_index('listing_id').reindex(ids)
        matrix[month] = g.target.to_numpy()
        if month-12 < pd.Period(cfg['train_months'][0], freq='M'):
            matrix[month-12] = g.lag_12.to_numpy()
        if first_month is None:
            first_month = month.ordinal - g.months_since_first_review.to_numpy()
    assert matrix.notna().all().all()
    return panel, matrix, months, first_month, cfg['train_months'], cfg['validation_months'], cfg['test_months']

if REBUILD_INPUTS:
    review_inputs = review_panel(latest_listings, latest_reviews, audit, sample_size=12000)
else:
    review_inputs = restore_review_inputs(audit)
display(review_inputs[0].head())
print('Rows:', len(review_inputs[0]), 'Listings:', len(review_inputs[1]))
""")
    md("""## 5. Fit and evaluate monthly and fixed-origin quarterly review models

Two tree complexities are compared on validation months only. Rolling monthly forecasts can update their history when an earlier test month becomes observed. The fixed three-month forecast has a single origin: predicted counts, rather than actual future counts, enter later lags.

MAE describes listing-level errors. Aggregate bias answers a different question: whether total activity is over- or underpredicted. Neither is a revenue-accuracy measure.
""")
    defs("review_metrics", "fit_reviews")
    code("""review_metrics_table = fit_reviews(review_inputs)
display(review_metrics_table.round(3))
""")
    md("""## 6. Inspect the additional historical snapshots and calendar

The four snapshots are not interchangeable. December has no price observations and lacks newer quote-context fields. March has substantially fewer listings. A listing not seen later is a missing outcome, not a verified closure.

The September calendar contains availability and stay restrictions but no price column. Availability is not booked occupancy. The first and last calendar months may be partial. The function below processes the calendar in chunks and verifies the monthly totals against source totals.
""")
    code(af["main"].replace("def main():", "def audit_archive_inputs():", 1))
    code("""if REBUILD_INPUTS:
    audit_archive_inputs()
archive = json.loads((ROOT/'results/archive_audit.json').read_text())
display(pd.read_csv(ROOT/'results/archive_snapshots.csv'))
display(pd.read_csv(ROOT/'results/archive_transitions.csv'))
print(json.dumps(archive['calendar'], indent=2))
""")
    md("""## 7. Build an earlier-snapshot forecasting experiment

This extension uses historical listing attributes and review histories from the source snapshot, then obtains outcomes from a later snapshot. The forecast month starts after every source listing was captured and ends before the follow-up batch starts. We train on February 2026, validate on May and test on July.

If a source listing is absent from the follow-up snapshot, its target remains missing and it is excluded from scoring. This reduces dependence on the final September survivor cohort, but evaluation is still conditional on being observed at follow-up. Only three target months are available, so seasonal stability cannot be established.

Prices are omitted because December's prices are entirely missing. Current-as-of-source ratings and host status are allowed here because they were captured before the forecast origin. No September attribute is used to predict an earlier target month.
""")
    temporal_tree = ast.parse(temporal_source)
    assignments = [ast.get_source_segment(temporal_source, n) for n in temporal_tree.body if isinstance(n, ast.Assign)]
    code("\n\n".join(assignments))
    code("\n\n\n".join(tf[n] for n in ["snapshot_path", "read_historical_snapshot", "historical_features", "build_temporal_panel"]))
    code("""if REBUILD_INPUTS:
    temporal_panel = build_temporal_panel()
else:
    temporal_panel = pd.read_csv(ROOT/'data/insideairbnb/temporal_review_panel.csv.gz', parse_dates=['last_scraped'])
assert temporal_panel.loc[~temporal_panel.followup_observed, 'target'].isna().all()
assert (temporal_panel.last_scraped < pd.to_datetime(temporal_panel.target_month+'-01')).all()
display(pd.read_csv(ROOT/'results/temporal_coverage.csv'))
display(pd.read_csv(ROOT/'results/historical_review_audit.csv'))
""")
    md("""## 8. Test whether historical property and location features help

The validation month chooses between 8 and 20 leaves. All feature specifications share that choice, and the July test does not tune the model. Baselines include zero reviews and the recent 30-/90-day review rate scaled to the target month's days. Feature sets are historical review activity, then property characteristics, then location.

Compare these models within this test. Do not compare their scores directly with the earlier rolling experiment: dates, population, inputs and observation windows differ.
""")
    code("\n\n\n".join(tf[n] for n in ["temporal_model", "fit_temporal_models", "temporal_figure"]))
    code("""temporal_metrics = fit_temporal_models(temporal_panel)
temporal_figure(temporal_metrics)
display(temporal_metrics.round(3))
display(Image(filename=str(ROOT/'figures/temporal_review_models.png')))
""")
    md("""## 9. Monthly, quarterly and annual revenue scenarios

Revenue requires an assumed opening plan and utilization rate because actual bookings are absent. The calculator uses a price benchmark and a seasonal review pattern only as scenario ingredients. It does not turn reviews into validated booking probabilities.

The example assumes ten open nights each month, three utilization levels and a constant nightly benchmark. Costs, financing and capital are unknown and remain `None`; consequently no personal ROI is reported. Edit the assumptions in the next cell to evaluate a conditional case. The annual night budget, if supplied, is an operating constraint over these twelve months, not a statement about rental law.
""")
    defs("seasonality", "scenario_calculation", "scenarios", "plots")
    assumptions = json.loads((ROOT / "scenario_inputs.json").read_text())
    code("assumptions = " + repr(assumptions) + "\n\nif REBUILD_INPUTS:\n    season_table, review_history = seasonality(latest_reviews, audit)\nelse:\n    season_table = pd.read_csv(ROOT/'results/seasonality.csv')\n    raw_history = pd.read_csv(ROOT/'results/monthly_reviews.csv', index_col=0)\n    review_history = raw_history.iloc[:, 0]\n    review_history.index = pd.PeriodIndex(review_history.index, freq='M')\nmonthly_scenarios = scenario_calculation(example_rate, season_table.set_index('month').daily_review_factor, assumptions)\nmonthly_scenarios.to_csv(ROOT/'results/notebook_monthly_scenarios.csv', index=False)\ndisplay(monthly_scenarios.head(12).round(2))\ndisplay(monthly_scenarios.groupby('scenario')[['gross_revenue_eur','scenario_booked_nights']].sum().round(2))\n")
    code("""# Sum corresponding months; do not multiply one month's revenue by 3 or 12.
quarterly_scenarios = monthly_scenarios.groupby(['scenario','calendar_quarter'], sort=False).agg(
    months_in_period=('month','size'), gross_revenue_eur=('gross_revenue_eur','sum'),
    open_nights=('open_nights','sum'), scenario_booked_nights=('scenario_booked_nights','sum'))
quarterly_scenarios.to_csv(ROOT/'results/notebook_quarterly_scenarios.csv')
print('Calendar-quarter totals; months_in_period identifies partial first/last quarters:')
display(quarterly_scenarios.round(2))
horizons = []
for name, group in monthly_scenarios.groupby('scenario', sort=False):
    for n in [1, 3, 12]:
        selected = group.head(n)
        cash = selected.cash_flow_eur.sum() if selected.cash_flow_eur.notna().all() else np.nan
        capital = assumptions.get('initial_cash_investment_eur')
        roi = cash/capital if n == 12 and capital and capital > 0 else np.nan
        horizons.append({'scenario':name, 'months':n, 'gross_revenue_eur':selected.gross_revenue_eur.sum(),
                         'cash_flow_eur':cash, 'annual_cash_return':roi})
horizon_table = pd.DataFrame(horizons)
horizon_table.to_csv(ROOT/'results/notebook_scenario_horizons.csv', index=False)
print('Cumulative totals over the first 1, 3 and 12 scenario months:')
display(horizon_table.round(2))
plots(price_data, price_metrics_table, review_metrics_table, season_table, review_history, monthly_scenarios)
for filename in ['prices.png','price_model_comparison.png','review_model_comparison.png','review_history.png','revenue_scenarios.png']:
    display(Image(filename=str(ROOT/'figures'/filename)))
""")
    md("""## 10. Independent checks and interpretation

These checks test temporal lag alignment, the distinction between missing and zero outcomes, saved prediction metrics and the scenario calculation. The synthetic EUR 100 case checks arithmetic only; it is not an empirical property forecast.
""")
    code("""ordered = review_inputs[0].sort_values(['listing_id','month'])
for lag in [1,2,3,6,12]:
    previous = ordered.groupby('listing_id').target.shift(lag)
    present = previous.notna()
    assert np.allclose(ordered.loc[present, f'lag_{lag}'], previous[present])
assert not temporal_panel.duplicated(['id','target_month']).any()
assert temporal_panel.loc[~temporal_panel.followup_observed,'target'].isna().all()
for protocol in ['host_holdout','spatial_holdout']:
    predicted = pd.read_csv(ROOT/f'results/price_predictions_{protocol}.csv')
    for model in ['median_baseline','internal','location','transit']:
        stored = price_metrics_table[(price_metrics_table.protocol==protocol)&(price_metrics_table.model==model)].mae_eur.iloc[0]
        assert np.isclose(np.abs(predicted.actual_price_eur-predicted[model]).mean(), stored)
example_inputs = dict(assumptions)
example_inputs.update(open_nights_by_month=[10]*12, occupancy_scenarios={'test':.5}, realised_price_multiplier=1.,
                      annual_booked_night_budget=None, variable_cost_rate=.2, fixed_cost_per_month_eur=100., financing_cash_per_month_eur=20.)
flat_season = pd.Series(1., index=range(1,13))
synthetic = scenario_calculation(100., flat_season, example_inputs)
assert np.allclose(synthetic.gross_revenue_eur, 500.)
assert np.allclose(synthetic.cash_flow_eur, 280.)
example_inputs['annual_booked_night_budget'] = 7
assert np.isclose(scenario_calculation(100.,flat_season,example_inputs).scenario_booked_nights.sum(),7.)
assert (monthly_scenarios.scenario_booked_nights <= monthly_scenarios.open_nights).all()
assert np.allclose(quarterly_scenarios.groupby('scenario').gross_revenue_eur.sum().sort_index(),
                   monthly_scenarios.groupby('scenario').gross_revenue_eur.sum().sort_index())
for name, group in monthly_scenarios.groupby('scenario', sort=False):
    for n in [1,3,12]:
        stored = horizon_table[(horizon_table.scenario==name)&(horizon_table.months==n)].gross_revenue_eur.iloc[0]
        assert np.isclose(stored, group.head(n).gross_revenue_eur.sum())
audit['price_deployment_feature_set'] = chosen_price_features
audit['package_versions'] = {'pandas':pd.__version__,'numpy':np.__version__,'scikit-learn':sklearn.__version__}
dump(ROOT/'results/data_audit.json',audit)
print('All notebook checks passed.')
""")
    md("""## Conclusions and remaining data

- Report price and review errors as different tasks. Neither is a real-revenue score.
- Read the location and transit comparisons directly from their held-out errors; external features are not guaranteed to help.
- In the earlier-snapshot July test, the 90-day review-rate baseline has MAE 0.5403 and total bias +0.26%, versus MAE 0.5820 and bias +18.51% for boosting with history, property and location. The simpler baseline wins on MAE; boosting has lower RMSE. The earlier rolling-cohort gain does not transfer automatically.
- Historical snapshots improve input timing, but missing follow-up and uneven snapshot coverage still limit inference.
- Actual booking/revenue records are needed for earnings validation. Costs, opening plans, financing and cash invested are needed for a personal return calculation.
- A longer sequence of complete target months is needed to validate seasonality and multi-month forecasts. Weather should be known at the forecast origin; neither weather nor neighborhood-income data is fitted in this version.

### Sources

- [Inside Airbnb downloads and archive list](https://insideairbnb.com/get-the-data/).
- [Inside Airbnb data assumptions](https://insideairbnb.com/data-assumptions/) and [data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit).
- [Official Île-de-France transport stations](https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/).

Original file URLs and hashes are retained in the audit manifests. See REPORT.html and DATA_NEEDED.md for the written interpretation and prioritized extensions.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.13.0"}},"nbformat":4,"nbformat_minor":5}
    destination = ROOT / "Airbnb_Project.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    for c in cells:
        if c["cell_type"] == "code": compile("".join(c["source"]),c["id"],"exec")
    print(f'Built {len(cells)} cells, including {sum(c["cell_type"]=="code" for c in cells)} executable code cells.')


if __name__ == "__main__":
    build()
