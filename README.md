# Paris Airbnb: Jupyter project

Start with **Airbnb_Project.ipynb**. This English notebook includes the cleaning, feature engineering, model fitting, charts, findings and verification code in its own cells. Its 22 code cells were executed successfully in a real Jupyter kernel. The default run reuses prepared inputs and refits all models; it does not just display saved scores.

**REPORT.html** is the illustrated report, with its images embedded. **REPORT.md** is the editable text version. The report includes the additional historical-data experiment.

## Open and run the notebook

1. Clone this repository, or use GitHub’s **Code → Download ZIP** and extract it. Keep the notebook beside its `data`, `results`, `figures` and `models` folders.
2. Open `Airbnb_Project.ipynb` in Jupyter Notebook, JupyterLab or an editor with notebook support. Select a Python environment with the packages listed in `requirements.txt`.
3. Choose **Restart Kernel and Run All**. Leave `REBUILD_INPUTS = False` to use the packaged analytical tables. The notebook's working directory must be the repository root: the folder containing the notebook and its data directory.
4. To repeat raw-file cleaning too, set `REBUILD_INPUTS = True`. Missing public files are downloaded when `DOWNLOAD_MISSING = True`. This requires internet access and several GB of disk space. September CSVs are downloaded into `data/raw` unless you configure another source directory. Existing raw files are reused without modification.

An optional fresh environment can be created from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt jupyterlab
python -m jupyter lab Airbnb_Project.ipynb
```

This project was tested with Python 3.13 and the pinned package versions. The existing scientific Python and Jupyter kernel were used; no packages were installed or upgraded on this machine. A default rerun needs no network access.

## What the project establishes

- **Quoted prices:** an entire-home benchmark, with unseen-host and spatial holdouts. Location helps; added station features do not reliably improve on location.
- **Review activity:** a retrospective rolling-month experiment, a fixed-origin three-month experiment, and an earlier-snapshot experiment that predicts a later month. In the last experiment, a simple prior-90-day review rate beats boosting on MAE and aggregate calibration; boosting has lower RMSE.
- **Revenue scenarios:** monthly, quarterly and annual arithmetic under explicit opening, utilization and price assumptions. Actual income is not observed. Profit and personal investment returns require your costs and capital inputs.

Do not interpret a quoted-price error or a review-count error as a revenue-prediction error. Calendar unavailability is not confirmed occupancy. The downloaded calendar contains no price fields.

## Files and data

- `Airbnb_Project.ipynb`: main executed deliverable; all implementation is visible in cells.
- `REPORT.html`, `REPORT.md`: methods, findings and limitations.
- `ARCHIVE_UPDATE.md`, `TEMPORAL_UPDATE.md`: coverage audit and the added historical forecasting experiment, also included in the main report.
- `DATA_NEEDED.md`: remaining requirements, particularly actual revenue and personal costs.
- `data/`: compressed analytical tables and official transport data, including the four-snapshot listing panel, monthly calendar summaries and the historical-review feature panel.
- `results/`: source manifests, hashes, cleaning counts, saved predictions, model metrics and verification results.
- `figures/`, `models/`: chart images and fitted price/rolling-review models. Load model files only from trusted sources.
- `analysis.py`, `audit_archives.py`, `longitudinal.py`: reusable versions of the analysis routines.
- `build_notebook.py`: assembles a fresh, **unexecuted** notebook from those routines; running it replaces the notebook, so only use it when rebuilding the artifact.
- `make_report.py`: refreshes the report from results while preserving the notebook.
- `verify.py`: independent checks of lags, metrics and scenario arithmetic.
- `predict_property.py`, `property_example.json`, `scenario_inputs.json`: a property benchmark and conditional scenario interface.

The original September listings and reviews were supplied locally and are not included in this repository. A full rebuild expects them at `data/raw/listings.csv` and `data/raw/reviews.csv`, or downloads them when missing. Historical listings and reviews for 13 December 2025, 21 March 2026 and 16 June 2026, plus the 16 September 2026 calendar, were obtained from Inside Airbnb. A full rebuild stores historical raw downloads in `data/insideairbnb`. This repository excludes the large raw CSV archives but includes all prepared tables required by the default notebook. Source URLs and hashes allow retrieval and provenance checks.

The analytical tables exclude review text, guest names, host names and profile text. The rolling review panel uses 12,000 established listings sampled with seed 42. The added historical experiment retains all source listings, marks missing follow-up outcomes explicitly and scores observed outcomes only.

## Use your own property assumptions

Edit `property_example.json` and `scenario_inputs.json`, then run:

```bash
python3 predict_property.py --property property_example.json --assumptions scenario_inputs.json --out results/my_property
```

The input expects a Paris entire home, a minimum stay of 1–29 nights, and the arrondissement name used by Inside Airbnb. The model was fitted on observed EUR 20–1,000 quotes. It gives a price benchmark; it has not been validated as an optimal price or a realized transaction rate. Unusual properties or dates can lie outside its training range.

The input example is fictional. The report uses a different, existing held-out listing, so the two examples have different scenario values. The notebook's assumptions are written directly in its scenario cell; changing `scenario_inputs.json` affects the command-line calculator, while notebook users should edit that cell.

Set operating costs, financing payments and initial cash invested to your own values. Zero means a known zero; `null` in JSON or `None` in Python means unknown. Avoid counting the same expense twice. The optional annual booked-night budget is an operating constraint over the twelve scenario months, not a statement about rental regulations.

The saved rolling-review model is a backtest model trained through May 2026. It is not a current demand service and is not used to convert reviews into bookings.

## Attribution

Credit [Inside Airbnb](https://insideairbnb.com/get-the-data/) for listing, review and calendar data, and [Île-de-France Mobilités](https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/) for station locations. The report and source manifests document dataset vintages and limitations.
