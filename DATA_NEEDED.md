# Additional data needed

The supplied files are sufficient for the included price benchmark, review-activity backtest and conditional revenue calculator. They cannot establish actual monthly income, profit or investment return.

Update, 11 November 2026: historical Paris listing **and review** snapshots dated 13 December 2025, 21 March 2026 and 16 June 2026 have now been downloaded, alongside the 16 September 2026 calendar. Together with the supplied September files, these support the completed earlier-snapshot forecasting experiment. See `ARCHIVE_UPDATE.md` for coverage and `TEMPORAL_UPDATE.md` for results. The September calendar contains **no price columns**. The original price and rolling-review experiments retain their definitions; the additional temporal experiment is evaluated separately.

| Priority | Data | Minimum useful fields | What it unlocks |
| --- | --- | --- | --- |
| Essential for real revenue validation | Anonymized host booking or monthly accounting records | Stable listing ID; stay dates or month; offered nights; occupied nights; accommodation revenue; cancellations/refunds; currency; fees included/excluded | Actual occupancy and revenue targets. Request consecutive months including zero-booking months and inactive periods. |
| Essential for a personal return calculation | Your operating plan and costs | Monthly open-night plan; realized-rate assumption; management/platform costs; cleaning and maintenance treatment; fixed expenses; financing cash payments; initial cash invested | Monthly cash flow, annual cash-on-cash return and break-even calculations. No personal ROI is reported while these are missing. |
| Acquired and analyzed; longer history still needed | Four dated listing and review snapshots | Listing ID; capture date; then-current attributes and visibility; review ID and review date. Check price schema changes. | Three full target months were built with predictors observed earlier. Longer histories and more complete follow-up are still needed to validate seasonality and performance over time. |
| Needed for weather research | Forecasts archived by issue time, aligned to the prediction horizon | Location; forecast issue timestamp; valid date; temperature/precipitation; lead time | Testing whether weather known at the forecast origin improves on month/season baselines. For monthly/annual horizons use suitable seasonal forecasts or climatology. Future realized weather is not a valid forecasting input. |
| Useful extension | INSEE Filosofi plus matching IRIS boundaries | IRIS code; boundary vintage; reference year; publication date; income/poverty variables; suppression flags | Socioeconomic geography beyond coordinates. Do not infer an individual's income from area statistics. |
| Useful extension | Dated tourism, restaurant and amenity locations | Feature ID; category; coordinates; extraction date; historical validity where possible | Tourism access and local amenities. Compare with a location-only baseline. |
| September calendar acquired | Inside Airbnb calendar snapshots | Listing ID; stay date; available status; minimum/maximum stay | Offered availability can now be summarized. This downloaded calendar has no price fields. Unavailable nights still cannot be identified as confirmed bookings. |
| Needed to tailor submission | Course brief or marking rubric | Required methods, report length, notebook/slide requirements, deadline | Alignment with the course assessment. English has been confirmed. No rubric was supplied. |

For real booking records, aim for at least two consecutive annual cycles where feasible and several independent hosts, with a later period held out. This is a practical starting point, not a guarantee of adequate sample size; precision depends on the number and diversity of properties. Do not provide guest names, contact details or payment information.

## Public sources

- [Inside Airbnb downloads and archive requests](https://insideairbnb.com/get-the-data/): quarterly public snapshots; older archives may require a request.
- [Inside Airbnb assumptions](https://insideairbnb.com/data-assumptions/): limitations of availability, coordinates and estimated occupancy.
- [Île-de-France Mobilités stations](https://data.iledefrance-mobilites.fr/explore/dataset/emplacement-des-gares-idf-data-generalisee/): already downloaded and tested in this project.
- [INSEE Filosofi 2021 at IRIS level](https://www.insee.fr/fr/statistiques/8229323): released July 2024, geography dated January 2022. An available historical example; match version and publication date to the forecast origin.
- [Open-Meteo archived forecast runs](https://open-meteo.com/en/docs/previous-runs-api): short-range lead-time archives. Match the archive type to the required horizon rather than using a hindsight weather series.

The main missing ingredient for your original earnings objective is actual host revenue data. More geographic variables cannot substitute for that target.
