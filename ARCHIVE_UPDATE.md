## Additional Paris data acquired on 10 November 2026

Following the supplied [Inside Airbnb download page](https://insideairbnb.com/get-the-data/), the three earlier listings snapshots and the September calendar were downloaded and audited. The existing September listings file was reused. The listing files now form a table of observations across four snapshots. The original price and rolling-review results retain their definitions; a separate experiment using the historical reviews is described in the following section.

| Snapshot label | Listings | Actual collection dates | Missing price |
| --- | --- | --- | --- |
| 2025-12-13 | 82,045 | 2025-12-13–2026-01-05 | 100.0% |
| 2026-03-21 | 38,075 | 2026-03-21–2026-04-02 | 5.6% |
| 2026-06-16 | 77,679 | 2026-06-17–2026-06-29 | 37.7% |
| 2026-09-16 | 77,149 | 2026-09-17–2026-09-24 | 43.6% |

Snapshot labels are batch dates, not necessarily each listing's capture date. For a forecasting experiment, use the row-level `last_scraped` date or a common origin after the batch finishes. Four quarterly snapshots cover roughly nine months between endpoints; they are not a continuous monthly panel or a full annual backtest.

### Listing continuity

| Snapshot pair | Seen in both | Not seen in later snapshot | Not seen in earlier snapshot |
| --- | --- | --- | --- |
| 2025-12-13 → 2026-03-21 | 34,329 | 47,716 | 3,746 |
| 2026-03-21 → 2026-06-16 | 34,285 | 3,790 | 43,394 |
| 2026-06-16 → 2026-09-16 | 71,588 | 6,091 | 5,561 |

These are visibility transitions, not verified business closures or new openings. A disappearance can reflect sampling, temporary unavailability or an actual exit. The derived table preserves each snapshot separately so historical attributes need not be replaced by final-snapshot values.

### What the calendar actually contains

The September calendar has **28,162,159 listing-date rows** covering **77,149 listings**, with dates from **2026-09-17 to 2027-09-23**. It has 0 rows outside the supplied September listing IDs and 0 supplied listings without calendar rows. Availability is marked true in 37.5% of rows. Monthly availability totals have been prepared and reconcile to the source row and available-day counts. The first and last observed months can be partial.

**This particular calendar has no price or adjusted-price column.** Its fields are listing ID, date, availability, minimum nights and maximum nights. It therefore cannot supply a daily-price curve or recover missing listing prices. An unavailable date is not a confirmed booking: the source does not distinguish host blocks from reservations. [Inside Airbnb limitations](https://insideairbnb.com/data-assumptions/).

The December listings also lack the newer quote-context fields present in the 2026 listings. Price collection definitions must be checked before interpreting cross-snapshot changes as comparable price movements.

### Implication for the project

Repeated public snapshots now support an initial forecasting extension using attributes captured before the prediction origin and outcomes observed later. Historical review files have also been acquired, and that experiment is reported in the next section. Missing follow-up remains distinct from zero activity. Calendar availability describes the advertised opening pattern at collection, not booked occupancy.

Actual booking revenue, personal operating costs, financing and initial cash investment remain missing. These public downloads do not remove the need for those inputs to validate actual earnings or calculate a personal return. The original price/review metrics are unchanged; no new forecast-accuracy claim is made from this audit alone.

Source URLs, file hashes and local locations are recorded in `results/archive_manifest.json`. Prepared files are `data/insideairbnb/listing_snapshot_panel.csv.gz` and `data/insideairbnb/calendar_monthly_availability.csv.gz`.
