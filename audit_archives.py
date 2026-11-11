"""Audit newly acquired Paris snapshots and calendar without changing model results."""
import json
from pathlib import Path
import pandas as pd
from analysis import ROOT, DEFAULT_DATA, sha256, dump


def main():
    source_root = "https://data.insideairbnb.com/france/ile-de-france/paris/"
    dates = ["2025-12-13", "2026-03-21", "2026-06-16", "2026-09-16"]
    folder = ROOT / "data/insideairbnb"
    rows, frames, manifest = [], {}, []
    for date in dates:
        path = DEFAULT_DATA / "listings.csv" if date == dates[-1] else folder / f"listings_{date}.csv.gz"
        cols = pd.read_csv(path, nrows=0).columns
        keep = [x for x in ["id", "host_id", "scrape_id", "last_scraped", "room_type", "property_type",
            "price", "latitude", "longitude", "accommodates", "bedrooms", "beds", "minimum_nights",
            "number_of_reviews", "number_of_reviews_ltm", "availability_365"] if x in cols]
        d = pd.read_csv(path, usecols=keep, low_memory=False)
        assert d.id.is_unique
        price = pd.to_numeric(d.price.astype("string").str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
        rows.append({"snapshot": date, "listings": len(d), "collection_start": d.last_scraped.min(),
            "collection_end": d.last_scraped.max(), "missing_price_fraction": float(price.isna().mean()),
            "quote_context_fields": "price_quote_checkin_date" in cols, "unique_scrape_ids": d.scrape_id.astype(str).unique().tolist()})
        d["snapshot"] = date
        frames[date] = d
        manifest.append({"snapshot": date, "kind": "listings", "url": source_root + date + "/data/listings.csv.gz",
                         "local_file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
                         "retrieval": "user-supplied" if date == dates[-1] else "2026-11-10"})
        print(f"Audited {date}: {len(d):,} listings", flush=True)
    latest_ids = set(frames[dates[-1]].id)
    coverage = []
    for earlier, later in zip(dates[:-1], dates[1:]):
        a, b = set(frames[earlier].id), set(frames[later].id)
        coverage.append({"earlier": earlier, "later": later, "earlier_listings": len(a), "later_listings": len(b),
            "seen_in_both": len(a & b), "not_seen_later": len(a - b), "not_seen_earlier": len(b - a),
            "fraction_earlier_seen_later": len(a & b) / len(a)})
    pd.DataFrame(rows).drop(columns="unique_scrape_ids").to_csv(ROOT / "results/archive_snapshots.csv", index=False)
    pd.DataFrame(coverage).to_csv(ROOT / "results/archive_transitions.csv", index=False)
    pd.concat(frames.values(), ignore_index=True).to_csv(folder / "listing_snapshot_panel.csv.gz", index=False, compression="gzip")
    cal = folder / "calendar_2026-09-16.csv.gz"
    fields = pd.read_csv(cal, nrows=0).columns.tolist()
    summary = {"columns": fields, "rows": 0, "available_rows": 0, "unknown_availability_rows": 0,
               "unmatched_listing_rows": 0, "date_min": None, "date_max": None,
               "has_price_columns": any(x in fields for x in ["price", "adjusted_price"])}
    ids, month_chunks = set(), []
    for chunk in pd.read_csv(cal, chunksize=1000000, usecols=["listing_id", "date", "available"]):
        summary["rows"] += len(chunk)
        summary["available_rows"] += int(chunk.available.eq("t").sum())
        summary["unknown_availability_rows"] += int((~chunk.available.isin(["t", "f"])).sum())
        summary["unmatched_listing_rows"] += int((~chunk.listing_id.isin(latest_ids)).sum())
        ids.update(chunk.listing_id.unique())
        lo, hi = chunk.date.min(), chunk.date.max()
        summary["date_min"] = min(lo, summary["date_min"] or lo)
        summary["date_max"] = max(hi, summary["date_max"] or hi)
        chunk["month"] = chunk.date.str[:7]
        chunk["available_count"] = chunk.available.eq("t").astype(int)
        month_chunks.append(chunk.groupby(["listing_id", "month"]).agg(
            observed_calendar_days=("date", "size"), offered_available_days=("available_count", "sum")))
    summary["unique_listings"] = len(ids)
    summary["listings_without_calendar"] = len(latest_ids - ids)
    summary["available_share"] = summary["available_rows"] / summary["rows"]
    monthly = pd.concat(month_chunks).groupby(level=[0, 1]).sum()
    monthly["observed_unavailable_days"] = monthly.observed_calendar_days - monthly.offered_available_days
    monthly.to_csv(folder / "calendar_monthly_availability.csv.gz", compression="gzip")
    assert int(monthly.observed_calendar_days.sum()) == summary["rows"]
    assert int(monthly.offered_available_days.sum()) == summary["available_rows"]
    manifest.append({"snapshot": dates[-1], "kind": "calendar", "url": source_root + dates[-1] + "/data/calendar.csv.gz",
                     "local_file": str(cal), "bytes": cal.stat().st_size, "sha256": sha256(cal), "retrieval": "2026-11-10"})
    dump(ROOT / "results/archive_manifest.json", manifest)
    dump(ROOT / "results/archive_audit.json", {"snapshots": rows, "transitions": coverage, "calendar": summary})
    print(json.dumps(summary, indent=2), flush=True)
    table = "\n".join(f"| {r['snapshot']} | {r['listings']:,} | {r['collection_start']}–{r['collection_end']} | {r['missing_price_fraction']:.1%} |" for r in rows)
    transition_table = "\n".join(f"| {r['earlier']} → {r['later']} | {r['seen_in_both']:,} | {r['not_seen_later']:,} | {r['not_seen_earlier']:,} |" for r in coverage)
    note = f"""## Additional Paris data acquired on 10 November 2026

Following the supplied [Inside Airbnb download page](https://insideairbnb.com/get-the-data/), the three earlier listings snapshots and the September calendar were downloaded and audited. The existing September listings file was reused. The listing files now form a table of observations across four snapshots. The original price and rolling-review results retain their definitions; a separate experiment using the historical reviews is described in the following section.

| Snapshot label | Listings | Actual collection dates | Missing price |
| --- | --- | --- | --- |
{table}

Snapshot labels are batch dates, not necessarily each listing's capture date. For a forecasting experiment, use the row-level `last_scraped` date or a common origin after the batch finishes. Four quarterly snapshots cover roughly nine months between endpoints; they are not a continuous monthly panel or a full annual backtest.

### Listing continuity

| Snapshot pair | Seen in both | Not seen in later snapshot | Not seen in earlier snapshot |
| --- | --- | --- | --- |
{transition_table}

These are visibility transitions, not verified business closures or new openings. A disappearance can reflect sampling, temporary unavailability or an actual exit. The derived table preserves each snapshot separately so historical attributes need not be replaced by final-snapshot values.

### What the calendar actually contains

The September calendar has **{summary['rows']:,} listing-date rows** covering **{summary['unique_listings']:,} listings**, with dates from **{summary['date_min']} to {summary['date_max']}**. It has {summary['unmatched_listing_rows']:,} rows outside the supplied September listing IDs and {summary['listings_without_calendar']:,} supplied listings without calendar rows. Availability is marked true in {summary['available_share']:.1%} of rows. Monthly availability totals have been prepared and reconcile to the source row and available-day counts. The first and last observed months can be partial.

**This particular calendar has no price or adjusted-price column.** Its fields are listing ID, date, availability, minimum nights and maximum nights. It therefore cannot supply a daily-price curve or recover missing listing prices. An unavailable date is not a confirmed booking: the source does not distinguish host blocks from reservations. [Inside Airbnb limitations](https://insideairbnb.com/data-assumptions/).

The December listings also lack the newer quote-context fields present in the 2026 listings. Price collection definitions must be checked before interpreting cross-snapshot changes as comparable price movements.

### Implication for the project

Repeated public snapshots now support an initial forecasting extension using attributes captured before the prediction origin and outcomes observed later. Historical review files have also been acquired, and that experiment is reported in the next section. Missing follow-up remains distinct from zero activity. Calendar availability describes the advertised opening pattern at collection, not booked occupancy.

Actual booking revenue, personal operating costs, financing and initial cash investment remain missing. These public downloads do not remove the need for those inputs to validate actual earnings or calculate a personal return. The original price/review metrics are unchanged; no new forecast-accuracy claim is made from this audit alone.

Source URLs, file hashes and local locations are recorded in `results/archive_manifest.json`. Prepared files are `data/insideairbnb/listing_snapshot_panel.csv.gz` and `data/insideairbnb/calendar_monthly_availability.csv.gz`.
"""
    (ROOT / "ARCHIVE_UPDATE.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
