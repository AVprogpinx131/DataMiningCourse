from datetime import datetime, date, timedelta, timezone
import os
import requests
import csv
import io
import sys
from typing import List, Tuple, Dict, Optional
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from utils import parse_timestamp_to_utc, format_local_time


DEFAULT_TZ = 'Europe/Tallinn'
BASE_URL = "https://dashboard.elering.ee/api/nps/price/csv"

HEADERS = {
    "User-Agent": "python-requests/2.x (+https://github.com/)",
    "Accept": "text/csv, */*;q=0.1"
}

# Weekly analysis configuration
WEEKS_BACK = 12
WEEK_TIMEZONE = 'Europe/Tallinn'


def build_start_end_iso_utc(delivery_date_str):
    """
    For Nord Pool day-ahead, the day starting 00:00 local (EET/EEST) corresponds to
    21:00 UTC previous day -> many scripts use start=(delivery_date-1 @21:00:00Z),
    end=(delivery_date @21:00:00Z) to fetch a full 24h.
    """
    d = date.fromisoformat(delivery_date_str)
    start_dt = datetime.combine(d - timedelta(days=1), datetime.min.time()
                                ).replace(hour=21, minute=0, second=0, tzinfo=timezone.utc)
    end_dt = datetime.combine(d,             datetime.min.time()).replace(
        hour=21, minute=0, second=0, tzinfo=timezone.utc)
    # Format as ISO with Z
    return start_dt.isoformat().replace("+00:00", "Z"), end_dt.isoformat().replace("+00:00", "Z")


def today_local(tz_name: str = DEFAULT_TZ) -> date:
    """Return today's date in the given timezone (no time part)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return now.date()


def fetch_csv(start_iso, end_iso, fields="ee"):
    params = {
        "start": start_iso,
        "end": end_iso,
        "fields": fields
    }
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def fetch_prices_range(start_date: date, end_date: date, fields: str = "ee") -> List[Tuple[datetime, float]]:
    """Fetch prices for a [start_date, end_date] inclusive window using 21:00 UTC boundaries.
    Returns a list of (utc_datetime, price).
    """
    start_iso, _ = build_start_end_iso_utc(
        (start_date + timedelta(days=1)).isoformat())
    _, end_iso = build_start_end_iso_utc(end_date.isoformat())

    csv_text = fetch_csv(start_iso, end_iso, fields=fields)
    rows = parse_prices_from_csv(csv_text)
    out: List[Tuple[datetime, float]] = []
    for ts, price in rows:
        dt = parse_timestamp_to_utc(ts)
        if dt:
            out.append((dt, price))
    # Sort by time
    out.sort(key=lambda x: x[0])
    return out


def weekly_stats(prices: List[Tuple[datetime, float]], tz_name: str = WEEK_TIMEZONE) -> List[Tuple[str, float, float, float]]:
    """Group hourly (datetime UTC, price) pairs into ISO weeks in tz_name and compute min/avg/max.
    Returns list of ("DD-MM-YYYY to DD-MM-YYYY", min, avg, max) sorted by week.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except ImportError:
        tz = timezone.utc

    # key by (iso_year, iso_week) and remember a representative local date for computing the range
    buckets: Dict[tuple, List[float]] = {}
    week_any_day: Dict[tuple, date] = {}
    for dt_utc, price in prices:
        try:
            dt_local = dt_utc.astimezone(tz)
        except Exception:
            dt_local = dt_utc
        iso_year, iso_week, _ = dt_local.isocalendar()
        key = (iso_year, iso_week)
        buckets.setdefault(key, []).append(price)
        week_any_day.setdefault(key, dt_local.date())

    result: List[Tuple[str, float, float, float]] = []
    for key in sorted(buckets.keys()):
        vals = buckets[key]
        mn = min(vals)
        mx = max(vals)
        avg = sum(vals) / len(vals)
        any_day = week_any_day[key]
        week_start = any_day - timedelta(days=any_day.weekday())
        week_end = week_start + timedelta(days=6)
        label = f"{week_start.strftime('%d-%m-%Y')} to {week_end.strftime('%d-%m-%Y')}"
        result.append((label, mn, avg, mx))
    return result


def weekly_day_stats(
    prices: List[Tuple[datetime, float]],
    tz_name: str = WEEK_TIMEZONE
) -> List[Tuple[str, List[Tuple[str, float, float, float]]]]:
    """For each ISO week in local tz_name, compute per-day (Mon..Sun) min/avg/max.
    Returns: [ ("DD-MM-YYYY to DD-MM-YYYY", [ ("Mon", min, avg, max), ... up to Sun ]) ]
    Only includes days that have data; missing days are omitted.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except ImportError:
        tz = timezone.utc

    # Map (iso_year, iso_week) -> { date_str -> [prices...] }
    week_day_prices: Dict[tuple, Dict[date, List[float]]] = {}
    for dt_utc, price in prices:
        dt_local = dt_utc.astimezone(tz)
        iso_year, iso_week, _ = dt_local.isocalendar()
        key = (iso_year, iso_week)
        day = dt_local.date()
        week_day_prices.setdefault(key, {}).setdefault(day, []).append(price)

    # Helper for day name
    day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    out: List[Tuple[str, List[Tuple[str, float, float, float]]]] = []
    for (iy, iw) in sorted(week_day_prices.keys()):
        day_map = week_day_prices[(iy, iw)]
        # Compute week range (Mon..Sun) from one date in the week
        any_day = min(day_map.keys())
        week_start = any_day - timedelta(days=any_day.weekday())
        week_end = week_start + timedelta(days=6)
        label = f"{week_start.strftime('%d-%m-%Y')} to {week_end.strftime('%d-%m-%Y')}"

        day_stats: List[Tuple[str, float, float, float]] = []
        for offset in range(7):
            d = week_start + timedelta(days=offset)
            if d in day_map:
                vals = day_map[d]
                mn = min(vals)
                mx = max(vals)
                av = sum(vals)/len(vals)
                day_stats.append((day_name[offset], mn, av, mx))
        out.append((label, day_stats))
    return out


def plot_weekly(stats: List[Tuple[str, float, float, float]], outfile: str = 'weekly_prices.png'):
    if not stats:
        print("No weekly stats to plot.")
        return
    if plt is None:
        print("matplotlib is not installed; skipping plot. Install with: pip install matplotlib")
        return

    weeks = [w for w, _, _, _ in stats]
    mins = [mn for _, mn, _, _ in stats]
    avgs = [av for _, _, av, _ in stats]
    maxs = [mx for _, _, _, mx in stats]

    plt.figure(figsize=(10, 5))
    plt.plot(weeks, mins, label='Min', marker='o')
    plt.plot(weeks, avgs, label='Average', marker='o')
    plt.plot(weeks, maxs, label='Max', marker='o')
    plt.title('EE day-ahead weekly prices (EUR/MWh)')
    plt.xlabel('Week (date range)')
    plt.ylabel('EUR/MWh')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def save_week_day_plots(
    week_day_stats_list: List[Tuple[str, List[Tuple[str, float, float, float]]]],
    outdir: str = 'weekly_day_plots',
    same_ylim: bool = True,
):
    """Save one bar chart per week showing each day's Min/Avg/Max.
    Filenames use the date range label (e.g., 19-09-2025_to_25-09-2025.png).
    """
    if not week_day_stats_list:
        print("No per-day weekly stats to plot.")
        return
    if plt is None:
        print("matplotlib is not installed; skipping per-week day plots. Install with: pip install matplotlib")
        return

    os.makedirs(outdir, exist_ok=True)

    # Determine y-limits if needed
    if same_ylim:
        vals = []
        for _, day_stats in week_day_stats_list:
            for _, mn, av, mx in day_stats:
                vals.extend([mn, av, mx])
        if vals:
            vmin, vmax = min(vals), max(vals)
            span = max(1.0, vmax - vmin)
            y_min = max(0.0, vmin - 0.05 * span)
            y_max = vmax + 0.05 * span
        else:
            y_min = y_max = None
    else:
        y_min = y_max = None

    for label, day_stats in week_day_stats_list:
        if not day_stats:
            continue
        days = [d for d, _, _, _ in day_stats]
        mins = [mn for _, mn, _, _ in day_stats]
        avgs = [av for _, _, av, _ in day_stats]
        maxs = [mx for _, _, _, mx in day_stats]

        x = list(range(len(days)))
        width = 0.25

        plt.figure(figsize=(8, 4))
        plt.bar([i - width for i in x], mins,
                width=width, label='Min', color='#2ca02c')
        plt.bar(x, avgs, width=width, label='Avg', color='#1f77b4')
        plt.bar([i + width for i in x], maxs,
                width=width, label='Max', color='#d62728')
        plt.xticks(x, days)
        plt.title(f'EE {label}')
        plt.ylabel('EUR/MWh')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        plt.tight_layout()
        safe_label = label.replace(' ', '_').replace(':', '').replace('/', '-')
        out_path = os.path.join(outdir, f'EE_{safe_label}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
    print(f"Saved {len(week_day_stats_list)} per-week day plots to {outdir}\\")


def parse_prices_from_csv(csv_text):
    """
    Expect CSV rows with a datetime column and a column for the 'ee' prices.
    Normalize decimal separators (commas -> dots) and convert to float.
    Returns list of tuples: (timestamp_str, price_float)
    """
    f = io.StringIO(csv_text)
    sample = csv_text[:1024]
    rows = []

    # Try to sniff CSV dialect (delimiter may be ';')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t'])
    except Exception:
        class _D:
            pass
        dialect = _D()
        dialect.delimiter = ';'

    # Helper to parse price string with comma decimals
    def parse_price_str(s: str):
        if s is None:
            return None
        p = s.strip().replace(' ', '').replace('\xa0', '')
        p = p.replace(',', '.')
        try:
            return float(p)
        except ValueError:
            return None

    # Helper to parse timestamp: try known formats or pass through
    def parse_timestamp(ts: str):
        if ts is None:
            return None
        s = ts.strip().strip('"')
        # try DD.MM.YYYY HH:MM
        for fmt in ("%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).isoformat()
            except Exception:
                pass
        # try epoch seconds
        if s.isdigit():
            try:
                return datetime.utcfromtimestamp(int(s)).isoformat() + 'Z'
            except Exception:
                return ts
        return ts

    # Token check helper: does a header contain 'ee' as a standalone token?
    import re
    token_re = re.compile(r"(?i)(?<![A-Za-z0-9])ee(?![A-Za-z0-9])")

    # First, try DictReader path (for headered CSV from API)
    f.seek(0)
    dict_reader = csv.DictReader(f, delimiter=dialect.delimiter)
    header = dict_reader.fieldnames or []
    if header:
        first_header = header[0]
        for r in dict_reader:
            price_val = None
            timestamp = None
            for key, val in r.items():
                if not isinstance(key, str):
                    continue
                key_l = key.strip().lower()
                if key_l in ("time", "date", "datetime", "timestamp", "timeutc") and timestamp is None:
                    timestamp = val
                # Prefer exact 'ee' column for Estonia price
                if key_l == "ee" and price_val is None:
                    price_val = val
                # Secondary: generic price/value columns
                if key_l in ("price", "value") and price_val is None:
                    price_val = val

            if timestamp is None and isinstance(first_header, str):
                timestamp = r.get(first_header)

            if price_val is None:
                # Search headers containing 'ee' as a token (avoid matching 'fee') or containing 'price'
                for k, v in r.items():
                    if not isinstance(k, str):
                        continue
                    k_low = k.lower()
                    if token_re.search(k) or "price" in k_low:
                        price_val = v
                        break

            if price_val is None and None in r and isinstance(r[None], list):
                for v in r[None]:
                    parsed = parse_price_str(v) if isinstance(v, str) else None
                    if parsed is not None:
                        price_val = v
                        break

            price = parse_price_str(price_val) if isinstance(
                price_val, str) else None
            if price is not None:
                rows.append((timestamp, price))
        if rows:
            return rows

    # Fallback: headerless CSV (e.g., "epoch";"human time";"price")
    f.seek(0)
    reader = csv.reader(f, delimiter=dialect.delimiter)
    for cols in reader:
        if not cols:
            continue
        # Trim quotes/spaces
        cols = [c.strip().strip('"') if isinstance(
            c, str) else c for c in cols]
        # Common patterns: [epoch, human_ts, price] or [human_ts, price]
        if len(cols) >= 3:
            ts = cols[1] or cols[0]
            price = parse_price_str(cols[2])
        elif len(cols) == 2:
            ts = cols[0]
            price = parse_price_str(cols[1])
        else:
            # try to find any numeric-like entry as price
            ts = cols[0]
            price = None
            for c in cols[1:]:
                price = parse_price_str(c)
                if price is not None:
                    break
        if price is not None:
            rows.append((parse_timestamp(ts), price))
    return rows


def find_min_max(prices):
    if not prices:
        return None
    min_item = min(prices, key=lambda x: x[1])
    max_item = max(prices, key=lambda x: x[1])
    return min_item, max_item


def main(delivery_date: Optional[str] = None):
    # Resolve date: if not provided, use today in Europe/Tallinn
    if delivery_date:
        date_for_day = delivery_date
    else:
        date_for_day = today_local().isoformat()

    start_iso, end_iso = build_start_end_iso_utc(date_for_day)
    print("Requesting:", start_iso, "->", end_iso)
    try:
        csv_text = fetch_csv(start_iso, end_iso, fields="ee")
    except Exception as e:
        print("Failed to fetch CSV:", e, file=sys.stderr)
        return

    prices = parse_prices_from_csv(csv_text)
    if not prices:
        print(
            "No price rows parsed. The CSV content may have a different format or be empty.")
        print("--- preview ---")
        print(csv_text[:500])
        return

    (min_ts, min_price), (max_ts, max_price) = find_min_max(prices)
    avg_price = sum(p for _, p in prices) / len(prices)

    print(f"--- Estonia (EE) day-ahead on {date_for_day} ---")

    # Convert timestamps to datetimes for formatting
    min_dt = parse_timestamp_to_utc(min_ts)
    max_dt = parse_timestamp_to_utc(max_ts)

    print(
        f"Lowest price:  {min_price:.2f} EUR at {format_local_time(min_dt) if min_dt else min_ts}")
    print(f"Average price: {avg_price:.2f} EUR over 24h")
    print(
        f"Highest price: {max_price:.2f} EUR at {format_local_time(max_dt) if max_dt else max_ts}")

    # --- Weekly analysis (last WEEKS_BACK weeks up to delivery_date week) ---
    # Compute last completed ISO week based on current local date
    now_local = today_local()
    curr_week_start = now_local - timedelta(days=now_local.weekday())  # Monday
    # Sunday of last completed week
    last_week_end = curr_week_start - timedelta(days=1)
    # Monday of last completed week
    last_week_start = last_week_end - timedelta(days=6)
    start_range = last_week_start - timedelta(weeks=WEEKS_BACK-1)
    end_range = last_week_end

    try:
        range_prices = fetch_prices_range(start_range, end_range, fields="ee")
        stats = weekly_stats(range_prices)
        if stats:
            # Print last week summary
            last_week, mn, av, mx = stats[-1]
            print(
                f"\nLatest week {last_week}: min={mn:.2f}, avg={av:.2f}, max={mx:.2f} EUR/MWh")
            plot_weekly(stats)
            # Per-week (Mon..Sun) day-of-week plots with date range labels
            wd_stats = weekly_day_stats(range_prices)
            save_week_day_plots(wd_stats)
        else:
            print("No weekly stats computed.")
    except Exception as e:
        print("Weekly analysis failed:", e, file=sys.stderr)


if __name__ == "__main__":
    arg_date = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_date)
