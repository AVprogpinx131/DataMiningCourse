
from flask import Flask, render_template, request, send_from_directory, url_for
import os
from datetime import date, datetime, timedelta, timezone
import time
import main as nordpool_api
from utils import parse_timestamp_to_utc, format_local_time

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def are_plots_recent(outdir: str, max_age_hours: int = 6) -> bool:
    """Check if weekly plots exist and are recent enough to skip regeneration."""
    try:
        if not os.path.exists(outdir):
            return False

        files = [f for f in os.listdir(outdir) if f.lower().endswith('.png')]
        if not files:
            return False

        # Check the modification time of the newest plot
        newest_time = 0
        for filename in files:
            filepath = os.path.join(outdir, filename)
            mtime = os.path.getmtime(filepath)
            newest_time = max(newest_time, mtime)

        # If newest plot is less than max_age_hours old, consider it recent
        age_hours = (time.time() - newest_time) / 3600
        return age_hours < max_age_hours
    except Exception:
        return False


def get_sorted_weekly_images(outdir: str) -> list:
    """Get weekly day plot image URLs sorted by date."""
    try:
        files = os.listdir(outdir)
        # Sort files by extracted start date (DD-MM-YYYY format in filename)

        def extract_date_from_filename(filename):
            try:
                # Extract date from format: EE_DD-MM-YYYY_to_DD-MM-YYYY.png
                if filename.startswith('EE_') and '_to_' in filename:
                    date_part = filename[3:].split('_to_')[0]  # Get DD-MM-YYYY
                    day, month, year = date_part.split('-')
                    return date(int(year), int(month), int(day))
            except (ValueError, IndexError):
                pass
            return date.min  # fallback for malformed filenames

        png_files = [f for f in files if f.lower().endswith('.png')]
        files = sorted(png_files, key=extract_date_from_filename)
        return [url_for('images', filename=f'weekly_day_plots/{f}') for f in files]
    except FileNotFoundError:
        return []


@app.route('/', methods=['GET'])
def index():
    # Default date is today in local (Europe/Tallinn) per helper
    default_date = nordpool_api.today_local().isoformat()
    return render_template('index.html',
                           date_for_day=default_date,
                           result=None,
                           weekly_prices_url=None,
                           weekly_day_images=[],
                           matplotlib_missing=(nordpool_api.plt is None))


@app.route('/run', methods=['POST'])
def run():
    # Read date input (YYYY-MM-DD)
    date_for_day = request.form.get(
        'date') or nordpool_api.today_local().isoformat()

    try:
        # Validate date format
        datetime.strptime(date_for_day, '%Y-%m-%d').date()
    except ValueError:
        return render_template('index.html',
                               date_for_day=date_for_day,
                               result={
                                   'error': 'Invalid date format. Use YYYY-MM-DD.'},
                               weekly_prices_url=None,
                               weekly_day_images=[],
                               matplotlib_missing=(nordpool_api.plt is None))

    start_iso, end_iso = nordpool_api.build_start_end_iso_utc(date_for_day)

    try:
        csv_text = nordpool_api.fetch_csv(start_iso, end_iso, fields="ee")
        raw_prices = nordpool_api.parse_prices_from_csv(csv_text)
    except Exception as e:
        return render_template('index.html',
                               date_for_day=date_for_day,
                               result={
                                   'error': f'Failed to fetch data: {str(e)}'},
                               weekly_prices_url=None,
                               weekly_day_images=[],
                               matplotlib_missing=(nordpool_api.plt is None))

    if not raw_prices:
        return render_template('index.html',
                               date_for_day=date_for_day,
                               result={
                                   'error': 'No price data found for this date.'},
                               weekly_prices_url=None,
                               weekly_day_images=[],
                               matplotlib_missing=(nordpool_api.plt is None))

    # Filter and process prices to ensure we only have the requested day's data
    filtered_prices = []
    start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))

    for ts_str, price in raw_prices:
        if not ts_str or price is None:
            continue

        # Parse timestamp using utility function
        dt = parse_timestamp_to_utc(ts_str)
        if not dt:
            continue

        # Only include data within the expected time window
        if dt and start_dt <= dt < end_dt:
            filtered_prices.append((dt, price))

    if not filtered_prices:
        return render_template('index.html',
                               date_for_day=date_for_day,
                               result={
                                   'error': 'No valid price data found in the expected time window.'},
                               weekly_prices_url=None,
                               weekly_day_images=[],
                               matplotlib_missing=(nordpool_api.plt is None))

    # Sort by timestamp for consistent ordering
    filtered_prices.sort(key=lambda x: x[0])

    # Calculate basic stats
    prices_only = [price for _, price in filtered_prices]
    min_price = min(prices_only)
    max_price = max(prices_only)
    avg_price = sum(prices_only) / len(prices_only)

    # Find times when min/max prices occurred
    min_price_entry = min(filtered_prices, key=lambda x: x[1])
    max_price_entry = max(filtered_prices, key=lambda x: x[1])
    min_time_dt, _ = min_price_entry
    max_time_dt, _ = max_price_entry

    # Find top 3 lowest and top 3 highest prices with their times
    sorted_by_price = sorted(filtered_prices, key=lambda x: x[1])
    top_3_min_by_price = sorted_by_price[:3]
    top_3_max_by_price = sorted(sorted_by_price[-3:], key=lambda x: x[1])

    # Convert to local time for display using utility function

    # Prepare result
    # Create delivery period description with clear time window explanation
    try:
        from zoneinfo import ZoneInfo
        estonia_tz = ZoneInfo('Europe/Tallinn')
        start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
        start_local = start_dt.astimezone(estonia_tz)
        end_local = end_dt.astimezone(estonia_tz)

        # The delivery period is for the requested date (local time 00:00 to 23:45)
        # Show this correctly instead of confusing UTC-based times
        delivery_date_obj = datetime.strptime(date_for_day, '%Y-%m-%d').date()
        end_date_local = delivery_date_obj  # Same day, not next day

        delivery_period = f"Prices for delivery date {date_for_day}: {delivery_date_obj.strftime('%Y-%m-%d')} 00:00 to {end_date_local.strftime('%Y-%m-%d')} 23:45 (Europe/Tallinn)"
    except Exception:
        # Fallback: show the requested date clearly
        delivery_period = f"Prices for delivery date {date_for_day}: 24-hour period (Europe/Tallinn)"

    result = {
        'delivery_date': date_for_day,
        'delivery_period': delivery_period,
        'num_hours': len(filtered_prices),
        'min_price': f"{min_price:.2f}",
        'max_price': f"{max_price:.2f}",
        'avg_price': f"{avg_price:.2f}",
        'min_time': format_local_time(min_time_dt),
        'max_time': format_local_time(max_time_dt),
        'top_3_min_prices': [
            {
                'time': format_local_time(dt),
                'price': f"{price:.2f}"
            }
            for dt, price in top_3_min_by_price
        ],
        'top_3_max_prices': [
            {
                'time': format_local_time(dt),
                'price': f"{price:.2f}"
            }
            for dt, price in top_3_max_by_price
        ]
    }

    # Weekly analysis
    weekly_prices_url = None
    weekly_day_images = []

    if nordpool_api.plt is not None:
        outdir = os.path.join(BASE_DIR, 'weekly_day_plots')
        weekly_png = os.path.join(BASE_DIR, 'weekly_prices.png')

        # Check if plots are recent enough to skip regeneration
        plots_are_recent = are_plots_recent(outdir, max_age_hours=6)
        weekly_png_exists = os.path.exists(weekly_png)

        if plots_are_recent and weekly_png_exists:
            # Use existing plots
            weekly_prices_url = url_for('images', filename='weekly_prices.png')
            weekly_day_images = get_sorted_weekly_images(outdir)
        else:
            # Generate new plots
            try:
                now_local = nordpool_api.today_local()
                curr_week_start = now_local - \
                    timedelta(days=now_local.weekday())  # Monday
                last_week_end = curr_week_start - timedelta(days=1)  # Sunday
                last_week_start = last_week_end - timedelta(days=6)
                start_range = last_week_start - \
                    timedelta(weeks=nordpool_api.WEEKS_BACK-1)
                end_range = last_week_end

                range_prices = nordpool_api.fetch_prices_range(
                    start_range, end_range, fields="ee")
                stats = nordpool_api.weekly_stats(range_prices)

                if stats:
                    # Combined weekly line plot
                    nordpool_api.plot_weekly(stats, weekly_png)
                    weekly_prices_url = url_for(
                        'images', filename='weekly_prices.png')

                    # Per-week day-of-week plots
                    wd_stats = nordpool_api.weekly_day_stats(range_prices)
                    nordpool_api.save_week_day_plots(wd_stats, outdir)
                    weekly_day_images = get_sorted_weekly_images(outdir)
            except Exception as e:
                print(f"Weekly analysis error: {e}")

    return render_template('index.html',
                           date_for_day=date_for_day,
                           result=result,
                           weekly_prices_url=weekly_prices_url,
                           weekly_day_images=weekly_day_images,
                           matplotlib_missing=(nordpool_api.plt is None))


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
