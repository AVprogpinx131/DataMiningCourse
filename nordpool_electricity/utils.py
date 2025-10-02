"""Utility functions shared between main.py and web_app.py."""

from datetime import datetime, timezone, timedelta
from typing import Optional


def parse_timestamp_to_utc(timestamp_str: str) -> Optional[datetime]:
    """Parse various timestamp formats to UTC datetime.

    Args:
        timestamp_str: Timestamp string in various formats

    Returns:
        UTC datetime object or None if parsing fails
    """
    if not timestamp_str:
        return None

    s = str(timestamp_str).strip()

    # Handle ISO format with Z suffix
    if s.endswith('Z'):
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        except Exception:
            pass

    # Handle ISO format with timezone info
    elif 'T' in s:
        try:
            dt = datetime.fromisoformat(s)
            # If naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass

    # Handle epoch seconds
    elif s.isdigit():
        try:
            return datetime.utcfromtimestamp(int(s)).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # Handle other formats (assume Estonia time)
    else:
        for fmt in ("%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt_naive = datetime.strptime(
                    s.replace('T', ' ').rstrip('Z'), fmt)
                try:
                    from zoneinfo import ZoneInfo
                    return dt_naive.replace(tzinfo=ZoneInfo('Europe/Tallinn')).astimezone(timezone.utc)
                except Exception:
                    return dt_naive.replace(tzinfo=timezone.utc)
            except Exception:
                continue

    return None


def format_local_time(dt_utc, target_tz='Europe/Tallinn'):
    """Format UTC datetime to Estonia local time string with timezone label."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(target_tz)
        local_dt = dt_utc.astimezone(tz)
        # Add timezone abbreviation for clarity
        tz_name = local_dt.strftime('%Z')  # EET or EEST
        return f"{local_dt.strftime('%Y-%m-%d %H:%M')} {tz_name}"
    except Exception:
        # Robust fallback: Calculate Estonia time manually
        # Estonia is UTC+2 (winter) or UTC+3 (summer)
        # DST: Last Sunday in March to last Sunday in October

        year = dt_utc.year

        # Calculate DST boundaries for the year
        # Last Sunday in March
        march_last_sunday = 31
        while datetime(year, 3, march_last_sunday).weekday() != 6:
            march_last_sunday -= 1

        # Last Sunday in October
        oct_last_sunday = 31
        while datetime(year, 10, oct_last_sunday).weekday() != 6:
            oct_last_sunday -= 1

        dst_start = datetime(year, 3, march_last_sunday,
                             1, 0, 0, tzinfo=timezone.utc)
        dst_end = datetime(year, 10, oct_last_sunday,
                           1, 0, 0, tzinfo=timezone.utc)

        # Determine if we're in DST
        if dst_start <= dt_utc < dst_end:
            offset_hours = 3  # EEST (UTC+3)
            tz_name = "EEST"
        else:
            offset_hours = 2  # EET (UTC+2)
            tz_name = "EET"

        local_dt = dt_utc + timedelta(hours=offset_hours)
        return f"{local_dt.strftime('%Y-%m-%d %H:%M')} {tz_name}"
