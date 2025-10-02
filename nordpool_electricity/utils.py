"""Utility functions shared between main.py and web_app.py."""

from datetime import datetime, timezone
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


def format_local_time(dt: datetime) -> str:
    """Format UTC datetime to Estonia local time string.

    Args:
        dt: UTC datetime object

    Returns:
        Formatted time string in Europe/Tallinn timezone
    """
    try:
        from zoneinfo import ZoneInfo
        local_dt = dt.astimezone(ZoneInfo('Europe/Tallinn'))
        return local_dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dt.strftime('%Y-%m-%d %H:%M')
