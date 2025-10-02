# Nord Pool Estonia Electricity Price Monitor

A Flask web application that fetches and displays Estonia's day-ahead electricity prices from the NordPool market via Elering's dashboard API.

## Features

- **Daily Price Analysis**: View min/max/average prices for any delivery date
- **Top 3 Price Rankings**: See the lowest and highest prices with timestamps
- **Weekly Trend Visualization**: 12-week historical analysis with matplotlib charts
- **Per-Week Day Analysis**: Compare price patterns across different days of the week

## How It Works

Day-ahead electricity prices are published 24 hours before delivery. For delivery date `2025-10-01`, the data covers the period from `2025-09-30 21:00` to `2025-10-01 21:00` (Europe/Tallinn timezone) - representing the full 24-hour delivery day in local time.

## Quick Start

### Requirements
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation & Running

```bash
# Clone the repository
git clone <your-repo-url>
cd nordpool_electricity

# Install dependencies and run web app
uv run web_app.py

# Or run CLI analysis for a specific date
uv run main.py 2025-10-01
```

Visit http://127.0.0.1:5000 to access the web interface.

### Dependencies

```bash
uv add requirements.txt
```

## Usage

### Web Interface
1. Enter a delivery date (YYYY-MM-DD format)
2. Click "Run" to fetch price data
3. View daily statistics and weekly trend charts


## Data Source

Fetches data from [Elering's dashboard API](https://dashboard.elering.ee/api/nps/price/csv) - Estonia's electricity transmission system operator. No API key required.


## Project Structure

- `main.py` - Core API client and CLI tool
- `web_app.py` - Flask web interface  
- `utils.py` - Timestamp parsing and timezone utilities
- `templates/` - HTML templates
- `weekly_day_plots/` - Generated chart files
