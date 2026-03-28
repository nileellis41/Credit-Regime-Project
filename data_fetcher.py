# data_fetcher.py
"""
Pulls macro series from FRED, resamples to monthly, caches locally as CSV.
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
from config import FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE, RESAMPLE_FREQ, DATA_DIR
from dotenv import load_dotenv

# Load keys from your specific .env file
load_dotenv(r"C:\Users\nilee\OneDrive\Documents\GitHub\Credit Regime Project\keys.env")

FRED_API_KEY = os.getenv("FRED_API_KEY")

def _get_fred_client() -> Fred:
    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY not found in keys.env at:\n"
            "  C:\\Users\\nilee\\OneDrive\\Documents\\GitHub\\Credit Regime Project\\keys.env\n"
            "Make sure it contains: FRED_API_KEY=your_key_here\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=FRED_API_KEY)

def fetch_raw(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch all configured FRED series, monthly-resample, outer-join into one DataFrame.
    Caches to data/raw_fred.csv to avoid repeat API calls.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_fred.csv")

    if use_cache and os.path.exists(cache_path):
        print(f"[data] Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print("[data] Fetching from FRED API...")
    fred = _get_fred_client()
    end = END_DATE or pd.Timestamp.today().strftime("%Y-%m-%d")

    frames = {}
    for label, series_id in FRED_SERIES.items():
        print(f"  → {label:15s} ({series_id})")
        s = fred.get_series(series_id, observation_start=START_DATE, observation_end=end)
        s.name = label
        frames[label] = s

    # Outer-join all series on their native dates
    df_raw = pd.concat(frames.values(), axis=1)
    df_raw.index = pd.DatetimeIndex(df_raw.index)

    # Resample to month-start, forward-fill quarterly/weekly series
    df_monthly = df_raw.resample(RESAMPLE_FREQ).last()
    df_monthly = df_monthly.ffill()

    df_monthly = df_monthly.loc[START_DATE:]
    df_monthly.to_csv(cache_path)
    print(f"[data] Saved to {cache_path}  shape={df_monthly.shape}")
    return df_monthly


def refresh_cache() -> pd.DataFrame:
    """Force re-fetch from FRED regardless of cache."""
    cache_path = os.path.join(DATA_DIR, "raw_fred.csv")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return fetch_raw(use_cache=False)


if __name__ == "__main__":
    df = fetch_raw()
    print(df.tail())
    print(df.describe().round(3))
