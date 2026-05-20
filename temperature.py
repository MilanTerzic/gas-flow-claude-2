"""
Temperature data sources for the Belgrade load forecast.

Order of preference (highest first):

  1. Manual upload (CSV / XLSX with columns: date, temperature_c)
  2. Open-Meteo public forecast/archive API (no key required, generous limits)
  3. weather.com 10-day scrape (best-effort; HTML structure changes often)

Each helper returns a DataFrame ``date | temperature_c`` (daily mean °C).
"""

from __future__ import annotations

import re
from datetime import date
from typing import IO, Union

import pandas as pd
import requests

# Belgrade coordinates (Stari Grad)
BELGRADE_LAT = 44.8176
BELGRADE_LON = 20.4569

WEATHER_COM_URL = (
    "https://weather.com/weather/tenday/l/Belgrade+Stari+Grad+Serbia?"
    "canonicalCityId=75a7efac519144dfa5cc9f8c3801ba18591e63ada5845e5928117976aceb16a2"
)


def read_uploaded(upload: Union[IO, bytes]) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a clean temperature frame."""
    name = getattr(upload, "name", "")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    cols_lower = {c.lower().strip(): c for c in df.columns}
    date_col = cols_lower.get("date") or cols_lower.get("day") or list(df.columns)[0]
    temp_col = None
    for cand in ["temperature_c", "temperature", "temp", "temp_c", "t"]:
        if cand in cols_lower:
            temp_col = cols_lower[cand]
            break
    if temp_col is None:
        for c in df.columns:
            if c == date_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                temp_col = c
                break
    if temp_col is None:
        raise ValueError("Temperature column not found in upload.")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
            "temperature_c": pd.to_numeric(df[temp_col], errors="coerce"),
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def fetch_open_meteo(start: date, end: date) -> pd.DataFrame:
    """Fetch Belgrade daily mean temperature (archive + forecast)."""
    today = date.today()
    frames: list[pd.DataFrame] = []

    if start <= today:
        archive_end = min(end, today)
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": BELGRADE_LAT,
                "longitude": BELGRADE_LON,
                "start_date": start.isoformat(),
                "end_date": archive_end.isoformat(),
                "daily": "temperature_2m_mean",
                "timezone": "Europe/Belgrade",
            },
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json().get("daily", {})
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(payload.get("time", [])),
                    "temperature_c": payload.get("temperature_2m_mean", []),
                }
            )
        )

    if end > today:
        forecast_start = max(start, today)
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": BELGRADE_LAT,
                "longitude": BELGRADE_LON,
                "start_date": forecast_start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_mean",
                "timezone": "Europe/Belgrade",
            },
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json().get("daily", {})
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(payload.get("time", [])),
                    "temperature_c": payload.get("temperature_2m_mean", []),
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["date", "temperature_c"])

    out = pd.concat(frames, ignore_index=True).dropna()
    out = out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return out


def fetch_weather_com(start: date, end: date) -> pd.DataFrame:
    """Best-effort scrape of weather.com's 10-day Belgrade forecast."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(WEATHER_COM_URL, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    max_match = re.search(r'"temperatureMax"\s*:\s*\[(.*?)\]', html)
    min_match = re.search(r'"temperatureMin"\s*:\s*\[(.*?)\]', html)
    if not (max_match and min_match):
        raise RuntimeError("weather.com page did not contain a parseable forecast.")

    highs = [float(v) for v in max_match.group(1).split(",") if v.strip().lstrip("-").isdigit()]
    lows = [float(v) for v in min_match.group(1).split(",") if v.strip().lstrip("-").isdigit()]
    n = min(len(highs), len(lows))
    if n == 0:
        raise RuntimeError("weather.com payload contained no temperature values.")

    means_f = [(highs[i] + lows[i]) / 2 for i in range(n)]
    means_c = [(f - 32) * 5 / 9 for f in means_f]

    dates = pd.date_range(date.today(), periods=n, freq="D")
    return pd.DataFrame({"date": dates, "temperature_c": means_c})


def align(temp_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex onto the dashboard date range and interpolate small gaps."""
    df = temp_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").groupby("date", as_index=False).last()
    df = df.set_index("date").reindex(pd.DatetimeIndex(date_index).normalize())
    df["temperature_c"] = df["temperature_c"].astype(float)
    df["temperature_c"] = df["temperature_c"].interpolate(method="linear").ffill().bfill()
    out = df.reset_index().rename(columns={"index": "date"})
    return out
