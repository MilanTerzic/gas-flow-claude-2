"""
ENTSOG Transparency Platform fetcher.

Pulls daily Allocation data for the four Serbian import points and returns a
wide frame in mcm/day, ready to drop into the balance calculation.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from config import CONVERSION_MCM_TO_KWH, ENTSOG_POINT_DIRECTIONS

BASE = "https://transparency.entsog.eu/api/v1"


def _month_chunks(start: date, end: date):
    """Yield monthly [start, end) windows so the API doesn't time out."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        yield max(start, cur), min(end + timedelta(days=1), nxt)
        cur = nxt


def _fetch_point(
    pd_key: str,
    start: date,
    end: date,
    token: Optional[str] = None,
    indicator: str = "Allocation",
) -> pd.DataFrame:
    headers = {"User-Agent": "serbia-gas-dashboard/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    rows: list[dict] = []
    for win_start, win_end in _month_chunks(start, end):
        params = {
            "pointDirection": pd_key,
            "from": win_start.isoformat(),
            "to": win_end.isoformat(),
            "indicator": indicator,
            "periodType": "day",
            "timezone": "CET",
            "limit": -1,
        }
        r = requests.get(f"{BASE}/operationaldata", params=params, headers=headers, timeout=60)
        if r.status_code >= 400:
            if indicator == "Allocation":
                params["indicator"] = "Physical Flow"
                r = requests.get(
                    f"{BASE}/operationaldata", params=params, headers=headers, timeout=60
                )
        r.raise_for_status()
        payload = r.json()
        records = (
            payload.get("operationaldatas")
            or payload.get("operationaldata")
            or payload.get("data")
            or []
        )
        if isinstance(records, list):
            rows.extend(records)
        time.sleep(0.1)
    return pd.DataFrame(rows)


def fetch_flows(start: date, end: date, token: Optional[str] = None) -> pd.DataFrame:
    """Fetch the four canonical points and return a wide frame in mcm/day."""
    frames: list[pd.DataFrame] = []
    for canonical, pd_key in ENTSOG_POINT_DIRECTIONS.items():
        raw = _fetch_point(pd_key, start, end, token=token)
        if raw.empty:
            continue

        period_col = next(
            (c for c in raw.columns if c.lower() in ("periodfrom", "from", "period")),
            None,
        )
        value_col = next((c for c in raw.columns if c.lower() == "value"), None)
        unit_col = next((c for c in raw.columns if c.lower() == "unit"), None)
        if period_col is None or value_col is None:
            continue

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[period_col], errors="coerce").dt.normalize(),
                "value": pd.to_numeric(raw[value_col], errors="coerce"),
                "unit": raw[unit_col].astype(str).str.lower() if unit_col else "kwh/d",
            }
        ).dropna()

        def to_mcm(row):
            u = str(row["unit"]).lower()
            v = row["value"]
            if "mwh" in u:
                return v / (CONVERSION_MCM_TO_KWH / 1000)
            return v / CONVERSION_MCM_TO_KWH

        df["mcm_per_day"] = df.apply(to_mcm, axis=1)
        df = df.groupby("date", as_index=False)["mcm_per_day"].sum()
        df.rename(columns={"mcm_per_day": canonical}, inplace=True)
        frames.append(df)

    if not frames:
        raise RuntimeError("ENTSOG returned no data for any Serbian point in the requested range.")

    out = frames[0]
    for nxt in frames[1:]:
        out = out.merge(nxt, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)
