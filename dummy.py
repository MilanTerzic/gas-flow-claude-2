"""
Realistic dummy data so the dashboard renders end-to-end without any external
connection. Numbers are tuned to the seasonal range that the v8 Kalotina
workbook actually shows for Serbian flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CAPACITY_DEFS


def temperature_series(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """A smooth seasonal curve + light noise, centred on Belgrade climate."""
    rng = np.random.default_rng(seed=42)
    doy = date_index.dayofyear.values
    seasonal = 11 + 13 * np.cos(2 * np.pi * (doy - 200) / 365)
    noise = rng.normal(0, 1.4, size=len(date_index))
    temp = seasonal + noise
    return pd.DataFrame({"date": date_index, "temperature_c": temp})


def flow_series(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-point flows in mcm/day with realistic seasonal pattern."""
    rng = np.random.default_rng(seed=7)
    n = len(date_index)
    doy = date_index.dayofyear.values
    winter_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365)

    kireevo = 6.0 + 2.5 * winter_factor + rng.normal(0, 0.25, n)
    kkd_2 = 0.6 + 0.5 * winter_factor + rng.normal(0, 0.1, n)
    kkd_hu = 4.5 + 4.0 * winter_factor + rng.normal(0, 0.3, n)
    kalotina = 1.5 + 1.5 * winter_factor + rng.normal(0, 0.2, n)
    # ~30% of HU flows are MET-contract
    kkd_hu_met = np.clip(kkd_hu * 0.3 + rng.normal(0, 0.15, n), 0, None)

    df = pd.DataFrame(
        {
            "date": date_index,
            "kiskundorozsma_hu": np.clip(kkd_hu, 0, None),
            "kireevo": np.clip(kireevo, 0, None),
            "kiskundorozsma_2": np.clip(kkd_2, 0, None),
            "kalotina": np.clip(kalotina, 0, None),
            "kiskundorozsma_hu_met": kkd_hu_met,
        }
    )
    return df


def capacity_bookings() -> pd.DataFrame:
    """Synthetic capacity bookings covering all TSO × point × period combinations."""
    rng = np.random.default_rng(seed=11)
    rows: list[dict] = []
    periods = ["Day", "D-1", "D-2", "D-3", "D-4"]

    offered_baseline = {
        "FGSZ_exit": 105_000,
        "Bulgartransgaz_exit": 95_000,
        "Gastrans_entry_kireevo": 95_000,
        "Gastrans_exit_kkd2": 70_000,
        "FGSZ_entry_kkd2": 70_000,
    }

    def lookup(d):
        key = d["tso"]
        if "Kireevo" in d["border_point"]:
            key += f"_{d['direction']}_kireevo"
        elif "Kiskundorozsma 2" in d["border_point"]:
            key += f"_{d['direction']}_kkd2"
        else:
            key += f"_{d['direction']}"
        return offered_baseline.get(key, 80_000)

    for d in CAPACITY_DEFS:
        offered = lookup(d)
        for product in ["daily", "monthly", "quarterly"]:
            for period in periods:
                booked = offered * float(rng.uniform(0.45, 0.95))
                if d["currency"] == "HUF":
                    price = float(rng.uniform(0.0015, 0.0040))
                else:
                    price = float(rng.uniform(0.00002, 0.00012))
                rows.append(
                    {
                        "tso": d["tso"],
                        "border_point": d["border_point"],
                        "direction": d["direction"],
                        "product": product,
                        "period": period,
                        "offered_mwh": round(offered, 0),
                        "booked_mwh": round(booked, 0),
                        "utilisation_pct": round(booked / offered * 100, 1),
                        "price": price,
                        "currency": d["currency"],
                        "pct_of_100": round(booked / offered * 100, 1),
                    }
                )
    return pd.DataFrame(rows)
