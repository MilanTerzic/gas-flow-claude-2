"""
Capacity-booking helpers.

Canonical schema (one row per TSO × border-point × direction × product × period):

  tso             FGSZ | Bulgartransgaz | Gastrans
  border_point    Free text — see CAPACITY_DEFS in config.py
  direction       entry | exit
  product         daily | monthly | quarterly
  period          Day | D-1 | D-2 | D-3 | D-4 | M | Q
  offered_mwh     float, MWh/day
  booked_mwh      float, MWh/day
  utilisation_pct float, 0–100  (= booked / offered × 100)
  price           float, in currency units per kWh/h/day
  currency        EUR | HUF
  pct_of_100      float, 0–100  (optional secondary % column from source files)
"""

from __future__ import annotations

from typing import IO, Union

import pandas as pd

REQUIRED_COLS = [
    "tso",
    "border_point",
    "direction",
    "product",
    "period",
    "offered_mwh",
    "booked_mwh",
    "utilisation_pct",
    "price",
    "currency",
    "pct_of_100",
]


def read_uploaded(upload: Union[IO, bytes]) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX capacity booking file."""
    name = getattr(upload, "name", "")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for key in REQUIRED_COLS:
        if key in cols_lower:
            rename[cols_lower[key]] = key
    df = df.rename(columns=rename)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None

    df["offered_mwh"] = pd.to_numeric(df["offered_mwh"], errors="coerce")
    df["booked_mwh"] = pd.to_numeric(df["booked_mwh"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["pct_of_100"] = pd.to_numeric(df["pct_of_100"], errors="coerce")

    mask = df["utilisation_pct"].isna() & df["offered_mwh"].notna() & (df["offered_mwh"] > 0)
    df.loc[mask, "utilisation_pct"] = (
        df.loc[mask, "booked_mwh"] / df.loc[mask, "offered_mwh"]
    ) * 100
    return df[REQUIRED_COLS]


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the bookings frame for display: rounded numbers, '-' / 'N/A' for
    missing values, sorted by TSO → border point → product → period.
    """
    if df.empty:
        return df

    period_order = {p: i for i, p in enumerate(["Day", "D-1", "D-2", "D-3", "D-4", "M", "Q"])}
    out = df.copy()
    out["_period_order"] = out["period"].map(period_order).fillna(99)
    out = out.sort_values(
        ["tso", "border_point", "direction", "product", "_period_order"]
    ).drop(columns="_period_order")

    def fmt_num(v, digits=0):
        if pd.isna(v):
            return "-"
        return f"{v:,.{digits}f}"

    def fmt_pct(v):
        if pd.isna(v):
            return "N/A"
        return f"{v:,.1f}%"

    def fmt_price(row):
        v = row["price"]
        if pd.isna(v):
            return "-"
        ccy = row["currency"] or ""
        return f"{v:,.4f} {ccy}"

    display = pd.DataFrame(
        {
            "TSO": out["tso"],
            "Border point": out["border_point"],
            "Type": out["direction"],
            "Product": out["product"],
            "Period": out["period"],
            "Offered (MWh/day)": [fmt_num(v) for v in out["offered_mwh"]],
            "Booked (MWh/day)": [fmt_num(v) for v in out["booked_mwh"]],
            "Booked %": [fmt_pct(v) for v in out["utilisation_pct"]],
            "Price": [fmt_price(r) for _, r in out.iterrows()],
            "Currency": out["currency"].fillna("-"),
            "% of 100": [fmt_pct(v) for v in out["pct_of_100"]],
        }
    )
    return display
