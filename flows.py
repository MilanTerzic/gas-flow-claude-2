"""
Flow data loading and unit conversion.

The canonical internal representation is:

    date              datetime64[ns]
    kiskundorozsma_hu  float (mcm/day)
    kireevo            float (mcm/day)
    kiskundorozsma_2   float (mcm/day)
    kalotina           float (mcm/day)
    [kiskundorozsma_hu_met]  optional float (mcm/day)
"""

from __future__ import annotations

from typing import IO, Union

import pandas as pd

from config import CONVERSION_MCM_TO_MWH, POINTS

CANONICAL_POINTS = list(POINTS.keys())


def _normalise_point_name(name: str) -> str:
    """Map any incoming point label to one of the canonical keys."""
    s = str(name).lower().strip()
    s = s.replace("/", " ").replace("-", " ").replace("_", " ")
    if "kalotina" in s:
        return "kalotina"
    if "kireevo" in s or "kireovo" in s or "kirevo" in s or "zaychar" in s:
        return "kireevo"
    if "kiskundorozsma" in s and ("2" in s or "ii" in s):
        return "kiskundorozsma_2"
    if "kiskundorozsma" in s and "met" in s:
        return "kiskundorozsma_hu_met"
    if "kiskundorozsma" in s:
        return "kiskundorozsma_hu"
    return s.replace(" ", "_")


def read_uploaded(upload: Union[IO, bytes]) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX of flow data; returns a wide frame in mcm/day."""
    name = getattr(upload, "name", "")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    cols_lower = {c.lower(): c for c in df.columns}
    if "point" in cols_lower:
        # Long format
        date_col = cols_lower.get("date", list(df.columns)[0])
        point_col = cols_lower["point"]
        if "mcm_per_day" in cols_lower:
            value_col = cols_lower["mcm_per_day"]
            factor = 1.0
        elif "mwh_per_day" in cols_lower:
            value_col = cols_lower["mwh_per_day"]
            factor = 1 / CONVERSION_MCM_TO_MWH
        else:
            value_col = next(
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            )
            factor = 1.0

        long = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col]).dt.normalize(),
                "point": df[point_col].map(_normalise_point_name),
                "value": pd.to_numeric(df[value_col], errors="coerce") * factor,
            }
        ).dropna()
        wide = long.pivot_table(
            index="date", columns="point", values="value", aggfunc="sum"
        ).reset_index()
    else:
        # Wide format
        date_col = cols_lower.get("date", list(df.columns)[0])
        wide = df.rename(columns={date_col: "date"}).copy()
        wide["date"] = pd.to_datetime(wide["date"]).dt.normalize()
        rename_map = {c: _normalise_point_name(c) for c in wide.columns if c != "date"}
        wide = wide.rename(columns=rename_map)

        numeric_cols = [c for c in wide.columns if c != "date"]
        if numeric_cols and wide[numeric_cols].abs().max().max() > 1000:
            for c in numeric_cols:
                wide[c] = wide[c] / CONVERSION_MCM_TO_MWH

    for c in CANONICAL_POINTS:
        if c not in wide.columns:
            wide[c] = 0.0

    cols_order = ["date"] + CANONICAL_POINTS + [
        c for c in wide.columns if c not in (["date"] + CANONICAL_POINTS)
    ]
    return wide[cols_order].sort_values("date").reset_index(drop=True)


def align(flow_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex flow data onto the master date_index, filling gaps with 0."""
    df = flow_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset="date").set_index("date").reindex(date_index)
    df = df.fillna(0.0)
    return df.reset_index().rename(columns={"index": "date"})
