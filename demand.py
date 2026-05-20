"""
Demand forecast and gas-balance calculation.

Implements the model documented in
`Serbian_natural_gas_balance_v8__Kalotina.xlsx` →
sheet "Serbian Gas Cons. forecast":

    Required (est.)        =  poly(avg_temp_C)        # mcm/day
    Import from BG (net)   =  Kireevo - Kiskundorozsma_2
    Bosnia consumption     =  bih_share × Import from BG (net)
    Available supply       =  Kiskundorozsma_HU
                              + Import from BG (net)
                              + Kalotina
                              + Production
                              - Bosnia consumption
    Storage +/-            =  Available supply - Required (est.)

The output frame uses the explicit ``*_mcm`` column naming required by
the dashboard layer:

    import_kalotina_mcm
    import_bg_mcm
    production_mcm
    import_hu_others_mcm
    import_hu_met_mcm
    required_actual_mcm
    required_forecast_mcm
    temperature_actual_c
    temperature_forecast_c
    bosnia_consumption_mcm
    serbian_available_supply_mcm
    storage_imbalance_mcm
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def forecast_demand(
    temperature_c: pd.Series,
    poly_coeffs: Sequence[float],
    linear_coeffs: Sequence[float],
    use_polynomial: bool = True,
    curve_shift: float = 0.0,
    curve_distortion: float = 1.0,
) -> pd.Series:
    """Return Serbian daily demand in mcm/day."""
    x = temperature_c.astype(float)
    coeffs = poly_coeffs if use_polynomial else linear_coeffs
    base = pd.Series(np.polyval(list(coeffs), x), index=x.index)

    if curve_distortion is None or curve_distortion == 0:
        curve_distortion = 1.0
    return base * curve_distortion + curve_shift


def rolling_avg_temperature(temp_series: pd.Series, window: int = 2) -> pd.Series:
    """Workbook column 'Avg. Temp' is a simple 2-day rolling mean."""
    return temp_series.rolling(window=window, min_periods=1).mean()


def build_balance(
    date_index: pd.DatetimeIndex,
    today_ts: pd.Timestamp,
    flow_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    poly_coeffs: Sequence[float],
    linear_coeffs: Sequence[float],
    use_polynomial: bool = True,
    curve_shift: float = 0.0,
    curve_distortion: float = 1.0,
    bih_share: float = 0.08,
    domestic_production: float = 0.5,
) -> pd.DataFrame:
    """
    Build the daily Serbian gas balance frame on the master date_index.

    All series are reindexed onto ``date_index`` so historical and forecast
    rows live in one continuous frame with no gaps. Historical vs forecast
    is decided by ``is_forecast = date > today_ts``.
    """
    df = pd.DataFrame({"date": date_index})

    # ---- Temperature (single column actual+forecast; we split for plotting) --
    temp = temp_df.set_index("date").reindex(date_index)["temperature_c"]
    df["temperature_c"] = temp.values
    df["avg_temperature_c"] = rolling_avg_temperature(temp).values

    is_forecast = df["date"] > today_ts
    df["is_forecast"] = is_forecast.values

    df["temperature_actual_c"] = np.where(is_forecast, np.nan, df["temperature_c"])
    df["temperature_forecast_c"] = np.where(is_forecast, df["temperature_c"], np.nan)

    # ---- Demand ------------------------------------------------------------
    demand = forecast_demand(
        df["avg_temperature_c"],
        poly_coeffs=poly_coeffs,
        linear_coeffs=linear_coeffs,
        use_polynomial=use_polynomial,
        curve_shift=curve_shift,
        curve_distortion=curve_distortion,
    ).values
    df["demand_mcm"] = demand
    df["required_actual_mcm"] = np.where(is_forecast, np.nan, demand)
    df["required_forecast_mcm"] = np.where(is_forecast, demand, np.nan)

    # ---- Flows (already in mcm/day from the flows module) ------------------
    flow_aligned = flow_df.set_index("date").reindex(date_index).fillna(0.0)
    kkd_hu = flow_aligned.get("kiskundorozsma_hu", pd.Series(0.0, index=date_index))
    kireevo = flow_aligned.get("kireevo", pd.Series(0.0, index=date_index))
    kkd_2 = flow_aligned.get("kiskundorozsma_2", pd.Series(0.0, index=date_index))
    kalotina = flow_aligned.get("kalotina", pd.Series(0.0, index=date_index))

    # MET vs Others on the Hungarian side
    if "kiskundorozsma_hu_met" in flow_aligned.columns:
        met = flow_aligned["kiskundorozsma_hu_met"].fillna(0.0)
        others = (kkd_hu - met).clip(lower=0.0)
    else:
        met = pd.Series(0.0, index=date_index)
        others = kkd_hu

    df["import_kalotina_mcm"] = kalotina.values
    df["import_bg_mcm"] = (kireevo - kkd_2).clip(lower=0.0).values
    df["import_hu_others_mcm"] = others.values
    df["import_hu_met_mcm"] = met.values
    df["production_mcm"] = float(domestic_production)
    df["bosnia_consumption_mcm"] = bih_share * df["import_bg_mcm"]

    df["serbian_available_supply_mcm"] = (
        df["import_hu_others_mcm"]
        + df["import_hu_met_mcm"]
        + df["import_bg_mcm"]
        + df["import_kalotina_mcm"]
        + df["production_mcm"]
        - df["bosnia_consumption_mcm"]
    )
    df["storage_imbalance_mcm"] = df["serbian_available_supply_mcm"] - df["demand_mcm"]

    return df
