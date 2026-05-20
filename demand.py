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

    kalotina_entry_mcm
    kiskundorozsma_entry_mcm
    imports_from_bulgaria_mcm   = kireevo_entry - kiskundorozsma_2_entry
    domestic_production_mcm     = 0.5 mcm/day (configurable)
    serbian_available_supply_mcm
    required_actual_mcm
    required_forecast_mcm
    temperature_actual_c
    temperature_forecast_c
    bosnia_consumption_mcm      (kept for reference / KPI, not in main stack)
    storage_imbalance_mcm
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


SUPPLY_COMPONENT_COLUMNS = [
    "imports_from_bulgaria_available_mcm",
    "kalotina_entry_mcm",
    "kiskundorozsma_entry_mcm",
    "domestic_production_mcm",
]


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


def _normalize_daily_index(date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return one sorted midnight timestamp per dashboard date."""
    return (
        pd.DatetimeIndex(pd.to_datetime(date_index))
        .normalize()
        .drop_duplicates()
        .sort_values()
    )


def _one_row_per_date(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and keep one complete daily row if sources overlap."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date")
    return out.groupby("date", as_index=False).last()


def _apply_current_day_flow_estimate(
    flow_aligned: pd.DataFrame,
    today_ts: pd.Timestamp,
    flow_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Current-day ENTSOG daily data may be incomplete during the gas day.

    If today's flow is missing, or is zero while yesterday had a positive
    value, temporarily use yesterday's actual value for today only. Future
    forecast days are deliberately left untouched.
    """
    out = flow_aligned.copy()
    out["is_current_day_estimate"] = False
    out["current_day_estimated_components"] = ""

    if today_ts not in out.index:
        return out

    yesterday = today_ts - pd.Timedelta(days=1)
    if yesterday not in out.index:
        return out

    estimated_components: list[str] = []
    for col in flow_columns:
        if col not in out.columns:
            continue

        today_value = out.at[today_ts, col]
        yesterday_value = out.at[yesterday, col]
        yesterday_is_usable = pd.notna(yesterday_value) and float(yesterday_value) > 0.0
        today_is_missing = pd.isna(today_value)
        today_is_artificial_zero = (
            pd.notna(today_value)
            and float(today_value) == 0.0
            and yesterday_is_usable
        )

        if yesterday_is_usable and (today_is_missing or today_is_artificial_zero):
            out.at[today_ts, col] = yesterday_value
            estimated_components.append(col)

    if estimated_components:
        out.at[today_ts, "is_current_day_estimate"] = True
        out.at[today_ts, "current_day_estimated_components"] = ", ".join(
            estimated_components
        )

    return out


def _append_estimated_component(existing: str, component: str) -> str:
    parts = [p.strip() for p in str(existing).split(",") if p.strip()]
    if component not in parts:
        parts.append(component)
    return ", ".join(parts)


def _apply_current_day_derived_estimate(
    df: pd.DataFrame,
    today_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Guard derived chart components against partial same-day ENTSOG reporting.

    Kireevo and the Kiskundorozsma-2 deduction can publish at different times.
    If today's net Bulgaria import is zero/missing, or jumps well above
    yesterday's net value, use yesterday's final net value for today only.
    """
    out = df.copy()
    today_mask = out["date"] == today_ts
    yesterday_mask = out["date"] == today_ts - pd.Timedelta(days=1)
    if not today_mask.any() or not yesterday_mask.any():
        return out

    today_idx = out.index[today_mask][0]
    yesterday_idx = out.index[yesterday_mask][0]

    def should_use_yesterday(col: str, spike_ratio: float = 1.5, spike_abs: float = 3.0) -> bool:
        today_value = out.at[today_idx, col]
        yesterday_value = out.at[yesterday_idx, col]
        if pd.isna(yesterday_value) or float(yesterday_value) <= 0.0:
            return False
        if pd.isna(today_value) or float(today_value) == 0.0:
            return True
        return float(today_value) > max(
            float(yesterday_value) * spike_ratio,
            float(yesterday_value) + spike_abs,
        )

    if should_use_yesterday("imports_from_bulgaria_mcm"):
        for col in [
            "imports_from_bulgaria_mcm",
            "bosnia_consumption_mcm",
            "imports_from_bulgaria_available_mcm",
        ]:
            out.at[today_idx, col] = out.at[yesterday_idx, col]
        out.at[today_idx, "is_current_day_estimate"] = True
        out.at[today_idx, "current_day_estimated_components"] = (
            _append_estimated_component(
                out.at[today_idx, "current_day_estimated_components"],
                "imports_from_bulgaria_mcm",
            )
        )

    for col in ["kalotina_entry_mcm", "kiskundorozsma_entry_mcm"]:
        if should_use_yesterday(col):
            out.at[today_idx, col] = out.at[yesterday_idx, col]
            out.at[today_idx, "is_current_day_estimate"] = True
            out.at[today_idx, "current_day_estimated_components"] = (
                _append_estimated_component(
                    out.at[today_idx, "current_day_estimated_components"],
                    col,
                )
            )

    return out


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
    max_storage_injection: float = 2.7,
    max_storage_withdrawal: float = 5.0,
) -> pd.DataFrame:
    """
    Build the daily Serbian gas balance frame on the master date_index.

    All series are reindexed onto ``date_index`` so historical and forecast
    rows live in one continuous frame with no gaps. Historical vs forecast
    is decided by ``is_forecast = date > today_ts``.
    """
    date_index = _normalize_daily_index(date_index)
    today_ts = pd.Timestamp(today_ts).normalize()
    temp_df = _one_row_per_date(temp_df)
    flow_df = _one_row_per_date(flow_df)

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
    flow_columns = ["kireevo", "kiskundorozsma_2", "kalotina", "kiskundorozsma_hu"]
    flow_aligned = flow_df.set_index("date").reindex(date_index)
    flow_aligned = _apply_current_day_flow_estimate(
        flow_aligned=flow_aligned,
        today_ts=today_ts,
        flow_columns=flow_columns,
    )
    estimate_flags = flow_aligned[
        ["is_current_day_estimate", "current_day_estimated_components"]
    ].copy()
    flow_aligned = flow_aligned.drop(
        columns=["is_current_day_estimate", "current_day_estimated_components"]
    ).fillna(0.0)
    kkd_hu = flow_aligned.get("kiskundorozsma_hu", pd.Series(0.0, index=date_index))
    kireevo = flow_aligned.get("kireevo", pd.Series(0.0, index=date_index))
    kkd_2 = flow_aligned.get("kiskundorozsma_2", pd.Series(0.0, index=date_index))
    kalotina = flow_aligned.get("kalotina", pd.Series(0.0, index=date_index))
    imports_from_bulgaria = (kireevo - kkd_2.clip(lower=0.0)).clip(lower=0.0)
    bosnia_consumption = (imports_from_bulgaria * float(bih_share)).clip(lower=0.0)

    # ---- Four supply components (no MET / others split) -------------------
    # kiskundorozsma_hu is the public HU>RS point-direction; if ENTSOG returns
    # zero, the plotted component remains zero. Kiskundorozsma-2/Horgos is
    # deducted from Kireevo before the Bosnia percentage is applied.
    df["kalotina_entry_mcm"] = kalotina.values
    df["kiskundorozsma_entry_mcm"] = kkd_hu.values
    df["imports_from_bulgaria_mcm"] = imports_from_bulgaria.values
    df["bosnia_consumption_pct"] = float(bih_share) * 100.0
    df["bosnia_consumption_mcm"] = bosnia_consumption.values
    df["imports_from_bulgaria_available_mcm"] = (
        df["imports_from_bulgaria_mcm"] - df["bosnia_consumption_mcm"]
    ).clip(lower=0.0)
    df["domestic_production_mcm"] = float(domestic_production)
    df["is_current_day_estimate"] = estimate_flags["is_current_day_estimate"].values
    df["current_day_estimated_components"] = estimate_flags[
        "current_day_estimated_components"
    ].values
    df = _apply_current_day_derived_estimate(df, today_ts)

    df["serbian_supply_before_bosnia_mcm"] = (
        df["imports_from_bulgaria_mcm"]
        + df["kalotina_entry_mcm"]
        + df["kiskundorozsma_entry_mcm"]
        + df["domestic_production_mcm"]
    )
    df["serbian_available_supply_mcm"] = (
        df["serbian_supply_before_bosnia_mcm"] - df["bosnia_consumption_mcm"]
    )

    df["storage_imbalance_raw_mcm"] = df["serbian_available_supply_mcm"] - df["demand_mcm"]
    capped_storage = df["storage_imbalance_raw_mcm"].clip(
        lower=-float(max_storage_withdrawal),
        upper=float(max_storage_injection),
    )
    df["storage_imbalance_mcm"] = capped_storage
    df["storage_injection_mcm"] = capped_storage.clip(lower=0.0)
    df["storage_withdrawal_mcm"] = -capped_storage.clip(upper=0.0)
    df["unserved_deficit_after_storage_limit_mcm"] = (
        -df["storage_imbalance_raw_mcm"] - float(max_storage_withdrawal)
    ).clip(lower=0.0)
    df["uncaptured_surplus_after_storage_limit_mcm"] = (
        df["storage_imbalance_raw_mcm"] - float(max_storage_injection)
    ).clip(lower=0.0)

    return df


def validate_balance_for_plot(
    df: pd.DataFrame,
    today_ts: pd.Timestamp,
    high_total_multiplier: float = 1.6,
    lookaround_days: int = 3,
) -> dict:
    """
    Return lightweight diagnostics for the Streamlit debug panel.

    The chart is daily and wide, so each date should appear once and each
    row should belong to either the historical or forecast side, never both.
    """
    check = df.copy()
    check["date"] = pd.to_datetime(check["date"]).dt.normalize()
    today_ts = pd.Timestamp(today_ts).normalize()

    components = [c for c in SUPPLY_COMPONENT_COLUMNS if c in check.columns]
    check["stacked_supply_total_mcm"] = check[components].sum(axis=1)

    duplicate_dates = (
        check[check["date"].duplicated(keep=False)]
        .sort_values("date")
        .copy()
    )
    hist_fcst_overlap = check[
        check["required_actual_mcm"].notna()
        & check["required_forecast_mcm"].notna()
    ].copy()

    median_total = check["stacked_supply_total_mcm"].median()
    if pd.isna(median_total) or median_total <= 0:
        high_total_threshold = np.nan
        high_totals = check.iloc[0:0].copy()
    else:
        high_total_threshold = float(median_total * high_total_multiplier)
        high_totals = check[
            check["stacked_supply_total_mcm"] > high_total_threshold
        ].copy()

    around_today = check[
        check["date"].between(
            today_ts - pd.Timedelta(days=max(lookaround_days, 1)),
            today_ts + pd.Timedelta(days=lookaround_days),
        )
    ].copy()
    current_day_estimates = check[
        check.get("is_current_day_estimate", False) == True  # noqa: E712
    ].copy()

    return {
        "components": components,
        "around_today": around_today,
        "current_day_estimates": current_day_estimates,
        "duplicate_dates": duplicate_dates,
        "hist_fcst_overlap": hist_fcst_overlap,
        "high_totals": high_totals,
        "high_total_threshold": high_total_threshold,
    }
