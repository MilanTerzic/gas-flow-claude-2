"""
Serbia Gas Balance and Capacity Dashboard
==========================================

Streamlit dashboard for Serbian natural gas flows, supply structure, demand
forecast and cross-border capacity bookings.

Tabs:
  1. Gas Balance        — KPIs + 3 vertically-aligned compact charts
  2. Flow Details       — per-point flow series and tables
  3. Capacity Bookings  — Excel-style booking tables and capacity charts
  4. Model & Assumptions

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

import capacity, charts, demand, dummy, entsog, flows, model, temperature
from config import (
    BIH_SHARE,
    CURVE_DISTORTION_DEFAULT,
    CURVE_SHIFT_DEFAULT,
    DOMESTIC_PRODUCTION_MCM,
    LINEAR_COEFFS,
    POINTS,
    POLY_COEFFS,
)

# -----------------------------------------------------------------------------
# Page setup — white, compact, operational
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Serbia Gas Balance & Capacity Dashboard",
    page_icon="🇷🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1400px; }
      div[data-testid="stMetricValue"] { font-size: 1.05rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.72rem; }
      .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
      h1 { font-size: 1.6rem !important; margin-bottom: 0.1rem; }
      .stCaption { color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🇷🇸 Serbia Gas Balance & Capacity Dashboard")
st.caption(
    "Daily supply composition · Belgrade temperature · Storage +/- · "
    "Cross-border capacity bookings (FGSZ · Bulgartransgaz · Gastrans)"
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    today = date.today()
    # Default: ~10 historical days + today + ~10 forecast days
    default_start = today - timedelta(days=10)
    default_end = today + timedelta(days=10)

    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        help="Rolling window centred on today. Keep it short for a readable chart.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, default_end

    st.divider()
    st.subheader("Data sources")

    use_dummy = st.toggle(
        "Use dummy demonstration data",
        value=True,
        help="ON → realistic synthetic data so the dashboard renders without network access.",
    )
    entsog_token = st.text_input(
        "ENTSOG API token (optional)", type="password",
        help="Public ENTSOG endpoints don't require a token.",
    )
    temp_source = st.selectbox(
        "Temperature source",
        options=["Open-Meteo (auto)", "weather.com scrape", "Manual / Upload"],
        index=0,
    )

    st.divider()
    st.subheader("Manual fallbacks")
    flow_upload = st.file_uploader("Flow data (CSV/XLSX)", type=["csv", "xlsx"])
    capacity_upload = st.file_uploader("Capacity bookings (CSV/XLSX)", type=["csv", "xlsx"])
    temp_upload = st.file_uploader("Temperature data (CSV/XLSX)", type=["csv", "xlsx"])
    model_upload = st.file_uploader("Regression model workbook (XLSX)", type=["xlsx"])

    st.divider()
    st.subheader("Model overrides")
    use_polynomial = st.toggle("Use polynomial regression", value=True)
    curve_shift = st.number_input("Curve shift (mcm/d)", value=CURVE_SHIFT_DEFAULT, step=0.1)
    curve_distortion = st.number_input(
        "Curve distortion factor", value=CURVE_DISTORTION_DEFAULT, step=0.1,
        help="1.0 = no distortion (default). Workbook uses 2.8 only in extreme cold scenarios.",
    )
    bih_pct = st.slider(
        "Bosnia consumption / export (% of Import from BG)",
        min_value=0.0, max_value=20.0, value=BIH_SHARE * 100, step=0.5,
    ) / 100.0
    production_mcm = st.number_input(
        "Domestic production (mcm/day)", value=DOMESTIC_PRODUCTION_MCM, step=0.1,
    )


# -----------------------------------------------------------------------------
# Build the master date index and load all series
# -----------------------------------------------------------------------------

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Single master date_index that every series is reindexed to
date_index = pd.date_range(start=start_date, end=end_date, freq="D")
today_ts = pd.Timestamp(today)

# 1) Coefficients
poly_coeffs = POLY_COEFFS
linear_coeffs = LINEAR_COEFFS
if model_upload is not None:
    try:
        poly_coeffs, linear_coeffs = model.read_coefficients_from_xlsx(model_upload)
        st.sidebar.success("Coefficients loaded from uploaded workbook.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not read coefficients from workbook: {exc}")

# 2) Temperature
temp_df: Optional[pd.DataFrame] = None
if temp_upload is not None:
    try:
        temp_df = temperature.read_uploaded(temp_upload)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not parse uploaded temperature file: {exc}")

if temp_df is None and not use_dummy:
    if temp_source == "Open-Meteo (auto)":
        try:
            temp_df = temperature.fetch_open_meteo(start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Open-Meteo fetch failed — falling back to dummy: {exc}")
    elif temp_source == "weather.com scrape":
        try:
            temp_df = temperature.fetch_weather_com(start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"weather.com scrape failed — falling back to dummy: {exc}")

if temp_df is None:
    temp_df = dummy.temperature_series(date_index)

temp_df = temperature.align(temp_df, date_index)

# 3) Flows
flow_df: Optional[pd.DataFrame] = None
if flow_upload is not None:
    try:
        flow_df = flows.read_uploaded(flow_upload)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not parse uploaded flow file: {exc}")

if flow_df is None and not use_dummy:
    try:
        flow_df = entsog.fetch_flows(start_date, end_date, token=entsog_token or None)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"ENTSOG fetch failed — falling back to dummy: {exc}")

if flow_df is None:
    flow_df = dummy.flow_series(date_index)

flow_df = flows.align(flow_df, date_index)

# 4) Capacity
cap_df: Optional[pd.DataFrame] = None
if capacity_upload is not None:
    try:
        cap_df = capacity.read_uploaded(capacity_upload)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not parse uploaded capacity file: {exc}")
if cap_df is None:
    cap_df = dummy.capacity_bookings()

# 5) Build balance — single source of truth, all on date_index
balance = demand.build_balance(
    date_index=date_index,
    today_ts=today_ts,
    flow_df=flow_df,
    temp_df=temp_df,
    poly_coeffs=poly_coeffs,
    linear_coeffs=linear_coeffs,
    use_polynomial=use_polynomial,
    curve_shift=curve_shift,
    curve_distortion=curve_distortion,
    bih_share=bih_pct,
    domestic_production=production_mcm,
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_balance, tab_flows, tab_capacity, tab_model = st.tabs(
    ["📊 Gas Balance", "🔁 Flow Details", "📋 Capacity Bookings", "🧮 Model & Assumptions"],
)


# =============================================================================
# TAB 1 — GAS BALANCE
# =============================================================================
with tab_balance:
    # KPI row — pick the "today" row (or middle of range if today is outside)
    today_row = balance[balance["date"] == today_ts]
    if today_row.empty:
        kpi_row = balance.iloc[len(balance) // 2]
    else:
        kpi_row = today_row.iloc[0]

    # First row of KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecasted demand", f"{kpi_row['demand_mcm']:,.2f} mcm/d")
    k2.metric("Total available supply", f"{kpi_row['serbian_available_supply_mcm']:,.2f} mcm/d")
    storage_val = kpi_row["storage_imbalance_mcm"]
    k3.metric(
        "Storage +/-",
        f"{storage_val:+,.2f} mcm/d",
        delta="Injection" if storage_val >= 0 else "Withdrawal",
        delta_color="normal" if storage_val >= 0 else "inverse",
    )
    k4.metric(
        "Belgrade temp",
        f"{kpi_row['temperature_c']:.1f} °C",
        delta=f"avg: {kpi_row['avg_temperature_c']:.1f} °C",
    )

    # Second row of KPIs
    k5, k6, k7, k8 = st.columns(4)
    import_hu_total = kpi_row["import_hu_others_mcm"] + kpi_row["import_hu_met_mcm"]
    k5.metric("Import from HU", f"{import_hu_total:,.2f} mcm/d")
    k6.metric("Import from BG (net)", f"{kpi_row['import_bg_mcm']:,.2f} mcm/d")
    k7.metric("Import Kalotina", f"{kpi_row['import_kalotina_mcm']:,.2f} mcm/d")
    k8.metric("Production", f"{kpi_row['production_mcm']:,.2f} mcm/d")

    # Third KPI row — Bosnia export
    k9, _, _, _ = st.columns(4)
    k9.metric("Bosnia consumption/export", f"{kpi_row['bosnia_consumption_mcm']:,.2f} mcm/d")

    st.markdown("")  # small gap

    # ---- Three compact, vertically-aligned charts -------------------------
    st.plotly_chart(
        charts.plot_gas_balance_chart(balance, today_ts),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        charts.plot_temperature_chart(balance, today_ts),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        charts.plot_storage_chart(balance, today_ts),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    with st.expander("📄 Show daily balance table"):
        show_cols = [
            "date",
            "temperature_c",
            "avg_temperature_c",
            "demand_mcm",
            "import_kalotina_mcm",
            "import_bg_mcm",
            "import_hu_others_mcm",
            "import_hu_met_mcm",
            "production_mcm",
            "bosnia_consumption_mcm",
            "serbian_available_supply_mcm",
            "storage_imbalance_mcm",
            "is_forecast",
        ]
        display = balance[show_cols].copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download balance as CSV",
            data=balance[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="serbia_gas_balance.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 2 — FLOW DETAILS
# =============================================================================
with tab_flows:
    st.subheader("Physical flows by point")
    st.caption(
        "Daily allocations at each ENTSOG point in mcm/day. "
        "Note: *Import from BG (net)* = Kireevo − Kiskundorozsma-2."
    )

    # Plot only the 4 canonical points (skip MET sub-split here)
    flow_plot_df = flow_df[["date", "kiskundorozsma_hu", "kireevo", "kiskundorozsma_2", "kalotina"]]
    st.plotly_chart(
        charts.plot_flow_details_chart(flow_plot_df, today_ts, POINTS),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Raw flow data (mcm/d)**")
        st.dataframe(
            flow_df.assign(date=flow_df["date"].dt.strftime("%Y-%m-%d")),
            use_container_width=True, hide_index=True,
        )
    with cols[1]:
        st.markdown("**Unit conversion**")
        st.dataframe(
            pd.DataFrame(
                {
                    "From": ["MWh/day", "kWh/day", "GWh/day", "mcm/day"],
                    "To":   ["mcm/day", "mcm/day", "mcm/day", "GWh/day"],
                    "Factor": ["÷ 10,550", "÷ 10,550,000", "÷ 10.55", "× 10.55"],
                    "Basis": ["1 mcm = 10.55 GWh"] * 4,
                }
            ),
            hide_index=True, use_container_width=True,
        )


# =============================================================================
# TAB 3 — CAPACITY BOOKINGS
# =============================================================================
with tab_capacity:
    st.subheader("Cross-border capacity bookings")
    st.caption("FGSZ · Bulgartransgaz · Gastrans — daily / monthly / quarterly products")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        tso_filter = st.multiselect(
            "TSO", sorted(cap_df["tso"].dropna().unique()),
            default=sorted(cap_df["tso"].dropna().unique()),
        )
    with f2:
        bp_filter = st.multiselect(
            "Border point", sorted(cap_df["border_point"].dropna().unique()),
            default=sorted(cap_df["border_point"].dropna().unique()),
        )
    with f3:
        dir_filter = st.multiselect(
            "Direction", sorted(cap_df["direction"].dropna().unique()),
            default=sorted(cap_df["direction"].dropna().unique()),
        )
    with f4:
        prod_filter = st.multiselect(
            "Product", sorted(cap_df["product"].dropna().unique()),
            default=sorted(cap_df["product"].dropna().unique()),
        )

    cap_view = cap_df[
        cap_df["tso"].isin(tso_filter)
        & cap_df["border_point"].isin(bp_filter)
        & cap_df["direction"].isin(dir_filter)
        & cap_df["product"].isin(prod_filter)
    ].copy()

    st.markdown("### Bookings table")
    grouped = capacity.format_table(cap_view)
    st.dataframe(grouped, use_container_width=True, hide_index=True)

    # Two charts side by side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.plot_capacity_booked_chart(cap_view),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        st.plotly_chart(
            charts.plot_capacity_utilisation_chart(cap_view),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Prices — separate HUF / EUR
    st.markdown("**Price comparison** — HUF and EUR shown separately (different magnitudes)")
    huf_fig, eur_fig = charts.plot_capacity_price_chart(cap_view)
    pc1, pc2 = st.columns(2)
    with pc1:
        if huf_fig is not None:
            st.plotly_chart(huf_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No HUF-priced bookings in current filter.")
    with pc2:
        if eur_fig is not None:
            st.plotly_chart(eur_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No EUR-priced bookings in current filter.")

    with st.expander("Raw capacity data"):
        st.dataframe(cap_view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download as CSV",
            data=cap_view.to_csv(index=False).encode("utf-8"),
            file_name="serbia_capacity_bookings.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 4 — MODEL & ASSUMPTIONS
# =============================================================================
with tab_model:
    st.subheader("Regression model")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Polynomial regression (active)**")
        st.latex(r"y = 0.0007\,x^{3} - 0.0188\,x^{2} - 0.3194\,x + 11.987")
        st.write("y = Serbian daily demand (mcm/d), x = Belgrade 2-day avg temperature (°C).")
        st.dataframe(
            pd.DataFrame({"Term": ["x³", "x²", "x", "constant"], "Coefficient": list(poly_coeffs)}),
            hide_index=True, use_container_width=True,
        )
    with cB:
        st.markdown("**Linear regression (fallback)**")
        st.latex(r"y = -0.354\,x + 11.396")
        st.dataframe(
            pd.DataFrame({"Term": ["x", "constant"], "Coefficient": list(linear_coeffs)}),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    st.subheader("Assumptions")
    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": [
                    "Domestic Serbian production",
                    "Bosnia consumption / export share",
                    "Import from BG (net)",
                    "Serbian available supply",
                    "Storage balance / imbalance",
                    "Energy conversion",
                    "Curve shift",
                    "Curve distortion",
                ],
                "Value": [
                    f"{production_mcm:.2f} mcm/day",
                    f"{bih_pct*100:.1f}% of Import from BG",
                    "Kireevo − Kiskundorozsma-2",
                    "KKD HU + Import BG + Kalotina + Production − Bosnia",
                    "Available supply − Required demand",
                    "1 mcm = 10.55 GWh",
                    f"{curve_shift}",
                    f"{curve_distortion}",
                ],
            }
        ),
        hide_index=True, use_container_width=True,
    )

    st.divider()
    st.subheader("Temperature & demand series")
    fc = balance[["date", "temperature_c", "avg_temperature_c", "demand_mcm", "is_forecast"]].copy()
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")
    fc = fc.rename(
        columns={
            "temperature_c": "Temperature (°C)",
            "avg_temperature_c": "2-day Avg Temp (°C)",
            "demand_mcm": "Demand (mcm/d)",
            "is_forecast": "Forecast?",
        }
    )
    st.dataframe(fc, use_container_width=True, hide_index=True)
