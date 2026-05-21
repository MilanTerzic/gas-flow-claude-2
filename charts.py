"""
Pure plotting functions.

Each function takes a fully-prepared DataFrame plus a ``today`` timestamp and
returns a configured ``plotly.graph_objects.Figure``. No data fetching, no
calculation, no Streamlit calls — easy to unit-test and easy to reuse.

Style targets
-------------
* white background (no dark theme), plotly_white template
* compact heights (330–380 / 170–220 / 200–240 px)
* uniform margins  margin=dict(l=50, r=20, t=40, b=35)
* light grey gridlines  rgba(220,220,220,0.7)
* horizontal legend above the plot
* narrow red "today" vrect spanning exactly one day, identical on every chart
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    COLOR_BG_IMPORT,
    COLOR_DEMAND,
    COLOR_HU_MET,
    COLOR_HU_OTHERS,
    COLOR_KALOTINA,
    COLOR_PRODUCTION,
    COLOR_STORAGE_NEG,
    COLOR_STORAGE_POS,
    COLOR_TEMP,
    COLOR_TODAY,
    GRID_COLOR,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEFAULT_MARGIN = dict(l=60, r=20, t=50, b=40)


def _today_band(fig: go.Figure, today: pd.Timestamp) -> None:
    """Add a narrow, semi-transparent red band covering exactly one day."""
    fig.add_vrect(
        x0=today - pd.Timedelta(hours=12),
        x1=today + pd.Timedelta(hours=12),
        fillcolor=COLOR_TODAY,
        opacity=0.15,
        line_width=1,
        line_color=COLOR_TODAY,
        layer="below",
    )


def _apply_common_layout(
    fig: go.Figure,
    title: Optional[str],
    y_title: str,
    height: int,
    show_legend: bool = True,
) -> None:
    layout_title = dict(text="")
    margin = DEFAULT_MARGIN.copy()
    legend_y = 1.02
    if title:
        layout_title = dict(
            text=title,
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=14),
        )
        margin["t"] = 80
        legend_y = 1.12

    fig.update_layout(
        title=layout_title,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=margin,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.15,
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickformat="%b %d",
        showline=True,
        linewidth=1,
        linecolor="rgba(150,150,150,0.5)",
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=GRID_COLOR,
        showline=True,
        linewidth=1,
        linecolor="rgba(150,150,150,0.5)",
        zeroline=False,
    )


def _split_hist_fcst(df: pd.DataFrame, today: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_hist = df["date"] <= today
    return df[is_hist].copy(), df[~is_hist].copy()


# ---------------------------------------------------------------------------
# 1) Gas balance — daily composition of Serbia demand (stacked area)
# ---------------------------------------------------------------------------

# Order = bottom-up in the stack and in the legend.
# Four components only — no HU MET/others split.
SUPPLY_COMPONENTS: List[Tuple[str, str, str]] = [
    ("imports_from_bulgaria_available_mcm",  "Imports from Bulgaria", COLOR_BG_IMPORT),
    ("kalotina_entry_mcm",         "Kalotina entry",        COLOR_KALOTINA),
    ("kiskundorozsma_entry_mcm",   "Kiskundorozsma entry",  COLOR_HU_OTHERS),
    ("domestic_production_mcm",    "Domestic production",   COLOR_PRODUCTION),
]


def plot_gas_balance_chart(df: pd.DataFrame, today: pd.Timestamp) -> go.Figure:
    """
    Cumulative stacked-area supply composition + red required-demand line.

    Each component is rendered as TWO Scatter traces sharing a legendgroup:
    a solid-fill historical area (rows where date <= today) and a
    lighter-fill forecast area (rows where date > today). Both go into the
    same ``stackgroup`` so the visual stack remains continuous across the
    today boundary.

    The red demand line sits on top — solid for historical, dashed for
    forecast.
    """
    fig = go.Figure()

    # Pre-split each component into hist / fcst series. We blank the "other
    # half" by setting it to None so the same stackgroup doesn't double-count
    # at the today boundary.
    is_hist = df["date"] <= today
    has_fcst = (~is_hist).any()

    for col, label, color in SUPPLY_COMPONENTS:
        if col not in df.columns or df[col].abs().sum() == 0:
            continue

        hist_y = df[col].where(is_hist)
        fcst_y = df[col].where(~is_hist)

        # Historical area — solid
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=hist_y,
                name=label,
                mode="lines",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                stackgroup="supply",
                legendgroup=label,
                hovertemplate=f"{label}: %{{y:.2f}} mcm/d<extra></extra>",
            )
        )
        # Forecast area — same color, lighter (semi-transparent) so the
        # boundary is visually obvious.
        if has_fcst:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=fcst_y,
                    name=f"{label} (fcst)",
                    mode="lines",
                    line=dict(width=0.5, color=color, dash="dot"),
                    fillcolor=_with_alpha(color, 0.45),
                    stackgroup="supply",
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=f"{label} (fcst): %{{y:.2f}} mcm/d<extra></extra>",
                )
            )

    # ---- Required demand on top of the stack -----------------------------
    hist, fcst = _split_hist_fcst(df, today)

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["required_actual_mcm"],
            mode="lines",
            name="Required (est.)",
            line=dict(color=COLOR_DEMAND, width=2.5),
            legendgroup="demand",
            hovertemplate="Required: %{y:.2f} mcm/d<extra></extra>",
        )
    )
    if not fcst.empty:
        # Bridge from the last historical point so the line is continuous.
        bridge_x = ([hist["date"].iloc[-1]] if not hist.empty else []) + list(fcst["date"])
        bridge_y = (
            ([hist["required_actual_mcm"].iloc[-1]] if not hist.empty else [])
            + list(fcst["required_forecast_mcm"])
        )
        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines",
                name="Required (fcst)",
                line=dict(color=COLOR_DEMAND, width=2.5, dash="dash"),
                legendgroup="demand",
                hovertemplate="Required (fcst): %{y:.2f} mcm/d<extra></extra>",
            )
        )

    _today_band(fig, today)
    _apply_common_layout(
        fig,
        title=None,
        y_title="mcm/day",
        height=360,
    )
    return fig


def _with_alpha(color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string into an rgba() string with the given alpha."""
    c = color.lstrip("#")
    if len(c) != 6:
        return color
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# 2) Belgrade temperature
# ---------------------------------------------------------------------------

def plot_temperature_chart(df: pd.DataFrame, today: pd.Timestamp) -> go.Figure:
    """Solid blue actual + dashed blue forecast."""
    fig = go.Figure()
    hist, fcst = _split_hist_fcst(df, today)

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["temperature_actual_c"],
            mode="lines+markers",
            name="Temp (actual)",
            line=dict(color=COLOR_TEMP, width=2),
            marker=dict(size=4, color=COLOR_TEMP),
            hovertemplate="%{y:.1f} °C<extra></extra>",
        )
    )

    if not fcst.empty:
        bridge = pd.DataFrame(
            {
                "date": ([hist["date"].iloc[-1]] if not hist.empty else []) + list(fcst["date"]),
                "temperature_forecast_c": (
                    ([hist["temperature_actual_c"].iloc[-1]] if not hist.empty else [])
                    + list(fcst["temperature_forecast_c"])
                ),
            }
        )
        fig.add_trace(
            go.Scatter(
                x=bridge["date"],
                y=bridge["temperature_forecast_c"],
                mode="lines+markers",
                name="Temp (fcst)",
                line=dict(color=COLOR_TEMP, width=2, dash="dash"),
                marker=dict(size=4, color=COLOR_TEMP, symbol="circle-open"),
                hovertemplate="%{y:.1f} °C (fcst)<extra></extra>",
            )
        )

    _today_band(fig, today)
    _apply_common_layout(
        fig,
        title=None,
        y_title="°C",
        height=200,
    )
    return fig


# ---------------------------------------------------------------------------
# 3) Storage +/-
# ---------------------------------------------------------------------------

def plot_storage_chart(df: pd.DataFrame, today: pd.Timestamp) -> go.Figure:
    """
    Positive storage_imbalance → injection (green bars above zero).
    Negative → withdrawal (red bars below zero).
    Forecast bars are hatched.
    """
    fig = go.Figure()
    hist, fcst = _split_hist_fcst(df, today)

    # Historical
    fig.add_trace(
        go.Bar(
            x=hist["date"],
            y=hist["storage_imbalance_mcm"],
            name="Storage +/- (hist)",
            marker_color=np.where(
                hist["storage_imbalance_mcm"] >= 0,
                COLOR_STORAGE_POS,
                COLOR_STORAGE_NEG,
            ),
            marker_line_width=0,
            showlegend=False,
            hovertemplate="%{y:+.2f} mcm/d<extra></extra>",
        )
    )

    # Forecast (hatched)
    if not fcst.empty:
        fig.add_trace(
            go.Bar(
                x=fcst["date"],
                y=fcst["storage_imbalance_mcm"],
                name="Storage +/- (fcst)",
                marker_color=np.where(
                    fcst["storage_imbalance_mcm"] >= 0,
                    COLOR_STORAGE_POS,
                    COLOR_STORAGE_NEG,
                ),
                marker_pattern_shape="/",
                marker_pattern_solidity=0.35,
                marker_pattern_fgcolor="white",
                marker_line_width=0,
                showlegend=False,
                hovertemplate="%{y:+.2f} mcm/d (fcst)<extra></extra>",
            )
        )

    # Strong horizontal zero line
    fig.add_hline(y=0, line_color=COLOR_DEMAND, line_width=2)

    # Auto-scale around the values but keep zero clearly visible
    y_min = float(df["storage_imbalance_mcm"].min())
    y_max = float(df["storage_imbalance_mcm"].max())
    if abs(y_max - y_min) < 1e-6:
        pad = max(0.5, abs(y_max) * 0.5)
    else:
        pad = (y_max - y_min) * 0.15
    # Ensure zero stays in the visible window
    y_lo = min(y_min - pad, -pad / 2)
    y_hi = max(y_max + pad, pad / 2)

    _today_band(fig, today)
    _apply_common_layout(
        fig,
        title=None,
        y_title="mcm/day",
        height=220,
        show_legend=False,
    )
    fig.update_yaxes(range=[y_lo, y_hi])
    return fig


# ---------------------------------------------------------------------------
# Flow details chart
# ---------------------------------------------------------------------------

def plot_flow_details_chart(
    flow_df: pd.DataFrame,
    today: pd.Timestamp,
    point_labels: dict,
) -> go.Figure:
    """Per-point flow lines in mcm/day."""
    fig = go.Figure()
    palette = {
        "kiskundorozsma_hu": COLOR_HU_OTHERS,
        "kireevo": COLOR_BG_IMPORT,
        "kiskundorozsma_2": COLOR_HU_MET,
        "kalotina": COLOR_KALOTINA,
    }
    for point in flow_df.columns:
        if point == "date":
            continue
        color = palette.get(point, None)
        sub = flow_df[["date", point]].rename(columns={point: "value"})
        hist = sub[sub["date"] <= today]
        fcst = sub[sub["date"] > today]

        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["value"],
                mode="lines+markers",
                name=point_labels.get(point, point),
                line=dict(width=2, color=color),
                marker=dict(size=4),
                legendgroup=point,
            )
        )
        if not fcst.empty:
            bridge_x = ([hist["date"].iloc[-1]] if not hist.empty else []) + list(fcst["date"])
            bridge_y = ([hist["value"].iloc[-1]] if not hist.empty else []) + list(fcst["value"])
            fig.add_trace(
                go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines+markers",
                    name=f"{point_labels.get(point, point)} (fcst)",
                    line=dict(width=2, color=color, dash="dash"),
                    marker=dict(size=4, symbol="circle-open"),
                    legendgroup=point,
                    showlegend=False,
                )
            )

    _today_band(fig, today)
    _apply_common_layout(
        fig,
        title="Physical flows by point",
        y_title="mcm/day",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Capacity charts
# ---------------------------------------------------------------------------

def plot_capacity_booked_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Grouped bar of booked capacity by border point × period."""
    fig = go.Figure()
    for product in sorted(cap_df["product"].dropna().unique()):
        sub = cap_df[cap_df["product"] == product]
        fig.add_trace(
            go.Bar(
                x=sub["border_point"] + " (" + sub["period"].astype(str) + ")",
                y=sub["booked_mwh"],
                name=str(product).capitalize(),
            )
        )
    fig.update_layout(barmode="group")
    _apply_common_layout(
        fig,
        title="Booked capacity (MWh/day)",
        y_title="MWh/day",
        height=340,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_capacity_utilisation_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Grouped bar of utilisation % with 100% reference line."""
    fig = go.Figure()
    for product in sorted(cap_df["product"].dropna().unique()):
        sub = cap_df[cap_df["product"] == product]
        fig.add_trace(
            go.Bar(
                x=sub["border_point"] + " (" + sub["period"].astype(str) + ")",
                y=sub["utilisation_pct"],
                name=str(product).capitalize(),
            )
        )
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color=COLOR_DEMAND,
        annotation_text="100 %",
        annotation_position="top right",
    )
    fig.update_layout(barmode="group")
    _apply_common_layout(
        fig,
        title="Capacity utilisation (%)",
        y_title="%",
        height=340,
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_capacity_price_chart(cap_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Two separate price charts — one per currency — because HUF and EUR
    magnitudes differ by ~100× and can't share a y-axis cleanly.
    Returns (huf_fig, eur_fig). Either may be None if no data.
    """
    figs = {}
    for ccy in ["HUF", "EUR"]:
        sub = cap_df[cap_df["currency"] == ccy]
        if sub.empty:
            figs[ccy] = None
            continue
        fig = go.Figure()
        for product in sorted(sub["product"].dropna().unique()):
            ssub = sub[sub["product"] == product]
            fig.add_trace(
                go.Bar(
                    x=ssub["border_point"] + " (" + ssub["period"].astype(str) + ")",
                    y=ssub["price"],
                    name=str(product).capitalize(),
                )
            )
        fig.update_layout(barmode="group")
        _apply_common_layout(
            fig,
            title=f"Price comparison ({ccy}/kWh/h/day)",
            y_title=f"{ccy}/kWh/h/day",
            height=320,
        )
        fig.update_xaxes(tickangle=-30)
        figs[ccy] = fig
    return figs.get("HUF"), figs.get("EUR")


# New ENTSOG capacity booking charts. These definitions intentionally override
# the legacy compact charts above while keeping older imports stable.
def _capacity_hover_fields(cap_df: pd.DataFrame) -> np.ndarray:
    fields = [
        cap_df.get("border_point_full", pd.Series("", index=cap_df.index)),
        cap_df.get("tso", pd.Series("", index=cap_df.index)),
        cap_df.get("direction", pd.Series("", index=cap_df.index)),
        cap_df.get("offered_mwh", pd.Series(np.nan, index=cap_df.index)),
        cap_df.get("booked_mwh", pd.Series(np.nan, index=cap_df.index)),
        cap_df.get("utilisation_pct", pd.Series(np.nan, index=cap_df.index)),
        cap_df.get("price_original", pd.Series("", index=cap_df.index)),
        cap_df.get("price_unit_detected", pd.Series("", index=cap_df.index)),
        cap_df.get("price_eur_per_mwh", pd.Series(np.nan, index=cap_df.index)),
        cap_df.get("price_conversion_note", pd.Series("", index=cap_df.index)),
    ]
    return np.stack([s.to_numpy() for s in fields], axis=-1)


def plot_capacity_booked_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Booked capacity by delivery period, color-coded by short border point."""
    fig = go.Figure()
    if cap_df.empty:
        _apply_common_layout(fig, "Booked capacity by delivery period", "MWh/day", 360)
        return fig
    for bp in sorted(cap_df["border_point_short"].dropna().unique()):
        sub = cap_df[cap_df["border_point_short"] == bp].sort_values("delivery_sort")
        fig.add_trace(
            go.Scatter(
                x=sub["delivery_period"],
                y=sub["booked_mwh"],
                customdata=_capacity_hover_fields(sub),
                name=str(bp),
                mode="lines+markers",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Period: %{x}<br>"
                    "TSO: %{customdata[1]}<br>"
                    "Type: %{customdata[2]}<br>"
                    "Offered: %{customdata[3]:,.0f} MWh/day<br>"
                    "Booked: %{customdata[4]:,.0f} MWh/day<br>"
                    "Booked %%: %{customdata[5]:.1f}<br>"
                    "Original price: %{customdata[6]} %{customdata[7]}<br>"
                    "EUR/MWh: %{customdata[8]:.4f}<br>"
                    "%{customdata[9]}<extra></extra>"
                ),
            )
        )
    _apply_common_layout(fig, "Booked capacity by delivery period", "MWh/day", 360)
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_offered_vs_booked_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Grouped offered vs booked capacity by border point."""
    fig = go.Figure()
    if cap_df.empty:
        _apply_common_layout(fig, "Offered vs booked capacity", "MWh/day", 340)
        return fig
    grouped = (
        cap_df.groupby("border_point_short", as_index=False)
        .agg(offered_mwh=("offered_mwh", "sum"), booked_mwh=("booked_mwh", "sum"))
    )
    grouped["booked_pct"] = np.where(
        grouped["offered_mwh"] > 0,
        grouped["booked_mwh"] / grouped["offered_mwh"] * 100.0,
        np.nan,
    )
    for col, label in [("offered_mwh", "Offered"), ("booked_mwh", "Booked")]:
        fig.add_trace(
            go.Bar(
                x=grouped["border_point_short"],
                y=grouped[col],
                name=label,
                customdata=np.stack([grouped["booked_pct"].to_numpy()], axis=-1),
                hovertemplate="%{x}<br>" + label + ": %{y:,.0f} MWh/day<br>Booked: %{customdata[0]:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    _apply_common_layout(fig, "Offered vs booked capacity", "MWh/day", 340)
    fig.update_xaxes(tickangle=-25)
    return fig


def plot_capacity_utilisation_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Booking utilization heatmap by border point and delivery period."""
    fig = go.Figure()
    if cap_df.empty:
        _apply_common_layout(fig, "Booking utilization heatmap", "%", 340, show_legend=False)
        return fig
    pivot = cap_df.pivot_table(
        index="border_point_short",
        columns="delivery_period",
        values="utilisation_pct",
        aggfunc="mean",
    )
    ordered_cols = (
        cap_df[["delivery_period", "delivery_sort"]]
        .drop_duplicates()
        .sort_values("delivery_sort")["delivery_period"]
        .tolist()
    )
    pivot = pivot.reindex(columns=[c for c in ordered_cols if c in pivot.columns])
    fig.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar=dict(title="%"),
            hovertemplate="Border point: %{y}<br>Period: %{x}<br>Booked: %{z:.1f}%<extra></extra>",
        )
    )
    _apply_common_layout(fig, "Booking utilization heatmap", "%", 340, show_legend=False)
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_capacity_price_chart(cap_df: pd.DataFrame) -> go.Figure:
    """Converted EUR/MWh price comparison by delivery period."""
    fig = go.Figure()
    sub = cap_df[cap_df["price_eur_per_mwh"].notna()].copy()
    if sub.empty:
        _apply_common_layout(fig, "Converted price comparison", "EUR/MWh", 320)
        return fig
    for bp in sorted(sub["border_point_short"].dropna().unique()):
        ssub = sub[sub["border_point_short"] == bp].sort_values("delivery_sort")
        fig.add_trace(
            go.Scatter(
                x=ssub["delivery_period"],
                y=ssub["price_eur_per_mwh"],
                customdata=_capacity_hover_fields(ssub),
                name=str(bp),
                mode="lines+markers",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Period: %{x}<br>"
                    "Original price: %{customdata[6]} %{customdata[7]}<br>"
                    "EUR/MWh: %{y:.4f}<br>"
                    "%{customdata[9]}<extra></extra>"
                ),
            )
        )
    _apply_common_layout(fig, "Converted price comparison", "EUR/MWh", 320)
    fig.update_xaxes(tickangle=-30)
    return fig
