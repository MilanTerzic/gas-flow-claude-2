"""
Central configuration and constants for the Serbia Gas Balance dashboard.

All values here originate from the workbook
`Serbian_natural_gas_balance_v8__Kalotina.xlsx`
(sheet: "Serbian Gas Cons. forecast").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Energy conversion: ENTSOG returns kWh/day or MWh/day; the model uses mcm/day.
# 1 mcm of natural gas ≈ 10.55 GWh (Higher Heating Value as used by Srbijagas).
# ---------------------------------------------------------------------------
CONVERSION_MCM_TO_GWH = 10.55          # 1 mcm = 10.55 GWh
CONVERSION_MCM_TO_MWH = 10550.0        # 1 mcm = 10,550 MWh
CONVERSION_MCM_TO_KWH = 10_550_000.0   # 1 mcm = 10,550,000 kWh


def mwh_per_day_to_mcm_per_day(mwh_per_day: float) -> float:
    """Convert MWh/day to mcm/day."""
    return mwh_per_day / CONVERSION_MCM_TO_MWH


def kwh_per_day_to_mcm_per_day(kwh_per_day: float) -> float:
    """Convert kWh/day to mcm/day."""
    return kwh_per_day / CONVERSION_MCM_TO_KWH


# ---------------------------------------------------------------------------
# Demand regression — read directly from the workbook.
#   Polynomial: y = 0.0007*x^3 - 0.0188*x^2 - 0.3194*x + 11.987
#   Linear:     y = -0.354*x + 11.396
# y = mcm/day,   x = Belgrade avg daily temperature in °C.
#
# Stored in NumPy convention (highest power first), matching numpy.polyval.
# ---------------------------------------------------------------------------
POLY_COEFFS = (0.0007, -0.0188, -0.3194, 11.987)
LINEAR_COEFFS = (-0.354, 11.396)


# ---------------------------------------------------------------------------
# Operational assumptions
# ---------------------------------------------------------------------------
DOMESTIC_PRODUCTION_MCM = 0.5     # mcm/day, constant
BIH_SHARE = 0.08                  # 8% of Import from BG goes to Bosnia
CURVE_SHIFT_DEFAULT = 0.0         # additive shift, mcm/day
CURVE_DISTORTION_DEFAULT = 1.0    # multiplier; 1.0 = no distortion


# ---------------------------------------------------------------------------
# Cross-border points (used throughout the app).
# ---------------------------------------------------------------------------
POINTS = {
    "kiskundorozsma_hu": "Kiskundorozsma HU (HU→RS)",
    "kireevo":           "Kireevo / Zaychar (BG→RS)",
    "kiskundorozsma_2":  "Kiskundorozsma 2 / Horgos transit",
    "kalotina":          "Kalotina (BG→RS)",
}

ENTSOG_POINT_DIRECTIONS = {
    "kiskundorozsma_hu": "hu-tso-0001itp-00055exit",
    "kireevo":           "bg-tso-0001itp-00529exit",
    "kiskundorozsma_2":  "hu-tso-0001itp-10013entry",
    "kalotina":          "bg-tso-0001itp-00134exit",
}


# ---------------------------------------------------------------------------
# Visual palette — keep consistent across all charts.
# Tuned to match a clean Excel/Power-BI operational look.
# ---------------------------------------------------------------------------
COLOR_KALOTINA       = "#1B7F3A"   # dark green
COLOR_BG_IMPORT      = "#7FB6E2"   # light blue
COLOR_PRODUCTION     = "#34526F"   # dark blue / grey-blue
COLOR_HU_OTHERS      = "#2E75B6"   # medium blue
COLOR_HU_MET         = "#ED7D31"   # orange
COLOR_DEMAND         = "#C00000"   # red
COLOR_TEMP           = "#1F77B4"   # blue
COLOR_STORAGE_POS    = "#2E7D32"   # green
COLOR_STORAGE_NEG    = "#C00000"   # red
COLOR_TODAY          = "#C00000"   # red band

GRID_COLOR           = "rgba(220,220,220,0.7)"


# ---------------------------------------------------------------------------
# Capacity booking schema — TSO / border point / direction / pricing
# ---------------------------------------------------------------------------
CAPACITY_DEFS = [
    {
        "tso": "FGSZ",
        "border_point": "Kiskundorozsma (HU)/Kiskundorozsma (RS)",
        "direction": "exit",
        "price_unit": "HUF/kWh/h/day",
        "currency": "HUF",
    },
    {
        "tso": "Bulgartransgaz",
        "border_point": "Kireevo (BG)/Zaychar (RS)",
        "direction": "exit",
        "price_unit": "EUR/kWh/h/day",
        "currency": "EUR",
    },
    {
        "tso": "Gastrans",
        "border_point": "Kireevo (BG)/Zaychar (RS)",
        "direction": "entry",
        "price_unit": "EUR/kWh/h/day",
        "currency": "EUR",
    },
    {
        "tso": "Gastrans",
        "border_point": "Kiskundorozsma 2",
        "direction": "exit",
        "price_unit": "EUR/kWh/h/day",
        "currency": "EUR",
    },
    {
        "tso": "FGSZ",
        "border_point": "Kiskundorozsma 2",
        "direction": "entry",
        "price_unit": "HUF/kWh/h/day",
        "currency": "HUF",
    },
]
