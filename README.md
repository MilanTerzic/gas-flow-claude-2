# Serbia Gas Balance & Capacity Dashboard

A Streamlit dashboard for Serbian natural gas flows, supply structure, demand
forecast and cross-border capacity bookings — designed as a compact, white,
operational dashboard (Excel / Power BI feel).

The centrepiece of Tab 1 is three vertically-stacked, date-aligned charts that
share one daily x-axis and one narrow red "today" band.

---

## What's inside

```
serbia_dashboard/
├── app.py                      # Streamlit entry point — UI only
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── config.py               # constants, regression coefficients, palette
    ├── demand.py               # demand forecast + balance calculation
    ├── temperature.py          # Open-Meteo / weather.com / upload fallbacks
    ├── flows.py                # per-point flow loading & unit conversion
    ├── entsog.py               # ENTSOG Transparency Platform fetcher
    ├── capacity.py             # capacity-booking parsing & table formatting
    ├── model.py                # read regression coefficients from XLSX
    ├── dummy.py                # realistic dummy data for offline preview
    └── charts.py               # pure Plotly plotting functions
```

Calculation logic lives in `demand.py`; plotting lives in `charts.py`. The two
do not mix.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens with **"Use dummy demonstration data" = ON** so you can preview the
layout without network access. Switch off to fetch live data.

---

## Visual style

* White background (`template="plotly_white"`, `plot_bgcolor="white"`,
  `paper_bgcolor="white"`).
* Light-grey gridlines (`rgba(220,220,220,0.7)`).
* Compact heights: main 360 px, temperature 200 px, storage 220 px.
* Uniform margins `dict(l=50, r=20, t=40, b=35)` on every chart.
* Horizontal legend above each plot.
* Narrow red "today" `vrect` spanning **exactly one day**, identical on all
  three charts.

---

## Tab 1 — Gas Balance

### KPIs (above the charts)

| KPI | Source |
|---|---|
| Forecasted Serbian demand | `demand_mcm` |
| Total available supply | `serbian_available_supply_mcm` |
| Storage +/- | `storage_imbalance_mcm` (Injection / Withdrawal label) |
| Belgrade temperature | `temperature_c` (+ 2-day avg) |
| Import from HU | `import_hu_others_mcm + import_hu_met_mcm` |
| Import from BG (net) | `import_bg_mcm` |
| Import Kalotina | `import_kalotina_mcm` |
| Production | `production_mcm` |
| Bosnia consumption/export | `bosnia_consumption_mcm` |

### Three aligned charts

| Row | Chart | Notes |
|----|-------|-------|
| 1 | Daily composition of Serbia demand | `barmode="stack"`. Solid bars = historical, hatched (`marker_pattern_shape="/"`) = forecast. Required-demand line: solid red historical, dashed red forecast. |
| 2 | Belgrade temperature (°C) | Solid blue actual, dashed blue forecast. Legend: *Temp (actual)*, *Temp (fcst)*. |
| 3 | Storage +/- | Green bars above zero (injection), red bars below zero (withdrawal). Forecast bars hatched. Strong horizontal zero line in red (`fig.add_hline(y=0, line_color="red", line_width=2)`). Y-axis auto-scaled to keep zero clearly visible. |

### Colour palette

| Series | Hex |
|---|---|
| Import Kalotina | `#1B7F3A` (dark green) |
| Import from BG | `#7FB6E2` (light blue) |
| Production | `#34526F` (dark blue / grey-blue) |
| Import HU (others) | `#2E75B6` (medium blue) |
| Import HU (MET) | `#ED7D31` (orange) |
| Required demand | `#C00000` (red) |
| Temperature | `#1F77B4` (blue) |
| Storage + | `#2E7D32` (green) |
| Storage − | `#C00000` (red) |

---

## Tab 2 — Flow Details
Per-point allocations in mcm/day for Kiskundorozsma HU, Kireevo,
Kiskundorozsma 2, Kalotina. Solid lines = historical, dashed = forecast.

## Tab 3 — Capacity Bookings

Excel-style grouped table covering five TSO / border-point combinations:

| # | TSO | Border point | Direction | Price unit |
|---|-----|--------------|-----------|------------|
| 1 | FGSZ | Kiskundorozsma (HU)/Kiskundorozsma (RS) | exit | HUF/kWh/h/day |
| 2 | Bulgartransgaz | Kireevo (BG)/Zaychar (RS) | exit | EUR/kWh/h/day |
| 3 | Gastrans | Kireevo (BG)/Zaychar (RS) | entry | EUR/kWh/h/day |
| 4 | Gastrans | Kiskundorozsma 2 | exit | EUR/kWh/h/day |
| 5 | FGSZ | Kiskundorozsma 2 | entry | HUF/kWh/h/day |

Periods: **Day, D-1, D-2, D-3, D-4** × **daily / monthly / quarterly**.
Missing values show `-`; uncalculable utilisations show `N/A`.

Charts: booked capacity (MWh/day), utilisation % (with 100 % reference line),
price comparison — **HUF and EUR shown in separate panels** because the
magnitudes differ by ~100×.

## Tab 4 — Model & Assumptions
Regression coefficients, all assumptions, full daily temperature/demand table.

---

## Demand model

**Polynomial (active):**
```
y = 0.0007 · x³ − 0.0188 · x² − 0.3194 · x + 11.987
```
**Linear fallback:**
```
y = −0.354 · x + 11.396
```
where *y* = Serbian daily demand (**mcm/day**) and *x* = Belgrade 2-day
rolling-average temperature (**°C**).

Upload a different workbook (same sheet name / cell layout) via the sidebar
to override the coefficients without code changes.

---

## Balance equations

```
Domestic Serbian production       = 0.5 mcm/day (configurable)
Import from BG (net)              = Kireevo − Kiskundorozsma-2
Bosnia consumption / export       = 0.08 × Import from BG  (configurable %)
Serbian available supply          = Kiskundorozsma HU entry
                                  + Import from BG
                                  + Kalotina entry
                                  + Domestic production
                                  − Bosnia consumption/export
Required Serbian demand           = poly( 2-day avg Belgrade temp °C )
Storage / imbalance               = Serbian available supply
                                  − Required Serbian demand
```

`storage_imbalance_mcm > 0` → **injection / surplus** (green)
`storage_imbalance_mcm < 0` → **withdrawal / deficit** (red)

---

## DataFrame columns

The `demand.build_balance()` output frame uses these explicit `*_mcm`
columns (matching the dashboard contract):

```
date
temperature_c                # combined actual+forecast
temperature_actual_c         # NaN on forecast rows
temperature_forecast_c       # NaN on historical rows
avg_temperature_c            # 2-day rolling avg
demand_mcm                   # combined required demand
required_actual_mcm          # NaN on forecast rows
required_forecast_mcm        # NaN on historical rows
import_kalotina_mcm
import_bg_mcm
production_mcm
import_hu_others_mcm
import_hu_met_mcm
bosnia_consumption_mcm
serbian_available_supply_mcm
storage_imbalance_mcm
is_forecast                  # bool
```

---

## Data sources

| Source | Default | Fallback |
|---|---|---|
| **Flows** | ENTSOG `operationaldata` (Allocation → Physical Flow) | CSV/XLSX upload — wide *or* long format |
| **Temperature** | Open-Meteo (archive + forecast, no key) | weather.com scrape · CSV/XLSX upload |
| **Capacity** | CSV/XLSX upload | Built-in synthetic dummy set |
| **Coefficients** | Built-in (read from v8 Kalotina workbook) | Upload any XLSX with the same sheet/cell layout |

### Flow upload schemas

**Wide:**
```
date, kiskundorozsma_hu, kireevo, kiskundorozsma_2, kalotina[, kiskundorozsma_hu_met]
```
Values may be in mcm/day or MWh/day — auto-detected and converted.

**Long:**
```
date, point, mwh_per_day      # or mcm_per_day
```
Point labels are normalised: "Kireevo / Zaychar", "kiskundorozsma 2",
"Kalotina (BG→RS)" all map to the right canonical key.

### Capacity upload schema
```
tso, border_point, direction, product, period,
offered_mwh, booked_mwh, utilisation_pct, price, currency, pct_of_100
```
Missing `utilisation_pct` is recomputed from `booked / offered`.

---

## Deploying on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On <https://share.streamlit.io>, create a new app, point at `app.py`.
3. (Optional) Add `ENTSOG_TOKEN` under *Secrets*.
4. App auto-installs from `requirements.txt`.

No further configuration required — the dashboard defaults to dummy data
until live sources are enabled via the sidebar.

---

## Error handling

The dashboard fails soft on every external dependency:

* ENTSOG failure → warning banner, falls back to dummy.
* Open-Meteo failure → warning banner, falls back to weather.com → dummy.
* weather.com layout change → raises, dashboard offers upload box.
* Workbook upload missing the expected sheet → uses built-in coefficients.
* Missing capacity values → table shows `-`; missing utilisations show `N/A`.

The "today" band is computed from the local server date, so it remains correct
as the date rolls forward.
