"""Capacity booking normalization, price conversion, and quality checks."""

from __future__ import annotations

import calendar
import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

ENTSOG_BASE_URL = "https://transparency.entsog.eu/api/v1"
DEFAULT_CAPACITY_INDICATORS = [
    "Firm Technical",
    "Firm Offered",
    "Firm Booked",
    "Firm Available",
]

TARGET_BORDER_POINTS = [
    {
        "canonical_key": "kiskundorozsma_2_horgos",
        "label": "Kiskundorozsma-2 (HU) / Horgos (RS)",
        "country_from": "HU",
        "country_to": "RS",
        "direction_hint": "entry",
        "patterns": ["kiskundorozsma", "horg", "2"],
    },
    {
        "canonical_key": "kiskundorozsma_hu_rs",
        "label": "Kiskundorozsma (HU > RS)",
        "country_from": "HU",
        "country_to": "RS",
        "direction_hint": "exit",
        "patterns": ["kiskundorozsma"],
    },
    {
        "canonical_key": "kalotina_dimitrovgrad",
        "label": "Kalotina (BG) / Dimitrovgrad (RS)",
        "country_from": "BG",
        "country_to": "RS",
        "direction_hint": "exit",
        "patterns": ["kalotina", "dimitrovgrad"],
    },
    {
        "canonical_key": "kireevo_zajecar",
        "label": "Kireevo/Kirevo (BG) / Zajecar (RS)",
        "country_from": "BG",
        "country_to": "RS",
        "direction_hint": "exit",
        "patterns": ["kiree", "kirevo", "zajec", "zaychar"],
    },
]


BORDER_POINT_SHORT_NAMES = {
    "Kireevo (BG)/Zaychar (RS)": "BG->RS Kireevo",
    "Kireevo / Zaychar": "BG->RS Kireevo",
    "Kiskundorozsma 2": "RS->HU Kisk. 2",
    "Kiskundorozsma (HU)/Kiskundorozsma (RS)": "HU->RS Kisk.",
    "Kiskundorozsma": "HU->RS Kisk.",
    "Kalotina": "BG->RS Kalotina",
    "Horgos": "HU->RS Horgos",
    "Zvornik": "BA/RS Zvornik",
    "Mokrin": "RS storage / Mokrin",
}

PRODUCT_ORDER = ["Within-day", "Day-ahead", "Daily", "Monthly", "Quarterly", "Yearly", "Unknown"]
FX_CURRENCIES = ["EUR", "HUF", "RSD", "BGN", "RON", "USD", "CHF", "GBP"]

COLUMN_ALIASES = {
    "tso": ["tso", "operator", "transmission system operator"],
    "border_point": ["border point", "border_point", "point", "interconnection point"],
    "direction": ["type", "direction", "entry/exit"],
    "product": ["product", "product type", "runtime period"],
    "period": ["period", "delivery period", "gas day", "date"],
    "offered_mwh": [
        "offered (mwh/day)",
        "offered (mwh/d)",
        "offered (kwh/h)",
        "offered (kwh/day)",
        "offered",
        "offered_mwh",
        "offered capacity",
        "offered_capacity_mwh_day",
    ],
    "booked_mwh": [
        "booked (mwh/day)",
        "booked (mwh/d)",
        "booked (kwh/h)",
        "booked (kwh/day)",
        "booked",
        "booked_mwh",
        "booked capacity",
        "booked_capacity_mwh_day",
    ],
    "utilisation_pct": ["booked %", "booked%", "utilisation_pct", "utilization_pct", "utilisation"],
    "price": ["price", "reserve price", "tariff"],
    "currency": ["currency", "ccy"],
    "pct_of_100": ["% of 100", "pct_of_100"],
    "price_unit": ["price unit", "unit", "price_unit"],
    "source_timestamp": ["source timestamp", "retrieval date", "retrieved at", "created at", "updated at", "timestamp"],
}


def _clean_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _first_present(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    lookup = {_clean_key(c): c for c in df.columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def empty_capacity_frame() -> pd.DataFrame:
    columns = [
        "tso",
        "border_point",
        "border_point_full",
        "border_point_short",
        "direction",
        "type",
        "product",
        "product_type",
        "product_level",
        "period",
        "delivery_period",
        "delivery_start",
        "delivery_end",
        "delivery_sort",
        "period_start",
        "period_end",
        "period_days",
        "offered_mwh",
        "booked_mwh",
        "utilisation_pct",
        "offered_capacity_mwh_day",
        "booked_capacity_mwh_day",
        "booked_percentage",
        "price_original",
        "price_currency",
        "price_unit_detected",
        "price_eur_per_mwh",
        "booked_energy_mwh_for_period",
        "data_quality_warning",
        "has_quality_warning",
        "source_retrieval_date",
    ]
    return pd.DataFrame(columns=columns)


def _to_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("\u00a0", " ")
    if not text:
        return np.nan
    text = text.replace("%", "")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _capacity_to_mwh_day(value: Any, source_col: Optional[str]) -> float:
    numeric = _to_number(value)
    if pd.isna(numeric):
        return np.nan
    unit = _clean_key(source_col or "")
    unit = unit.replace(" ", "")
    if "kwh/h" in unit or "kwhperh" in unit:
        return numeric * 0.024
    if "kwh/day" in unit or "kwh/d" in unit:
        return numeric / 1000.0
    if "gwh/day" in unit or "gwh/d" in unit:
        return numeric * 1000.0
    return numeric


def _parse_date(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    dayfirst = not bool(re.match(r"^\d{4}[./-]", text))
    ts = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def _days_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return max(1, int((end - start).days) + 1)


def fetch_latest_fx_rates(timeout: int = 10) -> dict[str, Any]:
    """Fetch latest available FX rates as original-currency units converted to EUR.

    Frankfurter is backed by ECB reference rates for ECB currencies. BGN is fixed
    by currency board at 1 EUR = 1.95583 BGN, so it is included explicitly even
    when the API is unavailable.
    """
    rates = {
        "EUR": {
            "fx_rate_to_eur": 1.0,
            "fx_rate_source": "EUR base",
            "fx_source": "EUR base",
            "fx_rate_date": pd.Timestamp.today().date().isoformat(),
            "note": "EUR price; no FX conversion needed.",
        },
        "BGN": {
            "fx_rate_to_eur": 1.0 / 1.95583,
            "fx_rate_source": "Bulgarian lev fixed parity",
            "fx_source": "Bulgarian lev fixed parity",
            "fx_rate_date": pd.Timestamp.today().date().isoformat(),
            "note": "BGN converted using fixed parity: 1 EUR = 1.95583 BGN.",
        },
    }
    errors: list[str] = []

    try:
        response = requests.get(
            "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
            timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        cubes = root.findall(".//{*}Cube/{*}Cube/{*}Cube")
        rate_date = ""
        dated_cube = root.find(".//{*}Cube/{*}Cube[@time]")
        if dated_cube is not None:
            rate_date = dated_cube.attrib.get("time", "")
        for cube in cubes:
            currency = cube.attrib.get("currency", "").upper()
            eur_to_currency = cube.attrib.get("rate")
            if currency in FX_CURRENCIES and eur_to_currency:
                rates[currency] = {
                    "fx_rate_to_eur": 1.0 / float(eur_to_currency),
                    "fx_rate_source": "ECB euro foreign exchange reference rates",
                    "fx_source": "ECB euro foreign exchange reference rates",
                    "fx_rate_date": rate_date or pd.Timestamp.today().date().isoformat(),
                    "note": f"{currency} converted using ECB latest available euro reference rate.",
                }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ECB FX fetch failed: {exc}")

    missing = [c for c in FX_CURRENCIES if c not in rates]
    if missing:
        try:
            symbols = ",".join(c for c in missing if c not in {"EUR", "BGN"})
            if symbols:
                response = requests.get(
                    "https://api.frankfurter.app/latest",
                    params={"from": "EUR", "to": symbols},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                rate_date = payload.get("date") or pd.Timestamp.today().date().isoformat()
                for currency, eur_to_currency in payload.get("rates", {}).items():
                    if eur_to_currency:
                        rates[currency.upper()] = {
                            "fx_rate_to_eur": 1.0 / float(eur_to_currency),
                            "fx_rate_source": "Frankfurter / ECB reference rates",
                            "fx_source": "Frankfurter / ECB reference rates",
                            "fx_rate_date": rate_date,
                            "note": f"{currency.upper()} converted using Frankfurter latest available reference rate.",
                        }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Frankfurter FX fetch failed: {exc}")

    missing = [c for c in FX_CURRENCIES if c not in rates]
    if "RSD" in missing:
        # NBS publishes official middle RSD exchange rates. The public web page
        # exposes EUR/RSD as RSD per 1 EUR; invert it so each RSD amount can be
        # converted into EUR. If parsing fails, the app leaves RSD conversion
        # empty and flags the affected rows.
        try:
            response = requests.get(
                "https://webappcenter.nbs.rs/ExchangeRateWebApp/CultureInfo/OpenPage",
                params={"culture": "en-Us", "pageUrl": "/ExchangeRateWebApp/ExchangeRate/CurrentMiddleRate"},
                timeout=timeout,
            )
            response.raise_for_status()
            match = re.search(r"\bEUR\b(?:.|\n){0,300}?(\d{2,3}[.,]\d{2,6})", response.text)
            if match:
                eur_rsd = float(match.group(1).replace(",", "."))
                rates["RSD"] = {
                    "fx_rate_to_eur": 1.0 / eur_rsd,
                    "fx_rate_source": "National Bank of Serbia official middle exchange rate",
                    "fx_source": "National Bank of Serbia official middle exchange rate",
                    "fx_rate_date": pd.Timestamp.today().date().isoformat(),
                    "note": "RSD converted using NBS official middle EUR/RSD rate.",
                }
            else:
                errors.append("NBS RSD FX fetch failed: EUR/RSD rate could not be parsed.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"NBS RSD FX fetch failed: {exc}")

    if all(currency in rates for currency in FX_CURRENCIES):
        errors = []
    return {"rates": rates, "errors": errors}


def parse_product_period(product: Any, period: Any, reference_date: Optional[date] = None) -> dict[str, Any]:
    """Interpret ENTSOG product/period text into product type and delivery window."""
    product_text = "" if pd.isna(product) else str(product).strip()
    period_text = "" if pd.isna(period) else str(period).strip()
    combined = f"{product_text} {period_text}".strip()
    lower = combined.lower()
    ref = pd.Timestamp(reference_date or date.today()).normalize()

    product_lower = product_text.lower()
    product_type = "Unknown"
    if any(token in product_lower for token in ["within", "intraday", "within-day"]):
        product_type = "Within-day"
    elif any(token in product_lower for token in ["day-ahead", "day ahead"]):
        product_type = "Day-ahead"
    elif any(token in product_lower for token in ["daily", "day", "gas day"]):
        product_type = "Daily"
    elif any(token in product_lower for token in ["monthly", "month"]):
        product_type = "Monthly"
    elif any(token in product_lower for token in ["quarterly", "quarter"]):
        product_type = "Quarterly"
    elif any(token in product_lower for token in ["yearly", "annual", "year", "gas year"]):
        product_type = "Yearly"
    elif any(token in lower for token in ["within", "intraday", "within-day"]):
        product_type = "Within-day"
    elif any(token in lower for token in ["day-ahead", "day ahead", "d-1"]):
        product_type = "Day-ahead"
    elif any(token in lower for token in ["monthly", "month"]):
        product_type = "Monthly"
    elif any(token in lower for token in ["quarterly", "quarter", " q"]):
        product_type = "Quarterly"
    elif any(token in lower for token in ["yearly", "annual", "year", "gas year"]):
        product_type = "Yearly"
    elif any(token in lower for token in ["daily", "day", "gas day", "d+"]):
        product_type = "Daily"

    start = pd.NaT
    end = pd.NaT
    label = period_text or product_text or "Unknown"
    note = ""

    date_match = re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}", combined)
    if date_match:
        start = _parse_date(date_match.group(0))
        end = start

    rel_match = re.fullmatch(r"d(?:ay)?\s*([+-]\s*\d+)?", period_text.lower().replace(" ", ""))
    if pd.isna(start) and rel_match and product_type in {"Daily", "Day-ahead", "Within-day"}:
        offset = int(rel_match.group(1).replace(" ", "")) if rel_match.group(1) else 0
        start = ref + pd.Timedelta(days=offset)
        end = start

    month_match = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s./-]+(20\d{2})\b",
        lower,
    )
    ym_match = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", combined)
    if pd.isna(start) and (month_match or ym_match):
        if month_match:
            month_name, year_text = month_match.groups()
            month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(month_name[:3]) + 1
            year = int(year_text)
        else:
            year = int(ym_match.group(1))
            month = int(ym_match.group(2))
        start = pd.Timestamp(year=year, month=month, day=1)
        end = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
        if product_type == "Unknown":
            product_type = "Monthly"

    q_match = re.search(r"\bq([1-4])[\s./-]*(20\d{2})\b|\b(20\d{2})[\s./-]*q([1-4])\b", lower)
    if q_match:
        q = int(q_match.group(1) or q_match.group(4))
        year = int(q_match.group(2) or q_match.group(3))
        month = (q - 1) * 3 + 1
        start = pd.Timestamp(year=year, month=month, day=1)
        end_month = month + 2
        end = pd.Timestamp(year=year, month=end_month, day=calendar.monthrange(year, end_month)[1])
        product_type = "Quarterly"

    gy_match = re.search(r"(?:gy|gas year)?\s*(20\d{2})\s*/\s*(20\d{2})", lower)
    year_match = re.search(r"\b(20\d{2})\b", lower)
    if product_type == "Yearly" and pd.isna(start):
        if gy_match:
            year = int(gy_match.group(1))
            start = pd.Timestamp(year=year, month=10, day=1)
            end = pd.Timestamp(year=year + 1, month=9, day=30)
        elif year_match:
            year = int(year_match.group(1))
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)

    if product_type in {"Daily", "Day-ahead", "Within-day"} and not pd.isna(start):
        label = start.strftime("%d %b %Y")
    elif product_type == "Monthly" and not pd.isna(start):
        label = start.strftime("%b %Y")
    elif product_type == "Quarterly" and not pd.isna(start):
        label = f"Q{((start.month - 1) // 3) + 1} {start.year}"
    elif product_type == "Yearly" and not pd.isna(start):
        label = f"GY {start.year}/{start.year + 1}" if start.month == 10 else str(start.year)
    elif product_type == "Unknown" or pd.isna(start):
        note = "Product or delivery period could not be parsed reliably."

    return {
        "product_type": product_type,
        "delivery_start": start,
        "delivery_end": end,
        "delivery_period": label,
        "period_days": _days_between(start, end),
        "period_parse_note": note,
    }


def parse_price_and_currency(price: Any, currency: Any = None, price_unit: Any = None) -> dict[str, Any]:
    """Parse numeric price, currency, and unit from mixed ENTSOG price text."""
    original = "" if pd.isna(price) else str(price).strip()
    unit_text = "" if pd.isna(price_unit) else str(price_unit).strip()
    ccy_text = "" if pd.isna(currency) else str(currency).strip().upper()
    text = " ".join(part for part in [original, ccy_text, unit_text] if part)
    numeric = _to_number(original)

    ccy_match = re.search(r"\b(EUR|HUF|RON|BGN|USD|GBP|CHF|RSD)\b", text.upper())
    parsed_currency = ccy_match.group(1) if ccy_match else ccy_text or None

    unit_source = unit_text or original
    unit_source = re.sub(r"[-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?", "", unit_source).strip()
    unit_source = unit_source or (f"{parsed_currency}/kWh/h/day" if parsed_currency and "/" in text else "")

    return {
        "price_original": original,
        "price_numeric": numeric,
        "price_currency": parsed_currency,
        "price_unit_detected": unit_source,
    }


def convert_price_to_eur_per_mwh(
    price_numeric: float,
    currency: Optional[str],
    unit: Any,
    booked_mwh_per_day: float,
    period_days: float,
    fx_rates: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Convert capacity tariffs to an effective EUR/MWh over the delivery period.

    ENTSOG capacity tariffs such as EUR/kWh/h/day price hourly capacity. A row's
    booked MWh/day is converted to kWh/h, cost is calculated for the tariff
    period, then divided by booked energy over the same delivery period.
    """
    base = {
        "fx_rate_source": "",
        "fx_rate_date": "",
        "fx_rate_to_eur": np.nan,
        "price_converted_eur": np.nan,
        "converted_price_eur": np.nan,
        "price_conversion_status": "failed",
    }
    if pd.isna(price_numeric):
        return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing price."}
    if not currency:
        return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing currency."}
    currency = str(currency).upper()
    fx_lookup = (fx_rates or {}).get("rates", fx_rates or {})
    fx = fx_lookup.get(currency)
    if not fx:
        return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": f"Missing FX rate for {currency}; price not converted."}
    fx_rate = float(fx["fx_rate_to_eur"])
    price_eur = price_numeric * fx_rate
    base.update(
        {
            "fx_source": fx.get("fx_source", ""),
            "fx_rate_source": fx.get("fx_rate_source", fx.get("fx_source", "")),
            "fx_rate_date": fx.get("fx_rate_date", ""),
            "fx_rate_to_eur": fx_rate,
            "price_converted_eur": price_eur,
            "converted_price_eur": price_eur,
        }
    )
    if pd.isna(booked_mwh_per_day) or booked_mwh_per_day <= 0:
        return {**base, "price_conversion_status": "warning", "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Cannot calculate EUR/MWh because booked capacity is zero or missing."}
    if pd.isna(period_days) or period_days <= 0:
        return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing delivery period length."}

    unit_clean = re.sub(r"\s+", "", str(unit).lower())
    booked_energy = booked_mwh_per_day * period_days

    if "mwh" in unit_clean and "kwh/h" not in unit_clean:
        return {**base, "price_conversion_status": "success", "price_eur_per_mwh": price_eur, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": f"Interpreted as {currency}/MWh and converted to EUR/MWh."}

    if "kwh/h" in unit_clean:
        # ENTSOG capacity tariffs price capacity, not energy. A booked
        # capacity of 1 kWh/h delivers 24 kWh/day = 0.024 MWh/day. We calculate
        # total capacity cost for the tariff period and divide it by the booked
        # MWh over the exact delivery period.
        hourly_capacity_kwh = booked_mwh_per_day * 1000.0 / 24.0
        tariff_period_days = period_days
        if any(token in unit_clean for token in ["/day", "/d"]):
            charged_periods = period_days
        elif "/month" in unit_clean:
            charged_periods = max(1.0, period_days / 30.4375)
        elif "/quarter" in unit_clean:
            charged_periods = max(1.0, period_days / 91.3125)
        elif "/year" in unit_clean:
            charged_periods = max(1.0, period_days / 365.0)
        elif "/period" in unit_clean or unit_clean.endswith("/p"):
            charged_periods = 1.0
        else:
            return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": f"Price unit '{unit}' could not be parsed."}
        total_cost = price_eur * hourly_capacity_kwh * charged_periods
        note = f"Converted from {currency}/kWh/h capacity tariff."
        note += f" Used {tariff_period_days:.0f} delivery days and {charged_periods:.2f} tariff period(s)."
        if currency == "BGN":
            note += " BGN uses fixed parity: 1 EUR = 1.95583 BGN."
        return {**base, "price_conversion_status": "success", "price_eur_per_mwh": total_cost / booked_energy, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": note}

    return {**base, "price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": f"Price unit '{unit}' could not be parsed."}


def border_point_short_name(border_point: Any, direction: Any = None) -> str:
    full = "" if pd.isna(border_point) else str(border_point).strip()
    if full in BORDER_POINT_SHORT_NAMES:
        return BORDER_POINT_SHORT_NAMES[full]
    compact = re.sub(r"\s*\([^)]*\)", "", full).strip()
    if compact in BORDER_POINT_SHORT_NAMES:
        return BORDER_POINT_SHORT_NAMES[compact]
    return compact[:24] + "..." if len(compact) > 27 else compact


def plotted_series_validation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exactly what source rows feed the booked-capacity chart."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_tso",
                "entsog_point_name",
                "point_key",
                "direction",
                "delivery_period",
                "offered_capacity_mwh_day",
                "booked_capacity_mwh_day",
                "original_unit",
                "converted_unit",
                "source_timestamp",
            ]
        )
    grouped = (
        df.groupby(
            [
                "tso",
                "border_point_full",
                "border_point_short",
                "direction",
                "delivery_period",
                "delivery_sort",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            offered_capacity_mwh_day=("offered_mwh", "sum"),
            booked_capacity_mwh_day=("booked_mwh", "sum"),
            offered_source_column=("offered_source_column", "first"),
            booked_source_column=("booked_source_column", "first"),
            original_unit=("price_unit_detected", lambda s: ", ".join(sorted({str(v) for v in s if str(v).strip()}))),
            converted_unit=("price_conversion_status", lambda s: "EUR/MWh" if (s == "success").any() else ""),
            source_timestamp=("source_retrieval_date", lambda s: ", ".join(sorted({str(v) for v in s if str(v).strip()}))),
        )
        .sort_values(["delivery_sort", "tso", "border_point_short", "direction"])
    )
    grouped["point_key"] = grouped["border_point_short"] + " | " + grouped["tso"] + " | " + grouped["direction"]
    return grouped.rename(
        columns={
            "tso": "source_tso",
            "border_point_full": "entsog_point_name",
        }
    )[
        [
            "source_tso",
            "entsog_point_name",
            "point_key",
            "direction",
            "delivery_period",
            "offered_capacity_mwh_day",
            "booked_capacity_mwh_day",
            "offered_source_column",
            "booked_source_column",
            "original_unit",
            "converted_unit",
            "source_timestamp",
        ]
    ]


def prepare_chart_data(df: pd.DataFrame, fx_rates: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Return normalized capacity booking rows ready for tables and charts."""
    if df is None or df.empty:
        return empty_capacity_frame()
    out = pd.DataFrame()
    source_cols: dict[str, Optional[str]] = {}
    for target, aliases in COLUMN_ALIASES.items():
        src = _first_present(df, aliases)
        source_cols[target] = src
        out[target] = df[src] if src else np.nan

    out["tso"] = out["tso"].fillna("").astype(str).str.strip()
    out["border_point_full"] = out["border_point"].fillna("").astype(str).str.strip()
    out["direction"] = out["direction"].fillna("").astype(str).str.strip().str.lower()
    out["border_point_short"] = [border_point_short_name(bp, d) for bp, d in zip(out["border_point_full"], out["direction"])]
    out["product"] = out["product"].fillna("").astype(str).str.strip()
    out["period"] = out["period"].fillna("").astype(str).str.strip()
    out["source_timestamp"] = out["source_timestamp"].fillna("").astype(str).str.strip()
    out["offered_source_column"] = source_cols.get("offered_mwh") or ""
    out["booked_source_column"] = source_cols.get("booked_mwh") or ""
    out["offered_mwh"] = [
        _capacity_to_mwh_day(value, source_cols.get("offered_mwh"))
        for value in out["offered_mwh"]
    ]
    out["booked_mwh"] = [
        _capacity_to_mwh_day(value, source_cols.get("booked_mwh"))
        for value in out["booked_mwh"]
    ]
    out["utilisation_pct"] = out["utilisation_pct"].map(_to_number)
    out["pct_of_100"] = out["pct_of_100"].map(_to_number)

    period_rows = [parse_product_period(p, per) for p, per in zip(out["product"], out["period"])]
    out = pd.concat([out, pd.DataFrame(period_rows)], axis=1)

    price_rows = [parse_price_and_currency(p, c, u) for p, c, u in zip(out["price"], out["currency"], out["price_unit"])]
    out = pd.concat([out, pd.DataFrame(price_rows)], axis=1)

    conversion_rows = [
        convert_price_to_eur_per_mwh(price, ccy, unit, booked, days, fx_rates=fx_rates)
        for price, ccy, unit, booked, days in zip(
            out["price_numeric"],
            out["price_currency"],
            out["price_unit_detected"],
            out["booked_mwh"],
            out["period_days"],
        )
    ]
    out = pd.concat([out, pd.DataFrame(conversion_rows)], axis=1)

    missing_util = out["utilisation_pct"].isna() & out["booked_mwh"].notna() & out["offered_mwh"].gt(0)
    out.loc[missing_util, "utilisation_pct"] = out.loc[missing_util, "booked_mwh"] / out.loc[missing_util, "offered_mwh"] * 100.0
    out["delivery_sort"] = out["delivery_start"].fillna(pd.Timestamp.max)
    out["type"] = out["direction"]
    out["product_level"] = out["product_type"].replace({"Yearly": "Annual"})
    out["period_start"] = out["delivery_start"]
    out["period_end"] = out["delivery_end"]
    out["offered_capacity_mwh_day"] = out["offered_mwh"]
    out["booked_capacity_mwh_day"] = out["booked_mwh"]
    out["booked_percentage"] = out["utilisation_pct"]
    out["original_price"] = out["price_original"]
    out["original_currency"] = out["price_currency"]
    out["price_original_currency"] = out["price_currency"]
    out["original_price_unit"] = out["price_unit_detected"]
    out["price_original_unit"] = out["price_unit_detected"]
    out["fx_source"] = out["fx_rate_source"]
    out["source_retrieval_date"] = out["source_timestamp"].where(
        out["source_timestamp"].astype(str).str.strip().astype(bool),
        pd.Timestamp.today().date().isoformat(),
    )
    out["data_quality_warning"] = ""
    return out


def run_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    warnings: list[dict[str, Any]] = []
    warning_columns = [
        "warning_type",
        "tso",
        "border_point",
        "border_point_short",
        "delivery_period",
        "product_type",
        "price_original",
        "price_eur_per_mwh",
        "explanation",
    ]

    duplicate_cols = ["tso", "border_point_full", "direction", "product", "period"]
    duplicate_mask = df.duplicated(duplicate_cols, keep=False) if all(c in df.columns for c in duplicate_cols) else pd.Series(False, index=df.index)

    def add(idx: int, warning_type: str, explanation: str) -> None:
        row = df.loc[idx]
        warnings.append(
            {
                "warning_type": warning_type,
                "tso": row.get("tso"),
                "border_point": row.get("border_point_full"),
                "border_point_short": row.get("border_point_short"),
                "delivery_period": row.get("delivery_period"),
                "product_type": row.get("product_type"),
                "price_original": row.get("price_original"),
                "price_eur_per_mwh": row.get("price_eur_per_mwh"),
                "explanation": explanation,
            }
        )

    for idx, row in df.iterrows():
        offered = row.get("offered_mwh")
        booked = row.get("booked_mwh")
        util = row.get("utilisation_pct")
        if not str(row.get("border_point_full", "")).strip():
            add(idx, "missing_border_point", "Border point is missing.")
        if row.get("product_type") == "Unknown" or not str(row.get("period", "")).strip():
            add(idx, "missing_or_unparsed_period", row.get("period_parse_note") or "Product or period is missing.")
        if pd.isna(offered):
            add(idx, "missing_offered_capacity", "Offered capacity is missing or not numeric.")
        if pd.isna(booked):
            add(idx, "missing_booked_capacity", "Booked capacity is missing or not numeric.")
        if not str(row.get("price_original", "")).strip():
            add(idx, "missing_price", "Price is missing.")
        if str(row.get("price_original", "")).strip() and not str(row.get("price_unit_detected", "")).strip():
            add(idx, "missing_price_unit", "Price exists but unit is missing.")
        if str(row.get("price_currency", "")).strip() and row.get("price_currency") not in FX_CURRENCIES:
            add(idx, "unsupported_currency", "Currency is not in the supported FX list.")
        if pd.notna(offered) and pd.notna(booked) and booked > offered * 1.001:
            add(idx, "booked_greater_than_offered", "Booked capacity is greater than offered capacity.")
        if pd.notna(offered) and offered > 0 and pd.notna(booked) and pd.notna(util):
            implied = booked / offered * 100.0
            if abs(implied - util) > 1.0:
                add(idx, "booked_percentage_inconsistent", f"Booked % is {util:.1f}, but booked/offered implies {implied:.1f}.")
        if str(row.get("price_original", "")).strip() and pd.isna(row.get("price_eur_per_mwh")):
            add(idx, "price_not_converted", row.get("price_conversion_note") or "Price exists but conversion failed.")
        if str(row.get("price_currency", "")).strip() and pd.isna(row.get("fx_rate_to_eur")):
            add(idx, "missing_fx_rate", "FX rate is missing; non-EUR price cannot be converted.")
        if pd.notna(row.get("price_numeric")) and row.get("price_numeric") == 0 and pd.notna(booked) and booked > 0:
            add(idx, "zero_price_with_booked_capacity", "Price is zero while booked capacity is positive.")
        eur_mwh = row.get("price_eur_per_mwh")
        if pd.notna(eur_mwh) and eur_mwh < 0.0001:
            add(idx, "suspiciously_low_eur_mwh", "Converted EUR/MWh is suspiciously low.")
        if pd.notna(eur_mwh) and eur_mwh > 50:
            add(idx, "suspiciously_high_eur_mwh", "Converted EUR/MWh is suspiciously high.")
        if duplicate_mask.loc[idx]:
            add(idx, "duplicate_row", "Duplicate row for the same TSO, border point, direction, product, and period.")

    return pd.DataFrame(warnings, columns=warning_columns)


def attach_quality_warnings(df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["data_quality_warning"] = ""
    out["has_quality_warning"] = False
    if quality_df.empty:
        return out
    grouped = (
        quality_df.groupby(["tso", "border_point", "delivery_period"])["warning_type"]
        .apply(lambda values: "; ".join(sorted(set(map(str, values)))))
        .reset_index()
    )
    warning_map = {
        (str(row["tso"]), str(row["border_point"]), str(row["delivery_period"])): row["warning_type"]
        for _, row in grouped.iterrows()
    }
    warnings = []
    for tso, bp, period in zip(out["tso"], out["border_point_full"], out["delivery_period"]):
        value = warning_map.get((str(tso), str(bp), str(period)), "")
        warnings.append(value)
    out["data_quality_warning"] = warnings
    out["has_quality_warning"] = out["data_quality_warning"].astype(bool)
    return out


def read_uploaded(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "tso",
        "border_point_full",
        "border_point_short",
        "type",
        "product_type",
        "product_level",
        "product",
        "delivery_period",
        "period_start",
        "period_end",
        "period_days",
        "offered_capacity_mwh_day",
        "booked_capacity_mwh_day",
        "booked_percentage",
        "original_price",
        "original_currency",
        "original_price_unit",
        "price_original_currency",
        "price_original_unit",
        "fx_rate_to_eur",
        "fx_rate_date",
        "fx_rate_source",
        "converted_price_eur",
        "price_eur_per_mwh",
        "price_conversion_status",
        "price_conversion_note",
        "data_quality_warning",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    rename = {
        "tso": "TSO",
        "border_point_full": "Full border point",
        "border_point_short": "Border point",
        "type": "Type",
        "product_type": "Product type",
        "product_level": "Product level",
        "product": "Product",
        "delivery_period": "Delivery period",
        "period_start": "Period start",
        "period_end": "Period end",
        "period_days": "Days",
        "offered_capacity_mwh_day": "Offered (MWh/day)",
        "booked_capacity_mwh_day": "Booked (MWh/day)",
        "booked_percentage": "Booked %",
        "original_price": "Original price",
        "original_currency": "Currency",
        "original_price_unit": "Unit",
        "price_original_currency": "Original currency",
        "price_original_unit": "Original unit",
        "fx_rate_to_eur": "FX to EUR",
        "fx_rate_date": "FX date",
        "fx_rate_source": "FX source",
        "converted_price_eur": "Price in EUR",
        "price_eur_per_mwh": "EUR/MWh",
        "price_conversion_status": "Price status",
        "price_conversion_note": "Conversion note",
        "data_quality_warning": "Data quality warning",
    }
    return out.rename(columns=rename).sort_values(
        by=[c for c in ["Product type", "Delivery period", "Border point"] if c in out.rename(columns=rename).columns]
    )


def _entsog_month_chunks(start_date: date, end_date: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        next_month = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        chunk_start = max(start_date, cur)
        chunk_end = min(end_date, (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date())
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        cur = next_month
    return chunks


def _entsog_get_json(path: str, params: dict[str, Any], timeout: int = 60) -> tuple[list[dict[str, Any]], str]:
    url = f"{ENTSOG_BASE_URL}/{path.lstrip('/')}"
    print(f"[capacity][ENTSOG] GET {url} params={params}")
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    records = (
        payload.get(path.strip("/"))
        or payload.get("operationaldatas")
        or payload.get("operatorpointdirections")
        or payload.get("data")
        or []
    )
    return records if isinstance(records, list) else [], response.url


def _to_period_start_wet(period_value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(period_value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert("WET").tz_localize(None).normalize()


def _entsog_energy_to_mcm_day(value: Any, unit: Any, gcv_kwh_per_m3: float = 10.55) -> tuple[float, str]:
    numeric = _to_number(value)
    if pd.isna(numeric):
        return np.nan, "missing_value"
    if gcv_kwh_per_m3 <= 0:
        return np.nan, "invalid_gcv"
    unit_key = str(unit or "").lower().replace(" ", "")
    if "kwh" in unit_key:
        kwh_day = numeric
    elif "mwh" in unit_key:
        kwh_day = numeric * 1000.0
    elif "gwh" in unit_key:
        kwh_day = numeric * 1_000_000.0
    else:
        return np.nan, f"unsupported_unit:{unit or 'unknown'}"
    return kwh_day / float(gcv_kwh_per_m3) / 1_000_000.0, "ok"


def fetch_operator_point_directions(
    operator_keywords: Optional[list[str]] = None,
    point_keywords: Optional[list[str]] = None,
    direction_key: Optional[str] = None,
) -> pd.DataFrame:
    operator_keywords = operator_keywords or ["Bulgartransgaz", "FGSZ"]
    point_keywords = point_keywords or ["kireevo", "zaychar", "kiskundorozsma", "horgos", "kalotina", "serbia", "hungary", "bulgaria"]
    records, _ = _entsog_get_json("operatorpointdirections", {"hasData": 1, "limit": -1}, timeout=60)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["operatorLabel", "pointLabel", "directionKey", "pointDirection"]:
        if col not in df.columns:
            df[col] = ""
    op_mask = df["operatorLabel"].astype(str).str.lower().apply(lambda s: any(k.lower() in s for k in operator_keywords))
    pt_mask = df["pointLabel"].astype(str).str.lower().apply(lambda s: any(k.lower() in s for k in point_keywords))
    out = df[op_mask & pt_mask].copy()
    if direction_key:
        out = out[out["directionKey"].astype(str).str.lower() == str(direction_key).lower()]
    out = out.drop_duplicates(subset=["pointDirection"]).reset_index(drop=True)
    print(f"[capacity][ENTSOG] operatorPointDirections returned={len(out)}")
    if not out.empty:
        print(f"[capacity][ENTSOG] operatorPointDirections sample={out.iloc[0].to_dict()}")
    return out


def fetch_entsog_capacity_bookings(
    start_date: date,
    end_date: date,
    granularity: str = "daily",
    gcv_kwh_per_m3: float = 10.55,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality: dict[str, Any] = {
        "api_calls": 0,
        "records_fetched": 0,
        "empty_responses": 0,
        "failed_calls": 0,
        "date_range_fetched": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "operators_included": [],
        "points_included": [],
        "warnings": [],
        "latest_lastUpdateDateTime": None,
        "unit_conversion_assumptions": f"gcv_kwh_per_m3={gcv_kwh_per_m3}",
    }
    opd = fetch_operator_point_directions()
    if opd.empty:
        quality["warnings"].append("No operator-point-direction rows found from ENTSOG /operatorpointdirections?hasData=1.")
        return pd.DataFrame(), quality

    quality["operators_included"] = sorted(opd["operatorLabel"].dropna().astype(str).unique().tolist())
    quality["points_included"] = sorted(opd["pointLabel"].dropna().astype(str).unique().tolist())

    month_chunks = _entsog_month_chunks(start_date, end_date)
    rows: list[dict[str, Any]] = []

    for _, op_row in opd.iterrows():
        point_direction = str(op_row.get("pointDirection", "")).strip()
        if not point_direction:
            continue
        found_data_for_point = False
        for win_start, win_end in month_chunks:
            params = {
                "indicator": "Firm Booked",
                "pointDirection": point_direction,
                "from": win_start.isoformat(),
                "to": win_end.isoformat(),
                "periodType": "day",
                "timeZone": "WET",
                "limit": 1000,
            }
            try:
                records, _ = _entsog_get_json("operationaldatas", params=params, timeout=60)
                quality["api_calls"] += 1
            except Exception as exc:  # noqa: BLE001
                quality["failed_calls"] += 1
                quality["warnings"].append(f"Failed call for {point_direction} [{win_start}..{win_end}]: {exc}")
                continue

            if not records:
                quality["empty_responses"] += 1
                fallback_params = dict(params)
                fallback_params["limit"] = -1
                fallback_params["to"] = min(win_start + pd.Timedelta(days=14), pd.Timestamp(win_end)).date().isoformat()
                try:
                    records, _ = _entsog_get_json("operationaldatas", params=fallback_params, timeout=60)
                    quality["api_calls"] += 1
                except Exception:
                    quality["failed_calls"] += 1
                    records = []
            if not records:
                continue
            found_data_for_point = True
            quality["records_fetched"] += len(records)
            print(f"[capacity][ENTSOG] selected pointDirection={point_direction} rows={len(records)}")
            print(f"[capacity][ENTSOG] sample record={records[0]}")
            for rec in records:
                row = dict(rec)
                row["operatorLabel"] = row.get("operatorLabel", op_row.get("operatorLabel", ""))
                row["pointLabel"] = row.get("pointLabel", op_row.get("pointLabel", ""))
                row["directionKey"] = row.get("directionKey", op_row.get("directionKey", ""))
                row["pointDirection"] = row.get("pointDirection", point_direction)
                rows.append(row)
        if not found_data_for_point:
            quality["warnings"].append(
                "No Firm Booked data returned for this operator-point-direction and period. "
                f"({op_row.get('operatorLabel','')} | {op_row.get('pointLabel','')} | {point_direction})"
            )

    if not rows:
        return pd.DataFrame(), quality

    raw = pd.DataFrame(rows)
    raw["periodFrom"] = raw.get("periodFrom", raw.get("from"))
    raw["period"] = raw["periodFrom"].apply(_to_period_start_wet)
    raw["value"] = pd.to_numeric(raw.get("value", np.nan), errors="coerce")
    raw["original_unit"] = raw.get("unit", "")
    conv = [_entsog_energy_to_mcm_day(v, u, gcv_kwh_per_m3=gcv_kwh_per_m3) for v, u in zip(raw["value"], raw["original_unit"])]
    raw["converted_mcm_day"] = [v for v, _ in conv]
    raw["conversion_status"] = [s for _, s in conv]
    print("[capacity][ENTSOG] unit conversion sample:", raw[["value", "original_unit", "converted_mcm_day", "conversion_status"]].head(1).to_dict("records"))

    gran_map = {"daily": "D", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    gran = gran_map.get(str(granularity).lower(), "D")
    raw["period_bucket"] = raw["period"].dt.to_period(gran).dt.to_timestamp()
    raw["periodType"] = raw.get("periodType", "day").fillna("day")
    raw["indicator"] = raw.get("indicator", "Firm Booked").fillna("Firm Booked")

    out = (
        raw.groupby(
            ["period_bucket", "operatorLabel", "pointLabel", "directionKey", "pointDirection", "indicator", "periodType", "original_unit"],
            as_index=False,
            dropna=False,
        )
        .agg(
            value=("value", "mean"),
            converted_mcm_day=("converted_mcm_day", "mean"),
            lastUpdateDateTime=("lastUpdateDateTime", "max"),
            conversion_status=("conversion_status", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .rename(columns={"period_bucket": "period"})
        .sort_values(["period", "operatorLabel", "pointLabel", "directionKey"])
    )
    if out["lastUpdateDateTime"].notna().any():
        quality["latest_lastUpdateDateTime"] = str(out["lastUpdateDateTime"].dropna().max())
    return out, quality


def _normalize_text(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(v or "").lower()).strip()


def _match_target_border_point(point_label: str) -> dict[str, Any] | None:
    text = _normalize_text(point_label)
    for point in TARGET_BORDER_POINTS:
        if all(p in text for p in point["patterns"]):
            return point
    for point in TARGET_BORDER_POINTS:
        if any(p in text for p in point["patterns"]):
            return point
    return None


def _infer_auction_product_type(date_from: pd.Timestamp, date_to: pd.Timestamp, period_type: str) -> str:
    pt = str(period_type or "").lower().strip()
    if pt in {"year", "yearly"}:
        return "yearly"
    if pt in {"quarter", "quarterly"}:
        return "quarterly"
    if pt in {"month", "monthly"}:
        return "monthly"
    if pt in {"day", "daily"}:
        return "daily"
    if pd.isna(date_from) or pd.isna(date_to):
        return "daily"
    days = max(1, int((date_to - date_from).days) + 1)
    if days >= 330:
        return "yearly"
    if days >= 80:
        return "quarterly"
    if days >= 27:
        return "monthly"
    return "daily"


def _indicator_column_name(indicator: str) -> str:
    i = str(indicator or "").lower()
    if "technical" in i:
        return "technical_capacity"
    if "offered" in i:
        return "offered_capacity"
    if "booked" in i or "allocated" in i:
        return "booked_capacity"
    if "available" in i:
        return "available_capacity"
    return ""


def fetch_entsog_cross_border_capacity_year(
    year: int,
    include_points: Optional[list[str]] = None,
    direction_filter: Optional[list[str]] = None,
    preferred_unit: str = "mcm/day",
    gcv_kwh_per_m3: float = 10.55,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_date = date(int(year), 1, 1)
    end_date = date(int(year), 12, 31)
    quality: dict[str, Any] = {
        "last_successful_fetch": None,
        "records_fetched": 0,
        "api_errors": [],
        "query_urls": [],
        "missing_points": [],
        "missing_products": [],
        "data_warnings": [],
        "matched_point_directions": [],
        "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "unit": preferred_unit,
        "gcv_kwh_per_m3": gcv_kwh_per_m3,
    }

    # Resolve live valid pointDirection values from ENTSOG metadata.
    opd_records, opd_url = _entsog_get_json("operatorpointdirections", {"hasData": 1, "limit": -1}, timeout=60)
    quality["query_urls"].append(opd_url)
    if not opd_records:
        quality["api_errors"].append("operatorpointdirections returned no records.")
        return pd.DataFrame(), quality
    opd = pd.DataFrame(opd_records)
    for col in ["operatorLabel", "pointLabel", "pointDirection", "directionKey", "pointKey", "tsoEicCode"]:
        if col not in opd.columns:
            opd[col] = ""

    matched_rows: list[pd.Series] = []
    for _, row in opd.iterrows():
        matched = _match_target_border_point(row.get("pointLabel"))
        if not matched:
            continue
        if include_points and matched["canonical_key"] not in include_points:
            continue
        dkey = str(row.get("directionKey", "")).lower()
        if direction_filter and dkey not in {d.lower() for d in direction_filter}:
            continue
        r = row.copy()
        r["canonical_key"] = matched["canonical_key"]
        r["canonical_label"] = matched["label"]
        r["country_from"] = matched["country_from"]
        r["country_to"] = matched["country_to"]
        matched_rows.append(r)

    if not matched_rows:
        quality["api_errors"].append("No matching ENTSOG pointDirection values found for requested border points.")
        quality["missing_points"] = [p["label"] for p in TARGET_BORDER_POINTS]
        return pd.DataFrame(), quality

    matched_df = pd.DataFrame(matched_rows).drop_duplicates(subset=["pointDirection"])
    quality["matched_point_directions"] = matched_df["pointDirection"].astype(str).tolist()

    all_records: list[dict[str, Any]] = []
    for _, md in matched_df.iterrows():
        pd_value = str(md.get("pointDirection", "")).strip()
        if not pd_value:
            continue
        for indicator in DEFAULT_CAPACITY_INDICATORS:
            for chunk_start, chunk_end in _entsog_month_chunks(start_date, end_date):
                params = {
                    "pointDirection": pd_value,
                    "from": chunk_start.isoformat(),
                    "to": chunk_end.isoformat(),
                    "indicator": indicator,
                    "periodType": "day",
                    "timeZone": "WET",
                    "limit": 1000,
                }
                try:
                    records, final_url = _entsog_get_json("operationaldatas", params=params, timeout=60)
                    quality["query_urls"].append(final_url)
                except Exception as exc:  # noqa: BLE001
                    quality["api_errors"].append(f"{indicator} {pd_value} {chunk_start}-{chunk_end}: {exc}")
                    continue
                if not records:
                    continue
                for rec in records:
                    row = dict(rec)
                    row["_query_indicator"] = indicator
                    row["_query_url"] = final_url
                    row["_canonical_key"] = md["canonical_key"]
                    row["_canonical_label"] = md["canonical_label"]
                    row["_country_from"] = md["country_from"]
                    row["_country_to"] = md["country_to"]
                    row["_pointDirection"] = pd_value
                    row["_directionKey"] = md.get("directionKey", row.get("directionKey", ""))
                    row["_pointKey"] = md.get("pointKey", row.get("pointKey", ""))
                    row["_tsoEicCode"] = md.get("tsoEicCode", row.get("tsoEicCode", ""))
                    all_records.append(row)

    if not all_records:
        quality["missing_points"] = [p["label"] for p in TARGET_BORDER_POINTS]
        quality["api_errors"].append("No operationaldatas rows found for configured indicators/points.")
        return pd.DataFrame(), quality

    raw = pd.DataFrame(all_records)
    raw["date_from"] = pd.to_datetime(raw.get("periodFrom", raw.get("from")), errors="coerce", utc=True).dt.tz_convert("WET").dt.tz_localize(None)
    raw["date_to"] = pd.to_datetime(raw.get("periodTo", raw.get("to")), errors="coerce", utc=True).dt.tz_convert("WET").dt.tz_localize(None)
    raw["gas_day"] = raw["date_from"].dt.normalize()
    raw["interconnection_point_name"] = raw.get("pointLabel", "")
    raw["interconnection_point_code"] = raw.get("pointKey", raw.get("_pointKey", ""))
    raw["TSO"] = raw.get("operatorLabel", "")
    raw["TSO_code"] = raw.get("tsoEicCode", raw.get("_tsoEicCode", ""))
    raw["direction"] = raw.get("directionKey", raw.get("_directionKey", ""))
    raw["pointDirection"] = raw.get("pointDirection", raw.get("_pointDirection", ""))
    raw["indicator"] = raw.get("indicator", raw.get("_query_indicator", ""))
    raw["unit"] = raw.get("unit", "")
    raw["value"] = pd.to_numeric(raw.get("value", np.nan), errors="coerce")

    mcm_conv = [_entsog_energy_to_mcm_day(v, u, gcv_kwh_per_m3=gcv_kwh_per_m3)[0] for v, u in zip(raw["value"], raw["unit"])]
    raw["value_mcm_day"] = mcm_conv

    column_for_indicator = raw["indicator"].map(_indicator_column_name)
    raw["metric_col"] = column_for_indicator
    raw = raw[raw["metric_col"] != ""].copy()
    if raw.empty:
        quality["api_errors"].append("Indicators returned data but none mapped to required capacity columns.")
        return pd.DataFrame(), quality

    key_cols = [
        "date_from",
        "date_to",
        "gas_day",
        "_country_from",
        "_country_to",
        "TSO",
        "TSO_code",
        "interconnection_point_name",
        "interconnection_point_code",
        "direction",
        "pointDirection",
        "_canonical_key",
        "_canonical_label",
        "unit",
        "periodType",
    ]
    pivot = (
        raw.pivot_table(
            index=key_cols,
            columns="metric_col",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    mcm_pivot = (
        raw.pivot_table(
            index=key_cols,
            columns="metric_col",
            values="value_mcm_day",
            aggfunc="mean",
        )
        .reset_index()
    )
    mcm_cols = {c: f"{c}_mcm_day" for c in ["technical_capacity", "offered_capacity", "booked_capacity", "available_capacity"] if c in mcm_pivot.columns}
    mcm_pivot = mcm_pivot.rename(columns=mcm_cols)
    merged = pivot.merge(mcm_pivot, on=key_cols, how="left")
    merged["country_from"] = merged["_country_from"]
    merged["country_to"] = merged["_country_to"]
    merged["country_pair"] = merged["country_from"] + ">" + merged["country_to"]
    merged["auction_product_type"] = [
        _infer_auction_product_type(df, dt, pt)
        for df, dt, pt in zip(merged["date_from"], merged["date_to"], merged.get("periodType", "day"))
    ]
    merged["source_url"] = ENTSOG_BASE_URL + "/operationaldatas"
    merged["query_metadata"] = "indicator + pointDirection + from + to + periodType=day + timeZone=WET"
    merged["booked_pct_of_technical"] = np.where(
        merged.get("technical_capacity", 0).fillna(0) > 0,
        merged.get("booked_capacity", np.nan) / merged.get("technical_capacity", np.nan) * 100.0,
        np.nan,
    )
    merged["point_match_ok"] = merged["interconnection_point_name"].astype(str).apply(lambda x: _match_target_border_point(x) is not None)
    merged["direction_match_ok"] = merged["direction"].astype(str).str.lower().isin(["entry", "exit"])
    merged["suspicious_zero"] = (
        merged.get("technical_capacity", 0).fillna(0).gt(0)
        & merged.get("booked_capacity", 0).fillna(0).eq(0)
    )
    merged["duplicate_key"] = merged.duplicated(subset=["gas_day", "pointDirection", "auction_product_type"], keep=False)
    merged["warning"] = ""
    merged.loc[~merged["point_match_ok"], "warning"] += "point_mismatch;"
    merged.loc[~merged["direction_match_ok"], "warning"] += "direction_mismatch;"
    merged.loc[merged["suspicious_zero"], "warning"] += "booked_zero_with_technical_positive;"
    merged.loc[merged["duplicate_key"], "warning"] += "duplicate_record;"
    merged["warning"] = merged["warning"].str.strip(";")

    # Missing products per point
    required_products = {"yearly", "quarterly", "monthly", "daily"}
    for point_key, g in merged.groupby("_canonical_key"):
        missing = sorted(required_products - set(g["auction_product_type"].dropna().astype(str)))
        if missing:
            quality["missing_products"].append(f"{point_key}: {', '.join(missing)}")

    found_points = set(merged["_canonical_key"].dropna().astype(str))
    quality["missing_points"] = [p["label"] for p in TARGET_BORDER_POINTS if p["canonical_key"] not in found_points]
    quality["records_fetched"] = len(merged)
    quality["last_successful_fetch"] = pd.Timestamp.utcnow().isoformat()
    quality["data_warnings"] = sorted(set([w for w in merged["warning"].astype(str).tolist() if w]))

    if preferred_unit.lower() == "mcm/day":
        for col in ["technical_capacity", "offered_capacity", "booked_capacity", "available_capacity"]:
            mcol = f"{col}_mcm_day"
            if mcol in merged.columns:
                merged[col] = merged[mcol]
        merged["unit"] = "mcm/day"

    return merged.reset_index(drop=True), quality
