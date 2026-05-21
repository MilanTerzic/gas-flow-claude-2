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
    "offered_mwh": ["offered (mwh/day)", "offered", "offered_mwh", "offered capacity", "offered_capacity_mwh_day"],
    "booked_mwh": ["booked (mwh/day)", "booked", "booked_mwh", "booked capacity", "booked_capacity_mwh_day"],
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
            "original_unit",
            "converted_unit",
            "source_timestamp",
        ]
    ]


def prepare_chart_data(df: pd.DataFrame, fx_rates: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Return normalized capacity booking rows ready for tables and charts."""
    out = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        src = _first_present(df, aliases)
        out[target] = df[src] if src else np.nan

    out["tso"] = out["tso"].fillna("").astype(str).str.strip()
    out["border_point_full"] = out["border_point"].fillna("").astype(str).str.strip()
    out["direction"] = out["direction"].fillna("").astype(str).str.strip().str.lower()
    out["border_point_short"] = [border_point_short_name(bp, d) for bp, d in zip(out["border_point_full"], out["direction"])]
    out["product"] = out["product"].fillna("").astype(str).str.strip()
    out["period"] = out["period"].fillna("").astype(str).str.strip()
    out["source_timestamp"] = out["source_timestamp"].fillna("").astype(str).str.strip()
    out["offered_mwh"] = out["offered_mwh"].map(_to_number)
    out["booked_mwh"] = out["booked_mwh"].map(_to_number)
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

    return pd.DataFrame(warnings)


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
