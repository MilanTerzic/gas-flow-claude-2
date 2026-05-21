"""Capacity booking normalization, price conversion, and quality checks."""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd


BORDER_POINT_SHORT_NAMES = {
    "Kireevo (BG)/Zaychar (RS)": "BG->RS Kireevo",
    "Kiskundorozsma 2": "RS->HU Kisk. 2",
    "Kiskundorozsma (HU)/Kiskundorozsma (RS)": "HU->RS Kisk.",
    "Kiskundorozsma": "HU->RS Kisk.",
    "Kalotina": "BG->RS Kalotina",
}

PRODUCT_ORDER = ["Within-day", "Day-ahead", "Daily", "Monthly", "Quarterly", "Yearly", "Unknown"]

COLUMN_ALIASES = {
    "tso": ["tso", "operator", "transmission system operator"],
    "border_point": ["border point", "border_point", "point", "interconnection point"],
    "direction": ["type", "direction", "entry/exit"],
    "product": ["product", "product type", "runtime period"],
    "period": ["period", "delivery period", "gas day", "date"],
    "offered_mwh": ["offered (mwh/day)", "offered", "offered_mwh", "offered capacity"],
    "booked_mwh": ["booked (mwh/day)", "booked", "booked_mwh", "booked capacity"],
    "utilisation_pct": ["booked %", "booked%", "utilisation_pct", "utilization_pct", "utilisation"],
    "price": ["price", "reserve price", "tariff"],
    "currency": ["currency", "ccy"],
    "pct_of_100": ["% of 100", "pct_of_100"],
    "price_unit": ["price unit", "unit", "price_unit"],
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
) -> dict[str, Any]:
    """Convert capacity tariffs to an effective EUR/MWh over the delivery period.

    ENTSOG capacity tariffs such as EUR/kWh/h/day price hourly capacity. A row's
    booked MWh/day is converted to kWh/h, cost is calculated for the tariff
    period, then divided by booked energy over the same delivery period.
    """
    if pd.isna(price_numeric):
        return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing price."}
    if not currency:
        return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing currency."}
    if str(currency).upper() != "EUR":
        return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": f"Currency {currency} is not converted because no FX rate is available."}
    if pd.isna(booked_mwh_per_day) or booked_mwh_per_day <= 0:
        return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing or zero booked capacity; effective EUR/MWh cannot be calculated."}
    if pd.isna(period_days) or period_days <= 0:
        return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": np.nan, "price_conversion_note": "Missing delivery period length."}

    unit_clean = re.sub(r"\s+", "", str(unit).lower())
    booked_energy = booked_mwh_per_day * period_days

    if "mwh" in unit_clean and "day" in unit_clean and "kwh/h" not in unit_clean:
        return {"price_eur_per_mwh": price_numeric, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": "Interpreted as EUR/MWh/day energy price."}

    if "kwh/h" in unit_clean:
        hourly_capacity_kwh = booked_mwh_per_day * 1000.0 / 24.0
        if any(token in unit_clean for token in ["/day", "/d"]):
            charged_days = period_days
        elif "/month" in unit_clean:
            charged_days = period_days
        elif "/quarter" in unit_clean:
            charged_days = period_days
        elif "/year" in unit_clean:
            charged_days = period_days
        elif "/period" in unit_clean or unit_clean.endswith("/p"):
            charged_days = 1.0 / 24.0
        else:
            return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": f"Price unit '{unit}' could not be parsed."}
        total_cost = price_numeric * hourly_capacity_kwh * 24.0 * charged_days
        return {"price_eur_per_mwh": total_cost / booked_energy, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": "Converted from EUR/kWh/h capacity tariff."}

    if "eur/mwh" in unit_clean:
        return {"price_eur_per_mwh": price_numeric, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": "Already in EUR/MWh."}

    return {"price_eur_per_mwh": np.nan, "booked_energy_mwh_for_period": booked_energy, "price_conversion_note": f"Price unit '{unit}' could not be parsed."}


def border_point_short_name(border_point: Any, direction: Any = None) -> str:
    full = "" if pd.isna(border_point) else str(border_point).strip()
    if full in BORDER_POINT_SHORT_NAMES:
        return BORDER_POINT_SHORT_NAMES[full]
    compact = re.sub(r"\s*\([^)]*\)", "", full).strip()
    if compact in BORDER_POINT_SHORT_NAMES:
        return BORDER_POINT_SHORT_NAMES[compact]
    return compact[:24] + "..." if len(compact) > 27 else compact


def prepare_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return normalized capacity booking rows ready for tables and charts."""
    out = pd.DataFrame()
    for target, aliases in COLUMN_ALIASES.items():
        src = _first_present(df, aliases)
        out[target] = df[src] if src else np.nan

    out["tso"] = out["tso"].fillna("").astype(str).str.strip()
    out["border_point_full"] = out["border_point"].fillna("").astype(str).str.strip()
    out["border_point_short"] = [border_point_short_name(bp, d) for bp, d in zip(out["border_point_full"], out["direction"])]
    out["direction"] = out["direction"].fillna("").astype(str).str.strip().str.lower()
    out["product"] = out["product"].fillna("").astype(str).str.strip()
    out["period"] = out["period"].fillna("").astype(str).str.strip()
    out["offered_mwh"] = out["offered_mwh"].map(_to_number)
    out["booked_mwh"] = out["booked_mwh"].map(_to_number)
    out["utilisation_pct"] = out["utilisation_pct"].map(_to_number)
    out["pct_of_100"] = out["pct_of_100"].map(_to_number)

    period_rows = [parse_product_period(p, per) for p, per in zip(out["product"], out["period"])]
    out = pd.concat([out, pd.DataFrame(period_rows)], axis=1)

    price_rows = [parse_price_and_currency(p, c, u) for p, c, u in zip(out["price"], out["currency"], out["price_unit"])]
    out = pd.concat([out, pd.DataFrame(price_rows)], axis=1)

    conversion_rows = [
        convert_price_to_eur_per_mwh(price, ccy, unit, booked, days)
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
        if pd.notna(offered) and pd.notna(booked) and booked > offered * 1.001:
            add(idx, "booked_greater_than_offered", "Booked capacity is greater than offered capacity.")
        if pd.notna(offered) and offered > 0 and pd.notna(booked) and pd.notna(util):
            implied = booked / offered * 100.0
            if abs(implied - util) > 1.0:
                add(idx, "booked_percentage_inconsistent", f"Booked % is {util:.1f}, but booked/offered implies {implied:.1f}.")
        if str(row.get("price_original", "")).strip() and pd.isna(row.get("price_eur_per_mwh")):
            add(idx, "price_not_converted", row.get("price_conversion_note") or "Price exists but conversion failed.")
        if pd.notna(row.get("price_numeric")) and row.get("price_numeric") == 0 and pd.notna(booked) and booked > 0:
            add(idx, "zero_price_with_booked_capacity", "Price is zero while booked capacity is positive.")
        eur_mwh = row.get("price_eur_per_mwh")
        if pd.notna(eur_mwh) and (eur_mwh < 0.0001 or eur_mwh > 50):
            add(idx, "converted_price_outlier", "Converted EUR/MWh is outside the expected operational range.")
        if duplicate_mask.loc[idx]:
            add(idx, "duplicate_row", "Duplicate row for the same TSO, border point, direction, product, and period.")

    return pd.DataFrame(warnings)


def read_uploaded(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return prepare_chart_data(raw)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "tso",
        "border_point_short",
        "border_point_full",
        "direction",
        "product_type",
        "delivery_period",
        "offered_mwh",
        "booked_mwh",
        "utilisation_pct",
        "price_original",
        "price_currency",
        "price_unit_detected",
        "price_eur_per_mwh",
        "price_conversion_note",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    rename = {
        "tso": "TSO",
        "border_point_short": "Border point",
        "border_point_full": "Full border point",
        "direction": "Type",
        "product_type": "Product type",
        "delivery_period": "Delivery period",
        "offered_mwh": "Offered (MWh/day)",
        "booked_mwh": "Booked (MWh/day)",
        "utilisation_pct": "Booked %",
        "price_original": "Original price",
        "price_currency": "Currency",
        "price_unit_detected": "Unit detected",
        "price_eur_per_mwh": "EUR/MWh",
        "price_conversion_note": "Conversion note",
    }
    return out.rename(columns=rename).sort_values(
        by=[c for c in ["Product type", "Delivery period", "Border point"] if c in out.rename(columns=rename).columns]
    )
