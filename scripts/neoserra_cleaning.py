import numpy as np
import pandas as pd

from scripts.name_cleaning import clean_name, _clean_spaces
from scripts.columns import NEOSERRA_COLUMNS

center_map = {
    "Long Beach SBDC": "LBCC",
    "Pacific Coast Regional": "PCR",
    "Pasadena City College": "PCC",
    "College of the Canyons SBDC": "COC",
    "El Camino College SBDC": "ECC",
    "University of La Verne SBDC": "LV",
    "Economic Development Collaborative": "EDC",
    "LEAD Center": "LEAD",
}


def clean_email_series(series: pd.Series) -> pd.Series:
    """
    Standardize emails for matching:
    - strip spaces
    - lowercase
    - remove internal spaces
    - convert blanks to NaN
    """
    s = series.fillna("").astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", "", regex=True)  # remove any whitespace inside
    s = s.replace("", np.nan)
    return s


def prepare_neoserra_clients(
    df: pd.DataFrame,
    keep_columns: list[str] = NEOSERRA_COLUMNS,
) -> pd.DataFrame:
    """
    Prepares NeoSerra client data for webinar matching.

    - Selects a known, fixed set of NeoSerra columns
    - Creates standardized matching keys:
        - full_name_clean
        - email_clean
    - Preserves original NeoSerra fields for context

    Returns a cleaned client reference DataFrame.
    """
    out = df[keep_columns].copy()

    # Prefer "Email Address" if present, otherwise fall back to "Email"
    email_source = out["Email Address"].where(
        out["Email Address"].notna(), out["Email"]
    )

    out["email_clean"] = clean_email_series(email_source)

    # Canonical name cleaning (shared with webinar)
    out["full_name_clean"] = out["Primary Contact"].map(clean_name).replace("", pd.NA)

    # Optional but recommended: drop rows with no usable match keys
    out = out.loc[~(out["email_clean"].isna() & out["full_name_clean"].isna())].copy()

    # cleaning extra spaces in Center (Bixel Exchange)
    out["Center"] = _clean_spaces(out["Center"])

    # Adding center abbreviation
    out["Center Abbr"] = out["Center"].map(center_map).fillna(out["Center"])

    return out
