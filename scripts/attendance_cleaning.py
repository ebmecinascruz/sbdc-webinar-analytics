import pandas as pd
import re
from pathlib import Path

from scripts.name_cleaning import _clean_spaces, clean_name


def add_email_clean(
    df: pd.DataFrame,
    email_col: str = "Email",
    out_col: str = "email_clean",
) -> pd.DataFrame:
    """
    Adds a standardized email column for matching:
    - strip whitespace
    - lowercase
    - blank -> NA
    Keeps all original columns.
    """
    out = df.copy()
    out[out_col] = out[email_col].astype(str).str.strip().str.lower().replace("", pd.NA)
    return out


def add_full_name(
    df: pd.DataFrame,
    first_col: str = "First Name",
    last_col: str = "Last Name",
) -> pd.DataFrame:
    """
    Adds:
    - full_name: human-readable combined name
    - full_name_clean: canonical cleaned version for matching

    Uses the shared clean_name() function so webinar and NeoSerra
    names live in the same normalized space.
    """
    out = df.copy()

    # human-readable (light cleaning only)
    first_raw = (
        _clean_spaces(out[first_col])
        if first_col in out.columns
        else pd.Series("", index=out.index)
    )
    last_raw = (
        _clean_spaces(out[last_col])
        if last_col in out.columns
        else pd.Series("", index=out.index)
    )

    out["full_name"] = _clean_spaces((first_raw + " " + last_raw).str.strip())
    out.loc[out["full_name"] == "", "full_name"] = pd.NA

    # canonical matching key
    out["full_name_clean"] = out["full_name"].map(clean_name).replace("", pd.NA)

    return out


def normalize_attended(
    df: pd.DataFrame, attended_col: str = "Attended"
) -> pd.DataFrame:
    """
    Normalizes webinar Attended column to boolean.
    Maps:
      - Yes -> True
      - No -> False
    """
    out = df.copy()

    out[attended_col] = (
        out[attended_col]
        .astype("string")
        .str.strip()
        .map(
            {
                "Yes": True,
                "No": False,
            }
        )
    )
    out = out[out[attended_col].isin([True, False])].copy()
    return out


def aggregate_attendance_person_key(
    df: pd.DataFrame,
    attended_col: str,
    suffix: str,
    *,
    email_col: str = "email_clean",
    name_col: str = "full_name_clean",
) -> pd.DataFrame:
    """
    Aggregates webinar attendance to one row per person_key.
    person_key = email_clean if present else full_name_clean.
    """
    out = df.copy()
    out["person_key"] = out[email_col].where(out[email_col].notna(), out[name_col])

    return (
        out.dropna(subset=["person_key"])
        .groupby("person_key", as_index=False)
        .agg(**{f"attended_{suffix}": (attended_col, "max")})
    )


def dedupe_first_record_person_key(
    df: pd.DataFrame,
    *,
    email_col: str = "email_clean",
    name_col: str = "full_name_clean",
) -> pd.DataFrame:
    """
    Keeps the FIRST record per person_key (first appearance in file),
    where person_key = email_clean if present else full_name_clean.
    """
    out = df.copy()
    out["person_key"] = out[email_col].where(out[email_col].notna(), out[name_col])

    out = out.dropna(subset=["person_key"])
    out = out.drop_duplicates(subset=["person_key"], keep="first")

    return out.drop(columns=["person_key"])


def apply_attendance_then_dedupe(
    webinar_df: pd.DataFrame,
    *,
    attended_col: str = "attended",
    email_col: str = "email_clean",
    name_col: str = "full_name_clean",
    out_attended_col: str = "Attended_Final",
) -> pd.DataFrame:
    """
    For a single webinar export:
      1) Build person_key = email_clean if present else full_name_clean
      2) Aggregate attendance truth across reconnect rows (max => any True wins)
      3) Merge attendance truth back
      4) Keep the FIRST record per person_key (first appearance in file)
    """
    out = webinar_df.copy()

    # 1) Build person_key (email preferred, fallback to name)
    out["person_key"] = out[email_col].where(out[email_col].notna(), out[name_col])

    # require a usable key
    out = out.dropna(subset=["person_key"]).copy()

    # 2) Aggregate "ever attended" across all rows for a person
    flags = out.groupby("person_key", as_index=False).agg(
        _attended_truth=(attended_col, "max")
    )
    flags["_attended_truth"] = flags["_attended_truth"].astype("boolean").fillna(False)

    # 3) Merge attendance truth back
    out = out.merge(flags, on="person_key", how="left")

    # single source of truth
    out[out_attended_col] = out["_attended_truth"]

    # drop temp truth col
    out = out.drop(columns=["_attended_truth"])

    # 4) Keep first record per person (first appearance)
    out = out.drop_duplicates(subset=["person_key"], keep="first")

    return out.drop(columns=["person_key"])


def parse_attendance_filename(file_path: str | Path) -> tuple[str, str]:
    """
    Expected: attendee_{webinar_id}_YYYY_MM_DD(.csv)
    Returns: (webinar_id, yyyymmdd_underscored)
    """
    name = Path(file_path).stem  # no extension
    m = re.match(r"attendee_(.+)_(20\d{2}_\d{2}_\d{2})$", name)

    if not m:
        raise ValueError(
            f"Filename '{name}' doesn't match expected pattern "
            "'attendee_{webinar_id}_YYYY_MM_DD'"
        )

    webinar_id = m.group(1)
    date_suffix = m.group(2)  # YYYY_MM_DD
    return webinar_id, date_suffix


def is_valid_email_series(s: pd.Series) -> pd.Series:
    """
    Practical email validation:
    - exactly one '@'
    - no spaces
    - at least 1 char before '@'
    - domain contains a '.'
    """
    s = s.astype("string")

    has_space = s.str.contains(r"\s", regex=True).fillna(False)
    at_count = s.str.count("@").fillna(0)

    # split safely
    parts = s.str.split("@", n=1, expand=True)
    local = parts[0]
    domain = parts[1]

    local_ok = local.notna() & (local.str.len() > 0)
    domain_ok = domain.notna() & domain.str.contains(r"\.", regex=True)

    # domain shouldn't start/end with dot
    domain_ok = domain_ok & ~domain.str.startswith(".") & ~domain.str.endswith(".")

    valid = (at_count == 1) & ~has_space & local_ok & domain_ok
    return valid.fillna(False)


def split_invalid_emails_from_clean(
    df: pd.DataFrame,
    *,
    email_clean_col: str = "email_clean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits rows based on validity of email_clean.
    Returns (valid_df, invalid_df).
    """
    out = df.copy()

    out["email_is_valid"] = is_valid_email_series(out[email_clean_col])

    missing = out[email_clean_col].isna()
    invalid_fmt = ~missing & ~out["email_is_valid"]

    out["invalid_reason"] = pd.NA
    out.loc[missing, "invalid_reason"] = "email_missing"
    out.loc[invalid_fmt, "invalid_reason"] = "email_invalid_format"

    valid_df = out[out["email_is_valid"]].copy()
    invalid_df = out[~out["email_is_valid"]].copy()

    return valid_df, invalid_df
