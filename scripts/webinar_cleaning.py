# Importing necessary packages
import pandas as pd

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


def prepare_webinar_df(
    df: pd.DataFrame,
    first_col: str = "First Name",
    last_col: str = "Last Name",
    email_col: str = "Email",
) -> pd.DataFrame:
    out = add_full_name(df, first_col=first_col, last_col=last_col)
    out = add_email_clean(out, email_col=email_col)

    return out
