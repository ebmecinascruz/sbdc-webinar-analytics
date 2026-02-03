import pandas as pd
import pgeocode

from scripts.zip_codes import clean_zip_5


def ensure_state_from_zip(
    df: pd.DataFrame,
    *,
    raw_zip_col: str = "Zip/Postal Code",
    zip_col: str = "zip_clean",
    state_col: str = "State/Province",
    country: str = "US",
) -> pd.DataFrame:
    out = df.copy()

    # 🔹 always normalize ZIPs first
    out = clean_zip_5(out, raw_zip_col=raw_zip_col, out_col=zip_col)

    # If state exists and has values, we're done
    if state_col in out.columns and out[state_col].notna().any():
        return out

    if state_col not in out.columns:
        out[state_col] = pd.NA

    unique_zips = out[zip_col].dropna().drop_duplicates().tolist()
    if not unique_zips:
        return out

    nomi = pgeocode.Nominatim(country)
    ref = nomi.query_postal_code(unique_zips).reset_index()

    if "postal_code" not in ref.columns and "index" in ref.columns:
        ref = ref.rename(columns={"index": "postal_code"})

    ref = ref.rename(
        columns={
            "postal_code": "zip_clean",
            "state_code": "state_code",
        }
    )
    ref["zip_clean"] = ref["zip_clean"].astype("string").str.zfill(5)

    out = out.merge(
        ref[["zip_clean", "state_code"]],
        how="left",
        on="zip_clean",
    )

    out[state_col] = (
        out[state_col]
        .astype("string")
        .where(
            out[state_col].notna() & (out[state_col].str.strip() != ""),
            out["state_code"],
        )
        .str.upper()
    )

    out = out.drop(columns=["state_code"], errors="ignore")
    return out


def ensure_columns_exist(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Ensure columns exist in DataFrame.
    Missing columns are added and filled with NA.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out
