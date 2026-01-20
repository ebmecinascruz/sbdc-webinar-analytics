import pandas as pd


def build_ns_lookup(
    ns: pd.DataFrame,
    key_col: str,
    keep_cols: list[str],
    counseling_col: str = "Last Counseling",
) -> pd.DataFrame:
    """
    Build a NeoSerra lookup table keyed by `key_col`
    (e.g., email_clean or full_name_clean).

    Collision resolution:
    - If a key appears once → keep it
    - If a key appears multiple times →
        keep the row with the most recent counseling date
        (NaT sorts last)

    Guarantees:
    - One row per key
    - Deterministic collision resolution
    - No duplicate column labels
    """

    df = ns.copy()

    if key_col not in df.columns:
        raise KeyError(f"Key column '{key_col}' not found in NeoSerra dataframe.")

    # Remove key_col from keep_cols to avoid duplicate labels
    keep_cols = [c for c in keep_cols if c != key_col]

    # Keep only rows with a usable key
    df = df[df[key_col].notna()].copy()

    # Parse counseling date safely
    if counseling_col in df.columns:
        df["_last_counseling_dt"] = pd.to_datetime(df[counseling_col], errors="coerce")
    else:
        # If missing, treat all as equal priority
        df["_last_counseling_dt"] = pd.NaT

    # Identify collisions
    dup_mask = df.duplicated(subset=[key_col], keep=False)

    # Resolve collisions by latest counseling
    df_collide = df.loc[dup_mask].sort_values(
        by=[key_col, "_last_counseling_dt"],
        ascending=[True, False],
        na_position="last",
    )

    df_single = df.loc[~dup_mask]

    df_collide_best = df_collide.drop_duplicates(subset=[key_col], keep="first")

    # Combine + final guard
    out = pd.concat([df_single, df_collide_best], ignore_index=True)
    out = out.drop_duplicates(subset=[key_col], keep="first")

    # Select output columns
    out = out.loc[:, [key_col, *keep_cols]].copy()

    # Safety check
    if out.columns.duplicated().any():
        raise ValueError("Output lookup has duplicate column labels.")

    return out
