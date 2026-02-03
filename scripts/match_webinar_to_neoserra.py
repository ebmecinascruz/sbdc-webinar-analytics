import pandas as pd

from scripts.neoserra_helper import build_ns_lookup
from scripts.zip_codes import clean_zip_5


def match_webinar_to_neoserra(
    webinar: pd.DataFrame,
    ns: pd.DataFrame,
    ns_keep_cols: list[str],
    counseling_col: str = "Last Counseling",
    protect_webinar_cols: bool = True,
    validate_email_unchanged: bool = True,
    webinar_zip_col: str = "Zip/Postal Code",
    ns_zip_col: str = "Physical Address ZIP Code",
    zip_clean_col: str = "zip_clean",
) -> pd.DataFrame:
    """
    Match webinar records to NeoSerra with:
      1) email_clean (preferred)
      2) full_name_clean + zip_clean (fallback if email match not found)
      3) full_name_clean (final fallback if zip match not found)

    NeoSerra collisions are resolved by selecting the row with the latest `counseling_col`
    via build_ns_lookup().

    Safeguards:
    - Never overwrites webinar key columns (email_clean, full_name_clean).
    - If protect_webinar_cols=True, never overwrites ANY existing webinar-native columns
      during fallback fills (prevents surprises).
    - If validate_email_unchanged=True, raises if email_clean changes.

    Adds:
    - ns_match_source: "email" | "name_zip" | "name" | "none"
    - is_client: bool (True if "Client ID" present after matching)
    - client_status: "client" | "non_client"
    """

    required = {"email_clean", "full_name_clean"}
    missing = required - set(webinar.columns)
    if missing:
        raise KeyError(f"Webinar dataframe missing required columns: {sorted(missing)}")

    out = webinar.copy()

    if validate_email_unchanged:
        email_before = out["email_clean"].fillna("").to_numpy()

    # Never allow keys inside keep cols (prevents duplicate labels + accidental overwrites)
    blocked = {"email_clean", "full_name_clean"}
    ns_keep_cols = [c for c in ns_keep_cols if c not in blocked]

    if protect_webinar_cols:
        webinar_cols = set(out.columns)
        ns_keep_cols_fill = [c for c in ns_keep_cols if c not in webinar_cols]
    else:
        ns_keep_cols_fill = ns_keep_cols

    # ----------------------------
    # Normalize ZIP to zip_clean
    # ----------------------------
    if zip_clean_col not in out.columns:
        if webinar_zip_col in out.columns:
            out = clean_zip_5(out, raw_zip_col=webinar_zip_col, out_col=zip_clean_col)
        else:
            out[zip_clean_col] = pd.NA

    ns_norm = ns.copy()
    if zip_clean_col not in ns_norm.columns:
        if ns_zip_col in ns_norm.columns:
            ns_norm = clean_zip_5(
                ns_norm, raw_zip_col=ns_zip_col, out_col=zip_clean_col
            )
        else:
            ns_norm[zip_clean_col] = pd.NA

    # ----------------------------
    # Build NeoSerra lookups
    # ----------------------------
    ns_email = build_ns_lookup(ns_norm, "email_clean", ns_keep_cols, counseling_col)

    # Name + ZIP lookup (composite key)
    ns_for_zip = ns_norm.copy()
    ns_for_zip["_name_zip_key"] = (
        ns_for_zip["full_name_clean"].astype("string").fillna("")
        + "|"
        + ns_for_zip[zip_clean_col].astype("string").fillna("")
    )
    ns_name_zip = build_ns_lookup(
        ns_for_zip, "_name_zip_key", ns_keep_cols, counseling_col
    )

    # Name-only lookup
    ns_name = build_ns_lookup(ns_norm, "full_name_clean", ns_keep_cols, counseling_col)

    # ----------------------------
    # Tier 1: Email merge
    # ----------------------------
    out = out.merge(
        ns_email,
        on="email_clean",
        how="left",
        suffixes=("", "_ns"),
        indicator="_email_merge_status",
    )

    email_matched = out["_email_merge_status"].eq("both")

    out["ns_match_source"] = "none"
    out.loc[email_matched, "ns_match_source"] = "email"

    # ----------------------------
    # Tier 2: Name + ZIP fallback
    # ----------------------------
    # Only where email failed, and we have both name + zip
    out["_name_zip_key"] = (
        out["full_name_clean"].astype("string").fillna("")
        + "|"
        + out[zip_clean_col].astype("string").fillna("")
    )

    need_name_zip = (
        out["_email_merge_status"].eq("left_only")
        & out["full_name_clean"].notna()
        & out[zip_clean_col].notna()
    )

    name_zip_fill = out.loc[need_name_zip, ["_name_zip_key"]].merge(
        ns_name_zip,
        on="_name_zip_key",
        how="left",
    )

    name_zip_matched = pd.Series(False, index=out.index)
    if "Client ID" in name_zip_fill.columns:
        name_zip_matched.loc[need_name_zip] = (
            name_zip_fill["Client ID"].notna().to_numpy()
        )
    else:
        # fallback: if any keep col landed, treat as matched; but Client ID is the best signal
        landed_any = (
            name_zip_fill.drop(columns=["_name_zip_key"], errors="ignore")
            .notna()
            .any(axis=1)
        )
        name_zip_matched.loc[need_name_zip] = landed_any.to_numpy()

    # Fill only safe columns (prevents overwriting webinar columns)
    for col in ns_keep_cols_fill:
        if col in out.columns and col in name_zip_fill.columns:
            out.loc[need_name_zip, col] = name_zip_fill[col].to_numpy()

    out.loc[need_name_zip & name_zip_matched, "ns_match_source"] = "name_zip"

    # ----------------------------
    # Tier 3: Name-only fallback
    # ----------------------------
    # Only where email failed AND name_zip didn't match
    need_name = (
        out["_email_merge_status"].eq("left_only")
        & (out["ns_match_source"] == "none")
        & out["full_name_clean"].notna()
    )

    name_fill = out.loc[need_name, ["full_name_clean"]].merge(
        ns_name,
        on="full_name_clean",
        how="left",
    )

    name_matched = pd.Series(False, index=out.index)
    if "Client ID" in name_fill.columns:
        name_matched.loc[need_name] = name_fill["Client ID"].notna().to_numpy()
    else:
        landed_any = (
            name_fill.drop(columns=["full_name_clean"], errors="ignore")
            .notna()
            .any(axis=1)
        )
        name_matched.loc[need_name] = landed_any.to_numpy()

    for col in ns_keep_cols_fill:
        if col in out.columns and col in name_fill.columns:
            out.loc[need_name, col] = name_fill[col].to_numpy()

    out.loc[need_name & name_matched, "ns_match_source"] = "name"

    # Cleanup indicators / temp keys
    out = out.drop(columns=["_email_merge_status", "_name_zip_key"], errors="ignore")

    # Client flags
    if "Client ID" in out.columns:
        out["is_client"] = out["Client ID"].notna()
        out["client_status"] = out["is_client"].map(
            {True: "client", False: "non_client"}
        )
    else:
        out["is_client"] = False
        out["client_status"] = "non_client"

    # Validate email_clean unchanged
    if validate_email_unchanged:
        changed = (out["email_clean"].fillna("").to_numpy() != email_before).sum()
        if changed:
            raise ValueError(
                f"email_clean changed for {changed} rows inside match_webinar_to_neoserra()."
            )

    return out
