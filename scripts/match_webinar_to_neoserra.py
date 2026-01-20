import pandas as pd

from scripts.neoserra_helper import build_ns_lookup


def match_webinar_to_neoserra(
    webinar: pd.DataFrame,
    ns: pd.DataFrame,
    ns_keep_cols: list[str],
    counseling_col: str = "Last Counseling",
    protect_webinar_cols: bool = True,
    validate_email_unchanged: bool = True,
) -> pd.DataFrame:
    """
    Match webinar records to NeoSerra with:
      1) email_clean (preferred)
      2) full_name_clean (fallback if email match not found)

    NeoSerra collisions are resolved by selecting the row with the latest `counseling_col`
    via build_ns_lookup().

    Safeguards:
    - Never overwrites webinar key columns (email_clean, full_name_clean).
    - If protect_webinar_cols=True, never overwrites ANY existing webinar-native columns
      during name fallback fill (prevents surprises).
    - If validate_email_unchanged=True, raises if email_clean changes.

    Adds:
    - ns_match_source: "email" | "name" | "none"
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

    # If requested, don't overwrite any columns that already exist in webinar
    if protect_webinar_cols:
        webinar_cols = set(out.columns)
        ns_keep_cols_fill = [c for c in ns_keep_cols if c not in webinar_cols]
    else:
        ns_keep_cols_fill = ns_keep_cols

    # Build NeoSerra lookups (dedup only on collisions, latest counseling wins)
    ns_email = build_ns_lookup(ns, "email_clean", ns_keep_cols, counseling_col)
    ns_name = build_ns_lookup(ns, "full_name_clean", ns_keep_cols, counseling_col)

    # 1) Email merge
    out = out.merge(
        ns_email,
        on="email_clean",
        how="left",
        suffixes=("", "_ns"),
        indicator="_email_merge_status",
    )

    email_matched = out["_email_merge_status"].eq("both")

    # 2) Name fallback (only where email failed)
    need_name = (
        out["_email_merge_status"].eq("left_only") & out["full_name_clean"].notna()
    )

    name_fill = out.loc[need_name, ["full_name_clean"]].merge(
        ns_name,
        on="full_name_clean",
        how="left",
    )

    # Correctly detect which of the need_name rows matched by name
    name_matched = pd.Series(False, index=out.index)
    if "Client ID" in name_fill.columns:
        name_matched.loc[need_name] = name_fill["Client ID"].notna().to_numpy()

    # Fill only the safe columns (prevents overwriting webinar columns)
    for col in ns_keep_cols_fill:
        if col in out.columns and col in name_fill.columns:
            out.loc[need_name, col] = name_fill[col].to_numpy()

    # Match source
    out["ns_match_source"] = "none"
    out.loc[email_matched, "ns_match_source"] = "email"
    out.loc[need_name & name_matched, "ns_match_source"] = "name"

    out = out.drop(columns=["_email_merge_status"])

    # Client flags
    if "Client ID" in out.columns:
        out["is_client"] = out["Client ID"].notna()
    else:
        out["is_client"] = False

    if validate_email_unchanged:
        changed = (out["email_clean"].fillna("").to_numpy() != email_before).sum()
        if changed:
            raise ValueError(
                f"email_clean changed for {changed} rows inside match_webinar_to_neoserra()."
            )

    return out
