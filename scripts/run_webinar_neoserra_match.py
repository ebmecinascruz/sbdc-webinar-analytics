from pathlib import Path
import pandas as pd

from scripts.attendance_cleaning import (
    parse_attendance_filename,
    apply_attendance_then_dedupe,
)
from scripts.smallbiz_func import process_zoom_attendance_file_full
from scripts.match_webinar_to_neoserra import match_webinar_to_neoserra
from scripts.neoserra_cleaning import prepare_neoserra_clients
from scripts.columns import (
    NS_OUTPUT_COLUMNS,
    NS_RENAME_COLS,
    ZIP_MERGE_KEEP_COLS,
    FINAL_KEEP_COLS,
)

from scripts.master_tables import (
    split_people_and_attendance,
    update_attendance_master,
    update_people_master,
)

from scripts.run_summary import (
    RunSummary,
    _attendance_key,
    find_people_enriched,
    print_run_summary,
    get_enriched_deltas,
)

from scripts.zip_codes import map_webinar_centers_for_nonclients


def run_webinar_neoserra_match(
    webinar_file: str | Path,
    neoserra_file: str | Path,
    centers_file: str | Path,
    *,
    output_path: str | Path | None = None,
    people_master_path: str | Path = "people_master.csv",
    attendance_master_path: str | Path = "attendance_master.csv",
    cache_path="../data/reference/zip_to_center_lookup.csv",
    print_summary: bool = True,
) -> dict:
    webinar_file = Path(webinar_file)
    neoserra_file = Path(neoserra_file)
    centers_file = Path(centers_file)

    centers = pd.read_csv(centers_file)

    webinar_id, webinar_date = parse_attendance_filename(webinar_file)

    # ---- Load masters BEFORE (for delta calculations) ----
    people_before_df = (
        pd.read_csv(people_master_path)
        if Path(people_master_path).exists()
        else pd.DataFrame()
    )
    attendance_before_df = (
        pd.read_csv(attendance_master_path)
        if Path(attendance_master_path).exists()
        else pd.DataFrame()
    )

    people_before = len(people_before_df) if not people_before_df.empty else 0
    attendance_before = (
        len(attendance_before_df) if not attendance_before_df.empty else 0
    )

    # -------------------------
    # 1) Load & clean webinar
    # -------------------------
    webinar_clean, webinar_non_emails = process_zoom_attendance_file_full(webinar_file)
    webinar_clean = apply_attendance_then_dedupe(webinar_clean, attended_col="attended")

    webinar_clean["webinar_id"] = webinar_id
    webinar_clean["webinar_date"] = webinar_date

    # -------------------------
    # 2) Load NeoSerra
    # -------------------------
    ns_raw = pd.read_csv(neoserra_file)
    ns = prepare_neoserra_clients(ns_raw)

    # -------------------------
    # 3) Match webinar → NeoSerra
    # -------------------------
    merged = match_webinar_to_neoserra(
        webinar=webinar_clean,
        ns=ns,
        ns_keep_cols=NS_OUTPUT_COLUMNS,
        protect_webinar_cols=True,
    )

    merged = merged.rename(columns=NS_RENAME_COLS)
    merged = merged[FINAL_KEEP_COLS].copy()

    # -------------------------
    # 3.5) Map webinar center for non-clients
    # -------------------------
    merged = map_webinar_centers_for_nonclients(merged, centers, cache_path)

    # -------------------------
    # 4) Split into tables
    # -------------------------
    people_session, attendance_session = split_people_and_attendance(merged)

    # ---- Attendance delta keys (added vs overwritten) ----
    # Only compute if attendance_before_df exists with needed cols
    if not attendance_before_df.empty:
        before_keys = set(_attendance_key(attendance_before_df).tolist())
    else:
        before_keys = set()

    session_keys = set(_attendance_key(attendance_session).tolist())

    attendance_overwritten = len(session_keys & before_keys)
    attendance_added = len(session_keys - before_keys)

    # -------------------------
    # 5) Update masters
    # -------------------------
    attendance_after_df = update_attendance_master(
        attendance_session, master_path=attendance_master_path
    )
    people_after_df = update_people_master(
        people_session, master_path=people_master_path
    )

    attendance_after = len(attendance_after_df)
    people_after = len(people_after_df)

    # ---- People deltas ----
    # New people inserted = emails in session not previously present
    if not people_before_df.empty and "email_clean" in people_before_df.columns:
        before_people_keys = set(
            people_before_df["email_clean"].astype("string").fillna("").tolist()
        )
    else:
        before_people_keys = set()

    session_people_keys = set(
        people_session["email_clean"].astype("string").fillna("").tolist()
    )
    people_new = len(session_people_keys - before_people_keys)

    # If this is the first run (no existing people_master on disk), there is no "before"
    if people_before_df.empty or "email_clean" not in people_before_df.columns:
        enriched_keys = []
        enriched_before_df = pd.DataFrame()
        enriched_after_df = pd.DataFrame()
        people_enriched = 0
    else:
        enriched_keys = find_people_enriched(
            people_before_df,
            people_after_df,
            key="email_clean",
        )
        people_enriched = len(enriched_keys)

        enriched_before_df, enriched_after_df = get_enriched_deltas(
            people_before_df,
            people_after_df,
            enriched_keys,
            key="email_clean",
        )

    # -------------------------
    # 6) Save session output
    # -------------------------
    if output_path is None:
        output_path = webinar_file.with_name(webinar_file.stem + "_with_neoserra.csv")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged[ZIP_MERGE_KEEP_COLS].to_csv(output_path, index=False)

    # -------------------------
    # 7) Print summary
    # -------------------------
    summary = RunSummary(
        webinar_id=str(webinar_id),
        webinar_date=str(webinar_date),
        session_rows=len(merged),
        session_unique_emails=merged["email_clean"].nunique(dropna=True)
        if "email_clean" in merged.columns
        else 0,
        attendance_before=attendance_before,
        attendance_after=attendance_after,
        attendance_added=attendance_added,
        attendance_overwritten=attendance_overwritten,
        people_before=people_before,
        people_after=people_after,
        people_new=people_new,
        people_enriched=people_enriched,
        people_name_collision_groups=0,
        people_name_collision_rows=0,
        people_name_collision_new_groups=0,
    )

    if print_summary:
        print_run_summary(summary)

    return {
        "session": merged,
        "people_session": people_session,
        "attendance_session": attendance_session,
        "people_master": people_after_df,
        "attendance_master": attendance_after_df,
        "summary": summary,
        # enrichment inspection
        "people_enriched_keys": enriched_keys,
        "people_enriched_before": enriched_before_df,
        "people_enriched_after": enriched_after_df,
        # invalid emails
        "webinar_invalid_emails": webinar_non_emails,
    }
