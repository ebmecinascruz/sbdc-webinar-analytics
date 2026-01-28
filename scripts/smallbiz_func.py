import pandas as pd
from pathlib import Path

from scripts.attendance_cleaning import (
    add_email_clean,
    add_full_name,
    normalize_attended,
    parse_attendance_filename,
    split_invalid_emails_from_clean,
)
from scripts.columns import WEBINAR_KEEP_COLS
from scripts.webinar_cleaning import ensure_state_from_zip


def detect_zoom_header_skiprows(
    file_path: str | Path,
    *,
    marker: str = "Attendee Details",
    max_lines: int = 200,
    encoding: str = "utf-8-sig",
) -> int:
    """
    Returns the number of rows to skip so that pd.read_csv reads the real header row.

    Zoom exports often have metadata rows, then a line containing 'Attendee Details',
    then the real header row (e.g., 'First Name,Last Name,Email,...') on the next line.

    We return i+1 where i is the line index containing the marker.
    """
    file_path = Path(file_path)

    with file_path.open("r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if marker.lower() in line.lower():
                return i + 1  # next line is the header row

    raise ValueError(
        f"Could not find marker '{marker}' in first {max_lines} lines of {file_path}"
    )


def process_zoom_attendance_file_full(
    file_path: str | Path,
    *,
    skiprows: int | None = None,
    email_col_in: str = "Email",
    attended_col_in: str = "Attended",
    webinar_keep_cols: list[str] = WEBINAR_KEEP_COLS,
    approval_col_in: str = "Approval Status",
    drop_cancelled_registrations: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads a Zoom attendee export and returns:
      (valid_df, invalid_df)

    Both outputs:
      - keep selected webinar columns
      - include webinar_id + webinar_date
      - include email_clean
      - include full_name (if First/Last exist)
      - include attended normalized to boolean where possible
    """
    if skiprows is None:
        skiprows = detect_zoom_header_skiprows(file_path)

    df = pd.read_csv(file_path, skiprows=skiprows, index_col=False)

    # ----------------------------
    # Drop cancelled registrations
    # ----------------------------
    if drop_cancelled_registrations and approval_col_in in df.columns:
        _approval = df[approval_col_in].astype(str).str.strip().str.lower()
        # Keep ONLY approved (drops "cancelled by self/host" and anything else)
        df = df[_approval == "approved"].copy()

    # add missing column
    df = ensure_state_from_zip(df)
    # Keep only columns that exist (Zoom changes headers sometimes)
    keep = [c for c in webinar_keep_cols if c in df.columns]
    out = df[keep].copy()

    webinar_id, date_suffix = parse_attendance_filename(file_path)
    out["webinar_id"] = webinar_id
    out["webinar_date"] = date_suffix  # YYYY_MM_DD string for now

    # Build clean email
    out = add_email_clean(out, email_col=email_col_in, out_col="email_clean")

    # Optional: make full name early (helps review queues too)
    if "First Name" in out.columns and "Last Name" in out.columns:
        out = add_full_name(out, first_col="First Name", last_col="Last Name")

    # Standardize attended column name BEFORE any split
    if attended_col_in in out.columns and attended_col_in != "attended":
        out = out.rename(columns={attended_col_in: "attended"})

    # Normalize attended for ALL rows (valid + invalid email rows)
    if "attended" in out.columns:
        out = normalize_attended(out, attended_col="attended")

        # HARD GUARD: if anything non-bool survives, tag it now
        bad_mask = out["attended"].notna() & ~out["attended"].isin([True, False])
        if bad_mask.any():
            # Keep them in the invalid bucket by forcing email_clean to NA (optional)
            # but at minimum, label them:
            out.loc[bad_mask, "attended_invalid_flag"] = True
        else:
            out["attended_invalid_flag"] = False

    # Now split invalid emails using the CLEAN COLUMN (IMPORTANT)
    valid_df, invalid_df = split_invalid_emails_from_clean(
        out, email_clean_col="email_clean"
    )

    return valid_df, invalid_df
