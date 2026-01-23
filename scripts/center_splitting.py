import pandas as pd
from pathlib import Path


def _to_date_series(s: pd.Series) -> pd.Series:
    """Coerce a column to datetime.date (safe for mixed formats)."""
    dt = pd.to_datetime(s, format="%Y_%m_%d", errors="coerce")
    return dt.dt.date


def filter_attendance_for_dates(
    attendance_master: pd.DataFrame,
    *,
    date_col: str = "Webinar Date",
    attended_col: str = "Attended",
    include_dates: list[str] | None = None,  # ["2026-01-20","2026-01-21"]
    date_range: tuple[str, str] | None = None,  # ("2026-01-01","2026-01-31")
) -> pd.DataFrame:
    df = attendance_master.copy()

    if date_col not in df.columns:
        raise KeyError(f"attendance_master missing date column '{date_col}'")
    if attended_col not in df.columns:
        raise KeyError(f"attendance_master missing attended column '{attended_col}'")

    df["_date"] = _to_date_series(df[date_col])

    # Filter dates
    if include_dates is not None:
        inc = pd.to_datetime(pd.Series(include_dates), errors="coerce").dt.date
        inc = set(inc.dropna().tolist())
        df = df[df["_date"].isin(inc)]
    elif date_range is not None:
        start, end = date_range
        start_d = pd.to_datetime(start, errors="coerce").date()
        end_d = pd.to_datetime(end, errors="coerce").date()
        df = df[(df["_date"] >= start_d) & (df["_date"] <= end_d)]

    # Only attended
    # (handles True/False, "Yes"/"No", 1/0 — adjust if needed)
    attended = df[attended_col]
    if attended.dtype == bool:
        df = df[attended]
    else:
        df = df[
            attended.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        ]

    return df.drop(columns=["_date"])


def keep_latest_attendance_per_person(
    attendance_df: pd.DataFrame,
    *,
    key_col: str = "email_clean",
    date_col: str = "Webinar Date",
) -> pd.DataFrame:
    """
    Given attendance rows (already filtered to attended==True and desired dates),
    keep only the record with the latest webinar_date per person.

    If ties exist (same person, same date, multiple rows), keeps the first one
    after stable sorting.
    """
    df = attendance_df.copy()

    if key_col not in df.columns:
        raise KeyError(f"Missing '{key_col}'")
    if date_col not in df.columns:
        raise KeyError(f"Missing '{date_col}'")

    # Normalize to datetime for correct max
    df["_webinar_dt"] = _to_date_series(df[date_col])

    # Sort so latest date is first per person
    df = df.sort_values(
        [key_col, "_webinar_dt"], ascending=[True, False], kind="mergesort"
    )

    # Drop duplicates keeping latest
    out = df.drop_duplicates(subset=[key_col], keep="first").drop(
        columns=["_webinar_dt"]
    )

    return out.reset_index(drop=True)


def merge_people_with_attendance(
    people_master: pd.DataFrame,
    attendance_filtered: pd.DataFrame,
    *,
    key: str = "email_clean",
    people_keep_cols: list[str] | None = None,
    attendance_keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    if key not in people_master.columns:
        raise KeyError(f"people_master missing key '{key}'")
    if key not in attendance_filtered.columns:
        raise KeyError(f"attendance_master missing key '{key}'")

    p = people_master.copy()
    a = attendance_filtered.copy()

    if people_keep_cols is not None:
        keep = [key] + [c for c in people_keep_cols if c in p.columns and c != key]
        p = p[keep]

    if attendance_keep_cols is not None:
        keep = [key] + [c for c in attendance_keep_cols if c in a.columns and c != key]
        a = a[keep]

    # Many attendance rows can map to one person (multiple dates)
    out = a.merge(p, on=key, how="left", validate="m:1")
    return out


def add_final_center(
    df: pd.DataFrame,
    *,
    is_client_col: str = "Client?",
    client_center_col: str = "NS Center Abbr",
    nonclient_center_col: str = "Assigned Center Abbr",  # change to your actual column
    out_col: str = "Final Center",
) -> pd.DataFrame:
    out = df.copy()

    # default
    out[out_col] = pd.NA
    out["Center Source"] = pd.NA

    if is_client_col not in out.columns:
        # If we don't have is_client, we can still prefer client_center if present
        is_client = (
            out[client_center_col].notna()
            if client_center_col in out.columns
            else pd.Series(False, index=out.index)
        )
    else:
        is_client = out[is_client_col].fillna(False).astype(bool)

    # Clients -> assigned_center
    if client_center_col in out.columns:
        out.loc[is_client & out[client_center_col].notna(), out_col] = out.loc[
            is_client & out[client_center_col].notna(), client_center_col
        ]
        out.loc[is_client & out[client_center_col].notna(), "Center Source"] = "Client"

    # Non-clients -> inferred
    if nonclient_center_col in out.columns:
        mask = (~is_client) & out[nonclient_center_col].notna()
        out.loc[mask, out_col] = out.loc[mask, nonclient_center_col]
        out.loc[mask, "Center Source"] = "Zip_inferred"

    # Anything else -> unknown
    out.loc[out[out_col].isna(), "Center Source"] = "Unknown"

    return out


def split_by_center(
    df: pd.DataFrame,
    *,
    center_col: str = "Final Center",
) -> dict[str, pd.DataFrame]:
    """
    Returns dict: {center_name: dataframe}. Unknown/NA goes under "__UNKNOWN__".
    """
    out: dict[str, pd.DataFrame] = {}

    if center_col not in df.columns:
        raise KeyError(f"Missing '{center_col}' for center split")

    tmp = df.copy()
    tmp[center_col] = tmp[center_col].fillna("__UNKNOWN__").astype(str)

    for center, g in tmp.groupby(center_col, dropna=False):
        out[str(center)] = g.reset_index(drop=True)

    return out


def write_center_reports(
    center_dfs: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for center, df in center_dfs.items():
        safe_center = center.replace("/", "-").replace("\\", "-").strip()
        path = output_dir / f"{prefix}_{safe_center}.csv"
        df.to_csv(path, index=False)
        paths.append(path)

    return paths


def build_latest_attended_center_reports(
    *,
    attendance: pd.DataFrame,
    people: pd.DataFrame,
    output_dir: str | Path = "outputs/center_reports",
    prefix: str = "attendees_selected_dates",
    include_dates: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    # Column names (match your screenshot defaults)
    attendance_key: str = "email_clean",
    attendance_date_col: str = "Webinar Date",
    attendance_attended_col: str = "Attended",
    final_center_col: str = "Final Center",
) -> dict[str, object]:
    """
    End-to-end pipeline:
      1) filter attendance_master to selected dates + attended==True
      2) dedupe to keep latest webinar date per person
      3) merge with people_master
      4) add final center
      5) split to centers
      6) write CSVs

    Returns a dict with intermediate dfs + output paths.
    """

    # 1) Filter to selected dates + attended only
    att_filt = filter_attendance_for_dates(
        attendance_master=attendance,
        date_col=attendance_date_col,
        attended_col=attendance_attended_col,
        include_dates=include_dates,
        date_range=date_range,
    )

    # 2) Keep only latest attendance row per person
    att_latest = keep_latest_attendance_per_person(
        att_filt,
        key_col=attendance_key,
        date_col=attendance_date_col,
    )

    # 3) Merge with people_master (now 1:1)
    merged = merge_people_with_attendance(
        people_master=people,
        attendance_filtered=att_latest,
        key=attendance_key,
    )

    # 4) Compute final center + assignment source
    center_final = add_final_center(merged)

    # 5) Split into center dfs
    center_dfs = split_by_center(center_final, center_col=final_center_col)

    # 6) Write reports
    paths = write_center_reports(
        center_dfs,
        output_dir=Path(output_dir),
        prefix=prefix,
    )

    return {
        "att_filt": att_filt,
        "att_latest": att_latest,
        "merged": merged,
        "center_final": center_final,
        "center_dfs": center_dfs,
        "paths": paths,
    }
