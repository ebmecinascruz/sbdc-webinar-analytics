import pandas as pd
from pathlib import Path
import numpy as np

from scripts.run_summary import _attendance_key
from scripts.columns import ATTENDANCE_COLS, PEOPLE_COLS


def split_people_and_attendance(
    session_df: pd.DataFrame,
    *,
    people_cols: list[str] = PEOPLE_COLS,
    attendance_cols: list[str] = ATTENDANCE_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes the final session-level output (already renamed, already cleaned),
    and returns:
      - people_df: one row per email_clean
      - attendance_df: one row per email_clean per (webinar date/id)
    """
    missing_people = [c for c in people_cols if c not in session_df.columns]
    missing_att = [c for c in attendance_cols if c not in session_df.columns]
    if missing_people:
        raise KeyError(f"Session df missing PEOPLE columns: {missing_people}")
    if missing_att:
        raise KeyError(f"Session df missing ATTENDANCE columns: {missing_att}")

    # Attendance is session-grain already
    attendance_df = session_df[attendance_cols].copy()

    # People: one row per email_clean
    people_df = (
        session_df[people_cols]
        .drop_duplicates(subset=["email_clean"], keep="first")
        .copy()
    )

    return people_df, attendance_df


def update_attendance_master(
    attendance_session: pd.DataFrame,
    *,
    master_path: str | Path = "attendance_master.csv",
) -> pd.DataFrame:
    """
    Upsert attendance rows so rerunning the same webinar session does NOT create duplicates.

    Uniqueness is enforced by _attendance_key():
      (email_clean, webinar_id/Webinar ID, webinar_date/Webinar Date) with normalization.

    Strategy:
      - Load master (if exists)
      - Concat master + session
      - Drop duplicates on _attendance_key (keep='last' so new run overwrites old)
      - Save
    """
    master_path = Path(master_path)

    master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    out_master = master.copy()
    out_session = attendance_session.copy()

    # Build one canonical key column
    out_master["_att_key"] = (
        _attendance_key(out_master)
        if not out_master.empty
        else pd.Series(dtype="string")
    )
    out_session["_att_key"] = _attendance_key(out_session)

    combined = pd.concat([out_master, out_session], ignore_index=True)

    # Keep last so the session version wins
    combined = combined.drop_duplicates(subset=["_att_key"], keep="last")

    # Stable sort for readability (by key)
    combined = combined.sort_values("_att_key").reset_index(drop=True)

    # Cleanup
    combined = combined.drop(columns=["_att_key"])

    # Make Registration Time datetime type
    combined["Registration Time"] = pd.to_datetime(
        combined["Registration Time"],
        format="mixed",
        errors="coerce",
    )

    combined.to_csv(
        master_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    return combined


def _match_strength(series: pd.Series) -> pd.Series:
    """
    Higher = stronger match.
    Accepts values like: 'email', 'name', 'none' (case-insensitive)
    """
    s = series.astype("string").str.lower().fillna("none")
    return s.map({"none": 0, "name": 1, "email": 2}).fillna(0).astype(int)


def update_people_master(
    people_session: pd.DataFrame,
    master_path: str | Path = "people_master.csv",
) -> pd.DataFrame:
    """
    Upsert people by email_clean.

    Rules:
      - For general profile fields: fill missing in master with non-missing from new.
      - For NeoSerra fields: update if:
          * master is missing, OR
          * new NS Match Type is stronger than master, OR
          * Client? flips False->True
    """
    master_path = Path(master_path)
    key = "email_clean"

    if master_path.exists():
        master = pd.read_csv(master_path)
    else:
        master = pd.DataFrame(columns=people_session.columns)

    # Outer merge to align rows for update decisions
    merged = master.merge(
        people_session,
        on=key,
        how="outer",
        suffixes=("_m", "_n"),
        indicator=True,
    )

    # Columns to manage (exclude key)
    cols = [c for c in people_session.columns if c != key]

    # Create strength + client flags for decision-making
    m_strength = _match_strength(
        merged.get("NS Match Type_m", pd.Series(dtype="string"))
    )
    n_strength = _match_strength(
        merged.get("NS Match Type_n", pd.Series(dtype="string"))
    )

    m_client = merged.get("Client?_m", pd.Series(dtype="boolean"))
    n_client = merged.get("Client?_n", pd.Series(dtype="boolean"))

    # Normalize client columns to boolean where possible
    def _to_bool(x: pd.Series) -> pd.Series:
        # handles True/False, "TRUE"/"FALSE", 1/0, blanks
        s = x.astype("string").str.strip().str.lower()
        return s.map({"true": True, "false": False}).where(s.notna(), np.nan)

    # If already boolean-like, keep
    if m_client.dtype != "bool" and m_client.dtype.name != "boolean":
        m_client_bool = _to_bool(m_client)
    else:
        m_client_bool = m_client

    if n_client.dtype != "bool" and n_client.dtype.name != "boolean":
        n_client_bool = _to_bool(n_client)
    else:
        n_client_bool = n_client

    m_client_bool = m_client_bool.astype("boolean")
    n_client_bool = n_client_bool.astype("boolean")

    client_flip = (~m_client_bool.fillna(False)) & (n_client_bool.fillna(False))
    stronger_match = n_strength > m_strength

    # Define which columns are "NeoSerra-ish"
    ns_cols = [
        c for c in cols if c.startswith("NS ") or c in ["Client?", "NS Match Type"]
    ]

    # Build output frame with final columns
    out = pd.DataFrame({key: merged[key]})

    for c in cols:
        cm = f"{c}_m"
        cn = f"{c}_n"

        mval = merged.get(cm)
        nval = merged.get(cn)

        if c in ns_cols:
            # Update NS fields if master missing OR stronger match OR client flips to True
            take_new = mval.isna() | stronger_match | client_flip
        else:
            # General fields: only fill missing
            take_new = mval.isna()

        out[c] = mval
        mask = take_new & nval.notna()
        out[c] = out[c].where(~mask, nval)

    # Save
    out.to_csv(master_path, index=False)
    return out
