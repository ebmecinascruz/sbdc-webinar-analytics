# scripts/run_summary.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class RunSummary:
    # Identity
    webinar_id: str
    webinar_date: str

    # Session-level
    session_rows: int
    session_unique_emails: int

    # Attendance master deltas
    attendance_before: int
    attendance_after: int
    attendance_added: int
    attendance_overwritten: int

    # People master deltas
    people_before: int
    people_after: int
    people_new: int
    people_enriched: int

    # People master collisions
    people_name_collision_groups: int
    people_name_collision_rows: int
    people_name_collision_new_groups: int
    # people_name_collision_resolved_groups: int


def _pick_col(df: pd.DataFrame, options: list[str]) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of these columns found: {options}. Columns: {list(df.columns)}"
    )


def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _norm_date(s: pd.Series) -> pd.Series:
    x = _norm_str(s).str.replace("_", "-", regex=False)
    dt = pd.to_datetime(x, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d").astype("string").fillna("")


def _attendance_key(df: pd.DataFrame) -> pd.Series:
    email_c = _pick_col(df, ["email_clean", "Email Clean", "email", "Email"])
    wid_c = _pick_col(df, ["Webinar ID", "webinar_id"])
    wdt_c = _pick_col(df, ["Webinar Date", "webinar_date"])

    return (
        _norm_str(df[email_c])
        + "||"
        + _norm_str(df[wid_c])
        + "||"
        + _norm_date(df[wdt_c])
    )


def find_people_enriched(
    people_before_df: pd.DataFrame,
    people_after_df: pd.DataFrame,
    *,
    key: str = "email_clean",
) -> list[str]:
    """
    Return list of person keys that gained at least one previously-missing value.
    """
    if people_before_df.empty or people_after_df.empty:
        return []

    b = people_before_df.drop_duplicates(subset=[key], keep="last").set_index(key)
    a = people_after_df.drop_duplicates(subset=[key], keep="last").set_index(key)

    common = b.index.intersection(a.index)
    if len(common) == 0:
        return []

    cols = [c for c in a.columns if c in b.columns]

    def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
        m = df[cols].isna()
        s = df[cols].astype("string")
        empty = s.apply(lambda col: col.str.strip().eq(""))
        return m | empty

    b_missing = missing_mask(b.loc[common])
    a_missing = missing_mask(a.loc[common])

    gained_any = (b_missing & ~a_missing).any(axis=1)

    return gained_any[gained_any].index.tolist()


def get_enriched_deltas(
    people_before_df: pd.DataFrame,
    people_after_df: pd.DataFrame,
    enriched_keys: list[str],
    *,
    key: str = "email_clean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (before_df, after_df) for enriched people only.
    """
    before = (
        people_before_df.loc[people_before_df[key].isin(enriched_keys)]
        .sort_values(key)
        .copy()
    )

    after = (
        people_after_df.loc[people_after_df[key].isin(enriched_keys)]
        .sort_values(key)
        .copy()
    )

    return before, after


def print_run_summary(s: RunSummary) -> None:
    print("\n" + "=" * 60)
    print("SmallBiz Talks — Run Summary")
    print("=" * 60)
    print(f"Webinar ID:   {s.webinar_id}")
    print(f"Webinar Date: {s.webinar_date}")
    print("-" * 60)
    print(f"Session rows (final):        {s.session_rows:,}")
    print(f"Unique emails in session:    {s.session_unique_emails:,}")
    print("-" * 60)
    print("Attendance master:")
    print(f"  Before:                   {s.attendance_before:,}")
    print(f"  Added (new keys):          {s.attendance_added:,}")
    print(f"  Overwritten (re-run keys): {s.attendance_overwritten:,}")
    print(f"  After:                    {s.attendance_after:,}")
    print("-" * 60)
    print("People master:")
    print(f"  Before:                   {s.people_before:,}")
    print(f"  New people inserted:       {s.people_new:,}")
    print(f"  Existing people enriched:  {s.people_enriched:,}")
    print(f"  After:                    {s.people_after:,}")

    # Name collisions
    if s.people_name_collision_groups > 0:
        delta = s.people_name_collision_new_groups
        delta_str = f"(+{delta:,} new)" if delta > 0 else "(+0 new)"
        print(
            f"  Name collisions:        "
            f"{s.people_name_collision_groups:,} names "
            f"({s.people_name_collision_rows:,} rows) {delta_str}"
        )
    else:
        print("  Name collisions:           None")

    print("=" * 60 + "\n")
