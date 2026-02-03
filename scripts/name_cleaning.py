import re
import unicodedata

import pandas as pd


def _clean_spaces(s: pd.Series) -> pd.Series:
    """Trim and collaps internal whitespace to single spaces."""
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def clean_name(s: str) -> str:
    if pd.isna(s):
        return ""

    s = (
        unicodedata.normalize("NFKD", str(s).lower())
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    s = s.replace("'", "").replace("’", "")

    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def find_name_collisions(
    people: pd.DataFrame,
    *,
    name_col: str = "full_name_clean",
    email_col: str = "email_clean",
    min_distinct_emails: int = 2,
) -> tuple[list[str], pd.DataFrame]:
    """
    A "name collision" means:
      same full_name_clean appears with >= min_distinct_emails distinct email_clean values.

    Returns:
      - collision_names: list of full_name_clean values that collide
      - collisions_df: subset of people containing only those names
        plus:
          - name_count: total rows for that name
          - distinct_email_count: number of distinct emails for that name
    """
    # basic guardrails
    missing = {name_col, email_col} - set(people.columns)
    if missing:
        raise ValueError(f"people is missing columns: {sorted(missing)}")

    tmp = people[[name_col, email_col]].dropna(subset=[name_col]).copy()

    # count distinct emails per name
    distinct_email_counts = (
        tmp.groupby(name_col)[email_col]
        .nunique(dropna=True)
        .sort_values(ascending=False)
    )

    collision_names = distinct_email_counts[
        distinct_email_counts >= min_distinct_emails
    ].index.tolist()

    collisions = people[people[name_col].isin(collision_names)].copy()

    # add useful counts
    name_counts = people[name_col].value_counts(dropna=True)
    collisions["name_count"] = collisions[name_col].map(name_counts)
    collisions["distinct_email_count"] = collisions[name_col].map(distinct_email_counts)

    # sort: biggest problems first
    collisions = collisions.sort_values(
        ["distinct_email_count", "name_count", name_col],
        ascending=[False, False, True],
    )

    return collision_names, collisions


def collision_name_set(
    people: pd.DataFrame, name_col: str = "full_name_clean"
) -> set[str]:
    if people.empty or name_col not in people.columns:
        return set()

    s = people[name_col].astype("string")
    s = s[s.notna() & (s.str.strip() != "")]
    counts = s.value_counts()
    return set(counts[counts >= 2].index.tolist())
