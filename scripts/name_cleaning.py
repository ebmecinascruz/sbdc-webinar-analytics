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
    name_col: str = "full_name_clean",
    min_count: int = 2,
) -> tuple[list[str], pd.DataFrame]:
    """
    Returns:
      - dup_names: list of names that occur >= min_count times
      - collisions_df: subset of people with those names + name_count
    """
    counts = people[name_col].value_counts(dropna=True)
    dup_names = counts[counts >= min_count].index.tolist()

    collisions = people[people[name_col].isin(dup_names)].copy()
    collisions["name_count"] = collisions[name_col].map(counts)

    collisions = collisions.sort_values(
        ["name_count", name_col], ascending=[False, True]
    )

    return dup_names, collisions


def collision_name_set(
    people: pd.DataFrame, name_col: str = "full_name_clean"
) -> set[str]:
    if people.empty or name_col not in people.columns:
        return set()

    s = people[name_col].astype("string")
    s = s[s.notna() & (s.str.strip() != "")]
    counts = s.value_counts()
    return set(counts[counts >= 2].index.tolist())
