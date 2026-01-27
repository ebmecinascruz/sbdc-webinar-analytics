from __future__ import annotations
import pandas as pd


VALID_ACTIONS = {"KEEP", "REMOVE", "ADD"}


def create_people_overwrite_from_collisions(
    collisions_df: pd.DataFrame,
    *,
    action_col: str = "action",
    review_status_col: str = "review_status",
) -> pd.DataFrame:
    """
    Return collisions_df (as-is) + review columns for humans.

    Humans will fill:
      - action: KEEP/REMOVE/ADD (ADD rows can be appended manually)
      - review_status: pending/approved/rejected (optional but recommended)
      - reason/notes
    """
    df = collisions_df.copy()

    if action_col not in df.columns:
        df[action_col] = ""
    if review_status_col not in df.columns:
        df[review_status_col] = "pending"
    if "reason" not in df.columns:
        df["reason"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""

    # Put review columns first
    front = [action_col, review_status_col, "reason", "notes"]
    remaining = [c for c in df.columns if c not in front]
    return df[front + remaining]


def update_people_overwrite_with_new_collisions(
    overwrite_df: pd.DataFrame,
    collisions_df: pd.DataFrame,
    *,
    name_col: str = "full_name_clean",
    email_col: str = "email_clean",
    action_col: str = "action",
    review_status_col: str = "review_status",
) -> pd.DataFrame:
    """
    Update an existing overwrite_df by appending any NEW collision rows
    from collisions_df (based on (full_name_clean, email_clean)).

    Preserves:
      - existing action/review_status/reason/notes
      - any ADD rows the human created
    Adds:
      - still_colliding flag for convenience
    """
    # Ensure overwrite has review cols
    ow = overwrite_df.copy()
    for col, default in [
        (action_col, ""),
        (review_status_col, "pending"),
        ("reason", ""),
        ("notes", ""),
    ]:
        if col not in ow.columns:
            ow[col] = default

    # Ensure collisions has same review cols if we append from it
    new_base = create_people_overwrite_from_collisions(
        collisions_df,
        action_col=action_col,
        review_status_col=review_status_col,
    )

    # Key for collision rows
    def _key(df: pd.DataFrame) -> pd.Series:
        return (
            df[name_col].astype(str).fillna("")
            + "||"
            + df[email_col].astype(str).fillna("")
        )

    # Identify existing collision-rows in overwrite (exclude ADD rows so they don't block keys)
    ow_actions = ow[action_col].fillna("").str.upper().str.strip()
    ow_is_add = ow_actions.eq("ADD")

    ow_keys = set(_key(ow.loc[~ow_is_add]).tolist())
    new_keys = _key(new_base).tolist()

    to_add_mask = [k not in ow_keys for k in new_keys]
    append_df = new_base.loc[to_add_mask].copy().fillna("")

    # Align columns: union of both
    all_cols = list(dict.fromkeys(list(ow.columns) + list(append_df.columns)))
    ow = ow.reindex(columns=all_cols)
    append_df = append_df.reindex(columns=all_cols)

    updated = pd.concat([ow, append_df], ignore_index=True)

    # Add 'still_colliding' visibility flag
    collision_key_set = set(_key(new_base).tolist())
    updated["still_colliding"] = _key(updated).isin(collision_key_set)

    return updated


def get_unreviewed_overwrite_rows(
    overwrite_df: pd.DataFrame,
    *,
    action_col: str = "action",
    include_add: bool = False,
) -> pd.DataFrame:
    """
    Rows needing attention: action is blank (or invalid).
    By default ignores ADD because those are manual.
    """
    df = overwrite_df.copy()
    actions = df[action_col].fillna("").str.upper().str.strip()
    invalid_or_blank = ~actions.isin(VALID_ACTIONS)

    if not include_add:
        invalid_or_blank &= ~actions.eq("ADD")

    return df[invalid_or_blank].copy()


def apply_people_overwrites(
    people_master_df: pd.DataFrame,
    people_overwrite_df: pd.DataFrame,
    *,
    email_col: str = "email_clean",
    action_col: str = "action",
    review_status_col: str = "review_status",
    require_approved: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build people_final from people_master + overwrite decisions.

    - REMOVE / KEEP apply by email_clean
    - ADD rows are appended (aligned to people_master schema)
    """
    if email_col not in people_master_df.columns:
        raise ValueError(f"people_master_df missing '{email_col}'")
    if action_col not in people_overwrite_df.columns:
        raise ValueError(f"overwrite missing '{action_col}'")

    ow = people_overwrite_df.copy()

    # approved gating (optional but recommended)
    if require_approved:
        if review_status_col not in ow.columns:
            raise ValueError(
                f"overwrite missing '{review_status_col}' (require_approved=True)"
            )
        ow = ow[ow[review_status_col].fillna("").str.lower().eq("approved")].copy()

    actions = ow[action_col].fillna("").str.upper().str.strip()
    invalid = sorted(set(actions.unique()) - (VALID_ACTIONS | {""}))
    if invalid:
        raise ValueError(f"Invalid actions in overwrite file: {invalid}")

    # emails to remove/keep
    to_remove = set(ow.loc[actions == "REMOVE", email_col].dropna().astype(str))
    to_keep = set(ow.loc[actions == "KEEP", email_col].dropna().astype(str))
    to_remove = to_remove - to_keep  # KEEP wins

    # start final as filtered master
    final_df = people_master_df.copy()
    final_df[email_col] = final_df[email_col].astype(str)

    before = len(final_df)
    if to_remove:
        final_df = final_df[~final_df[email_col].isin(to_remove)].copy()
    removed_rows = before - len(final_df)

    # ADD rows: append as new people rows (align to master columns)
    add_rows = ow.loc[actions == "ADD"].copy()
    added_rows = 0
    if not add_rows.empty:
        if email_col not in add_rows.columns:
            raise ValueError(f"ADD rows require '{email_col}'")

        # align to people_master schema (ignore extra overwrite columns)
        add_aligned = add_rows.reindex(columns=people_master_df.columns)

        # must have email
        add_aligned[email_col] = add_aligned[email_col].astype(str)
        add_aligned = add_aligned[add_aligned[email_col].str.len() > 0].copy()

        final_df = pd.concat([final_df, add_aligned], ignore_index=True)
        added_rows = len(add_aligned)

    info = {
        "removed_emails": sorted(to_remove),
        "kept_emails": sorted(to_keep),
        "removed_rows": removed_rows,
        "added_rows": added_rows,
        "final_rows": len(final_df),
    }
    return final_df, info


def apply_attendance_removals_from_people_overwrite(
    attendance_master_df: pd.DataFrame,
    people_overwrite_df: pd.DataFrame,
    *,
    email_col: str = "email_clean",
    action_col: str = "action",
    review_status_col: str = "review_status",
    require_approved: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build attendance_final by removing rows for emails marked REMOVE in people_overwrite.
    KEEP wins over REMOVE.
    """
    if email_col not in attendance_master_df.columns:
        raise ValueError(f"attendance_master_df missing '{email_col}'")
    if action_col not in people_overwrite_df.columns:
        raise ValueError(f"overwrite missing '{action_col}'")

    ow = people_overwrite_df.copy()
    if require_approved:
        if review_status_col not in ow.columns:
            raise ValueError(
                f"overwrite missing '{review_status_col}' (require_approved=True)"
            )
        ow = ow[ow[review_status_col].fillna("").str.lower().eq("approved")].copy()

    actions = ow[action_col].fillna("").str.upper().str.strip()

    to_remove = set(ow.loc[actions == "REMOVE", email_col].dropna().astype(str))
    to_keep = set(ow.loc[actions == "KEEP", email_col].dropna().astype(str))
    to_remove = to_remove - to_keep

    final_df = attendance_master_df.copy()
    final_df[email_col] = final_df[email_col].astype(str)

    before = len(final_df)
    if to_remove:
        final_df = final_df[~final_df[email_col].isin(to_remove)].copy()

    info = {
        "removed_emails": sorted(to_remove),
        "removed_rows": before - len(final_df),
        "final_rows": len(final_df),
    }
    return final_df, info
