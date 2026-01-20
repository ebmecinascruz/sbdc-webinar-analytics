# scripts/kpis.py
from __future__ import annotations
from pathlib import Path
import pandas as pd


def _make_client_lookup(people_master: pd.DataFrame) -> pd.DataFrame:
    """
    One row per email_clean with a boolean Client flag.
    """
    lookup = (
        people_master.loc[:, ["email_clean", "Client?"]]
        .dropna(subset=["email_clean"])
        .drop_duplicates("email_clean")
        .copy()
    )
    lookup["Client"] = lookup["Client?"].astype(bool)
    return lookup[["email_clean", "Client"]]


def generate_webinar_kpis(
    attendance: pd.DataFrame,
    out_dir: Path,
    people_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate webinar-level KPI table using the standing-audience model.

    If people_master is provided, also computes client vs non-client attendee counts.

    Required columns in attendance:
      - email_clean
      - Webinar ID
      - Webinar Date (YYYY_MM_DD)
      - Attended (bool-like)
      - Registration Time (timestamp)

    Required columns in people_master (if provided):
      - email_clean
      - Client? (bool-like)
    """
    df = attendance.copy()

    required = {
        "email_clean",
        "Webinar ID",
        "Webinar Date",
        "Attended",
        "Registration Time",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in attendance: {sorted(missing)}")

    # Types
    df["Webinar Date"] = pd.to_datetime(
        df["Webinar Date"], format="%Y_%m_%d", errors="coerce"
    )
    if df["Webinar Date"].isna().any():
        raise ValueError(
            "Some Webinar Date values could not be parsed (expected YYYY_MM_DD)."
        )

    df["Registration Time"] = pd.to_datetime(
        df["Registration Time"],
        format="mixed",
        errors="coerce",
    )
    if df["Registration Time"].isna().any():
        raise ValueError("Some Registration Time values could not be parsed.")

    df["Attended"] = df["Attended"].fillna(False).astype(bool)

    # Optional enrichment: client flag
    if people_master is not None:
        lookup = _make_client_lookup(people_master)
        df = df.merge(lookup, on="email_clean", how="left")
        df["Client"] = df["Client"].fillna(False).astype(bool)
    else:
        df["Client"] = False

    # First-ever attendance per person
    first_attended = (
        df.loc[df["Attended"]]
        .groupby("email_clean")["Webinar Date"]
        .min()
        .rename("first_attended_date")
    )
    df = df.merge(first_attended, on="email_clean", how="left")

    df["is_attendee"] = df["Attended"]
    df["is_first_time_attendee"] = df["Attended"] & (
        df["Webinar Date"] == df["first_attended_date"]
    )
    df["is_repeat_attendee"] = df["Attended"] & (
        df["Webinar Date"] > df["first_attended_date"]
    )

    # Base per-webinar counts (overall)
    webinar_counts = (
        df.groupby(["Webinar ID", "Webinar Date"])
        .agg(
            Attendees=(
                "email_clean",
                lambda s: s[df.loc[s.index, "Attended"]].nunique(),
            ),
            First_Time_Attendees=(
                "email_clean",
                lambda s: s[df.loc[s.index, "is_first_time_attendee"]].nunique(),
            ),
            Repeat_Attendance=(
                "email_clean",
                lambda s: s[df.loc[s.index, "is_repeat_attendee"]].nunique(),
            ),
        )
        .reset_index()
        .sort_values("Webinar Date")
        .reset_index(drop=True)
    )

    # Total audience up to that webinar date (cumulative registrations by registration DATE)
    reg_by_person = df.groupby("email_clean")["Registration Time"].min().reset_index()

    reg_by_person["registration_date"] = reg_by_person[
        "Registration Time"
    ].dt.normalize()

    reg_dates_sorted = reg_by_person["registration_date"].sort_values().to_numpy()
    webinar_dates = webinar_counts["Webinar Date"].dt.normalize().sort_values().unique()

    audience_sizes = pd.Series(
        pd.Index(reg_dates_sorted).searchsorted(webinar_dates, side="right"),
        index=webinar_dates,
        name="Total_Audience",
    )

    # map back (normalize to match index)
    webinar_counts["_web_date_norm"] = webinar_counts["Webinar Date"].dt.normalize()
    webinar_counts["Total_Audience"] = (
        webinar_counts["_web_date_norm"].map(audience_sizes).astype(int)
    )
    webinar_counts = webinar_counts.drop(columns=["_web_date_norm"])

    # Derived audience metrics
    webinar_counts["No_Show"] = (
        webinar_counts["Total_Audience"] - webinar_counts["Attendees"]
    )
    webinar_counts["Engagement_Rate"] = (
        webinar_counts["Attendees"] / webinar_counts["Total_Audience"]
    ).fillna(0.0)

    # Repeat / first-time shares
    webinar_counts["Repeat_Rate"] = (
        webinar_counts["Repeat_Attendance"] / webinar_counts["Attendees"]
    ).fillna(0.0)
    webinar_counts["First_Time_Share"] = (
        webinar_counts["First_Time_Attendees"] / webinar_counts["Attendees"]
    ).fillna(0.0)

    # Client vs non-client attendee counts
    if people_master is not None:
        # Only attendees, and count unique people per webinar by Client flag
        client_breakdown = (
            df.loc[df["Attended"]]
            .groupby(["Webinar ID", "Webinar Date", "Client"])["email_clean"]
            .nunique()
            .unstack("Client", fill_value=0)
            .rename(columns={False: "NonClient_Attendees", True: "Client_Attendees"})
            .reset_index()
        )

        webinar_counts = webinar_counts.merge(
            client_breakdown,
            on=["Webinar ID", "Webinar Date"],
            how="left",
        )

        webinar_counts["Client_Attendees"] = (
            webinar_counts["Client_Attendees"].fillna(0).astype(int)
        )
        webinar_counts["NonClient_Attendees"] = (
            webinar_counts["NonClient_Attendees"].fillna(0).astype(int)
        )

        webinar_counts["Client_Attendee_Share"] = (
            webinar_counts["Client_Attendees"] / webinar_counts["Attendees"]
        ).fillna(0.0)

    # Output
    out_dir.mkdir(parents=True, exist_ok=True)
    webinar_counts.to_csv(out_dir / "webinar_kpis.csv", index=False)

    return webinar_counts
