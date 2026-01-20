import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import pandas as pd


def prepare_webinar_kpis_for_plotting(
    webinar_kpis: pd.DataFrame, window: int = 4
) -> pd.DataFrame:
    """
    Sorts, parses dates, and adds rolling-average columns used for plotting.

    Expects columns:
      - Webinar Date
      - Attendees
      - First_Time_Attendees
      - Repeat_Attendance
      - Engagement_Rate
    """
    df = webinar_kpis.sort_values("Webinar Date").copy()
    df["Webinar Date"] = pd.to_datetime(df["Webinar Date"])

    for col in [
        "Attendees",
        "First_Time_Attendees",
        "Repeat_Attendance",
        "Engagement_Rate",
    ]:
        df[f"{col}_roll"] = df[col].rolling(window, min_periods=1).mean()

    return df


def get_default_plot_style() -> dict:
    """
    Central place for styling knobs.
    """
    return {
        # line widths
        "raw_lw": 1.0,
        "trend_lw": 2.8,
        # alphas
        "raw_alpha_main": 0.35,
        "raw_alpha_dull": 0.20,
        "trend_alpha_main": 0.95,
        "trend_alpha_dull": 0.65,
        # COLORS
        "attendees_color": "#028F20",
        "first_time_color": "#05728d",
        "repeat_color": "#af0066",
        "rate_color": "#ff0e0e",
        # stacked bar colors
        "audience_attended_color": "#1f77b4",
        "audience_noshow_color": "#afadad",
        # stacked bar repeat and first time
        "repeat_stack_color": "#1f77b4",
        "first_time_stack_color": "#afadad",
        # stacked client vs non client colors
        "nonclient_color": "#afadad",
        "client_color": "#1f77b4",
    }


def apply_date_axis_formatting(ax, month_interval: int = 1):
    """Apply consistent monthly tick formatting and rotation."""
    month_locator = mdates.MonthLocator(interval=month_interval)
    month_fmt = mdates.DateFormatter("%b %Y")

    ax.xaxis.set_major_locator(month_locator)
    ax.xaxis.set_major_formatter(month_fmt)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def make_attendance_counts_figure(
    df: pd.DataFrame, window: int = 4, style: dict | None = None
):
    """
    Plot attendees, first-time attendees, repeat attendance (raw + rolling trend).
    Returns (fig, ax) so callers can use #plt.show() or st.pyplot(fig).
    """
    style = style or get_default_plot_style()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()

    # RAW (faint)
    ax.plot(
        df["Webinar Date"],
        df["Attendees"],
        color=style["attendees_color"],
        linewidth=style["raw_lw"],
        alpha=style["raw_alpha_main"],
        label="Attendees (raw)",
    )

    ax.plot(
        df["Webinar Date"],
        df["First_Time_Attendees"],
        color=style["first_time_color"],
        linewidth=style["raw_lw"],
        alpha=style["raw_alpha_dull"],
        label="First-time (raw)",
    )

    ax.plot(
        df["Webinar Date"],
        df["Repeat_Attendance"],
        color=style["repeat_color"],
        linewidth=style["raw_lw"],
        alpha=style["raw_alpha_dull"],
        label="Repeat (raw)",
    )

    # TREND (bold)
    ax.plot(
        df["Webinar Date"],
        df["Attendees_roll"],
        color=style["attendees_color"],
        linewidth=style["trend_lw"],
        alpha=style["trend_alpha_main"],
        label=f"Attendees (trend, {window})",
    )

    ax.plot(
        df["Webinar Date"],
        df["First_Time_Attendees_roll"],
        color=style["first_time_color"],
        linewidth=style["trend_lw"],
        alpha=style["trend_alpha_dull"],
        label=f"First-time (trend, {window})",
    )

    ax.plot(
        df["Webinar Date"],
        df["Repeat_Attendance_roll"],
        color=style["repeat_color"],
        linewidth=style["trend_lw"],
        alpha=style["trend_alpha_dull"],
        label=f"Repeat (trend, {window})",
    )

    ax.set_title("Webinar attendance over time")
    ax.set_xlabel("Webinar date")
    ax.set_ylabel("People")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    apply_date_axis_formatting(ax, month_interval=1)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # # Annotate peak TREND (less noisy than raw)
    # idx_max = df["Attendees_roll"].idxmax()
    # ax.annotate(
    #     f"Peak trend: {int(df.loc[idx_max, 'Attendees_roll'])}",
    #     xy=(df.loc[idx_max, "Webinar Date"], df.loc[idx_max, "Attendees_roll"]),
    #     xytext=(10, 10),
    #     textcoords="offset points",
    #     arrowprops=dict(arrowstyle="->", alpha=0.7),
    # )

    # Annotate peak RAW attendees (largest single webinar)
    idx_peak_raw = df["Attendees"].idxmax()
    peak_date = df.loc[idx_peak_raw, "Webinar Date"]
    peak_val = int(df.loc[idx_peak_raw, "Attendees"])

    ax.annotate(
        f"Peak attendees: {peak_val}",
        xy=(peak_date, peak_val),
        xytext=(10, -30),
        textcoords="offset points",
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            alpha=0.7,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    fig.tight_layout()
    return fig, ax


def make_engagement_rate_figure(
    df: pd.DataFrame, window: int = 4, style: dict | None = None
):
    """
    Plot engagement rate (raw + rolling trend) with percent axis formatting.
    Returns (fig, ax).
    """
    style = style or get_default_plot_style()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()

    ax.plot(
        df["Webinar Date"],
        df["Engagement_Rate"],
        color=style["rate_color"],
        linewidth=style["raw_lw"],
        alpha=0.35,
        label="Engagement rate (raw)",
    )

    ax.plot(
        df["Webinar Date"],
        df["Engagement_Rate_roll"],
        color=style["rate_color"],
        linewidth=style["trend_lw"],
        alpha=0.95,
        label=f"Engagement rate (trend, {window})",
    )

    ax.set_title("Engagement rate over time")
    ax.set_xlabel("Webinar date")
    ax.set_ylabel("Engagement rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    apply_date_axis_formatting(ax, month_interval=1)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # # Annotate peak TREND
    # idx_maxr = df["Engagement_Rate_roll"].idxmax()
    # ax.annotate(
    #     f"Peak trend: {df.loc[idx_maxr, 'Engagement_Rate_roll']:.2%}",
    #     xy=(df.loc[idx_maxr, "Webinar Date"], df.loc[idx_maxr, "Engagement_Rate_roll"]),
    #     xytext=(10, 10),
    #     textcoords="offset points",
    #     arrowprops=dict(arrowstyle="->", alpha=0.7),
    # )

    idx_peak_rate = df["Engagement_Rate"].idxmax()
    peak_date = df.loc[idx_peak_rate, "Webinar Date"]
    peak_rate = float(df.loc[idx_peak_rate, "Engagement_Rate"])

    ax.annotate(
        f"Peak rate: {df.loc[idx_peak_rate, 'Engagement_Rate']:.2%}",
        xy=(peak_date, peak_rate),
        xytext=(10, -25),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", alpha=0.7, connectionstyle="arc3,rad=0.2"),
    )

    fig.tight_layout()
    return fig, ax


def plot_webinar_kpis_pretty(webinar_kpis: pd.DataFrame, window: int = 4):
    """
    Backwards-compatible wrapper:
    prepares data, builds both figures, and shows them.
    """
    df = prepare_webinar_kpis_for_plotting(webinar_kpis, window=window)
    style = get_default_plot_style()

    fig1, _ = make_attendance_counts_figure(df, window=window, style=style)
    # plt.show()

    fig2, _ = make_engagement_rate_figure(df, window=window, style=style)
    # plt.show()
    return fig1, fig2


def plot_audience_participation_stacked(
    webinar_kpis: pd.DataFrame,
    style: dict | None = None,
):
    """
    Stacked bar chart showing total audience split into
    attendees vs no-shows per webinar.
    """
    style = style or get_default_plot_style()

    df = webinar_kpis.sort_values("Webinar Date").copy()
    df["Webinar Date"] = pd.to_datetime(df["Webinar Date"])

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 5  # days (weekly cadence)

    # Bottom: No-shows
    ax.bar(
        df["Webinar Date"],
        df["No_Show"],
        width=bar_width,
        label="Did not attend",
        color=style["audience_noshow_color"],
        alpha=0.8,
    )

    # Top: Attendees
    ax.bar(
        df["Webinar Date"],
        df["Attendees"],
        bottom=df["No_Show"],
        width=bar_width,
        label="Attended",
        color=style["audience_attended_color"],
        alpha=0.95,
    )

    ax.set_title("Audience participation per webinar")
    ax.set_xlabel("Webinar date")
    ax.set_ylabel("Total registered audience")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_attendance_composition(webinar_kpis: pd.DataFrame, style: dict | None = None):
    """
    100% stacked bar chart showing attendee composition:
    First-time vs Repeat attendees per webinar.
    """
    style = style or get_default_plot_style()

    df = webinar_kpis.sort_values("Webinar Date").copy()
    df["Webinar Date"] = pd.to_datetime(df["Webinar Date"])

    # Avoid division by zero
    df = df[df["Attendees"] > 0].copy()

    df["First_Time_Share"] = df["First_Time_Attendees"] / df["Attendees"]
    df["Repeat_Share"] = df["Repeat_Attendance"] / df["Attendees"]

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 5  # days

    # Bottom: Repeat attendees
    ax.bar(
        df["Webinar Date"],
        df["Repeat_Share"],
        width=bar_width,
        label="Repeat attendees",
        color=style["repeat_stack_color"],
        alpha=0.85,
    )

    # Top: First-time attendees
    ax.bar(
        df["Webinar Date"],
        df["First_Time_Share"],
        bottom=df["Repeat_Share"],
        width=bar_width,
        label="First-time attendees",
        color=style["first_time_stack_color"],
        alpha=0.95,
    )

    ax.set_title("Attendee composition per webinar")
    ax.set_xlabel("Webinar date")
    ax.set_ylabel("Share of attendees")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_client_composition_per_webinar(
    webinar_kpis: pd.DataFrame,
    style: dict | None = None,
):
    """
    100% stacked bar chart showing attendee composition by client status:
    Clients vs Non-clients per webinar.

    Requires columns:
      - Webinar Date
      - Attendees
      - Client_Attendees
      - NonClient_Attendees
    """
    style = style or get_default_plot_style()

    required = {"Webinar Date", "Attendees", "Client_Attendees", "NonClient_Attendees"}
    missing = required - set(webinar_kpis.columns)
    if missing:
        raise ValueError(f"Missing required columns for plot: {sorted(missing)}")

    df = webinar_kpis.sort_values("Webinar Date").copy()
    df["Webinar Date"] = pd.to_datetime(df["Webinar Date"])

    # Only webinars with attendees (avoid divide-by-zero)
    df = df[df["Attendees"] > 0].copy()

    df["Client_Share"] = df["Client_Attendees"] / df["Attendees"]
    df["NonClient_Share"] = df["NonClient_Attendees"] / df["Attendees"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 5  # days

    # Bottom: Clients
    ax.bar(
        df["Webinar Date"],
        df["Client_Share"],
        width=bar_width,
        label="Clients",
        color=style["client_color"],
        alpha=0.95,
    )

    # Top: Non-clients
    ax.bar(
        df["Webinar Date"],
        df["NonClient_Share"],
        bottom=df["Client_Share"],
        width=bar_width,
        label="Non-clients",
        color=style["nonclient_color"],
        alpha=0.85,
    )

    ax.set_title("Client composition of attendees per webinar")
    ax.set_xlabel("Webinar date")
    ax.set_ylabel("Share of attendees")

    # Percent y-axis (0–100%)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    # plt.show()
    return fig, ax
