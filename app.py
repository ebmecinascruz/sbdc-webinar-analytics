from __future__ import annotations

import os
from pathlib import Path
import re
import pandas as pd
import streamlit as st

from scripts.run_webinar_neoserra_match import (
    run_webinar_neoserra_match,
)
from scripts.center_loading import CENTERS_PATH
from scripts.name_cleaning import find_name_collisions
from scripts.overwriting import (
    create_people_overwrite_from_collisions,
    update_people_overwrite_with_new_collisions,
    get_unreviewed_overwrite_rows,
    apply_attendance_removals_from_people_overwrite,
    apply_people_overwrites,
)


from scripts.kpis import generate_webinar_kpis
from scripts.attendance_plots import (
    prepare_webinar_kpis_for_plotting,
    get_default_plot_style,
    make_attendance_counts_figure,
    make_engagement_rate_figure,
    plot_audience_participation_stacked,
    plot_attendance_composition,
    plot_client_composition_per_webinar,
)
from scripts.center_mapping import (
    map_centers_for_nonclients,
    map_centers_for_clients,
)
from scripts.center_splitting import (
    build_latest_attended_center_reports,
    _to_date_series,
)
import matplotlib.pyplot as plt


# ============================
# Page config + light styling
# ============================
st.set_page_config(page_title="SmallBiz Talks", layout="wide")

with st.sidebar:
    if st.button("Quit App"):
        os._exit(0)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      [data-testid="stMetricLabel"] > div { font-size: 0.9rem; opacity: 0.85; }
      .stAlert > div { padding-top: 0.65rem; padding-bottom: 0.65rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SmallBiz Talks — Webinar → NeoSerra Match")
st.caption("Runs locally on your computer. Files stay on this machine.")


# ============================
# Session state init
# ============================
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None

if "success_runs" not in st.session_state:
    # list[dict] where each dict holds per-success-run artifacts
    st.session_state.success_runs = []

if "output_paths" not in st.session_state:
    st.session_state.output_paths = []

if "last_run_meta" not in st.session_state:
    st.session_state.last_run_meta = {}


if "center_report_dates" not in st.session_state:
    st.session_state.center_report_dates = []


# ============================
# Helpers
# ============================
def _write_upload_to_disk(
    upload: st.runtime.uploaded_file_manager.UploadedFile, dest: Path
) -> Path:
    """Write a Streamlit upload to disk (so our pipeline can consume file paths)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(upload.getvalue())
    return dest


def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def _parse_webinar_filename(name: str) -> dict:
    """
    Best-effort parse:
      attendee_{webinar_id}_YYYY_MM_DD(.csv)
    Returns dict with webinar_id, webinar_date, and ok flag.
    """
    m = re.search(r"attendee_(?P<id>\d+?)_(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})", name)
    if not m:
        return {"ok": False, "webinar_id": None, "webinar_date": None}
    webinar_id = m.group("id")
    webinar_date = f"{m.group('y')}-{m.group('m')}-{m.group('d')}"
    return {"ok": True, "webinar_id": webinar_id, "webinar_date": webinar_date}


def render_and_close(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def save_fig_overwrite(fig, path: Path, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to disk, overwriting if it already exists.
    Also closes the figure to prevent memory leaks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _safe_multiselect_defaults(
    options: list, desired: list | None, *, fallback: str = "last"
) -> list:
    """
    Ensure multiselect defaults are always valid:
    - keep only values that exist in options
    - if nothing remains, fall back to [options[-1]] or [options[0]]
    """
    if not options:
        return []

    desired = desired or []
    opt_set = set(options)
    cleaned = [d for d in desired if d in opt_set]

    if cleaned:
        return cleaned

    return [options[-1]] if fallback == "last" else [options[0]]


def _load_masters(
    people_master_path: str,
    attendance_master_path: str,
    *,
    base_path: Path,
):
    """
    Prefer FINAL outputs if they exist (post-overwrite), else fall back to MASTER.
    Returns: (attendance_df, people_df, meta_dict)
    """
    # finals live under base_path/outputs/
    people_final_path, attendance_final_path = _final_paths(base_path)

    # choose paths
    ppl_path = (
        people_final_path if people_final_path.exists() else Path(people_master_path)
    )
    att_path = (
        attendance_final_path
        if attendance_final_path.exists()
        else Path(attendance_master_path)
    )

    if not (ppl_path.exists() and att_path.exists()):
        return None, None, {"people_source": None, "attendance_source": None}

    people_df = pd.read_csv(ppl_path)
    attendance_df = pd.read_csv(att_path)

    meta = {
        "people_source": "FINAL" if ppl_path == people_final_path else "MASTER",
        "attendance_source": "FINAL" if att_path == attendance_final_path else "MASTER",
        "people_path": str(ppl_path),
        "attendance_path": str(att_path),
    }
    return attendance_df, people_df, meta


def _final_paths(base_path: Path) -> tuple[Path, Path]:
    outputs_dir = base_path / "outputs"
    return outputs_dir / "people_final.csv", outputs_dir / "attendance_final.csv"


# ============================
# Sidebar: Inputs + settings
# ============================
with st.sidebar:
    st.header("1) Upload files")

    webinar_files = st.file_uploader(
        "Zoom webinar attendee CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or many webinar attendee exports. The pipeline will run once per file.",
    )
    neoserra_file = st.file_uploader("NeoSerra clients CSV", type=["csv"])
    centers_file = st.file_uploader(
        "Centers CSV (optional override)",
        type=["csv"],
        help="Leave blank to use the bundled centers.csv packaged with the app.",
    )

    st.caption(f"Default (bundled): {CENTERS_PATH}")

    st.divider()

    st.header("2) Output folder")
    base_dir = st.text_input(
        "Local output folder",
        value=str(Path.home() / "SmallBizTalksOutputs"),
        help="Where run outputs and review files will be saved.",
    )

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Defaults
    people_master_default = base_path / "people_master.csv"
    attendance_master_default = base_path / "attendance_master.csv"
    cache_default = base_path / "zip_to_center_lookup.csv"

    with st.expander("Advanced paths (optional)", expanded=False):
        people_master_path = st.text_input(
            "people_master.csv", value=str(people_master_default)
        )
        attendance_master_path = st.text_input(
            "attendance_master.csv", value=str(attendance_master_default)
        )
        cache_path = st.text_input(
            "zip_to_center_lookup.csv (cache)", value=str(cache_default)
        )

    st.divider()

    st.header("Batch options")
    overwrite_outputs = st.toggle(
        "Overwrite output CSVs",
        value=False,
        help="If an output CSV already exists for a webinar file, overwrite it.",
    )
    continue_on_error = st.toggle(
        "Continue on error",
        value=True,
        help="If one file fails, keep going with the rest.",
    )

    st.divider()

    run_btn = st.button("Run pipeline", type="primary", width="stretch")

    # Optional: clear stored batch results
    clear_btn = st.button("Clear last batch results", width="stretch")
    if clear_btn:
        st.session_state.batch_df = None
        st.session_state.success_runs = []
        st.session_state.output_paths = []
        st.session_state.last_run_meta = {}
        st.toast("Cleared last batch results.")


# ============================
# Main: Guided status
# ============================
st.divider()

tab_run, tab_kpis, tab_reports, tab_maps, tab_review = st.tabs(
    ["Run Pipeline", "Dashboard (KPIs)", "Center Reports", "Maps", "Batch + Review"]
)

# -------------------------
# TAB 1: Run Pipeline
# -------------------------
with tab_run:
    st.subheader("Run pipeline")
    ready = bool(webinar_files and neoserra_file)

    with st.container(border=True):
        st.subheader("How this works")
        a, b, c, d = st.columns(4)
        a.markdown("**1. Upload**")
        a.caption("Webinar file(s) + NeoSerra + Centers")
        b.markdown("**2. Run**")
        b.caption("Matches each webinar to NeoSerra")
        c.markdown("**3. Update masters**")
        c.caption("People + attendance master CSVs")
        d.markdown("**4. Review**")
        d.caption("Invalid emails, collisions, enriched deltas")

        if ready:
            st.success(f"Ready. {len(webinar_files)} webinar file(s) uploaded.")
        else:
            st.info(
                "Upload **NeoSerra** and at least **one webinar file** in the sidebar."
            )

    # ============================
    # Run (batch) — stores results into session_state
    # ============================
    if run_btn:
        if not ready:
            st.error("Please upload NeoSerra, centers, and at least one webinar file.")
            st.stop()

        run_dir = base_path / "_runs"
        run_dir.mkdir(parents=True, exist_ok=True)

        neoserra_path = _write_upload_to_disk(
            neoserra_file, run_dir / neoserra_file.name
        )
        # Centers: use bundled by default, but allow override upload
        if centers_file is not None:
            centers_path = _write_upload_to_disk(
                centers_file, run_dir / centers_file.name
            )
        else:
            centers_path = CENTERS_PATH
            if not centers_path.exists():
                st.error(f"Bundled centers.csv not found at: {centers_path}")
                st.stop()

        # Reset stored batch results (new run)
        st.session_state.batch_df = None
        st.session_state.success_runs = []
        st.session_state.output_paths = []
        st.session_state.last_run_meta = {
            "base_dir": str(base_path),
            "run_dir": str(run_dir),
            "neoserra_path": str(neoserra_path),
            "centers_path": str(centers_path),
            "people_master_path": str(people_master_path),
            "attendance_master_path": str(attendance_master_path),
            "cache_path": str(cache_path),
        }

        with st.container(border=True):
            st.subheader("Run plan")
            colL, colR = st.columns(2)
            with colL:
                st.markdown("**Inputs**")
                st.code(str(neoserra_path), language=None)
                st.code(str(centers_path), language=None)
            with colR:
                st.markdown("**Outputs**")
                st.code(str(Path(people_master_path)), language=None)
                st.code(str(Path(attendance_master_path)), language=None)
                st.code(str(Path(cache_path)), language=None)

        st.subheader("Batch run")

        progress = st.progress(0.0)
        status = st.empty()

        batch_rows: list[dict] = []
        output_paths: list[Path] = []

        total = len(webinar_files)

        for i, wf in enumerate(webinar_files, start=1):
            meta = _parse_webinar_filename(wf.name)
            label_bits = [wf.name]
            if meta["ok"]:
                label_bits.append(
                    f"(id={meta['webinar_id']} date={meta['webinar_date']})"
                )

            status.info(f"Running {i}/{total}: " + " ".join(label_bits))

            try:
                webinar_path = _write_upload_to_disk(wf, run_dir / wf.name)
                output_path = base_path / (Path(wf.name).stem + "_with_neoserra.csv")

                # skip if exists and not overwriting
                if output_path.exists() and not overwrite_outputs:
                    batch_rows.append(
                        {
                            "webinar_file": wf.name,
                            "webinar_id": meta.get("webinar_id"),
                            "webinar_date": meta.get("webinar_date"),
                            "status": "skipped (output exists)",
                            "output_csv": str(output_path),
                        }
                    )
                    output_paths.append(output_path)
                    progress.progress(i / total)
                    continue

                with st.spinner(f"Processing: {wf.name}"):
                    results = run_webinar_neoserra_match(
                        webinar_file=webinar_path,
                        neoserra_file=neoserra_path,
                        centers_file=centers_path,
                        output_path=output_path,
                        people_master_path=people_master_path,
                        attendance_master_path=attendance_master_path,
                        cache_path=cache_path,
                        print_summary=False,
                    )

                summary = results["summary"]

                # store per-run review artifacts in session_state
                st.session_state.success_runs.append(
                    {
                        "webinar_file": wf.name,
                        "output_path": str(output_path),
                        "summary": summary,
                        "results": {
                            "webinar_invalid_emails": results.get(
                                "webinar_invalid_emails"
                            ),
                            "people_name_collision_df": results.get(
                                "people_name_collision_df"
                            ),
                            "people_enriched_before": results.get(
                                "people_enriched_before"
                            ),
                            "people_enriched_after": results.get(
                                "people_enriched_after"
                            ),
                        },
                    }
                )

                batch_rows.append(
                    {
                        "webinar_file": wf.name,
                        "webinar_id": meta.get("webinar_id"),
                        "webinar_date": meta.get("webinar_date"),
                        "status": "ok",
                        "session_rows": getattr(summary, "session_rows", None),
                        "unique_emails": getattr(
                            summary, "session_unique_emails", None
                        ),
                        "attendance_added": getattr(summary, "attendance_added", None),
                        "attendance_overwritten": getattr(
                            summary, "attendance_overwritten", None
                        ),
                        "people_new": getattr(summary, "people_new", None),
                        "people_enriched": getattr(summary, "people_enriched", None),
                        "output_csv": str(output_path),
                    }
                )

                output_paths.append(output_path)

            except Exception as e:
                batch_rows.append(
                    {
                        "webinar_file": wf.name,
                        "webinar_id": meta.get("webinar_id"),
                        "webinar_date": meta.get("webinar_date"),
                        "status": f"error: {type(e).__name__}",
                        "error": str(e),
                    }
                )

                if not continue_on_error:
                    status.error(f"Stopped on error ({wf.name}): {e}")
                    break

            progress.progress(i / total)

        status.success("Batch complete.")

        st.session_state.batch_df = pd.DataFrame(batch_rows)
        st.session_state.output_paths = [str(p) for p in output_paths]

    # ============================
    # Post-batch: collisions -> overwrite -> finals
    # ============================
    with st.container(border=True):
        st.subheader("Post-batch review artifacts + final outputs")

        outputs_dir = base_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        overwrites_dir = base_path / "overwrites"
        overwrites_dir.mkdir(parents=True, exist_ok=True)

        people_master_df = None
        attendance_master_df = None
        name_collisions_final_path = None
        collision_groups_final = None
        collision_rows_final = None
        unreviewed_count = None

        # Paths
        name_collisions_master_path = outputs_dir / "name_collisions_master.csv"
        people_overwrite_path = overwrites_dir / "people_overwrite.xlsx"
        people_final_path = outputs_dir / "people_final.csv"
        attendance_final_path = outputs_dir / "attendance_final.csv"

        # 1) Load masters produced by the batch loop
        if not Path(people_master_path).exists():
            st.warning(
                "people_master.csv not found; skipping collisions/overwrites/finals."
            )
        else:
            people_master_df = pd.read_csv(people_master_path)

            attendance_master_df = None
            if Path(attendance_master_path).exists():
                attendance_master_df = pd.read_csv(attendance_master_path)

            with st.spinner("Computing name collisions master…"):
                _, collisions_df = find_name_collisions(
                    people_master_df,
                    name_col="full_name_clean",
                )
                collisions_df.to_csv(name_collisions_master_path, index=False)
                collision_groups_master = (
                    collisions_df["full_name_clean"].nunique()
                    if not collisions_df.empty
                    and "full_name_clean" in collisions_df.columns
                    else 0
                )
                collision_rows_master = len(collisions_df)

            with st.spinner("Creating/updating overwrite file…"):
                if not people_overwrite_path.exists():
                    overwrite_df = create_people_overwrite_from_collisions(
                        collisions_df
                    )
                else:
                    overwrite_existing = pd.read_excel(
                        people_overwrite_path, engine="openpyxl"
                    ).fillna("")
                    overwrite_df = update_people_overwrite_with_new_collisions(
                        overwrite_existing,
                        collisions_df,
                        name_col="full_name_clean",
                        email_col="email_clean",
                    )

                overwrite_df.to_excel(
                    people_overwrite_path, index=False, engine="openpyxl"
                )

            # Show unreviewed count
            unreviewed_df = get_unreviewed_overwrite_rows(
                overwrite_df, include_add=False
            )
            st.info(
                f"Overwrite rows needing review (blank/invalid action): {len(unreviewed_df)}"
            )
            unreviewed_count = len(unreviewed_df)

            # reviewed rows = collision rows - unreviewed rows (ignore ADD rows)
            overwrite_reviewed_rows = max(0, len(overwrite_df) - unreviewed_count)

            # 2) Apply overwrites to build finals
            with st.spinner("Building people_final + attendance_final…"):
                people_final_df, people_info = apply_people_overwrites(
                    people_master_df,
                    overwrite_df,
                    email_col="email_clean",
                    require_approved=True,
                )
                people_final_df.to_csv(people_final_path, index=False)

                att_info = None
                if attendance_master_df is not None:
                    attendance_final_df, att_info = (
                        apply_attendance_removals_from_people_overwrite(
                            attendance_master_df,
                            overwrite_df,
                            email_col="email_clean",
                            require_approved=True,
                        )
                    )
                    attendance_final_df.to_csv(attendance_final_path, index=False)

            # Compute collisions AFTER overwrites
            _, collisions_final_df = find_name_collisions(
                people_final_df,
                name_col="full_name_clean",
            )

            name_collisions_final_path = outputs_dir / "name_collisions_final.csv"
            collisions_final_df.to_csv(name_collisions_final_path, index=False)

            collision_groups_final = (
                collisions_final_df["full_name_clean"].nunique()
                if not collisions_final_df.empty
                and "full_name_clean" in collisions_final_df.columns
                else 0
            )
            collision_rows_final = len(collisions_final_df)

            # Small summary
            st.success(
                "Saved: name_collisions_master, people_overwrite, people_final"
                + (", attendance_final" if att_info else "")
            )
            st.caption(
                f"People removed: {people_info['removed_rows']} | "
                f"People added: {people_info['added_rows']} | "
                f"People final rows: {people_info['final_rows']}"
            )
            if att_info:
                st.caption(
                    f"Attendance removed: {att_info['removed_rows']} | "
                    f"Attendance final rows: {att_info['final_rows']} | "
                    f"Collisions after overwrites (FINAL): {collision_groups_final} group(s), {collision_rows_final} row(s). | "
                    f"Unreviewed overwrite rows: {unreviewed_count}."
                )

        if people_master_df is not None:
            st.session_state.last_run_meta.update(
                {
                    "name_collisions_master_path": str(name_collisions_master_path),
                    "people_overwrite_path": str(people_overwrite_path),
                    "people_final_path": str(people_final_path),
                    "attendance_final_path": str(attendance_final_path),
                }
            )

            if name_collisions_final_path is not None:
                st.session_state.last_run_meta.update(
                    {
                        "name_collisions_final_path": str(name_collisions_final_path),
                        "global_collision_groups_final": int(collision_groups_final),
                        "global_collision_rows_final": int(collision_rows_final),
                        "overwrite_unreviewed_rows": int(unreviewed_count),
                        "global_collision_groups_master": int(collision_groups_master),
                        "global_collision_rows_master": int(collision_rows_master),
                        "overwrite_reviewed_rows": int(overwrite_reviewed_rows),
                    }
                )

    # ============================
    # Render stored batch results (persists across reruns)
    # ============================
    batch_df = st.session_state.batch_df
    success_runs = st.session_state.success_runs
    stored_output_paths = (
        [Path(p) for p in st.session_state.output_paths]
        if st.session_state.output_paths
        else []
    )

# -------------------------
# TAB 2: KPIs Dashboard
# -------------------------
with tab_kpis:
    st.subheader("Webinar KPIs (overall)")

    attendance_master_df, people_master_df, src = _load_masters(
        people_master_path, attendance_master_path, base_path=base_path
    )
    st.caption(
        f"Using: people={src['people_source']}, attendance={src['attendance_source']}"
    )

    if attendance_master_df is None or people_master_df is None:
        st.info("Masters not found yet. Run the pipeline first.")
    else:
        kpi_out_dir_path = base_path / "kpis"
        kpi_out_dir_path.mkdir(parents=True, exist_ok=True)

        webinar_kpis = generate_webinar_kpis(
            attendance=attendance_master_df,
            out_dir=kpi_out_dir_path,
            people_master=people_master_df,
        )

        st.caption(f"Saved: {kpi_out_dir_path / 'webinar_kpis.csv'}")
        st.dataframe(webinar_kpis, width="stretch")

        # Figures
        df_plot = prepare_webinar_kpis_for_plotting(webinar_kpis, window=4)
        style = get_default_plot_style()

        fig_counts, _ = make_attendance_counts_figure(df_plot, window=4, style=style)
        fig_rate, _ = make_engagement_rate_figure(df_plot, window=4, style=style)

        fig_audience, _ = plot_audience_participation_stacked(webinar_kpis, style=style)
        fig_comp, _ = plot_attendance_composition(webinar_kpis, style=style)
        fig_client_comp, _ = plot_client_composition_per_webinar(
            webinar_kpis, style=style
        )

        # Save figures
        plots_dir = kpi_out_dir_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        save_fig_overwrite(fig_counts, plots_dir / "attendance_counts.png")
        save_fig_overwrite(fig_rate, plots_dir / "engagement_rate.png")
        save_fig_overwrite(fig_audience, plots_dir / "audience_participation.png")
        save_fig_overwrite(fig_comp, plots_dir / "attendance_composition.png")
        save_fig_overwrite(fig_client_comp, plots_dir / "client_composition.png")

        st.caption(f"Saved: {kpi_out_dir_path / 'plots' / 'webinar_kpis.csv'}")

        st.subheader("Attendance over time")
        render_and_close(fig_counts)

        st.subheader("Engagement rate")
        render_and_close(fig_rate)

        st.subheader("Audience participation (Total audience split)")
        render_and_close(fig_audience)

        st.subheader("Attendee composition (First-time vs Repeat)")
        render_and_close(fig_comp)

        st.subheader("Client composition of attendees")
        render_and_close(fig_client_comp)

# -------------------------
# TAB 3: Center Reports
# -------------------------
with tab_reports:
    st.subheader("Center reports (Latest attended per person)")

    attendance_master_df, people_master_df, src = _load_masters(
        people_master_path, attendance_master_path, base_path=base_path
    )
    st.caption(
        f"Using: people={src['people_source']}, attendance={src['attendance_source']}"
    )

    if attendance_master_df is None or people_master_df is None:
        st.info("Masters not found yet. Run the pipeline first.")
    else:
        DATE_COL = "Webinar Date"
        if DATE_COL not in attendance_master_df.columns:
            st.error(f"attendance_master is missing '{DATE_COL}'.")
        else:
            _dates = _to_date_series(attendance_master_df[DATE_COL])
            available_dates = sorted({d for d in _dates.dropna().tolist()})

            if not available_dates:
                st.info("No webinar dates found in attendance_master.")
            else:
                default_dates = _safe_multiselect_defaults(
                    options=available_dates,
                    desired=st.session_state.get("center_report_dates"),
                    fallback="last",  # or "first"
                )
                picked_dates = st.multiselect(
                    "Select webinar date(s) to include",
                    options=available_dates,
                    default=default_dates,
                    key="center_report_date_picker",
                    help="We keep only attended=True rows, then keep the latest date per person.",
                )
                st.session_state.center_report_dates = picked_dates

                report_prefix = st.text_input(
                    "Output file prefix",
                    value="latest_attended_selected_dates",
                    key="center_report_prefix",
                )

                report_out_dir = base_path / "center_reports" / report_prefix
                report_out_dir.mkdir(parents=True, exist_ok=True)

                run_center_report_btn = st.button(
                    "Generate center report CSVs",
                    type="primary",
                    use_container_width=True,
                    key="center_report_generate_btn",
                )

                if run_center_report_btn:
                    if not picked_dates:
                        st.error("Pick at least one webinar date.")
                        st.stop()

                    with st.spinner("Building center reports..."):
                        result = build_latest_attended_center_reports(
                            attendance=attendance_master_df,
                            people=people_master_df,
                            include_dates=[str(d) for d in picked_dates],
                            output_dir=report_out_dir,
                            prefix=report_prefix,
                            attendance_key="email_clean",
                            attendance_date_col=DATE_COL,
                            attendance_attended_col="Attended",
                            final_center_col="Final Center",
                        )

                    st.success(f"Saved {len(result['paths'])} center report file(s).")
                    st.caption(f"Folder: {report_out_dir}")

                # ---- Preview from disk (no session_state needed) ----
                st.divider()
                st.markdown("### Preview saved center reports")

                # Find CSVs matching the prefix
                csvs = sorted(report_out_dir.glob(f"{report_prefix}_*.csv"))

                if not csvs:
                    st.info(
                        "No center report CSVs found yet. Generate the report first."
                    )
                else:
                    # Show center names by stripping prefix_
                    def _label(p: Path) -> str:
                        name = p.stem  # no .csv
                        if name.startswith(report_prefix + "_"):
                            return name[len(report_prefix) + 1 :]
                        return name

                    center_options = {_label(p): p for p in csvs}
                    picked_center = st.selectbox(
                        "Preview a center",
                        options=sorted(center_options.keys()),
                        key="center_report_center_pick",
                    )

                    preview_path = center_options[picked_center]
                    st.caption(f"File: {preview_path}")

                    preview_df = pd.read_csv(preview_path)
                    st.dataframe(preview_df, width="stretch", hide_index=True)


# -------------------------
# TAB 4: Maps
# -------------------------
with tab_maps:
    st.subheader("Center mapping (clients vs non-clients)")

    meta = st.session_state.last_run_meta
    attendance_master_df, people_master_df, src = _load_masters(
        people_master_path, attendance_master_path, base_path=base_path
    )
    st.caption(
        f"Using: people={src['people_source']}, attendance={src['attendance_source']}"
    )

    required_keys = ["neoserra_path", "centers_path", "cache_path"]

    if any(k not in meta for k in required_keys):
        st.info("Run the pipeline first so inputs are available for post-processing.")
    elif people_master_df is None:
        st.info("People file not found yet. Run the pipeline first.")
    else:
        neoserra_path = Path(meta["neoserra_path"])
        centers_path = Path(meta["centers_path"])
        cache_path = Path(meta["cache_path"])

        if not neoserra_path.exists():
            st.info("NeoSerra file not found yet. Run the pipeline first.")
        elif not centers_path.exists():
            st.error(f"Centers file not found: {centers_path}")
        else:
            neoserra_raw_df = pd.read_csv(neoserra_path)
            centers_df = pd.read_csv(centers_path)

            zip_lookup_df = pd.read_csv(cache_path) if cache_path.exists() else None

            map_out_dir = base_path / "center_mapping"
            map_out_dir.mkdir(parents=True, exist_ok=True)

            out_nonclients_html = map_out_dir / "nonclients_zip_footprint.html"
            out_clients_html = map_out_dir / "clients_zip_footprint.html"

            # KEY improvement: don't auto-run unless user clicks
            if st.button("Generate maps", type="primary", use_container_width=True):
                with st.spinner("Generating non-client ZIP footprint map..."):
                    map_centers_for_nonclients(
                        people_master_df=people_master_df,
                        centers_df=centers_df,
                        zip_lookup_df=zip_lookup_df,
                        raw_zip_col="Zip/Postal Code",
                        out_html=out_nonclients_html,
                    )

                with st.spinner("Generating client ZIP footprint map..."):
                    map_centers_for_clients(
                        neoserra_df=neoserra_raw_df,
                        raw_zip_col="Physical Address ZIP Code",
                        out_html=out_clients_html,
                    )

                st.success("Center maps generated and saved.")

            show_prev = st.toggle("Preview latest maps", value=True)
            if show_prev:
                if out_nonclients_html.exists():
                    st.markdown("### Non-clients ZIP footprint")
                    st.components.v1.html(
                        out_nonclients_html.read_text(encoding="utf-8"), height=650
                    )
                else:
                    st.info("Non-client map not generated yet.")

                if out_clients_html.exists():
                    st.markdown("### Clients ZIP footprint")
                    st.components.v1.html(
                        out_clients_html.read_text(encoding="utf-8"), height=650
                    )
                else:
                    st.info("Client map not generated yet.")


# -------------------------
# TAB 5: Batch + Review
# -------------------------
with tab_review:
    st.subheader("Batch results + review")

    # --- Global (FINAL) collision status + overwrite status ---
    meta = st.session_state.last_run_meta or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Collisions in MASTER (groups)", meta.get("global_collision_groups_master", "—")
    )
    m2.metric("Collisions reviewed (rows)", meta.get("overwrite_reviewed_rows", "—"))
    m3.metric(
        "Collisions in FINAL (groups)", meta.get("global_collision_groups_final", "—")
    )
    m4.metric("Unreviewed (rows)", meta.get("overwrite_unreviewed_rows", "—"))

    if batch_df is None:
        st.info("No batch results yet. Run the pipeline first.")
    else:
        ok_count = (
            int((batch_df["status"] == "ok").sum())
            if "status" in batch_df.columns
            else 0
        )
        err_count = (
            int(batch_df["status"].astype(str).str.startswith("error").sum())
            if "status" in batch_df.columns
            else 0
        )
        skip_count = (
            int(batch_df["status"].astype(str).str.startswith("skipped").sum())
            if "status" in batch_df.columns
            else 0
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Succeeded", f"{ok_count:,}")
        k2.metric("Skipped", f"{skip_count:,}")
        k3.metric("Errors", f"{err_count:,}")
        k4.metric("Total files", f"{len(batch_df):,}")

        st.dataframe(batch_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Review (choose a successful run)")

    if not success_runs:
        st.info("No successful runs to review (yet).")
    else:
        labels = [
            f"{r['webinar_file']}  →  {Path(r['output_path']).name}"
            for r in success_runs
        ]
        idx = st.selectbox(
            "Select a successful run",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="review_pick",  # key makes selection stable
        )

        picked = success_runs[idx]
        picked_summary = picked["summary"]
        picked_results = picked["results"]
        picked_output_path = picked["output_path"]

        st.caption(f"Reviewing: `{picked_output_path}`")

        s_top = st.columns(4)
        s_top[0].metric(
            "Session rows", _fmt_int(getattr(picked_summary, "session_rows", None))
        )
        s_top[1].metric(
            "Unique emails",
            _fmt_int(getattr(picked_summary, "session_unique_emails", None)),
        )
        s_top[2].metric(
            "Attendance added",
            _fmt_int(getattr(picked_summary, "attendance_added", None)),
        )
        s_top[3].metric(
            "Attendance overwritten",
            _fmt_int(getattr(picked_summary, "attendance_overwritten", None)),
        )

        s_bot = st.columns(2)
        s_bot[0].metric(
            "People new", _fmt_int(getattr(picked_summary, "people_new", None))
        )
        s_bot[1].metric(
            "People enriched",
            _fmt_int(getattr(picked_summary, "people_enriched", None)),
        )

        tab_enriched, tab_invalid = st.tabs(["Enriched deltas", "Invalid emails"])

        with tab_enriched:
            before_df = picked_results.get("people_enriched_before")
            after_df = picked_results.get("people_enriched_after")

            if isinstance(before_df, pd.DataFrame) and not before_df.empty:
                st.caption(f"{len(before_df):,} enriched rows (before → after)")
                colL, colR = st.columns(2)
                with colL:
                    st.markdown("**Before**")
                    st.dataframe(before_df, width="stretch", hide_index=True)
                with colR:
                    st.markdown("**After**")
                    st.dataframe(after_df, width="stretch", hide_index=True)
            else:
                st.success("No enriched deltas for this run.")

        with tab_invalid:
            invalid_df = picked_results.get("webinar_invalid_emails")
            if isinstance(invalid_df, pd.DataFrame) and not invalid_df.empty:
                st.caption(f"{len(invalid_df):,} rows")
                st.dataframe(invalid_df, width="stretch", hide_index=True)
            else:
                st.success("No invalid emails returned for this run.")


# ---------------------------
# Helpful footer (always)
# ---------------------------
st.caption(
    "Tip: keep this app local-only. Don't deploy to public hosting with client data."
)
