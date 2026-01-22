from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import re
import pandas as pd
import streamlit as st

from scripts.run_webinar_neoserra_match import (
    run_webinar_neoserra_match,
)
from scripts.center_loading import CENTERS_PATH


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

import matplotlib.pyplot as plt


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
    # small dict with run directory paths, etc.
    st.session_state.last_run_meta = None


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
        st.session_state.last_run_meta = None
        st.toast("Cleared last batch results.")


# ============================
# Main: Guided status
# ============================
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
        st.info("Upload **NeoSerra** and at least **one webinar file** in the sidebar.")


# ============================
# Run (batch) — stores results into session_state
# ============================
if run_btn:
    if not ready:
        st.error("Please upload NeoSerra, centers, and at least one webinar file.")
        st.stop()

    run_dir = base_path / "_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    neoserra_path = _write_upload_to_disk(neoserra_file, run_dir / neoserra_file.name)
    # Centers: use bundled by default, but allow override upload
    if centers_file is not None:
        centers_path = _write_upload_to_disk(centers_file, run_dir / centers_file.name)
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
            label_bits.append(f"(id={meta['webinar_id']} date={meta['webinar_date']})")

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
                        "webinar_invalid_emails": results.get("webinar_invalid_emails"),
                        "people_name_collision_df": results.get(
                            "people_name_collision_df"
                        ),
                        "people_enriched_before": results.get("people_enriched_before"),
                        "people_enriched_after": results.get("people_enriched_after"),
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
                    "unique_emails": getattr(summary, "session_unique_emails", None),
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
# Render stored batch results (persists across reruns)
# ============================
batch_df = st.session_state.batch_df
success_runs = st.session_state.success_runs
stored_output_paths = (
    [Path(p) for p in st.session_state.output_paths]
    if st.session_state.output_paths
    else []
)

# ---- KPI summary (computed ONCE from disk masters) ----
st.divider()
st.subheader("Webinar KPIs (overall)")

att_master_path = Path(attendance_master_path)
ppl_master_path = Path(people_master_path)

if att_master_path.exists() and ppl_master_path.exists():
    attendance_master_df = pd.read_csv(att_master_path)
    people_master_df = pd.read_csv(ppl_master_path)

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
    fig_client_comp, _ = plot_client_composition_per_webinar(webinar_kpis, style=style)

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


else:
    st.info("Masters not found yet. Run the pipeline first.")


# ============================
# Post-processing: Center mapping (auto-run)
# ============================
st.divider()
st.subheader("Center mapping (clients vs non-clients)")

meta = st.session_state.last_run_meta
if not meta:
    st.info("Run the pipeline first so inputs are available for post-processing.")
else:
    neoserra_path = Path(meta["neoserra_path"])
    centers_path = Path(meta["centers_path"])
    cache_path = Path(meta["cache_path"])

    if not neoserra_path.exists():
        st.info("NeoSerra file not found yet. Run the pipeline first.")
    elif not centers_path.exists():
        st.error(f"Centers file not found: {centers_path}")
    else:
        # Load inputs
        neoserra_raw_df = pd.read_csv(neoserra_path)
        centers_df = pd.read_csv(centers_path)

        zip_lookup_df = None
        if cache_path.exists():
            zip_lookup_df = pd.read_csv(cache_path)

        map_out_dir = base_path / "center_mapping"
        map_out_dir.mkdir(parents=True, exist_ok=True)

        out_nonclients_html = map_out_dir / "nonclients_zip_footprint.html"
        out_clients_html = map_out_dir / "clients_zip_footprint.html"

        # Auto-run mapping every time this section is reached
        with st.spinner("Generating non-client ZIP footprint map..."):
            _nonclients_html_path, _ = map_centers_for_nonclients(
                people_master_df=people_master_df,
                centers_df=centers_df,
                zip_lookup_df=zip_lookup_df,  # ok if None
                raw_zip_col="Zip/Postal Code",
                out_html=out_nonclients_html,
            )

        with st.spinner("Generating client ZIP footprint map..."):
            _clients_html_path = map_centers_for_clients(
                neoserra_df=neoserra_raw_df,
                raw_zip_col="Physical Address ZIP Code",
                out_html=out_clients_html,
            )

        st.success("Center maps generated and saved.")
        st.caption(f"Saved: {out_nonclients_html}")
        st.caption(f"Saved: {out_clients_html}")

        show_prev = st.toggle("Preview latest maps", value=True)
        if show_prev:
            if out_nonclients_html.exists():
                st.markdown("### Non-clients ZIP footprint")
                st.components.v1.html(
                    out_nonclients_html.read_text(encoding="utf-8"),
                    height=650,
                )

            if out_clients_html.exists():
                st.markdown("### Clients ZIP footprint")
                st.components.v1.html(
                    out_clients_html.read_text(encoding="utf-8"),
                    height=650,
                )


if batch_df is not None:
    st.divider()
    st.subheader("Batch results")

    ok_count = (
        int((batch_df["status"] == "ok").sum()) if "status" in batch_df.columns else 0
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

        s_bot = st.columns(4)
        s_bot[0].metric(
            "People new", _fmt_int(getattr(picked_summary, "people_new", None))
        )
        s_bot[1].metric(
            "People enriched",
            _fmt_int(getattr(picked_summary, "people_enriched", None)),
        )
        s_bot[2].metric(
            "Collision groups",
            _fmt_int(getattr(picked_summary, "people_name_collision_groups", None)),
        )
        s_bot[3].metric(
            "Collision rows",
            _fmt_int(getattr(picked_summary, "people_name_collision_rows", None)),
        )

        with st.expander("Raw summary (debug)", expanded=False):
            try:
                st.json(asdict(picked_summary))
            except Exception:
                st.write(picked_summary)

        tab_collisions, tab_enriched, tab_invalid = st.tabs(
            ["Name collisions", "Enriched deltas", "Invalid emails"]
        )

        with tab_collisions:
            coll_df = picked_results.get("people_name_collision_df")
            if isinstance(coll_df, pd.DataFrame) and not coll_df.empty:
                st.caption(f"{len(coll_df):,} rows")
                q = st.text_input(
                    "Filter (contains)", value="", key=f"coll_filter_{idx}"
                )
                view = coll_df
                if q.strip():
                    mask = view.astype(str).apply(
                        lambda s: s.str.contains(q, case=False, na=False)
                    )
                    view = view[mask.any(axis=1)]
                st.dataframe(view, width="stretch", hide_index=True)
            else:
                st.success("No name collisions in this run.")

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
