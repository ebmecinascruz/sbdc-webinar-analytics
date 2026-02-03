import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster

from scripts.map_helper import _cluster_sum_people_icon_create_function, _add_zip_point


def add_center_legend(
    m,
    centers,
    color_map,
    center_counts,
    total_assigned,
    abbr_col="center_abbr",
    name_col="center_name",
    title="Centers (Non-clients)",
):
    rows = []
    for _, r in centers.iterrows():
        abbr = str(r.get(abbr_col, ""))
        name = str(r.get(name_col, "Center"))
        c = color_map.get(abbr, "gray")
        n = int(center_counts.get(abbr, 0))
        pct = 100 * n / total_assigned

        rows.append(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
              <span style="width:12px; height:12px; background:{c}; display:inline-block; border:1px solid #333;"></span>
              <span><b>{abbr}</b> — {name} ({n}, {pct:.1f}%)</span>
            </div>
            """
        )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 12px 14px;
        border: 1px solid #999;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-size: 13px;
        ">
      <div style="font-weight:700; margin-bottom:8px;">{title}</div>
      {"".join(rows)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def make_nonclient_zip_map_colored(
    centers: pd.DataFrame,
    non_clients_df: pd.DataFrame,  # people-level NONCLIENTS only
    zip_lookup: pd.DataFrame,  # zip-level lookup/cache with lat/lon + assignment
    out_html: str = "nonclients_zip_by_center.html",
    *,
    # centers columns
    center_lat_col: str = "lat",
    center_lon_col: str = "lon",
    center_abbr_col: str = "center_abbr",
    center_name_col: str = "center_name",
    # zip columns
    zip_col: str = "zip_clean",
    zip_lat_col: str = "zip_lat",
    zip_lon_col: str = "zip_lon",
    # assignment columns
    assigned_abbr_col_zip: str = "Assigned Center Abbr",
    assigned_abbr_col_people: str = "Assigned Center Abbr",
    distance_col: str = "distance_miles",
    # map behavior
    cluster: bool = True,
):
    """
    Non-client ZIP map color-coded by assigned center.

    Key guarantee:
    - ONLY ZIPs that appear in non_clients_df (this run) will be plotted.
      (Even if zip_lookup has historical ZIPs from old runs, they will be ignored.)

    Tooltip/popup includes:
    - ZIP
    - non-client count in that ZIP (from non_clients_df)
    - assigned center + distance (from zip_lookup)
    """

    # ----- Palette + color map -----
    palette = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkblue",
        "darkgreen",
        "pink",
        "gray",
        "black",
    ]

    center_abbrs = (
        centers[center_abbr_col].dropna().astype("string").str.strip().unique().tolist()
    )
    center_abbrs = [str(x) for x in center_abbrs]
    color_map = {
        abbr: palette[i % len(palette)] for i, abbr in enumerate(sorted(center_abbrs))
    }

    # ----- Center counts for legend (PEOPLE counts, derived from non_clients_df) -----
    center_counts = (
        non_clients_df[assigned_abbr_col_people]
        .dropna()
        .astype("string")
        .str.strip()
        .value_counts()
        .to_dict()
    )
    total_assigned = sum(center_counts.values()) or 1

    # ----- Build the "list of all nonclients" ZIPs + counts (source of truth) -----
    nonclient_zip_series = (
        non_clients_df[zip_col]
        .astype("string")
        .str.extract(r"(\d{5})", expand=False)
        .dropna()
        .str.zfill(5)
    )
    nonclient_zip_set = set(nonclient_zip_series.unique().tolist())

    zip_counts = (
        nonclient_zip_series.value_counts().to_dict()
    )  # zip -> #nonclients in that zip

    # If there are no usable ZIPs, bail early
    if not nonclient_zip_set:
        raise ValueError("No valid 5-digit ZIPs found in non_clients_df to plot.")

    # ----- Filter zip_lookup to ONLY the ZIPs present in non_clients_df -----
    zl = zip_lookup.copy()
    zl[zip_col] = (
        zl[zip_col]
        .astype("string")
        .str.extract(r"(\d{5})", expand=False)
        .astype("string")
        .str.zfill(5)
    )
    zl[assigned_abbr_col_zip] = zl[assigned_abbr_col_zip].astype("string").str.strip()

    zl = zl[
        zl[zip_col].isin(nonclient_zip_set)
        & zl[zip_lat_col].notna()
        & zl[zip_lon_col].notna()
        & zl[assigned_abbr_col_zip].notna()
    ].copy()

    if zl.empty:
        raise ValueError(
            "After filtering zip_lookup to current non-client ZIPs, nothing was left to plot. "
            "Check that zip_lookup contains lat/lon + assignments for these ZIPs."
        )

    # ----- Build base map -----
    map_center = [
        float(centers[center_lat_col].astype(float).mean()),
        float(centers[center_lon_col].astype(float).mean()),
    ]
    m = folium.Map(location=map_center, zoom_start=9, control_scale=True)

    # ----- Centers layer (house icons, color-coded) -----
    centers_fg = folium.FeatureGroup(name="Centers", show=True)

    for _, r in centers.iterrows():
        abbr = str(r.get(center_abbr_col, "")).strip()
        name = str(r.get(center_name_col, "Center")).strip()
        lat = float(r[center_lat_col])
        lon = float(r[center_lon_col])

        n_people = int(center_counts.get(abbr, 0))
        pct = 100 * n_people / total_assigned
        icon_color = color_map.get(abbr, "gray")

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(
                f"<b>{name}</b><br>{abbr}<br><b>{n_people}</b> people ({pct:.1f}%)",
                max_width=280,
            ),
            tooltip=f"{abbr} - {name} ({n_people} people, {pct:.1f}%)",
            icon=folium.Icon(icon="home", prefix="fa", color=icon_color),
        ).add_to(centers_fg)

    centers_fg.add_to(m)

    # ----- ZIP dots grouped by center -----
    for abbr, g in zl.groupby(assigned_abbr_col_zip):
        abbr = str(abbr).strip()
        fg = folium.FeatureGroup(name=f"ZIPs → {abbr}", show=True)

        icon_fn = _cluster_sum_people_icon_create_function()
        target = (
            MarkerCluster(
                icon_create_function=icon_fn,
                disable_clustering_at_zoom=13,  # <-- show individual circles earlier
                max_cluster_radius=35,  # <-- smaller = less clustering
                spiderfy_on_max_zoom=True,
                show_coverage_on_hover=False,
                zoom_to_bounds_on_click=True,
            ).add_to(fg)
            if cluster
            else fg
        )

        dot_color = color_map.get(abbr, "blue")

        for _, r in g.iterrows():
            z5 = str(r.get(zip_col, "")).strip()
            n_zip = int(zip_counts.get(z5, 0))

            dist = r.get(distance_col, None)
            dist_txt = (
                f"{float(dist):.1f} mi" if dist is not None and pd.notna(dist) else "NA"
            )

            lat = float(r[zip_lat_col])
            lon = float(r[zip_lon_col])

            popup_html = (
                f"ZIP: <b>{z5}</b>"
                f"<br>Non-clients: <b>{n_zip}</b>"
                f"<br>Assigned: <b>{abbr}</b>"
                f"<br>Distance: {dist_txt}"
            )
            tooltip = f"{z5} — {n_zip} non-clients → {abbr} ({dist_txt})"

            _add_zip_point(
                target,
                lat=lat,
                lon=lon,
                dot_color=dot_color,
                n_people=n_zip,  # <-- SUM THIS in cluster bubble
                popup_html=popup_html,
                tooltip=tooltip,
                use_cluster_marker=cluster,  # <-- important
            )

        fg.add_to(m)

    # ----- Legend -----
    add_center_legend(
        m=m,
        centers=centers,
        color_map=color_map,
        center_counts=center_counts,
        total_assigned=total_assigned,
        abbr_col=center_abbr_col,
        name_col=center_name_col,
        title="Non-clients by assigned center",
    )

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html, color_map


def make_client_zip_map_single_colored(
    *,
    people_df: pd.DataFrame,
    out_html: str = "clients_zip_footprint.html",
    zip_col: str = "zip_clean",
    zip_lat_col: str = "zip_lat",
    zip_lon_col: str = "zip_lon",
    dot_color: str = "blue",
    centers: pd.DataFrame | None = None,
    center_lat_col: str = "lat",
    center_lon_col: str = "lon",
    center_abbr_col: str = "center_abbr",
    center_name_col: str = "center_name",
):
    # 1) aggregate to ZIP-level
    zl = (
        people_df.dropna(subset=[zip_col, zip_lat_col, zip_lon_col])
        .groupby([zip_col, zip_lat_col, zip_lon_col])
        .size()
        .reset_index(name="n_people")
    )

    if zl.empty:
        raise ValueError("No ZIPs with lat/lon available to map.")

    # 2) map center based on ZIP distribution
    map_center = [
        float(zl[zip_lat_col].astype(float).mean()),
        float(zl[zip_lon_col].astype(float).mean()),
    ]
    m = folium.Map(location=map_center, zoom_start=7, control_scale=True)

    # 3) optional: centers overlay (neutral)
    if centers is not None and not centers.empty:
        centers_fg = folium.FeatureGroup(name="Centers", show=True)
        for _, r in centers.iterrows():
            abbr = str(r.get(center_abbr_col, ""))
            name = str(r.get(center_name_col, "Center"))
            lat = float(r[center_lat_col])
            lon = float(r[center_lon_col])

            folium.Marker(
                [lat, lon],
                popup=folium.Popup(f"<b>{name}</b><br>{abbr}", max_width=280),
                tooltip=f"{abbr} - {name}",
                icon=folium.Icon(icon="home", prefix="fa", color="lightgray"),
            ).add_to(centers_fg)

        centers_fg.add_to(m)

    # 4) ZIP dots (radius scaled by count)
    fg = folium.FeatureGroup(name="Client ZIPs", show=True)
    icon_fn = _cluster_sum_people_icon_create_function()
    cluster = MarkerCluster(
        icon_create_function=icon_fn,
        disable_clustering_at_zoom=13,
        max_cluster_radius=35,
        spiderfy_on_max_zoom=True,
        show_coverage_on_hover=False,
        zoom_to_bounds_on_click=True,
    ).add_to(fg)

    for _, r in zl.iterrows():
        n = int(r["n_people"])

        # radius scaling (log keeps big ZIPs from dominating)
        radius = min(18, 4 + np.log1p(n) * 4)

        lat = float(r[zip_lat_col])
        lon = float(r[zip_lon_col])

        popup_html = f"ZIP: <b>{r[zip_col]}</b><br>Clients: <b>{n}</b>"
        tooltip = f"{r[zip_col]} ({n} clients)"

        _add_zip_point(
            cluster,  # <-- cluster container
            lat=lat,
            lon=lon,
            dot_color=dot_color,
            n_people=n,  # <-- SUM THIS in cluster bubble
            popup_html=popup_html,
            tooltip=tooltip,
            use_cluster_marker=True,
        )

    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html
