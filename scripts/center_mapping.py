import pandas as pd
import numpy as np
from pathlib import Path
import folium
from folium.plugins import MarkerCluster

from scripts.colored_zip import make_nonclient_zip_map_colored
from scripts.zip_codes import clean_zip_5
from scripts.name_cleaning import _clean_spaces
from scripts.neoserra_helper import add_zip_geography
from scripts.neoserra_cleaning import center_map
from scripts.colored_zip import make_client_zip_map_single_colored
from scripts.map_helper import _cluster_sum_people_icon_create_function, _add_zip_point


def map_centers_for_nonclients(
    *,
    people_master_df: pd.DataFrame,
    centers_df: pd.DataFrame,
    zip_lookup_df: pd.DataFrame | None = None,
    raw_zip_col: str = "Zip/Postal Code",
    out_html: str | Path = "nonclients_zip_footprint.html",
) -> tuple[Path, dict]:
    """
    Create a ZIP footprint map for NON-clients only.

    Returns
    -------
    out_html_path : Path
        Path to the generated HTML map.
    legend : dict
        Center → color mapping used in the map.
    """
    # 1) ZIP cleanup
    people_with_zip = clean_zip_5(
        people_master_df,
        raw_zip_col=raw_zip_col,
    )

    # 2) Filter non-clients
    non_clients = people_with_zip[~people_with_zip["Client?"]].copy()

    # 3) Make map
    out_html_path, legend = make_nonclient_zip_map_colored(
        non_clients_df=non_clients,
        centers=centers_df,
        zip_lookup=zip_lookup_df,
        out_html=str(out_html),
    )

    return Path(out_html_path), legend


def map_centers_for_clients(
    *,
    neoserra_df: pd.DataFrame,
    raw_zip_col: str = "Physical Address ZIP Code",
    out_html: str | Path = "clients_zip_footprint.html",
) -> Path:
    """
    Create a ZIP footprint map for CLIENTS (NeoSerra).

    Returns
    -------
    Path to generated HTML map.
    """
    ns = neoserra_df.copy()

    # Clean center name spacing
    ns["Center"] = _clean_spaces(ns["Center"])

    # Add center abbreviation
    ns["Center Abbr"] = ns["Center"].map(center_map).fillna(ns["Center"])

    # Add ZIP geography
    ns_zip = add_zip_geography(
        ns,
        raw_zip_col=raw_zip_col,
    )

    # Make map
    out_html_path = make_client_zip_map_single_colored(
        people_df=ns_zip,
        out_html=str(out_html),
    )

    return Path(out_html_path)


def make_all_attendees_zip_map_single_colored(
    *,
    people_df: pd.DataFrame,
    out_html: str = "all_attendees_zip_footprint.html",
    zip_col: str = "zip_clean",
    zip_lat_col: str = "zip_lat",
    zip_lon_col: str = "zip_lon",
    dot_color: str = "purple",
    centers: pd.DataFrame | None = None,
    center_lat_col: str = "lat",
    center_lon_col: str = "lon",
    center_abbr_col: str = "center_abbr",
    center_name_col: str = "center_name",
):
    # 1) aggregate to ZIP-level (COUNT PEOPLE per ZIP)
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

    # 4) ZIP dots clustered (SUM counts in cluster bubble)
    fg = folium.FeatureGroup(name="Attendee ZIPs", show=True)
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
        lat = float(r[zip_lat_col])
        lon = float(r[zip_lon_col])

        popup_html = f"ZIP: <b>{r[zip_col]}</b><br>Attendees: <b>{n}</b>"
        tooltip = f"{r[zip_col]} ({n} attendees)"  # hover shows the count

        _add_zip_point(
            cluster,
            lat=lat,
            lon=lon,
            dot_color=dot_color,
            n_people=n,
            popup_html=popup_html,
            tooltip=tooltip,
            use_cluster_marker=True,
        )

    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html


def map_centers_for_run_clients(
    *,
    people_df: pd.DataFrame,
    out_html: str,
    zip_col: str = "zip_clean",
    zip_lat_col: str = "zip_lat",
    zip_lon_col: str = "zip_lon",
    dot_color: str = "blue",
):
    """
    Client ZIP footprint for clients that exist in THIS RUN.
    Mirrors map_centers_for_clients, but operates on people_master_df.
    """
    make_client_zip_map_single_colored(
        people_df=people_df,
        out_html=out_html,
        zip_col=zip_col,
        zip_lat_col=zip_lat_col,
        zip_lon_col=zip_lon_col,
        dot_color=dot_color,
        centers=None,
    )
