import pandas as pd
from pathlib import Path
from scripts.colored_zip import make_nonclient_zip_map_colored
from scripts.zip_codes import clean_zip_5
from scripts.name_cleaning import _clean_spaces
from scripts.neoserra_helper import add_zip_geography
from scripts.neoserra_cleaning import center_map
from scripts.colored_zip import make_client_zip_map_single_colored


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
