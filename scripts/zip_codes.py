import pandas as pd
import numpy as np
import os
import pgeocode

ALLOWED_STATES = {"CA"}
ALLOWED_COUNTIES = {"Los Angeles", "Ventura", "Santa Barbara"}

ZIP_COLS = [
    "zip_clean",
    "zip_lat",
    "zip_lon",
    "zip_state",
    "zip_county",
    "Assigned Center Abbr",
    "Assigned Center Name",
    "distance_miles",
]


def clean_zip_5(
    df: pd.DataFrame,
    raw_zip_col: str,
    out_col: str = "zip_clean",
) -> pd.DataFrame:
    """
    Normalize ZIP codes to 5-digit US ZIPs.

    Handles:
      - ZIP+4 (90814-8124 → 90814)
      - Excel artifacts (="90814-8124")
      - Extra whitespace / text
    """
    out = df.copy()

    out[out_col] = (
        out[raw_zip_col]
        .astype("string")
        # Remove Excel ="..." wrapper if present
        .str.replace(r'^="?|"$', "", regex=True)
        # Extract first 5 digits (US ZIP)
        .str.extract(r"(\d{5})", expand=False)
        .astype("string")
    )

    return out


def build_zip_ref_pgeocode(unique_zips: list[str]) -> pd.DataFrame:
    nomi = pgeocode.Nominatim("US")
    ref = nomi.query_postal_code(unique_zips).reset_index()

    # index column differs by version
    if "postal_code" not in ref.columns and "index" in ref.columns:
        ref = ref.rename(columns={"index": "postal_code"})

    ref = ref.rename(
        columns={
            "postal_code": "zip_clean",
            "latitude": "zip_lat",
            "longitude": "zip_lon",
            "state_code": "zip_state",
            "county_name": "zip_county",
            "county_code": "zip_county_code",
        }
    )

    ref["zip_clean"] = ref["zip_clean"].astype("string").str.strip().str.zfill(5)
    ref["zip_county"] = ref["zip_county"].astype("string").str.strip()

    return ref[
        [
            "zip_clean",
            "zip_lat",
            "zip_lon",
            "zip_state",
            "zip_county",
            "zip_county_code",
        ]
    ]


def add_zip_problems(
    df: pd.DataFrame,
    zip_col: str = "zip_clean",
    state_col: str = "zip_state",
    county_col: str = "zip_county",
    lat_col: str = "zip_lat",
) -> pd.DataFrame:
    out = df.copy()

    out["Zip Problem"] = "no_problem"

    # 1) null zip
    out.loc[out[zip_col].isna(), "Zip Problem"] = "zip_missing"

    # 2) invalid / not found (pgeocode returns NaN lat when unknown)
    out.loc[out[zip_col].notna() & out[lat_col].isna(), "Zip Problem"] = (
        "zip_invalid_or_not_found"
    )

    # 3) not CA (only if valid)
    valid = out[zip_col].notna() & out[lat_col].notna()
    out.loc[valid & (~out[state_col].isin(ALLOWED_STATES)), "Zip Problem"] = (
        "zip_not_in_ca"
    )

    # 4) outside service counties (only if valid + in CA)
    in_ca = valid & out[state_col].isin(ALLOWED_STATES)
    out.loc[in_ca & (~out[county_col].isin(ALLOWED_COUNTIES)), "Zip Problem"] = (
        "zip_outside_service_counties"
    )

    return out


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 3958.7613
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_zip_to_center(
    zip_ref_ok: pd.DataFrame, centers: pd.DataFrame
) -> pd.DataFrame:
    z = (
        zip_ref_ok.dropna(subset=["zip_clean", "zip_lat", "zip_lon"])
        .drop_duplicates("zip_clean")
        .copy()
    )
    if z.empty:
        return z.assign(
            assigned_center_abbr=pd.NA, assigned_center_name=pd.NA, distance_miles=pd.NA
        )

    lat1 = z["zip_lat"].to_numpy()
    lon1 = z["zip_lon"].to_numpy()

    dist_mat = np.vstack(
        [haversine_miles(lat1, lon1, c["lat"], c["lon"]) for _, c in centers.iterrows()]
    ).T

    idx = dist_mat.argmin(axis=1)
    z["distance_miles"] = dist_mat.min(axis=1)
    z["Assigned Center Abbr"] = centers.iloc[idx]["center_abbr"].to_numpy()
    z["Assigned Center Name"] = centers.iloc[idx]["center_name"].to_numpy()

    return z[ZIP_COLS]


def update_zip_center_cache(
    zip_to_center_new: pd.DataFrame,
    cache_path: str,
) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype={"zip_clean": "string"})
        cache["zip_clean"] = cache["zip_clean"].astype("string").str.zfill(5)
    else:
        cache = pd.DataFrame(columns=[ZIP_COLS])

    if cache.empty:
        updated = zip_to_center_new.copy()
    else:
        updated = pd.concat([cache, zip_to_center_new], ignore_index=True)

    updated = updated.drop_duplicates("zip_clean", keep="last").reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        updated.to_csv(cache_path, index=False)

    return updated


def map_people_to_centers(
    people: pd.DataFrame,
    centers: pd.DataFrame,
    raw_zip_col: str,
    cache_path: str,
) -> pd.DataFrame:
    # 1) clean zip
    out = clean_zip_5(people, raw_zip_col, out_col="zip_clean")

    # 2) pgeocode ref for unique zips
    unique_zips = out["zip_clean"].dropna().drop_duplicates().tolist()
    zip_ref = build_zip_ref_pgeocode(unique_zips)

    # 3) merge zip ref + compute problems
    out = out.merge(zip_ref, how="left", on="zip_clean")
    out = add_zip_problems(out)

    # 4) compute zip->center only for NO-PROBLEM zips, and cache it
    ok_zip_ref = out.loc[
        out["Zip Problem"].eq("no_problem"),
        ["zip_clean", "zip_lat", "zip_lon", "zip_state", "zip_county"],
    ].drop_duplicates("zip_clean")
    zip_to_center_new = compute_zip_to_center(ok_zip_ref, centers)
    zip_to_center_cache = update_zip_center_cache(zip_to_center_new, cache_path)

    # 5) merge assignments back
    out = out.merge(zip_to_center_cache, how="left", on="zip_clean")

    # 6) review flag
    out["Missing Center"] = out["Assigned Center Abbr"].isna()
    out["Needs Center Review"] = (
        out["Zip Problem"].ne("no_problem") | out["Missing Center"]
    )

    return out


def map_webinar_centers_for_nonclients(
    webinar_df: pd.DataFrame,
    centers: pd.DataFrame,
    cache_path: str,
    raw_zip_col: str = "Zip/Postal Code",
    client_col: str = "Client?",
    assigned_center_col: str = "Assigned Center Abbr",
) -> pd.DataFrame:
    out = webinar_df.copy()

    # preserve original row identity
    out["__row_id"] = out.index

    # client flag
    is_client = out[client_col].fillna(False).astype(bool)

    clients = out.loc[is_client].copy()
    non_clients = out.loc[~is_client].copy()

    # map only non-clients
    non_clients_mapped = map_people_to_centers(
        people=non_clients,
        centers=centers,
        raw_zip_col=raw_zip_col,
        cache_path=cache_path,
    )

    # bring back the row id (map_people_to_centers might drop/reset index)
    if "__row_id" not in non_clients_mapped.columns:
        non_clients_mapped = non_clients_mapped.merge(
            non_clients[["__row_id"]],
            left_index=True,
            right_index=True,
            how="left",
        )

    # set index to row_id for stable recombine
    clients = clients.set_index("__row_id", drop=True)
    non_clients_mapped = non_clients_mapped.set_index("__row_id", drop=True)

    # align columns (union)
    all_cols = clients.columns.union(non_clients_mapped.columns)
    clients = clients.reindex(columns=all_cols)
    non_clients_mapped = non_clients_mapped.reindex(columns=all_cols)

    # ---- Explicit dtype harmonization to avoid concat FutureWarning ----
    # Columns later set for clients:
    flag_cols = ["Missing Center", "Needs Center Review"]
    text_cols = ["Zip Problem"]  # Set to "client_skip" for clients

    for c in flag_cols:
        if c in clients.columns:
            clients[c] = clients[c].astype("boolean")
        if c in non_clients_mapped.columns:
            non_clients_mapped[c] = non_clients_mapped[c].astype("boolean")

    for c in text_cols:
        if c in clients.columns:
            clients[c] = clients[c].astype("string")
        if c in non_clients_mapped.columns:
            non_clients_mapped[c] = non_clients_mapped[c].astype("string")

    for c in all_cols:
        if c in flag_cols or c in text_cols:
            continue

        cd = clients[c].dtype
        nd = non_clients_mapped[c].dtype

        if (
            cd == "object"
            or nd == "object"
            or str(cd).startswith("string")
            or str(nd).startswith("string")
        ):
            clients[c] = clients[c].astype("string")
            non_clients_mapped[c] = non_clients_mapped[c].astype("string")

    # concat (keep __row_id index)
    out2 = pd.concat([clients, non_clients_mapped], axis=0)

    # restore original order + original index
    out2 = out2.reindex(out["__row_id"])
    out2.index = out.index
    out2 = out2.drop(columns=["__row_id"], errors="ignore")

    # ---- force clients to never be flagged (now dtype-safe) ----
    is_client2 = out2[client_col].fillna(False).astype(bool)

    for c in flag_cols:
        if c in out2.columns:
            out2[c] = out2[c].astype("boolean")  # ensure dtype
            out2.loc[is_client2, c] = False

    for c in text_cols:
        if c in out2.columns:
            out2[c] = out2[c].astype("string")  # ensure dtype
            out2.loc[is_client2, c] = "client_skip"

    return out2
