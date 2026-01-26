import numpy as np
import folium


def _cluster_sum_people_icon_create_function() -> str:
    return """
    function(cluster) {
        var markers = cluster.getAllChildMarkers();
        var sum = 0;

        for (var i = 0; i < markers.length; i++) {
            var opts = markers[i].options || {};
            var v = 0;

            // folium camelizes python keys: n_people -> nPeople
            if (opts.nPeople != null) v = opts.nPeople;
            else if (opts.n_people != null) v = opts.n_people;

            // force numeric
            v = Number(v) || 0;
            sum += v;
        }

        var c = ' marker-cluster-small';
        if (sum >= 100) c = ' marker-cluster-medium';
        if (sum >= 1000) c = ' marker-cluster-large';

        return L.divIcon({
            html: '<div><span>' + sum + '</span></div>',
            className: 'marker-cluster' + c,
            iconSize: new L.Point(40, 40)
        });
    }
    """


def _scaled_dot_divicon(color: str, n: int) -> folium.DivIcon:
    """
    Marker icon whose circle size scales with n (log-scaled).
    Returns a DivIcon so it clusters (MarkerCluster only clusters Markers).
    """
    n = int(n) if n is not None else 0

    # log scaling keeps huge ZIPs from becoming ridiculous
    size = 10 + np.log1p(n) * 6  # base 10px, grows with count
    size = float(np.clip(size, 10, 34))
    html = f"""
    <div style="
        width: {size}px;
        height: {size}px;
        background: {color};
        border: 2px solid white;
        border-radius: 50%;
        box-shadow: 0 0 2px rgba(0,0,0,0.6);
        opacity: 0.85;
    "></div>
    """

    # icon_anchor centers the circle on the lat/lon
    return folium.DivIcon(
        html=html,
        icon_size=(size, size),
        icon_anchor=(size / 2, size / 2),
    )


def _add_zip_point(
    target,
    *,
    lat: float,
    lon: float,
    dot_color: str,
    n_people: int,
    popup_html: str,
    tooltip: str,
    use_cluster_marker: bool,
):
    n_people = int(n_people) if n_people is not None else 0

    if use_cluster_marker:
        folium.Marker(
            location=[lat, lon],
            icon=_scaled_dot_divicon(dot_color, n_people),
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=tooltip,
            nPeople=n_people,
        ).add_to(target)
    else:
        radius = float(np.clip(4 + np.log1p(n_people) * 3.5, 5, 18))
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=dot_color,
            fill=True,
            fill_color=dot_color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=tooltip,
        ).add_to(target)
