"""Example Dash app: points, lines, arcs and segments on the globe.  python app.py"""

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html

import ecf_globe as eg
from demo import orbit

# ---- synthetic data -------------------------------------------------------
STATIONS = pd.DataFrame(
    [
        ("Washington", 38.9, -77.0),
        ("London", 51.5, -0.1),
        ("Tokyo", 35.7, 139.7),
        ("Sydney", -33.9, 151.2),
        ("Cape Town", -33.9, 18.4),
        ("Santiago", -33.4, -70.7),
        ("Anchorage", 61.2, -149.9),
    ],
    columns=["name", "lat", "lon"],
)
STATIONS[["x", "y", "z"]] = np.column_stack(
    eg.geodetic_to_ecf(STATIONS.lat, STATIONS.lon, 8.0)
)

LINKS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    (6, 2),
    (0, 6),
]  # station index pairs

ORBITS = pd.concat(
    [
        orbit("LEO-53", 6900, 53),
        orbit("Polar", 7200, 90),
        orbit("GEO-ish", 42164, 5, spin=False),
    ],
    ignore_index=True,
)
ORBITS["lat"], ORBITS["lon"], ORBITS["alt"] = eg.ecf_to_geodetic(
    ORBITS.x, ORBITS.y, ORBITS.z
)
ORBITS["k"] = ORBITS.groupby("sat").cumcount()  # time-step index within each orbit
N = ORBITS.k.max() + 1

# a tracked object with its ECF position (km) and a full covariance (km^2);
# the off-diagonal terms tilt the uncertainty ellipsoid
OBJECT = eg.geodetic_to_ecf(20, -30, 800)
OBJECT_COV = np.array(
    [
        [90_000, 60_000, 0],
        [60_000, 160_000, 30_000],
        [0, 30_000, 40_000],
    ]
)

LAYERS = {
    "stations": "Ground stations",
    "links": "Station links (arcs)",
    "orbits": "Orbit paths",
    "tracks": "Ground tracks",
    "sats": "Satellites (colour = altitude)",
    "nadir": "Nadir lines",
    "cov": "Position uncertainty (3σ)",
}


# ---- figure ---------------------------------------------------------------
def build(layers, i):
    # each eg.* call returns a new figure (px semantics), so keep reassigning
    fig = eg.globe_figure()

    if "links" in layers:
        a, b = zip(*LINKS)
        s0, s1 = STATIONS.iloc[list(a)], STATIONS.iloc[list(b)]
        fig = eg.arc_globe(
            s0.lat,
            s0.lon,
            s1.lat,
            s1.lon,
            color=["Comm link"] * len(LINKS),
            peak_alt=900,
            line_width=3,
            fig=fig,
        )
    if "orbits" in layers:
        fig = eg.line_globe(
            ORBITS, color="sat", hover_name="sat", hover_data=["t_min"], fig=fig
        )
    if "tracks" in layers:
        gx, gy, gz = eg.geodetic_to_ecf(ORBITS.lat, ORBITS.lon, 0.0)
        tr = pd.DataFrame(dict(x=gx, y=gy, z=gz, sat=ORBITS.sat + " track"))
        fig = eg.line_globe(tr, color="sat", line_width=1, opacity=0.6, fig=fig)

    now = ORBITS[ORBITS.k == i]
    if "nadir" in layers:
        gx, gy, gz = eg.geodetic_to_ecf(now.lat, now.lon, 0.0)
        fig = eg.segment_globe(
            now.x,
            now.y,
            now.z,
            gx,
            gy,
            gz,
            color=["Nadir"] * len(now),
            line_width=2,
            fig=fig,
        )
    if "cov" in layers:
        # synthetic covariance: 100 km everywhere, 400 km along the direction of travel
        j = min(i, N - 2)  # the last step reuses the one before it
        p0, p1 = (ORBITS[ORBITS.k == k][["x", "y", "z"]].to_numpy() for k in (j, j + 1))
        t = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1, keepdims=True)
        cov = 100**2 * np.eye(3) + (400**2 - 100**2) * t[:, :, None] * t[:, None, :]
        fig = eg.covariance_globe(
            now.x, now.y, now.z, cov, name=(now.sat + " 3σ").tolist(), fig=fig
        )
        fig = eg.covariance_globe(*OBJECT, OBJECT_COV, name="Tracked object", fig=fig)
    if "sats" in layers:
        fig = eg.scatter_globe(
            now,
            color="alt",
            size=7,
            hover_name="sat",
            hover_data=["t_min"],
            name="Satellites",
            fig=fig,
        )
    if "stations" in layers:
        fig = eg.scatter_globe(
            STATIONS, color="name", size=5, hover_name="name", fig=fig
        )
    return fig


# ---- app ------------------------------------------------------------------
app = Dash(__name__)
app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "background": "#0b0f1a",
        "color": "#ddd",
        "fontFamily": "sans-serif",
    },
    children=[
        html.Div(
            style={"width": "260px", "padding": "16px", "flex": "none"},
            children=[
                html.H3("ECF globe demo"),
                dcc.Checklist(
                    id="layers",
                    value=list(LAYERS),
                    labelStyle={"display": "block"},
                    options=[{"label": v, "value": k} for k, v in LAYERS.items()],
                ),
                html.Div("Time step", style={"marginTop": "20px"}),
                dcc.Slider(
                    id="t",
                    min=0,
                    max=N - 1,
                    step=1,
                    value=0,
                    marks=None,
                    updatemode="drag",
                ),
            ],
        ),
        dcc.Graph(
            id="globe",
            style={"flex": "1", "height": "100vh"},
            config={"displaylogo": False},
        ),
    ],
)


@app.callback(Output("globe", "figure"), Input("layers", "value"), Input("t", "value"))
def update(layers, t):
    return build(layers, t)


if __name__ == "__main__":
    app.run_server(debug=True)
