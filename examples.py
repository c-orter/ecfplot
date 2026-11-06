"""Example gallery: every feature, live, next to the code that draws it.  python examples.py

To add one, write a function below that returns a figure and list it in GROUPS.
Its docstring is the description. The page shows its source, plus whichever of
the np / pd / go / eg imports it uses, so keep each example self-contained.
"""

import inspect
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import ecf_globe as eg


# ---- examples ---------------------------------------------------------------
def points():
    """Markers at ECF positions. Columns default to x, y, z."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    return eg.scatter_globe(sat, size=3, name="Samples")


def lines():
    """A polyline through the rows in order; name= labels the legend entry."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    return eg.line_globe(sat, name="Orbit")


def arrays():
    """No DataFrame needed: x, y, z can be arrays. With one, name the columns."""
    fig = eg.scatter_globe(x=[7000, 0, -7000], y=[0, 7000, 0], z=[0, 0, 7000], size=8)
    df = pd.DataFrame(dict(X=[7000, 0, -7000], Y=[0, 7000, 0], Z=[0, 0, 7000]))
    return eg.line_globe(df, x="X", y="Y", z="Z", fig=fig)


def categorical_color():
    """Text colours give one trace and legend entry per value (click to toggle)."""
    t = np.linspace(0, 2 * np.pi, 200)
    equatorial = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0, name="Equatorial")
    )
    tilted = equatorial.assign(name="Tilted", z=equatorial.x * 0.5)
    return eg.line_globe(pd.concat([equatorial, tilted]), color="name")


def numeric_color():
    """Numbers give a colour scale and colour bar: on the markers here, or on a line
    that has none. Integer IDs count as numbers, so pass those as text."""
    t = np.linspace(0, 2 * np.pi, 200)
    r = 7000 + 1500 * np.cos(t)  # an eccentric orbit, so altitude varies
    sat = pd.DataFrame(dict(x=r * np.cos(t), y=r * np.sin(t), z=0.0))
    sat["alt"] = eg.ecf_to_geodetic(sat.x, sat.y, sat.z)[2]
    return eg.scatter_globe(sat, color="alt", size=4)


def hover():
    """Hover always shows the ECF position. hover_name adds a title,
    hover_data extra columns, and size can be a column."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0, name="Equatorial")
    )
    sat["minute"] = np.arange(len(sat))
    return eg.scatter_globe(
        sat, size=3 + sat.minute / 40, hover_name="name", hover_data=["minute"]
    )


def geodetic():
    """Start from lat/lon/alt with geodetic_to_ecf; ecf_to_geodetic goes back. The
    sites are 10 km up: the continents are drawn 6 km up, so anything at altitude 0
    is hidden under them."""
    cities = pd.DataFrame(
        [("Washington", 38.9, -77.0), ("Tokyo", 35.7, 139.7), ("Sydney", -33.9, 151.2)],
        columns=["city", "lat", "lon"],
    )
    x, y, z = eg.geodetic_to_ecf(cities.lat, cities.lon, 10)
    cities = cities.assign(x=x, y=y, z=z)
    return eg.scatter_globe(cities, color="city", hover_name="city", size=7)


def segments():
    """One segment per row, from (x0, y0, z0) to (x1, y1, z1): nadir lines, range
    vectors. The default column names are x0, y0, z0, x1, y1, z1."""
    t = np.linspace(0, 2 * np.pi, 12)
    x0, y0, z0 = 7000 * np.cos(t), 7000 * np.sin(t), 0 * t
    x1, y1, z1 = eg.geodetic_to_ecf(*eg.ecf_to_geodetic(x0, y0, z0)[:2], 10.0)
    nadir = pd.DataFrame(dict(x0=x0, y0=y0, z0=z0, x1=x1, y1=y1, z1=z1))
    return eg.segment_globe(nadir, name="Nadir")


def fan_out():
    """A ground station linked to a row of points arranged in an arc, each line
    coloured by the point's altitude. Numeric colour on lines gets a colour bar."""
    s = np.linspace(0, 1, 15)
    alt = 400 + 1600 * np.sin(np.pi * s)  # the pass climbs, then descends
    arc = eg.geodetic_to_ecf(np.linspace(15, 60, 15), np.linspace(-115, -40, 15), alt)
    ground = eg.geodetic_to_ecf(38.9, -77.0, 10)
    beams = pd.DataFrame(  # the single ground point broadcasts across the rows
        dict(x0=ground[0], y0=ground[1], z0=ground[2], x1=arc[0], y1=arc[1], z1=arc[2])
    ).assign(alt=alt)
    fig = eg.segment_globe(beams, color="alt", line_width=4)
    fig = eg.scatter_globe(x=arc[0], y=arc[1], z=arc[2], size=5, name="Pass", fig=fig)
    return eg.scatter_globe(x=ground[0], y=ground[1], z=ground[2], size=7, fig=fig)


def swath():
    """One surface point linked to a wide swath of surface points, coloured by
    distance. Each beam is 25 points whose altitude climbs to 0.3 of its length and
    falls back, so it flies out into space; line_group joins each beam's points."""
    # 10 km up, so the lines clear the continents (drawn 6 km up)
    lat, lon = np.meshgrid(np.linspace(-10, 50, 7), np.linspace(-110, 30, 15))
    ground = np.array(eg.geodetic_to_ecf(lat.ravel(), lon.ravel(), 10))
    source = np.array(eg.geodetic_to_ecf(25, -45, 10))
    distance = np.linalg.norm(ground - source[:, None], axis=0)

    s = np.linspace(0, 1, 25)  # position along each beam
    # (xyz, 25 points, beams): straight from the source to each target
    beam = source[:, None, None] + s[:, None] * (ground - source[:, None])[:, None]
    lat, lon, _ = eg.ecf_to_geodetic(*beam)
    alt = 10 + 0.3 * distance * np.sin(np.pi * s)[:, None]
    x, y, z = eg.geodetic_to_ecf(lat, lon, alt)  # each (25, beams)
    beams = pd.DataFrame(  # long form: one row per point, beam by beam
        dict(
            beam=np.repeat(np.arange(distance.size), 25),
            x=x.T.ravel(),
            y=y.T.ravel(),
            z=z.T.ravel(),
            distance=np.repeat(distance, 25),
        )
    )
    fig = eg.line_globe(beams, color="distance", line_group="beam")
    fig = eg.scatter_globe(x=ground[0], y=ground[1], z=ground[2], size=2, fig=fig)
    return eg.scatter_globe(x=source[0], y=source[1], z=source[2], size=8, fig=fig)


def interpolated():
    """steps=N smooths a sparse set of points: it adds N - 1 points between each pair
    on a spline through them, and numeric columns follow too (here altitude, which
    drives the marker colour). For points a few degrees apart, not distant ones."""
    t = np.linspace(0, np.pi, 9)  # a sparse pass over the equator, rising and falling
    r = 7000 + 1500 * np.sin(t)
    pass_ = pd.DataFrame(dict(x=r * np.cos(t), y=r * np.sin(t), z=0.0))
    pass_["alt"] = eg.ecf_to_geodetic(pass_.x, pass_.y, pass_.z)[2]
    return eg.scatter_globe(pass_, color="alt", size=4, mode="lines+markers", steps=4)


def covariance():
    """An uncertainty ellipsoid: sigma standard deviations out for an ECF covariance
    (units squared). The off-diagonal terms tilt it. The legend entry toggles it."""
    pos = eg.geodetic_to_ecf(20, -30, 800)
    cov = np.array(
        [[90_000, 60_000, 0], [60_000, 160_000, 30_000], [0, 30_000, 40_000]]
    )
    return eg.covariance_globe(*pos, cov, name="Tracked object")


def covariance_many():
    """One position and covariance per object. sigma=1 holds ~20% of a 3-D Gaussian,
    2 about 74%, 3 about 97%."""
    covs = np.stack([np.diag([50**2, 200**2, 50**2]), np.diag([100**2] * 3)])
    return eg.covariance_globe(
        [7000, 0], [0, 7000], [0, 0], covs, sigma=1, name=["Alpha", "Beta"]
    )


def composing():
    """Every function returns a new figure, so layers stack by reassigning fig."""
    t = np.linspace(0, 2 * np.pi, 200)
    orbit = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    fig = eg.globe_figure()
    fig = eg.line_globe(orbit, name="Orbit", fig=fig)
    fig = eg.scatter_globe(x=[7000], y=[0], z=[0], size=9, name="Start", fig=fig)
    return eg.covariance_globe(7000, 0, 0, 500**2 * np.eye(3), fig=fig)


def metres():
    """units="m" changes the globe, the hover text and the conversions together.
    A figure remembers its units; mixing them raises a ValueError."""
    fig = eg.globe_figure(units="m")
    return eg.scatter_globe(
        x=[7_000_000], y=[0], z=[0], size=8, units="m", name="Sat", fig=fig
    )


def styling():
    """No continents, a light background, a new camera and a title (which needs a
    top margin: the globe has none). The rest is plain plotly."""
    t = np.linspace(0, 2 * np.pi, 200)
    orbit = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    fig = eg.globe_figure(coast=False)
    fig.update_layout(
        title="Equatorial orbit",
        margin_t=40,
        paper_bgcolor="white",
        scene=dict(bgcolor="white", camera=dict(eye=dict(x=0, y=-2.5, z=0.5))),
    )
    fig.update_traces(
        selector=dict(name="Earth"), colorscale=[[0, "#789"], [1, "#789"]]
    )
    return eg.line_globe(orbit, name="Orbit", fig=fig)


def ecf_time_series():
    """A LEO orbit as ECF positions over three revolutions: the Earth turns under
    it, so the path spirals. Inertial (X, Y, Z) is rotated by the Earth rotation
    angle into ECF. The ground track is the same points dropped to 10 km."""
    t = np.linspace(0, 6 * np.pi, 300)  # orbit phase: three revolutions
    inc = np.radians(51.6)
    X, Y = 6900 * np.cos(t), 6900 * np.sin(t) * np.cos(inc)
    Z = 6900 * np.sin(t) * np.sin(inc)
    spin = 0.0662 * t  # Earth rotation angle: about 6.6% of a turn per orbit
    orbit = pd.DataFrame(
        dict(
            x=X * np.cos(spin) + Y * np.sin(spin),
            y=-X * np.sin(spin) + Y * np.cos(spin),
            z=Z,
            minute=t / (2 * np.pi) * 95,
        )
    )
    lat, lon, _ = eg.ecf_to_geodetic(orbit.x, orbit.y, orbit.z)
    track = orbit.assign(**dict(zip("xyz", eg.geodetic_to_ecf(lat, lon, 10))))
    fig = eg.line_globe(orbit, hover_data=["minute"], name="Orbit")
    return eg.line_globe(track, hover_data=["minute"], name="Ground track", fig=fig)


def access():
    """Access lines from a ground station, drawn only while the satellite is above a
    10 degree elevation mask and coloured by range. ecf_to_enu gives each satellite
    position as east / north / up from the station, which makes range, elevation and
    azimuth one line each; the last two show in the hover."""
    t = np.linspace(0, 6 * np.pi, 600)
    inc = np.radians(51.6)
    X, Y = 6900 * np.cos(t), 6900 * np.sin(t) * np.cos(inc)
    Z = 6900 * np.sin(t) * np.sin(inc)
    spin = 0.0662 * t
    sat = np.column_stack(
        [X * np.cos(spin) + Y * np.sin(spin), -X * np.sin(spin) + Y * np.cos(spin), Z]
    )

    site = (55.8, 37.6)  # Moscow: the track passes close enough to see it
    station = eg.geodetic_to_ecf(*site, 10)
    east, north, up = eg.ecf_to_enu(sat[:, 0], sat[:, 1], sat[:, 2], *station)
    seen = pd.DataFrame(
        dict(
            x0=station[0], y0=station[1], z0=station[2],
            x1=sat[:, 0], y1=sat[:, 1], z1=sat[:, 2],
            range=np.sqrt(east**2 + north**2 + up**2),
            elevation=np.degrees(np.arctan2(up, np.hypot(east, north))),
            azimuth=np.degrees(np.arctan2(east, north)) % 360,
        )
    )  # fmt: skip
    seen = seen[seen.elevation > 10]
    fig = eg.line_globe(
        x=sat[:, 0], y=sat[:, 1], z=sat[:, 2], name="Orbit", line_width=1
    )
    fig = eg.segment_globe(
        seen, color="range", hover_data=["elevation", "azimuth"], fig=fig
    )
    return eg.scatter_globe(
        x=station[0], y=station[1], z=station[2], size=7, name="Station", fig=fig
    )


def velocity_arrows():
    """Velocity vectors along a track: a segment from each position along its
    velocity (a finite difference of the ECF positions), with a cone as the
    arrow head. The cone is a plain plotly trace added to the same figure."""
    t = np.linspace(0, 2 * np.pi, 120)
    orbit = np.column_stack(
        [7000 * np.cos(t), 7000 * np.sin(t) * 0.8, 7000 * np.sin(t) * 0.6]
    )
    velocity = np.gradient(orbit, axis=0)
    direction = velocity / np.linalg.norm(velocity, axis=1, keepdims=True)
    arrow = 800 * direction  # 800 km long
    p, tip = orbit[::10], orbit[::10] + arrow[::10]
    fig = eg.line_globe(x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], name="Orbit")
    fig = eg.segment_globe(
        x0=p[:, 0], y0=p[:, 1], z0=p[:, 2], x1=tip[:, 0], y1=tip[:, 1], z1=tip[:, 2],
        name="Velocity", line_width=4, fig=fig,
    )  # fmt: skip
    shade = fig.data[-1].line.color  # the velocity line's colour
    return fig.add_trace(
        go.Cone(
            x=tip[:, 0], y=tip[:, 1], z=tip[:, 2],
            u=arrow[::10, 0], v=arrow[::10, 1], w=arrow[::10, 2],
            anchor="tip", sizemode="absolute", sizeref=300,
            colorscale=[[0, shade], [1, shade]], showscale=False,
            hoverinfo="skip",
        )
    )  # fmt: skip


def enu_covariance():
    """A covariance given in local east / north / up axes, rotated into ECF with
    C_ecf = R.T C R, where R = enu_matrix(x, y, z) at the object's ECF position has
    the east, north and up unit vectors (written in ECF) as its rows. The arrows
    show the three axes."""
    pos = np.array(eg.geodetic_to_ecf(25, 105, 800))  # near the limb: "up" not end-on
    R = eg.enu_matrix(*pos)
    cov = R.T @ np.diag([600, 200, 1200]) ** 2 @ R  # 1 sigma: E 600, N 200, U 1200 km
    axes = pd.DataFrame(
        dict(
            x0=pos[0], y0=pos[1], z0=pos[2],
            x1=pos[0] + 2000 * R[:, 0], y1=pos[1] + 2000 * R[:, 1], z1=pos[2] + 2000 * R[:, 2],
            axis=["East", "North", "Up"],
        )
    )  # fmt: skip
    fig = eg.covariance_globe(*pos, cov, sigma=1, name="Object (1 sigma)")
    return eg.segment_globe(axes, color="axis", line_width=4, fig=fig)


def fov_footprint():
    """A nadir sensor's footprint: the ring on the surface that a satellite sees
    within its field of view, and the edge rays down to it. Spherical-Earth geometry
    (central angle = asin((a / R) sin(half-angle)) - half-angle), fine for a picture."""
    sat = np.array(eg.geodetic_to_ecf(25, 45, 800))
    a, R = np.linalg.norm(sat), 6371.0
    eta = np.radians(45)  # sensor half-angle
    lam = np.arcsin(a / R * np.sin(eta)) - eta  # Earth central angle to the edge
    u = sat / a
    e1 = np.cross(u, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    phi = np.linspace(0, 2 * np.pi, 73)[:, None]
    ring = R * (np.cos(lam) * u + np.sin(lam) * (np.cos(phi) * e1 + np.sin(phi) * e2))
    ring = np.column_stack(eg.geodetic_to_ecf(*eg.ecf_to_geodetic(*ring.T)[:2], 10))
    edge = pd.DataFrame(
        dict(
            x0=sat[0],
            y0=sat[1],
            z0=sat[2],
            x1=ring[::6, 0],
            y1=ring[::6, 1],
            z1=ring[::6, 2],
        )
    )
    fig = eg.line_globe(
        x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], name="Footprint", line_width=4
    )
    fig = eg.segment_globe(edge, name="Edge rays", fig=fig)
    return eg.scatter_globe(
        x=sat[0], y=sat[1], z=sat[2], size=7, name="Satellite", fig=fig
    )


def constellation():
    """A Walker-style constellation: 3 planes of 8 satellites. One trace per plane,
    so each is one legend entry (click to hide it); the line is the plane's orbit and
    the markers are the satellites (a size of 0 hides a marker)."""
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    inc = np.radians(55)
    ring = 26560 * np.array(
        [np.cos(t), np.sin(t) * np.cos(inc), np.sin(t) * np.sin(inc)]
    )
    planes = []
    for k in range(3):
        a = 2 * np.pi * k / 3  # right ascension of this plane
        planes.append(
            pd.DataFrame(
                dict(
                    x=ring[0] * np.cos(a) - ring[1] * np.sin(a),
                    y=ring[0] * np.sin(a) + ring[1] * np.cos(a),
                    z=ring[2],
                    plane=f"Plane {k + 1}",
                    size=np.where(np.arange(120) % 15 == 0, 5, 0),  # 8 satellites
                )
            )
        )
    return eg.scatter_globe(
        pd.concat(planes),
        color="plane",
        size="size",
        mode="lines+markers",
    )


def animation():
    """A satellite moving along its track, using plotly's own frames and slider:
    nothing special in ecf_globe, since the figure is an ordinary go.Figure. The
    frames replace the position of one trace (the satellite marker)."""
    t = np.linspace(0, 4 * np.pi, 200)
    orbit = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t) * 0.8, z=7000 * np.sin(t) * 0.6)
    )
    fig = eg.line_globe(orbit, name="Orbit", line_width=1)
    fig = eg.scatter_globe(orbit.iloc[[0]], size=8, name="Satellite", fig=fig)
    moving = len(fig.data) - 1  # index of the satellite trace
    steps = range(0, len(orbit), 5)
    fig.frames = [
        go.Frame(
            data=[go.Scatter3d(x=[orbit.x[i]], y=[orbit.y[i]], z=[orbit.z[i]])],
            traces=[moving],
            name=str(i),
        )
        for i in steps
    ]
    play = dict(frame=dict(duration=50, redraw=True), fromcurrent=True)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None, play])],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        label=str(i),
                        args=[
                            [str(i)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                    )
                    for i in steps
                ]
            )
        ],
    )
    return fig


def axis_triad():
    """The ECF X, Y and Z axes as three coloured arrows with labels. The segments
    start inside the Earth, which hides that part. trace_kw (here mode and text
    position) is passed straight to the plotly trace."""
    axes = pd.DataFrame(
        dict(
            x0=0.0, y0=0.0, z0=0.0,
            x1=[10_000, 0, 0], y1=[0, 10_000, 0], z1=[0, 0, 10_000],
            axis=["X", "Y", "Z"],
        )
    )  # fmt: skip
    fig = eg.segment_globe(axes, color="axis", line_width=5)
    return eg.scatter_globe(
        axes, x="x1", y="y1", z="z1", hover_name="axis", size=3,
        trace_kw=dict(mode="markers+text", textposition="top center"), fig=fig,
    )  # fmt: skip


def labelled_sites():
    """Text labels on points: hover_name supplies the text, and trace_kw passes
    anything else to the plotly trace (here the mode, text position and symbol)."""
    sites = pd.DataFrame(
        [("London", 51.5, -0.1), ("Cairo", 30.0, 31.2), ("Delhi", 28.6, 77.2)],
        columns=["name", "lat", "lon"],
    )
    x, y, z = eg.geodetic_to_ecf(sites.lat, sites.lon, 10)
    return eg.scatter_globe(
        sites.assign(x=x, y=y, z=z), hover_name="name", size=7,
        trace_kw=dict(mode="markers+text", textposition="top center", marker_symbol="diamond"),
    )  # fmt: skip


GROUPS = {
    "Basics": [points, lines, arrays],
    "Colour and hover": [categorical_color, numeric_color, hover, labelled_sites],
    "Geometry": [geodetic, segments, fan_out, swath, interpolated],
    "Uncertainty": [covariance, covariance_many, enu_covariance],
    "Astrodynamics": [
        ecf_time_series,
        access,
        velocity_arrows,
        fov_footprint,
        constellation,
        animation,
        axis_triad,
    ],
    "The figure": [composing, metres, styling],
}
EXAMPLES = [f for fs in GROUPS.values() for f in fs]
BY_NAME = {f.__name__: f for f in EXAMPLES}
IMPORTS = [  # (alias, statement) in isort order: third party, then local
    [
        ("np", "import numpy as np"),
        ("pd", "import pandas as pd"),
        ("go", "import plotly.graph_objects as go"),
    ],
    [("eg", "import ecf_globe as eg")],
]


def title(f):
    return f.__name__.replace("_", " ").capitalize()


def listing(f):
    """The example's source with the imports it needs on top: what you'd paste."""
    source = textwrap.dedent(inspect.getsource(f))
    blocks = (
        "\n".join(s for a, s in group if re.search(rf"\b{a}\.", source))
        for group in IMPORTS
    )
    return "\n\n".join(filter(None, blocks)) + "\n\n\n" + source


# ---- app (styles live in examples_assets/examples.css) ------------------------
app = Dash(__name__, title="ecf_globe examples", assets_folder="examples_assets")
app.layout = html.Div(
    id="page",
    children=[
        dcc.Location(id="url"),  # the example is the URL hash, so links can be shared
        html.Nav(id="nav"),
        dcc.Graph(id="globe", config={"displaylogo": False}),
        html.Aside(
            [
                html.H2(id="heading"),
                html.P(id="about"),
                html.Div(
                    ["Code", dcc.Clipboard(id="copy", title="Copy a runnable script")],
                    className="bar",
                ),
                dcc.Markdown(id="code"),
            ]
        ),
    ],
)


def nav(current):
    items = []
    for group, funcs in GROUPS.items():
        items.append(html.H4(group))
        items += [
            html.A(
                title(f),
                href=f"#{f.__name__}",
                className="active" if f is current else "",
            )
            for f in funcs
        ]
    return [html.H3("ecf_globe"), *items]


@app.callback(
    Output("globe", "figure"),
    Output("nav", "children"),
    Output("heading", "children"),
    Output("about", "children"),
    Output("code", "children"),
    Output("copy", "content"),
    Input("url", "hash"),
)
def show(hash_):
    f = BY_NAME.get((hash_ or "").lstrip("#"), EXAMPLES[0])
    code = listing(f)
    runnable = f"{code}\n\nfig = {f.__name__}()\nfig.show()\n"
    return (
        f(),
        nav(f),
        title(f),
        inspect.getdoc(f),
        f"```python\n{code}```",
        runnable,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
