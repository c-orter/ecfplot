"""Example gallery: every feature, live, next to the code that draws it.  python examples.py

To add one, write a function below that returns a figure and list it in GROUPS.
Its docstring is the description. The page shows its source, plus whichever of
the np / pd / go / ep imports it uses, so keep each example self-contained.
"""

import inspect
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import ecfplot as ep


# ---- examples ---------------------------------------------------------------
def points():
    """Markers at ECF positions. Columns default to x, y, z."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    return ep.scatter_globe(sat, size=3, name="Samples")


def lines():
    """A polyline through the rows in order; name= labels the legend entry."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    return ep.line_globe(sat, name="Orbit")


def arrays():
    """No DataFrame needed: x, y, z can be arrays. With one, name the columns."""
    fig = ep.scatter_globe(x=[7000, 0, -7000], y=[0, 7000, 0], z=[0, 0, 7000], size=8)
    df = pd.DataFrame(dict(X=[7000, 0, -7000], Y=[0, 7000, 0], Z=[0, 0, 7000]))
    return ep.line_globe(df, x="X", y="Y", z="Z", fig=fig)


def categorical_color():
    """Text colours give one trace and legend entry per value (click to toggle)."""
    t = np.linspace(0, 2 * np.pi, 200)
    equatorial = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0, name="Equatorial")
    )
    tilted = equatorial.assign(name="Tilted", z=equatorial.x * 0.5)
    return ep.line_globe(pd.concat([equatorial, tilted]), color="name")


def numeric_color():
    """Numbers give a colour scale and colour bar: on the markers here, or on a line
    that has none. Integer IDs count as numbers, so pass those as text."""
    t = np.linspace(0, 2 * np.pi, 200)
    r = 7000 + 1500 * np.cos(t)  # an eccentric orbit, so altitude varies
    sat = pd.DataFrame(dict(x=r * np.cos(t), y=r * np.sin(t), z=0.0))
    sat["alt"] = ep.ecf_to_geodetic(sat.x, sat.y, sat.z)[2]
    return ep.scatter_globe(sat, color="alt", size=4)


def hover():
    """Hover always shows the ECF position. hover_name adds a title,
    hover_data extra columns, and size can be a column."""
    t = np.linspace(0, 2 * np.pi, 200)
    sat = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0, name="Equatorial")
    )
    sat["minute"] = np.arange(len(sat))
    return ep.scatter_globe(
        sat, size=3 + sat.minute / 40, hover_name="name", hover_data=["minute"]
    )


def geodetic():
    """Start from lat/lon/alt with geodetic_to_ecf; ecf_to_geodetic goes back. Sites
    at altitude 0 show on top of the continents."""
    cities = pd.DataFrame(
        [("Washington", 38.9, -77.0), ("Tokyo", 35.7, 139.7), ("Sydney", -33.9, 151.2)],
        columns=["city", "lat", "lon"],
    )
    x, y, z = ep.geodetic_to_ecf(cities.lat, cities.lon, 0)
    cities = cities.assign(x=x, y=y, z=z)
    return ep.scatter_globe(cities, color="city", hover_name="city", size=7)


def segments():
    """One segment per row, from (x0, y0, z0) to (x1, y1, z1): nadir lines, range
    vectors. The default column names are x0, y0, z0, x1, y1, z1."""
    t = np.linspace(0, 2 * np.pi, 12)
    x0, y0, z0 = 7000 * np.cos(t), 7000 * np.sin(t), 0 * t
    x1, y1, z1 = ep.geodetic_to_ecf(*ep.ecf_to_geodetic(x0, y0, z0)[:2], 10.0)
    nadir = pd.DataFrame(dict(x0=x0, y0=y0, z0=z0, x1=x1, y1=y1, z1=z1))
    return ep.segment_globe(nadir, name="Nadir")


def fan_out():
    """A ground station linked to a row of points arranged in an arc, each line
    coloured by the point's altitude. Numeric colour on lines gets a colour bar."""
    s = np.linspace(0, 1, 15)
    alt = 400 + 1600 * np.sin(np.pi * s)  # the pass climbs, then descends
    arc = ep.geodetic_to_ecf(np.linspace(15, 60, 15), np.linspace(-115, -40, 15), alt)
    ground = ep.geodetic_to_ecf(38.9, -77.0, 10)
    beams = pd.DataFrame(  # the single ground point broadcasts across the rows
        dict(x0=ground[0], y0=ground[1], z0=ground[2], x1=arc[0], y1=arc[1], z1=arc[2])
    ).assign(alt=alt)
    fig = ep.segment_globe(beams, color="alt", line_width=4)
    fig = ep.scatter_globe(x=arc[0], y=arc[1], z=arc[2], size=5, name="Pass", fig=fig)
    return ep.scatter_globe(x=ground[0], y=ground[1], z=ground[2], size=7, fig=fig)


def swath():
    """One surface point linked to a wide swath of surface points, coloured by
    distance. Each beam is 25 points whose altitude climbs to 0.3 of its length and
    falls back, so it flies out into space; line_group joins each beam's points."""
    # 10 km up, so the lines clear the continents (drawn 6 km up)
    lat, lon = np.meshgrid(np.linspace(-10, 50, 7), np.linspace(-110, 30, 15))
    ground = np.array(ep.geodetic_to_ecf(lat.ravel(), lon.ravel(), 10))
    source = np.array(ep.geodetic_to_ecf(25, -45, 10))
    distance = np.linalg.norm(ground - source[:, None], axis=0)

    s = np.linspace(0, 1, 25)  # position along each beam
    # (xyz, 25 points, beams): straight from the source to each target
    beam = source[:, None, None] + s[:, None] * (ground - source[:, None])[:, None]
    lat, lon, _ = ep.ecf_to_geodetic(*beam)
    alt = 10 + 0.3 * distance * np.sin(np.pi * s)[:, None]
    x, y, z = ep.geodetic_to_ecf(lat, lon, alt)  # each (25, beams)
    beams = pd.DataFrame(  # long form: one row per point, beam by beam
        dict(
            beam=np.repeat(np.arange(distance.size), 25),
            x=x.T.ravel(),
            y=y.T.ravel(),
            z=z.T.ravel(),
            distance=np.repeat(distance, 25),
        )
    )
    fig = ep.line_globe(beams, color="distance", line_group="beam")
    fig = ep.scatter_globe(x=ground[0], y=ground[1], z=ground[2], size=2, fig=fig)
    return ep.scatter_globe(x=source[0], y=source[1], z=source[2], size=8, fig=fig)


def covariance():
    """An uncertainty ellipsoid: sigma standard deviations out for an ECF covariance
    (km squared). The off-diagonal terms tilt it. The legend entry toggles it."""
    pos = ep.geodetic_to_ecf(20, -30, 800)
    cov = np.array(
        [[90_000, 60_000, 0], [60_000, 160_000, 30_000], [0, 30_000, 40_000]]
    )
    return ep.covariance_globe(*pos, cov, name="Tracked object")


def covariance_many():
    """One position and covariance per object. sigma=1 holds ~20% of a 3-D Gaussian,
    2 about 74%, 3 about 97%."""
    covs = np.stack([np.diag([50**2, 200**2, 50**2]), np.diag([100**2] * 3)])
    return ep.covariance_globe(
        [7000, 0], [0, 7000], [0, 0], covs, sigma=1, name=["Alpha", "Beta"]
    )


def composing():
    """Every function returns a new figure, so layers stack by reassigning fig."""
    t = np.linspace(0, 2 * np.pi, 200)
    orbit = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    fig = ep.globe_figure()
    fig = ep.line_globe(orbit, name="Orbit", fig=fig)
    fig = ep.scatter_globe(x=[7000], y=[0], z=[0], size=9, name="Start", fig=fig)
    return ep.covariance_globe(7000, 0, 0, 500**2 * np.eye(3), fig=fig)


def styling():
    """No continents, a light background, a new camera and a title (which needs a
    top margin: the globe has none). The rest is plain plotly."""
    t = np.linspace(0, 2 * np.pi, 200)
    orbit = pd.DataFrame(dict(x=7000 * np.cos(t), y=7000 * np.sin(t), z=0.0))
    fig = ep.globe_figure(coast=False)
    fig.update_layout(
        title="Equatorial orbit",
        margin_t=40,
        paper_bgcolor="white",
        scene=dict(bgcolor="white", camera=dict(eye=dict(x=0, y=-2.5, z=0.5))),
    )
    fig.update_traces(
        selector=dict(name="Earth"), colorscale=[[0, "#789"], [1, "#789"]]
    )
    return ep.line_globe(orbit, name="Orbit", fig=fig)


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
    lat, lon, _ = ep.ecf_to_geodetic(orbit.x, orbit.y, orbit.z)
    track = orbit.assign(**dict(zip("xyz", ep.geodetic_to_ecf(lat, lon, 10))))
    fig = ep.line_globe(orbit, hover_data=["minute"], name="Orbit")
    return ep.line_globe(track, hover_data=["minute"], name="Ground track", fig=fig)


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
    station = ep.geodetic_to_ecf(*site, 10)
    east, north, up = ep.ecf_to_enu(sat[:, 0], sat[:, 1], sat[:, 2], *station)
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
    fig = ep.line_globe(
        x=sat[:, 0], y=sat[:, 1], z=sat[:, 2], name="Orbit", line_width=1
    )
    fig = ep.segment_globe(
        seen, color="range", hover_data=["elevation", "azimuth"], fig=fig
    )
    return ep.scatter_globe(
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
    fig = ep.line_globe(x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], name="Orbit")
    fig = ep.segment_globe(
        x0=p[:, 0], y0=p[:, 1], z0=p[:, 2], x1=tip[:, 0], y1=tip[:, 1], z1=tip[:, 2],
        name="Velocity", line_width=4, fig=fig,
    )  # fmt: skip
    shade = fig.data[-1].line.color  # the velocity line's colour
    fig.add_trace(
        go.Cone(
            x=tip[:, 0], y=tip[:, 1], z=tip[:, 2],
            u=arrow[::10, 0], v=arrow[::10, 1], w=arrow[::10, 2],
            anchor="tip", sizemode="absolute", sizeref=300,
            colorscale=[[0, shade], [1, shade]], showscale=False,
            hoverinfo="skip",
        )
    )  # fmt: skip
    # a Cone trace skews the automatic axis scaling and stretches the globe
    # (an egg), so fix the axes to a cube instead
    return fig.update_layout(
        scene=dict(
            aspectmode="cube", **{f"{a}axis_range": [-8000, 8000] for a in "xyz"}
        )
    )


def enu_covariance():
    """A covariance given in local east / north / up axes, rotated into ECF with
    C_ecf = R.T C R, where R = enu_matrix(x, y, z) at the object's ECF position has
    the east, north and up unit vectors (written in ECF) as its rows. The arrows
    show the three axes."""
    pos = np.array(ep.geodetic_to_ecf(25, 105, 800))  # near the limb: "up" not end-on
    R = ep.enu_matrix(*pos)
    cov = R.T @ np.diag([600, 200, 1200]) ** 2 @ R  # 1 sigma: E 600, N 200, U 1200 km
    axes = pd.DataFrame(
        dict(
            x0=pos[0], y0=pos[1], z0=pos[2],
            x1=pos[0] + 2000 * R[:, 0], y1=pos[1] + 2000 * R[:, 1], z1=pos[2] + 2000 * R[:, 2],
            axis=["East", "North", "Up"],
        )
    )  # fmt: skip
    fig = ep.covariance_globe(*pos, cov, sigma=1, name="Object (1 sigma)")
    return ep.segment_globe(axes, color="axis", line_width=4, fig=fig)


def fov_footprint():
    """A nadir sensor's footprint: the ring on the surface that a satellite sees
    within its field of view, and the edge rays down to it. Spherical-Earth geometry
    (central angle = asin((a / R) sin(half-angle)) - half-angle), fine for a picture."""
    sat = np.array(ep.geodetic_to_ecf(25, 45, 800))
    a, R = np.linalg.norm(sat), 6371.0
    eta = np.radians(45)  # sensor half-angle
    lam = np.arcsin(a / R * np.sin(eta)) - eta  # Earth central angle to the edge
    u = sat / a
    e1 = np.cross(u, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    phi = np.linspace(0, 2 * np.pi, 73)[:, None]
    ring = R * (np.cos(lam) * u + np.sin(lam) * (np.cos(phi) * e1 + np.sin(phi) * e2))
    ring = np.column_stack(ep.geodetic_to_ecf(*ep.ecf_to_geodetic(*ring.T)[:2], 10))
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
    fig = ep.line_globe(
        x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], name="Footprint", line_width=4
    )
    fig = ep.segment_globe(edge, name="Edge rays", fig=fig)
    return ep.scatter_globe(
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
    return ep.scatter_globe(
        pd.concat(planes),
        color="plane",
        size="size",
        mode="lines+markers",
    )


def animation():
    """A satellite moving along its track, using plotly's own frames and slider:
    nothing special in ecfplot, since the figure is an ordinary go.Figure. The
    frames replace the position of one trace (the satellite marker)."""
    t = np.linspace(0, 4 * np.pi, 200)
    orbit = pd.DataFrame(
        dict(x=7000 * np.cos(t), y=7000 * np.sin(t) * 0.8, z=7000 * np.sin(t) * 0.6)
    )
    fig = ep.line_globe(orbit, name="Orbit", line_width=1)
    fig = ep.scatter_globe(orbit.iloc[[0]], size=8, name="Satellite", fig=fig)
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
    fig = ep.segment_globe(axes, color="axis", line_width=5)
    return ep.scatter_globe(
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
    x, y, z = ep.geodetic_to_ecf(sites.lat, sites.lon, 10)
    return ep.scatter_globe(
        sites.assign(x=x, y=y, z=z), hover_name="name", size=7,
        trace_kw=dict(mode="markers+text", textposition="top center", marker_symbol="diamond"),
    )  # fmt: skip


def gto_transfer():
    """Geostationary transfer orbit (GTO): a LEO parking orbit, an eccentric transfer
    arc with perigee and apogee burns, and the circular GEO ring at 42,164 km."""
    r_leo, r_geo = 6700.0, 42164.0
    a, e = (r_leo + r_geo) / 2, (r_geo - r_leo) / (r_geo + r_leo)
    nu = np.linspace(0, np.pi, 120)  # half-ellipse transfer arc
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    inc = np.radians(28.5)  # Cape Canaveral inclination
    t = np.linspace(0, 2 * np.pi, 120)

    leo = pd.DataFrame(
        dict(
            x=r_leo * np.cos(t),
            y=r_leo * np.sin(t) * np.cos(inc),
            z=r_leo * np.sin(t) * np.sin(inc),
            phase="LEO Parking",
        )
    )
    transfer = pd.DataFrame(
        dict(
            x=r * np.cos(nu),
            y=r * np.sin(nu) * np.cos(inc),
            z=r * np.sin(nu) * np.sin(inc),
            phase="GTO Arc",
        )
    )
    geo = pd.DataFrame(
        dict(x=r_geo * np.cos(t), y=r_geo * np.sin(t), z=0.0, phase="GEO Target")
    )
    fig = ep.line_globe(pd.concat([leo, transfer, geo]), color="phase", line_width=3)
    burns = pd.DataFrame(
        [("Perigee Burn", r_leo, 0, 0), ("Apogee Burn", -r_geo, 0, 0)],
        columns=["burn", "x", "y", "z"],
    )
    return ep.scatter_globe(burns, hover_name="burn", size=8, fig=fig)


def eclipse():
    """An orbit entering and exiting the Earth's shadow (cylindrical umbra): sunlit
    segments are separated from eclipse. The arrow shows the Sun vector."""
    t = np.linspace(0, 2 * np.pi, 240)
    inc = np.radians(65.0)
    orbit = pd.DataFrame(
        dict(
            x=7200 * np.cos(t),
            y=7200 * np.sin(t) * np.cos(inc),
            z=7200 * np.sin(t) * np.sin(inc),
        )
    )
    sun = np.array([np.cos(np.radians(23.4)), 0.0, np.sin(np.radians(23.4))])
    pos = orbit[["x", "y", "z"]].to_numpy()
    along = pos @ sun
    dist_sq = np.sum(pos**2, axis=1) - along**2
    orbit["regime"] = np.where(
        (along < 0) & (dist_sq < 6378.137**2), "Umbra (Eclipse)", "Sunlit"
    )
    orbit["segment"] = (orbit.regime != orbit.regime.shift()).cumsum()
    fig = ep.line_globe(orbit, color="regime", line_group="segment", line_width=4)
    sun_ray = pd.DataFrame(
        dict(
            x0=[0],
            y0=[0],
            z0=[0],
            x1=[12000 * sun[0]],
            y1=[12000 * sun[1]],
            z1=[12000 * sun[2]],
        )
    )
    return ep.segment_globe(sun_ray, name="Sun Vector", line_width=3, fig=fig)


def crosslinks():
    """Inter-satellite link (ISL) line-of-sight and occultation: line-of-sight
    between constellation nodes is blocked when the ray passes through the Earth."""
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    sats = pd.DataFrame(
        dict(
            name=[f"Node {i + 1}" for i in range(8)],
            x=7600 * np.cos(t),
            y=7600 * np.sin(t),
            z=0.0,
        )
    )
    p = sats[["x", "y", "z"]].to_numpy()
    links = []
    r_earth = 6378.137 + 100.0  # Earth + 100 km atmospheric buffer
    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            p0, p1 = p[i], p[j]
            v = p1 - p0
            t_min = np.clip(-np.dot(p0, v) / np.dot(v, v), 0, 1)
            d_min = np.linalg.norm(p0 + t_min * v)
            links.append(
                dict(
                    x0=p0[0],
                    y0=p0[1],
                    z0=p0[2],
                    x1=p1[0],
                    y1=p1[1],
                    z1=p1[2],
                    status="Clear LOS" if d_min >= r_earth else "Occulted",
                )
            )
    df = pd.DataFrame(links)
    fig = ep.segment_globe(
        df[df.status == "Clear LOS"], name="Active Link (LOS)", line_width=3
    )
    fig = ep.segment_globe(
        df[df.status == "Occulted"],
        name="Earth Occulted",
        line_width=1,
        trace_kw=dict(line=dict(dash="dot")),
        fig=fig,
    )
    return ep.scatter_globe(sats, hover_name="name", size=7, fig=fig)


def ground_network():
    """Multi-station tracking and handover: Deep Space Network stations track a
    high-eccentricity Molniya orbit, colored by the station with highest elevation."""
    stations = pd.DataFrame(
        [
            ("Goldstone", 35.4, -116.9),
            ("Madrid", 40.4, -4.2),
            ("Canberra", -35.4, 149.0),
        ],
        columns=["name", "lat", "lon"],
    )
    sx, sy, sz = ep.geodetic_to_ecf(stations.lat, stations.lon, 10.0)
    stations = stations.assign(x=sx, y=sy, z=sz)
    t = np.linspace(0, 2 * np.pi, 240)
    a, e, inc = 26600.0, 0.72, np.radians(63.4)
    r = a * (1 - e**2) / (1 + e * np.cos(t - np.pi / 2))
    sat_x, sat_y, sat_z = (
        r * np.cos(t),
        r * np.sin(t) * np.cos(inc),
        r * np.sin(t) * np.sin(inc),
    )
    best_station = []
    for i in range(len(t)):
        active = "No contact"
        best_el = 10.0
        for _, st in stations.iterrows():
            _, el, _ = ep.ecf_to_aer(sat_x[i], sat_y[i], sat_z[i], st.x, st.y, st.z)
            if el > best_el:
                best_el = el
                active = st["name"]
        best_station.append(active)
    orbit = pd.DataFrame(dict(x=sat_x, y=sat_y, z=sat_z, contact=best_station))
    orbit["segment"] = (orbit.contact != orbit.contact.shift()).cumsum()
    fig = ep.line_globe(orbit, color="contact", line_group="segment", line_width=3)
    return ep.scatter_globe(stations, hover_name="name", size=8, fig=fig)


GROUPS = {
    "Basics": [points, lines, arrays],
    "Colour and hover": [categorical_color, numeric_color, hover, labelled_sites],
    "Geometry": [geodetic, segments, fan_out, swath],
    "Uncertainty": [covariance, covariance_many, enu_covariance],
    "Astrodynamics": [
        ecf_time_series,
        gto_transfer,
        eclipse,
        crosslinks,
        ground_network,
        access,
        velocity_arrows,
        fov_footprint,
        constellation,
        animation,
        axis_triad,
    ],
    "The figure": [composing, styling],
}
EXAMPLES = [f for fs in GROUPS.values() for f in fs]
BY_NAME = {f.__name__: f for f in EXAMPLES}
IMPORTS = [  # (alias, statement) in isort order: third party, then local
    [
        ("np", "import numpy as np"),
        ("pd", "import pandas as pd"),
        ("go", "import plotly.graph_objects as go"),
    ],
    [("ep", "import ecfplot as ep")],
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
app = Dash(__name__, title="ecfplot examples", assets_folder="examples_assets")
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
    return [html.H3("ecfplot"), *items]


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
