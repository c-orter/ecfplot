"""plotly-express-style ECF plotting over an Earth.

    import ecf_globe as eg
    fig = eg.scatter_globe(df, x="x", y="y", z="z", color="sat", units="km")
    fig = eg.line_globe(df, x="x", y="y", z="z", color="sat", fig=fig)

Every function returns a new go.Figure; pass `fig=` to draw onto an existing
globe. The result works directly as the `figure=` of a dcc.Graph.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from plotly.colors import qualitative

from .coast import land_trace
from .earth import earth_trace, ecf_to_geodetic, geodetic_to_ecf

PALETTE = qualitative.D3[1:]  # D3 minus its blue, which vanishes on the Earth
BG = "#0b0f1a"


def globe_figure(units="km", coast=True):
    """Empty globe: Earth, land, hidden axes, equal aspect, dark background."""
    fig = go.Figure(earth_trace(units))
    if coast:
        fig.add_trace(land_trace(units))
    hidden = dict(visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG,
        font_color="#dddddd",
        uirevision="globe",  # Dash keeps the camera across callback updates
        meta=units,  # checked when composing with fig=
        scene=dict(
            xaxis=hidden,
            yaxis=hidden,
            zaxis=hidden,
            bgcolor=BG,
            aspectmode="data",
            dragmode="turntable",  # spins about the polar (z) axis, keeps north up
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
    )
    return fig


def scatter_globe(
    data_frame=None,
    x="x",
    y="y",
    z="z",
    color=None,
    size=3,
    hover_name=None,
    hover_data=None,
    units="km",
    fig=None,
    name=None,
    opacity=1.0,
    line_width=2,
    mode="markers",
):
    """Markers at ECF positions.

    x, y, z    column names (with data_frame) or array-likes, in `units`.
    color      column/array. Text -> one legend entry per value;
               numbers -> colorscale and colorbar (markers only; lines stay flat).
    size       marker size: a number or a column/array.
    hover_name, hover_data   as in px; hover_data is a list of column names.
               Hover always shows ECF position and geodetic lat/lon/alt.
    units      "km" or "m".
    fig        globe to draw onto (it is copied, not changed).
    name       legend entry when color is not categorical.
    """

    def col(v):  # column name or array-like -> array
        return np.asarray(data_frame[v] if isinstance(v, str) else v)

    d = pd.DataFrame({"x": col(x), "y": col(y), "z": col(z)})
    d["lat"], d["lon"], d["alt"] = ecf_to_geodetic(d.x, d.y, d.z, units)
    cd = ["lat", "lon", "alt"]
    hover = [
        f"ECF (%{{x:,.1f}}, %{{y:,.1f}}, %{{z:,.1f}}) {units}",
        (
            f"lat %{{customdata[0]:.3f}}°, lon %{{customdata[1]:.3f}}°, "
            f"alt %{{customdata[2]:,.1f}} {units}"
        ),
    ]
    for h in hover_data or []:
        cd.append(f"hd{len(cd)}")
        d[cd[-1]] = col(h)
        hover.append(f"{h}: %{{customdata[{len(cd) - 1}]}}")
    if hover_name is not None:
        d["text"] = col(hover_name)
        hover.insert(0, "<b>%{text}</b>")
    if color is not None:
        d["color"] = col(color)
    if np.ndim(size) or isinstance(size, str):
        d["size"] = col(size)

    if fig is None:
        fig = globe_figure(units)
    if fig.layout.meta not in (None, units):
        raise ValueError(f"fig uses units={fig.layout.meta!r}, got units={units!r}")
    fig = go.Figure(fig)  # copy, like px

    # categorical color: one trace (and legend entry) per value
    categorical = color is not None and not is_numeric_dtype(d["color"])
    groups = (
        d.groupby("color", sort=False, dropna=False)  # dropna: keep missing labels
        if categorical
        else [(name, d)]
    )
    for k, (label, g) in enumerate(groups):
        flat = PALETTE[k % len(PALETTE)]
        marker = dict(color=flat, size=g.get("size", size))
        if color is not None and not categorical and "markers" in mode:
            marker.update(
                color=g["color"],
                showscale=True,
                colorbar=dict(title=color if isinstance(color, str) else None),
            )
        fig.add_trace(
            go.Scatter3d(
                x=g.x,
                y=g.y,
                z=g.z,
                mode=mode,
                opacity=opacity,
                name=None if label is None else str(label),
                showlegend=label is not None,
                marker=marker,
                line=dict(color=flat, width=line_width),
                text=g.get("text"),
                customdata=g[cd].to_numpy(),
                hovertemplate="<br>".join(hover) + "<extra></extra>",
            )
        )
    return fig


def line_globe(data_frame=None, **kw):
    """Polyline through ECF positions in row order. Same arguments as
    scatter_globe; color= with a text column draws one line per value."""
    return scatter_globe(data_frame, mode="lines", **kw)


def _paths(x, y, z, color, **kw):
    """Each row of the 2-D x, y, z arrays is one line; a NaN after each lifts the pen."""

    def flat(a):
        a = np.asarray(a, float)
        return np.column_stack([a, np.full(len(a), np.nan)]).ravel()

    if color is not None:  # one label per row -> one per point
        color = np.repeat(np.broadcast_to(color, len(x)), np.shape(x)[1] + 1)
    return line_globe(x=flat(x), y=flat(y), z=flat(z), color=color, **kw)


def segment_globe(x0, y0, z0, x1, y1, z1, color=None, **kw):
    """Straight ECF segments p0 -> p1, one per element (e.g. nadir lines).
    color: a label, or one per segment. Other kwargs go to line_globe."""
    return _paths(
        np.column_stack([x0, x1]),
        np.column_stack([y0, y1]),
        np.column_stack([z0, z1]),
        color,
        **kw,
    )


def arc_globe(lat0, lon0, lat1, lon1, color=None, peak_alt=0.0, units="km", **kw):
    """Great-circle arcs between geodetic points (deg), one per element.
    peak_alt lifts mid-arc (in `units`) so links clear the surface.
    color: a label, or one per arc. Other kwargs go to line_globe."""
    s = np.linspace(0, 1, 64)
    lat0, lon0, lat1, lon1 = np.broadcast_arrays(lat0, lon0, lat1, lon1)
    p = np.reshape(geodetic_to_ecf(lat0, lon0, 0.0, units), (3, -1, 1))
    q = np.reshape(geodetic_to_ecf(lat1, lon1, 0.0, units), (3, -1, 1))
    # ponytail: points evenly spaced along the chord, so they bunch at the ends
    # of very long arcs; antipodal pairs are undefined. Use slerp if that shows.
    lat, lon, _ = ecf_to_geodetic(*((1 - s) * p + s * q), units)
    x, y, z = geodetic_to_ecf(lat, lon, peak_alt * np.sin(np.pi * s), units)
    return _paths(x, y, z, color, units=units, **kw)
