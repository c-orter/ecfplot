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
from numpy.typing import ArrayLike
from pandas.api.types import is_numeric_dtype
from plotly.colors import qualitative

from .coast import land_trace
from .earth import NO_HIGHLIGHT, UNITS, earth_trace

PALETTE = qualitative.D3[1:]  # D3 minus its blue, which vanishes on the Earth
BG = "#0b0f1a"
Col = str | ArrayLike  # a data_frame column name, or the values themselves


def globe_figure(units: str = "km", coast: bool = True) -> go.Figure:
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


def _canvas(fig: go.Figure | None, units: str) -> go.Figure:
    """A copy of fig (or of a new globe), checked to use the same units."""
    if units not in UNITS:
        raise ValueError(f"units must be one of {list(UNITS)}, got {units!r}")
    if fig is None:
        fig = globe_figure(units)
    if fig.layout.meta not in (None, units):
        raise ValueError(f"fig uses units={fig.layout.meta!r}, got units={units!r}")
    return go.Figure(fig)  # copy, like px


def _drawn(fig: go.Figure) -> int:
    """How many data traces fig already has, so a new layer continues the palette."""
    return sum(t.type == "scatter3d" for t in fig.data)


def _densify(g: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Rows of g with steps - 1 points inserted between each consecutive pair, on a
    smooth curve (a Catmull-Rom spline) through all the points. Numeric columns
    follow the same curve (floats only: text, integer and date columns take the
    earlier row's value). Meant for points a few degrees apart: the curve is not a
    great circle, so a long hop would cut under the surface."""
    n = len(g)
    ok = np.isfinite(g[["x", "y", "z"]]).all(axis=1).to_numpy()
    pieces = np.where(ok[:-1] & ok[1:], steps, 1)  # a pen lift (NaN) stays one row
    i = np.append(np.repeat(np.arange(n - 1), pieces), n - 1)  # row before each new row
    t = np.append(np.concatenate([np.arange(p) / p for p in pieces]), 0)  # how far on
    j = np.minimum(i + 1, n - 1)  # row after
    # the rows either side of those, unless a pen lift (or the end) is in the way
    h = np.where(ok[np.maximum(i - 1, 0)], np.maximum(i - 1, 0), i)
    k = np.where(ok[np.minimum(i + 2, n - 1)], np.minimum(i + 2, n - 1), j)
    out = g.iloc[i].reset_index(drop=True)
    for col in g.select_dtypes("floating").columns:
        v = g[col].to_numpy(float)
        a, b, c, d = v[h], v[i], v[j], v[k]
        curve = 0.5 * (
            2 * b
            + (c - a) * t
            + (2 * a - 5 * b + 4 * c - d) * t**2
            + (3 * b - a - 3 * c + d) * t**3
        )
        out[col] = np.where(t > 0, curve, b)
    return out


def scatter_globe(
    data_frame: pd.DataFrame | None = None,
    x: Col = "x",
    y: Col = "y",
    z: Col = "z",
    color: Col | None = None,
    size: float | Col = 3,
    hover_name: Col | None = None,
    hover_data: list[Col] | None = None,
    units: str = "km",
    fig: go.Figure | None = None,
    name: str | None = None,
    opacity: float = 1.0,
    line_width: float = 2,
    mode: str = "markers",
    steps: int = 1,
    line_group: Col | None = None,
    trace_kw: dict | None = None,
) -> go.Figure:
    """Markers at ECF positions.

    x, y, z    column names (with data_frame) or array-likes, in `units`.
    color      column/array. Text -> one legend entry per value;
               numbers -> colorscale and colorbar, on the markers (or on the
               line when mode has none; a line fades between its points).
               Integer IDs count as numbers: pass them as text (df.id.astype(str)).
               Each call continues the palette where the figure left off.
    size       marker size: a number or a column/array.
    hover_name, hover_data   as in px; hover_data is a list of column names.
               Hover always shows the ECF position.
    units      "km" or "m".
    fig        globe to draw onto (it is copied, not changed).
    name       legend entry when color is not categorical.
    steps      split each hop between consecutive points into this many pieces
               on a smooth spline through the points (float columns such as
               colour follow it). For points a few degrees apart: it is not a
               great circle, so a long hop would cut under the surface. Markers
               appear on the new points too.
    line_group column/array: rows with the same value are joined in order, and
               different values are separate lines (as in px.line).
    trace_kw   anything else for the plotly trace, e.g. marker_symbol="diamond",
               line_dash="dash", mode="markers+text" (labels from hover_name).
    """

    def col(v):  # column name or array-like -> array
        return np.atleast_1d(data_frame[v] if isinstance(v, str) else v)

    d = pd.DataFrame(
        dict(zip("xyz", np.broadcast_arrays(col(x), col(y), col(z)))), dtype=float
    )
    if units == "km" and d.abs().max().max() > 5e6:  # no LEO or GEO is this far out
        raise ValueError("positions are over 5 million km out: units='m' for metres?")
    cd = []  # hover_data columns, in customdata order
    hover = [f"ECF (%{{x:,.1f}}, %{{y:,.1f}}, %{{z:,.1f}}) {units}"]
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
    if line_group is not None and len(d):  # a NaN row after each group lifts the pen
        d["line"] = col(line_group)
        pen_up = dict(x=np.nan, y=np.nan, z=np.nan)
        d = pd.concat(
            [
                pd.concat([g, g.iloc[:1].assign(**pen_up)])
                for _, g in d.groupby("line", sort=False, dropna=False)
            ],
            ignore_index=True,
        ).drop(columns="line")

    fig = _canvas(fig, units)

    # categorical color: one trace (and legend entry) per value
    categorical = color is not None and not is_numeric_dtype(d["color"])
    groups = (
        d.groupby("color", sort=False, dropna=False)  # dropna: keep missing labels
        if categorical
        else [(name, d)]
    )
    used = _drawn(fig)
    bars = sum(  # colour bars already on the figure; each new one sits further left
        bool(t.marker.showscale or t.line.showscale)
        for t in fig.data
        if t.type == "scatter3d"
    )
    for k, (label, g) in enumerate(groups, used):
        if steps > 1 and len(g) > 1:
            g = _densify(g, steps)
        flat = PALETTE[k % len(PALETTE)]
        marker = dict(color=flat, size=g.get("size", size))
        line = dict(color=flat, width=line_width)
        if color is not None and not categorical:  # colour scale on the markers, or
            (marker if "markers" in mode else line).update(  # on a markerless line
                color=g["color"],
                showscale=True,
                colorbar=dict(
                    title=color if isinstance(color, str) else None,
                    x=1.02 - 0.1 * bars,
                ),
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
                line=line,
                text=g.get("text"),
                customdata=g[cd].to_numpy() if cd else None,
                hovertemplate="<br>".join(hover) + "<extra></extra>",
            )
        )
        fig.data[-1].update(trace_kw or {})
    return fig


def line_globe(data_frame: pd.DataFrame | None = None, **kw) -> go.Figure:
    """Polyline through ECF positions in row order. Same arguments as
    scatter_globe; color= with a text column draws one line per value,
    line_group= draws several lines without splitting colour, and steps=N
    smooths the line with N pieces per hop."""
    return scatter_globe(data_frame, mode="lines", **kw)


def segment_globe(
    data_frame: pd.DataFrame | None = None,
    x0: Col = "x0",
    y0: Col = "y0",
    z0: Col = "z0",
    x1: Col = "x1",
    y1: Col = "y1",
    z1: Col = "z1",
    color: Col | None = None,
    hover_name: Col | None = None,
    hover_data: list[str] | None = None,
    **kw,
) -> go.Figure:
    """One segment per row from ECF point 0 to point 1 (e.g. nadir lines).

    x0 .. z1   column names (with data_frame) or array-likes; they broadcast, so
               one point 0 can fan out to many point 1s.
    color, hover_name   column/array with one value per segment (numbers give
               a colour scale). hover_data is a list of column names.
    Other arguments (name, steps, line_width, units, fig, ...) go to line_globe.
    """

    def col(v):
        return np.atleast_1d(data_frame[v] if isinstance(v, str) else v)

    ends = np.broadcast_arrays(*(col(v) for v in (x0, y0, z0, x1, y1, z1)))
    n = len(ends[0])

    def per_end(v):  # one value per segment -> one per endpoint
        return None if v is None else np.repeat(np.broadcast_to(col(v), n), 2)

    d = pd.DataFrame(
        {k: np.column_stack(ends[i::3]).ravel() for i, k in enumerate("xyz")}
    )
    for v in (color, hover_name, *(hover_data or [])):
        if isinstance(v, str):  # keep the name for the colour bar and hover text
            d[v] = per_end(v)

    def pick(v):  # column names resolve in d; arrays are repeated
        return v if isinstance(v, str) else per_end(v)

    return line_globe(
        d,
        color=pick(color),
        hover_name=pick(hover_name),
        hover_data=hover_data,
        line_group=np.repeat(np.arange(n), 2),
        **kw,
    )


def covariance_globe(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    cov: ArrayLike,
    sigma: float = 3.0,
    name: str | list[str] | None = None,
    units: str = "km",
    fig: go.Figure | None = None,
    opacity: float = 0.3,
) -> go.Figure:
    """Uncertainty ellipsoids: a marker at each position, wrapped in the surface
    that lies `sigma` standard deviations out for its covariance.

    x, y, z    ECF position(s) in `units`: numbers, or one per object.
    cov        ECF position covariance in `units` squared: (3, 3), or (n, 3, 3)
               for n objects. A covariance in another frame (RIC, ...) has to be
               rotated into ECF first.
    sigma      ellipsoid size in standard deviations: 1 holds about 20% of a
               3-D Gaussian, 3 about 97%.
    name       legend label, or one per object. Clicking it toggles both parts.
    fig        globe to draw onto (it is copied, not changed).
    """
    p = np.column_stack(np.broadcast_arrays(x, y, z)).astype(float)  # (n, 3)
    cov = np.broadcast_to(cov, (len(p), 3, 3))
    if not np.allclose(cov, cov.swapaxes(-1, -2)):
        raise ValueError("cov must be symmetric")
    if name is None or isinstance(name, str):
        name = [name or f"object {i}" for i in range(len(p))]
    fig = _canvas(fig, units)
    used = _drawn(fig)
    th, ph = np.mgrid[0 : np.pi : 25j, 0 : 2 * np.pi : 49j]
    sphere = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    for i, (c, s, label) in enumerate(zip(p, cov, name, strict=True)):
        w, v = np.linalg.eigh(s)  # principal axes and variances along them
        if w.min() < -1e-9 * w.max():
            raise ValueError(
                "cov must be positive semi-definite (no negative variance)"
            )
        r = sigma * np.sqrt(np.clip(w, 0, None))  # semi-axis lengths
        sx, sy, sz = c[:, None, None] + np.einsum("ij,jkl->ikl", v * r, sphere)
        color = PALETTE[(used + i) % len(PALETTE)]
        axes = ", ".join(f"{a:,.2f}" for a in r[::-1])
        fig.add_trace(
            go.Surface(
                contours=NO_HIGHLIGHT,
                x=sx,
                y=sy,
                z=sz,
                opacity=opacity,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=label,
                legendgroup=label,
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>{sigma:g}σ semi-axes: {axes} {units}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers",
                marker=dict(color=color, size=4),
                name=label,
                legendgroup=label,
                hoverinfo="skip",
            )
        )
    return fig
