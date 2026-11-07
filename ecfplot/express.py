"""plotly-express-style ECF plotting over an Earth.

    import ecfplot as ep
    fig = ep.scatter_globe(df, x="x", y="y", z="z", color="sat")
    fig = ep.line_globe(df, x="x", y="y", z="z", color="sat", fig=fig)

Every function returns a new go.Figure; pass `fig=` to draw onto an existing
globe. The result works directly as the `figure=` of a dcc.Graph. Lengths are km.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from plotly.colors import qualitative

from .coast import land_trace
from .earth import NO_HIGHLIGHT, earth_trace

PALETTE = qualitative.D3[1:]  # D3 minus its blue, which vanishes on the Earth
BG = "#0b0f1a"
Col = str | ArrayLike  # a data_frame column name, or the values themselves


def globe_figure(coast: bool = True) -> go.Figure:
    """Empty globe: Earth, land, hidden axes, equal aspect, dark background."""
    fig = go.Figure(earth_trace())
    if coast:
        fig.add_trace(land_trace())
    hidden = dict(visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG,
        font_color="#dddddd",
        uirevision="globe",  # Dash keeps the camera across callback updates
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


def _col(df: pd.DataFrame | None, v: Col) -> np.ndarray:
    """Resolve a column name or array-like to a 1-D numpy array."""
    if isinstance(v, str):
        if df is None:
            raise ValueError(f"{v!r} is a column name but no data_frame was given")
        v = df[v]
    return np.atleast_1d(v)


def _canvas(fig: go.Figure | None) -> go.Figure:
    """A copy of fig (or of a new globe), like px."""
    return go.Figure(globe_figure() if fig is None else fig)


def _drawn(fig: go.Figure) -> int:
    """How many data traces fig already has, so a new layer continues the palette."""
    return sum(t.type == "scatter3d" for t in fig.data)


def scatter_globe(
    data_frame: pd.DataFrame | None = None,
    x: Col = "x",
    y: Col = "y",
    z: Col = "z",
    color: Col | None = None,
    size: float | Col = 3,
    hover_name: Col | None = None,
    hover_data: list[Col] | None = None,
    fig: go.Figure | None = None,
    name: str | None = None,
    opacity: float = 1.0,
    line_width: float = 2,
    mode: str = "markers",
    line_group: Col | None = None,
    trace_kw: dict | None = None,
) -> go.Figure:
    """Markers at ECF positions.

    x, y, z    column names (with data_frame) or array-likes, in km.
    color      column/array. Text -> one legend entry per value;
               numbers -> colorscale and colorbar, on the markers (or on the
               line when mode has none; a line fades between its points).
               Integer IDs count as numbers: pass them as text (df.id.astype(str)).
               Each call continues the palette where the figure left off.
    size       marker size: a number or a column/array.
    hover_name, hover_data   as in px; hover_data is a list of column names
               (or arrays, labelled hover_data[0], ...). Hover always shows the
               ECF position.
    fig        globe to draw onto (it is copied, not changed).
    name       legend entry when color is not categorical.
    line_group column/array: rows with the same value are joined in order, and
               different values are separate lines (as in px.line).
    trace_kw   anything else for the plotly trace, e.g. marker_symbol="diamond",
               line_dash="dash", mode="markers+text" (labels from hover_name).
    """

    col = lambda v: _col(data_frame, v)
    d = pd.DataFrame(
        dict(zip("xyz", np.broadcast_arrays(col(x), col(y), col(z)))), dtype=float
    )
    if len(d) and max(abs(d.x).max(), abs(d.y).max(), abs(d.z).max()) > 5e6:
        raise ValueError("positions are over 5 million km out: metres instead of km?")
    cd = []  # hover_data columns, in customdata order
    hover = ["ECF (%{x:,.1f}, %{y:,.1f}, %{z:,.1f}) km"]
    for i, h in enumerate(hover_data or []):
        cd.append(f"hd{i}")
        d[cd[-1]] = col(h)
        label = h if isinstance(h, str) else f"hover_data[{i}]"  # an array has no name
        hover.append(f"{label}: %{{customdata[{i}]}}")
    if hover_name is not None:
        d["text"] = col(hover_name)
        hover.insert(0, "<b>%{text}</b>")
    if color is not None:
        d["color"] = col(color)
        if is_datetime64_any_dtype(d["color"]):  # a scale in seconds from the first
            d["color"] = (d["color"] - d["color"].min()).dt.total_seconds()
    if np.ndim(size) or isinstance(size, str):
        d["size"] = col(size)
    if line_group is not None and len(d):  # joined in order within each group
        d["line"] = col(line_group)

    fig = _canvas(fig)

    # categorical color: one trace (and legend entry) per value
    categorical = color is not None and (
        is_bool_dtype(d["color"]) or not is_numeric_dtype(d["color"])
    )
    groups = (
        d.groupby("color", sort=False, dropna=False)  # dropna: keep missing labels
        if categorical
        else [(name, d)]
    )
    used = _drawn(fig)
    bars = sum(
        bool(
            getattr(t, "marker", None)
            and t.marker.showscale
            or getattr(t, "line", None)
            and t.line.showscale
        )
        for t in fig.data
        if t.type == "scatter3d"
    )
    for k, (label, g) in enumerate(groups, used):
        if "line" in g:
            chunks = [sub for _, sub in g.groupby("line", sort=False)]
            gap = chunks[0].iloc[[0]].assign(x=np.nan, y=np.nan, z=np.nan)
            g = pd.concat([r for c in chunks for r in (c, gap)][:-1], ignore_index=True)
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
        trace = go.Scatter3d(
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
            hovertemplate="<br>".join(hover)
            + f"<extra>{'' if label is None else label}</extra>",
        )
        if trace_kw:
            trace.update(trace_kw)
        fig.add_trace(trace)
    return fig


def line_globe(data_frame: pd.DataFrame | None = None, **kw) -> go.Figure:
    """Polyline through ECF positions in row order. Same arguments as
    scatter_globe; color= with a text column draws one line per value,
    and line_group= draws several lines without splitting colour."""
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
    Other arguments (name, line_width, fig, ...) go to line_globe.
    """
    col = lambda v: _col(data_frame, v)
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

    pick = lambda v: v if isinstance(v, str) else per_end(v)
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
    fig: go.Figure | None = None,
    opacity: float = 0.3,
) -> go.Figure:
    """Uncertainty ellipsoids: a marker at each position, wrapped in the surface
    that lies `sigma` standard deviations out for its covariance.

    x, y, z    ECF position(s) in km: numbers, or one per object.
    cov        ECF position covariance in km squared: (3, 3), or (n, 3, 3)
               for n objects. A covariance in another frame (RIC, ...) has to be
               rotated into ECF first.
    sigma      ellipsoid size in standard deviations: 1 holds about 20% of a
               3-D Gaussian, 3 about 97%.
    name       legend label, or one per object. Clicking it toggles both parts.
    fig        globe to draw onto (it is copied, not changed).
    """
    p = np.column_stack(np.broadcast_arrays(x, y, z)).astype(float)  # (n, 3)
    cov = np.broadcast_to(np.asarray(cov)[..., :3, :3], (len(p), 3, 3))  # 6x6 ok
    if not np.allclose(cov, cov.swapaxes(-1, -2)):
        raise ValueError("cov must be symmetric")
    fig = _canvas(fig)
    used = _drawn(fig)
    if name is None or isinstance(name, str):
        first = len({t.legendgroup for t in fig.data if t.legendgroup})
        name = [name or f"object {first + i}" for i in range(len(p))]
    if len(name) != len(p):
        raise ValueError(f"{len(name)} names for {len(p)} objects")
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
                    f"<b>{label}</b><br>{sigma:g}σ semi-axes: {axes} km<extra></extra>"
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
                showlegend=label not in name[:i],  # one entry per distinct label
                hoverinfo="skip",
            )
        )
    return fig
