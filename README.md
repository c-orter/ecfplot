# ecf_globe

plotly-express-style plotting of ECF (Earth-fixed) coordinates on a 3-D globe.
A WGS84 ellipsoid with low-poly continents, and a handful of functions that
return ordinary `plotly.graph_objects.Figure`s.

Built for an offline Anaconda 2023.07 install (Python 3.11, plotly 5.9, numpy,
pandas, dash 2.3). It needs nothing else.

```python
import ecf_globe as eg

fig = eg.scatter_globe(df, x="x", y="y", z="z", color="sat", units="km")
fig = eg.line_globe(df, color="sat", fig=fig)  # draw onto an existing globe
fig.show()
```

## Functions

| Function | Draws |
| --- | --- |
| `globe_figure(units, coast)` | the empty globe |
| `scatter_globe(df, x, y, z, ...)` | markers at ECF positions |
| `line_globe(df, x, y, z, ...)` | polylines through ECF positions; `line_group=` for several, `steps=N` smooths with a spline |
| `segment_globe(df, x0, y0, z0, x1, y1, z1)` | one segment per row, point 0 to point 1 |
| `covariance_globe(x, y, z, cov)` | uncertainty ellipsoids from 3×3 ECF covariances |
| `geodetic_to_ecf`, `ecf_to_geodetic` | WGS84 conversions |

Each drawing function takes `fig=` to add to a globe and returns a new figure
(the old one is untouched), `units="km"` or `"m"`, and, where it makes sense,
`color`, `size`, `hover_name`, `hover_data`, `name`, `opacity`. Every docstring
lists its arguments. `trace_kw=dict(...)` on `scatter_globe` and `line_globe`
passes anything else to the plotly trace (`marker_symbol`, `line_dash`,
`mode="markers+text"` for labels from `hover_name`).

## Examples

- `python examples.py` runs the example gallery: pick an example and see the
  globe beside the code that drew it. Add your own as a function in
  `examples.py`.
- `python app.py` runs a Dash demo with toggleable layers and a time slider.
- `python demo.py` is the smallest Dash example.

## Layout

```
ecf_globe/earth.py     conversions, Earth surface
ecf_globe/coast.py     hand-drawn continent outlines and their triangulation
ecf_globe/express.py   the plotting functions
examples.py            example gallery app
app.py                 layered demo app
```

## Notes

**Coordinates**
- Inputs are ECF (Earth-fixed, WGS84 / ITRF-aligned) in km or m. There is no ECI
  handling or time handling: rotate ECI data into ECF yourself (the gallery's
  `ecf_time_series` shows the rotation).
- Hover shows the ECF position only (plus any `hover_name` / `hover_data`).
- A covariance is ECF, in the position units squared. For one in another frame
  (RIC, ...): `C_ecf = R @ C @ R.T`, where the columns of `R` are that frame's axes
  written in ECF (see `ric_covariance`).

**Things that look like bugs but aren't**
- Draw lines and markers at least **10 km** above the surface. The continents are
  drawn 6 km up, so anything at altitude 0 is hidden under them or flickers over
  the ocean. `geodetic_to_ecf(lat, lon, 10)` is the usual way to get a ground point.
- Integer IDs in `color=` count as numbers: you get a colour bar and one line that
  jumps between satellites. Pass `df.id.astype(str)` to get one line each.
- Each call continues the colour palette where the figure left off, so layers stay
  distinct. Set a colour yourself with `trace_kw=dict(line_color="red")`.
- Two numeric-colour layers each get a colour bar, placed side by side.
- A figure title needs room: `fig.update_layout(title="...", margin_t=40)`.
- `steps=N` smooths with a spline; for points a few degrees apart. It runs low
  between points, by about 3 km at 5 degrees of spacing, 12 km at 10, 49 km at 20
  and 113 km at 30 (at a 6400 km radius), and it is not a great circle.
- Wrong units and swapped arguments are caught where cheaply possible: latitude
  beyond 90 degrees, km positions over 5 million km out, and a covariance that is
  not symmetric or positive semi-definite all raise.

**Other**
- The continents are rough and lifted about 6 km off the surface. They give
  context and are not map data. Edit the `COAST` dict in `coast.py` to change
  them.
- Formatting is `ruff format`; type hints sit on the public functions only.
