# ecfplot

plotly-express-style plotting of ECF (Earth-fixed) coordinates on a 3-D globe.
A WGS84 ellipsoid with simplified continents, and a handful of functions that
return ordinary `plotly.graph_objects.Figure`s.

Built for an offline Anaconda 2023.07 install (Python 3.11, plotly 5.9, numpy,
pandas, dash 2.3). It needs nothing else.

```python
import ecfplot as ep

fig = ep.scatter_globe(df, x="x", y="y", z="z", color="sat")
fig = ep.line_globe(df, color="sat", fig=fig)  # draw onto an existing globe
fig.show()
```

## Functions

| Function | Draws |
| --- | --- |
| `globe_figure(coast)` | the empty globe |
| `scatter_globe(df, x, y, z, ...)` | markers at ECF positions |
| `line_globe(df, x, y, z, ...)` | polylines through ECF positions; `line_group=` for several |
| `segment_globe(df, x0, y0, z0, x1, y1, z1)` | one segment per row, point 0 to point 1 |
| `covariance_globe(x, y, z, cov)` | uncertainty ellipsoids from 3×3 ECF covariances |
| `geodetic_to_ecf`, `ecf_to_geodetic` | WGS84 conversions |
| `enu_matrix`, `ecf_to_enu`, `enu_to_ecf` | local east / north / up at a site: the rotation, and points to and from it |
| `ecf_to_aer` | azimuth, elevation and range from a site |

Each drawing function takes `fig=` to add to a globe and returns a new figure
(the old one is untouched), and, where it makes sense,
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
ecfplot/earth.py       conversions, Earth surface
ecfplot/coast.py       Natural Earth 50m land (fitted to whole degrees) and its triangulation
ecfplot/express.py     the plotting functions
examples.py            example gallery app
app.py                 layered demo app
demo.py                smallest Dash example (also provides orbit() to the demo app)
packages.md            package versions in the target Anaconda 2023.07 install
```

## Notes

**Coordinates**
- Inputs are ECF (Earth-fixed, WGS84 / ITRF-aligned) in **km** throughout; for metres, divide by 1000. There is no ECI
  handling or time handling: rotate ECI data into ECF yourself (the gallery's
  `ecf_time_series` shows the rotation).
- Hover shows the ECF position only (plus any `hover_name` / `hover_data`).
- A covariance is ECF, in km squared. For one in local east /
  north / up axes at an ECF position: `R = enu_matrix(x0, y0, z0)` and
  `C_ecf = R.T @ C_enu @ R` (see `enu_covariance`). `ecf_to_enu` and `enu_to_ecf`
  convert points to and from east, north, up around a site given as an ECF position,
  which is what elevation masks and azimuths need (see `access`). Other frames
  (RIC, ...) are up to you.

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
- A plotly `go.Cone` (arrow heads) on the globe skews the automatic axis scaling and
  stretches the Earth into an egg. Fix the axes to a cube, as `velocity_arrows` does:
  `fig.update_layout(scene=dict(aspectmode="cube", xaxis_range=[-8000, 8000], ...))`.
- Mistakes are caught where cheaply possible: latitude beyond 90 degrees,
  positions over 5 million km out (metres?), and a covariance that is not
  symmetric or positive semi-definite all raise.

**Other**
- The continents are fitted to Natural Earth 50m land (public domain) on a
  whole-degree grid, sharpest over North America, Asia and the Middle East. They
  sit at altitude 0 and the ocean is drawn 6 km down, so points at altitude 0 show
  on top of the land. They give context and are not map data. Edit
  the `LAND` text in `coast.py` to change them: each landmass is a `Name:` line then
  `lon lat` pairs; Eurasia runs past 180 to 190 E, Antarctica runs along the pole,
  and lakes (the Caspian too) are filled in.
- Formatting is `ruff format`; type hints sit on the public functions only.
