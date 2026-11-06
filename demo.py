"""Synthetic ECF demo: two orbits + a ground station. Run: python demo.py"""

import numpy as np
import pandas as pd

import ecf_globe as eg


def orbit(name, a_km, inc_deg, n=400, spin=True):
    """Circular orbit in ECI, rotated into ECF by Earth rotation (so it spirals)."""
    t = np.linspace(0, 2 * np.pi * 1.5, n)
    inc = np.radians(inc_deg)
    x = a_km * np.cos(t)
    y = a_km * np.sin(t) * np.cos(inc)
    z = a_km * np.sin(t) * np.sin(inc)
    th = -t * (0.0 if not spin else 1 / 15)  # crude Earth rotation relative to orbit
    xe, ye = x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)
    return pd.DataFrame(dict(sat=name, x=xe, y=ye, z=z, t_min=t * 8))


def make_figure():
    df = pd.concat(
        [
            orbit("LEO-53", 6900, 53),
            orbit("Polar", 7200, 90),
            orbit("GEO-ish", 42164, 5, spin=False),
        ],
        ignore_index=True,
    )
    fig = eg.line_globe(df, color="sat", hover_name="sat", hover_data=["t_min"])
    gx, gy, gz = eg.geodetic_to_ecf(38.9, -77.0, 0.0)  # Washington DC
    return eg.scatter_globe(x=[gx], y=[gy], z=[gz], size=8, name="Ground stn", fig=fig)


if __name__ == "__main__":
    from dash import Dash, dcc, html

    app = Dash(__name__)
    app.layout = html.Div([dcc.Graph(figure=make_figure(), style={"height": "95vh"})])
    app.run_server(debug=True)
