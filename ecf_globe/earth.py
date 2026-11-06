"""WGS84 Earth: ECF <-> geodetic conversion and the Earth surface trace."""

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

A = 6378137.0  # WGS84 semi-major axis, metres
F = 1 / 298.257223563  # flattening
E2 = F * (2 - F)  # eccentricity squared
UNITS = {"m": 1.0, "km": 1e3}
OCEAN_COLOR = "#1f5fa8"
LIGHTING = dict(
    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1, roughness=0.9),
    lightposition=dict(x=3e4, y=2e4, z=2e4),
)
# plotly draws contour circles under the cursor on every Surface unless told not to
NO_HIGHLIGHT = {axis: dict(highlight=False) for axis in "xyz"}


def geodetic_to_ecf(
    lat: ArrayLike, lon: ArrayLike, alt: ArrayLike = 0.0, units: str = "km"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geodetic lat, lon (deg), alt -> ECF x, y, z. Lengths in `units`."""
    if np.any(np.abs(lat) > 90):
        raise ValueError("latitude beyond 90 degrees: are lat and lon swapped?")
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / UNITS[units] / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + alt) * np.sin(lat)
    return x, y, z


def ecf_to_geodetic(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, units: str = "km"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECF x, y, z -> geodetic lat (deg), lon (deg), alt. Lengths in `units`."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    a = A / UNITS[units]
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(5):  # error shrinks ~E2x per pass: sub-mm well before 5
        n = a / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + E2 * n * np.sin(lat), p)
    alt = p * np.cos(lat) + z * np.sin(lat) - a * np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), alt


def earth_trace(units: str = "km") -> go.Surface:
    """Solid blue, non-hoverable WGS84 Earth surface."""
    lat, lon = np.mgrid[-90:90:61j, -180:180:121j]  # 3-degree grid
    x, y, z = geodetic_to_ecf(lat, lon, 0.0, units)
    return go.Surface(
        contours=NO_HIGHLIGHT,
        x=np.round(x, 1),  # 0.1 km (or m): the JSON is a third the size
        y=np.round(y, 1),
        z=np.round(z, 1),
        colorscale=[[0, OCEAN_COLOR], [1, OCEAN_COLOR]],
        showscale=False,
        hoverinfo="skip",
        name="Earth",
        showlegend=False,
        **LIGHTING,
    )
