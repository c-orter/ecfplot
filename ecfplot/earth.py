"""WGS84 Earth: ECF <-> geodetic conversion and the Earth surface trace.

All lengths are in km (multiply by 1000 yourself if you work in metres).
"""

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike

A = 6378.137  # WGS84 semi-major axis, km
F = 1 / 298.257223563  # flattening
E2 = F * (2 - F)  # eccentricity squared
OCEAN_COLOR = "#1f5fa8"
LIGHTING = dict(
    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1, roughness=0.9),
    lightposition=dict(x=3e4, y=2e4, z=2e4),
)
# plotly draws contour circles under the cursor on every Surface unless told not to
NO_HIGHLIGHT = {axis: dict(highlight=False) for axis in "xyz"}


def geodetic_to_ecf(
    lat: ArrayLike, lon: ArrayLike, alt: ArrayLike = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geodetic lat, lon (deg), alt (km) -> ECF x, y, z (km)."""
    lat, lon, alt = (np.asarray(v, float) for v in (lat, lon, alt))
    if np.any(np.abs(lat) > 90):
        raise ValueError("latitude beyond 90 degrees: are lat and lon swapped?")
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + alt) * np.sin(lat)
    return x, y, z


def ecf_to_geodetic(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECF x, y, z (km) -> geodetic lat (deg), lon (deg), alt (km)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(5):  # error shrinks ~E2x per pass: sub-mm well before 5
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + E2 * n * np.sin(lat), p)
    alt = p * np.cos(lat) + z * np.sin(lat) - A * np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), alt


def enu_matrix(x0: ArrayLike, y0: ArrayLike, z0: ArrayLike) -> np.ndarray:
    """Rotation from ECF to local east, north, up at the ECF position x0, y0, z0.

    (3, 3), or (..., 3, 3) for arrays. The rows are the east, north and up unit
    vectors written in ECF ("up" is the ellipsoid normal, the usual ENU), so
    `R @ v_ecf` is `v_enu`, and a covariance goes back with `C_ecf = R.T @ C_enu @ R`.
    """
    lat, lon, _ = ecf_to_geodetic(x0, y0, z0)
    lat, lon = np.broadcast_arrays(np.radians(lat), np.radians(lon))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    east = np.stack([-so, co, 0 * so], -1)
    north = np.stack([-sl * co, -sl * so, cl], -1)
    up = np.stack([cl * co, cl * so, sl], -1)
    return np.stack([east, north, up], -2)


def ecf_to_enu(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, x0: float, y0: float, z0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECF points -> east, north, up from the site at the ECF position x0, y0, z0."""
    rho = np.column_stack(np.broadcast_arrays(x, y, z)) - [x0, y0, z0]  # site to point
    east, north, up = enu_matrix(x0, y0, z0) @ rho.T
    return east, north, up


def enu_to_ecf(
    east: ArrayLike, north: ArrayLike, up: ArrayLike, x0: float, y0: float, z0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East, north, up from the site at the ECF position x0, y0, z0 -> ECF points."""
    enu = np.column_stack(np.broadcast_arrays(east, north, up))
    x, y, z = enu_matrix(x0, y0, z0).T @ enu.T + np.array([[x0], [y0], [z0]])
    return x, y, z


def ecf_to_aer(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, x0: float, y0: float, z0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECF points -> azimuth (deg, clockwise from north), elevation (deg) and range
    (km) from the site at the ECF position x0, y0, z0."""
    e, n, u = ecf_to_enu(x, y, z, x0, y0, z0)
    rng = np.sqrt(e**2 + n**2 + u**2)
    return np.degrees(np.arctan2(e, n)) % 360, np.degrees(np.arcsin(u / rng)), rng


def earth_trace() -> go.Surface:
    """Solid blue, non-hoverable WGS84 Earth surface, 6 km down so the land
    (whose flat triangles sag up to 5 km) and anything at altitude 0 show on top."""
    lat, lon = np.mgrid[-90:90:61j, -180:180:121j]  # 3-degree grid
    x, y, z = geodetic_to_ecf(lat, lon, -6.0)
    return go.Surface(
        contours=NO_HIGHLIGHT,
        x=np.round(x, 1),  # 0.1 km: the JSON is a third the size
        y=np.round(y, 1),
        z=np.round(z, 1),
        colorscale=[[0, OCEAN_COLOR], [1, OCEAN_COLOR]],
        showscale=False,
        hoverinfo="skip",
        name="Earth",
        showlegend=False,
        **LIGHTING,
    )
