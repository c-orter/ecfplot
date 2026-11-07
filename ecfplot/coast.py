"""Land outlines, drawn as filled land.

LAND is fitted to Natural Earth 50m land (public domain) on a whole-degree grid,
with the most points spent on North America, Asia and the Middle East; lakes and
the Caspian are filled in. 39 landmasses, 401 points. Each entry is a name line
ending in ":" and then "lon lat" pairs in degrees. Each outline closes on its own
(the last point joins the first), and:
- Afro-Eurasia runs on past 180 to 190 E, so Chukotka is not cut off at the
  antimeridian;
- Antarctica runs from -180 -90 to 180 -90, so its closing edge is the pole;
- Hawaii and the Aleutians are too small for whole degrees and use tenths.
"""

from functools import lru_cache

import numpy as np
import plotly.graph_objects as go

from .earth import LIGHTING, geodetic_to_ecf

LAND = """
Afro-Eurasia:
    15 55, 4 60, 18 70, 28 71, 44 67, 66 70, 71 74, 74 72, 104 78, 114 76, 113 73,
    128 74, 130 71, 143 73, 169 69, 177 70, 190 66, 187 64, 181 66, 178 63, 169 60,
    163 60, 163 56, 157 51, 155 56, 166 63, 154 60, 142 59, 136 55, 142 53, 139 47,
    134 43, 131 43, 128 39, 130 36, 126 34, 127 37, 121 41, 118 38, 123 37, 120 36,
    122 29, 117 23, 106 20, 110 12, 104 8, 105 10, 101 13, 99 10, 103 5, 104 1, 101 3,
    100 7, 98 8, 99 12, 97 17, 94 15, 95 18, 91 22, 87 21, 81 16, 79 9, 77 8, 73 17,
    73 21, 70 21, 66 26, 62 25, 53 27, 49 31, 48 29, 52 24, 56 25, 60 22, 56 18, 44 12,
    42 18, 35 28, 34 27, 40 15, 45 10, 51 12, 49 5, 43 0, 39 -5, 41 -15, 35 -20, 36 -24,
    28 -33, 19 -35, 14 -22, 12 -18, 14 -9, 9 -1, 10 3, 3 7, 0 5, -9 5, -17 13, -16 24,
    -8 34, 5 37, 11 37, 10 34, 20 30, 21 33, 28 31, 35 32, 36 37, 28 36, 26 40, 33 42,
    42 41, 37 47, 29 46, 28 41, 24 41, 22 37, 19 42, 14 45, 2 43, -2 37, -9 37, -9 44,
    -2 43, 0 45, -3 48, 9 55, 14 54, 21 55, 22 66
Americas:
    -95 72, -87 68, -83 70, -81 67, -90 64, -95 60, -93 58, -83 55, -80 51, -77 57,
    -78 63, -72 62, -67 58, -65 61, -61 56, -55 53, -57 51, -65 50, -65 47, -59 46,
    -65 44, -68 45, -75 39, -76 35, -81 32, -80 24, -83 28, -83 30, -93 30, -97 28,
    -98 21, -94 18, -86 23, -89 16, -83 16, -84 10, -81 9, -71 12, -61 9, -52 5, -49 0,
    -35 -5, -41 -22, -48 -25, -50 -31, -65 -43, -70 -55, -75 -53, -74 -40, -72 -34,
    -70 -19, -75 -15, -82 -4, -77 5, -79 9, -81 7, -94 16, -98 16, -105 19, -106 23,
    -116 33, -108 21, -115 29, -118 34, -121 35, -124 40, -124 48, -141 61, -153 59,
    -159 56, -163 55, -157 58, -161 59, -167 62, -160 64, -168 65, -161 71, -138 70,
    -99 68
Antarctica:
    -180 -90, -180 -84, -147 -76, -91 -73, -74 -74, -66 -66, -59 -74, -80 -79, -56 -84,
    -36 -79, -7 -71, 34 -70, 54 -66, 74 -70, 85 -67, 106 -66, 135 -66, 169 -71, 159 -82,
    180 -84, 180 -90
Australia:
    146 -17, 154 -28, 149 -39, 141 -38, 132 -31, 117 -35, 113 -25, 115 -21, 120 -20,
    126 -14, 138 -11, 135 -15, 141 -18, 142 -10
Greenland:
    -31 84, -12 82, -23 70, -40 65, -43 60, -48 61, -52 64, -56 74, -59 76, -70 76,
    -67 81
New Guinea:
    134 -1, 135 -4, 137 -1, 145 -4, 151 -12, 145 -8, 141 -9, 137 -5, 134 -4, 130 0
Borneo:
    117 7, 118 6, 118 0, 116 -3, 111 -4, 109 0
Madagascar:
    48 -24, 44 -26, 44 -17, 50 -13
Baffin:
    -76 73, -62 67, -67 62, -73 64, -77 70, -89 70, -90 74
Sumatra:
    103 1, 106 -4, 105 -7, 94 7
Honshu:
    142 40, 140 35, 130 34, 140 38, 140 42
Victoria Island:
    -106 73, -101 69, -114 68, -120 73
Great Britain:
    1 53, -4 52, -6 60
Ellesmere:
    -63 83, -79 77, -90 76, -93 82
Sulawesi:
    126 1, 120 0, 123 -5, 119 -3, 120 1
Java:
    110 -6, 116 -9, 104 -7
Newfoundland:
    -56 52, -52 46, -56 48, -60 47
Cuba:
    -80 23, -74 20, -77 20
Luzon:
    121 19, 123 18, 121 13, 120 16
Mindanao:
    126 9, 127 6, 122 8
Hokkaido:
    146 43, 139 42, 142 44, 141 46
Sakhalin:
    142 55, 144 50, 142 46
Hispaniola:
    -68 19, -73 18, -73 20
Banks Island:
    -114 74, -124 71, -123 75
Sri Lanka:
    80 11, 82 7, 80 5
Hawaii:
    -155.6 19, -155.9 19.1, -156 19.7, -155.8 20, -155.8 20.3, -155.2 20, -154.8 19.5
Maui:
    -156.5 20.9, -156.3 21, -156 20.8, -156.4 20.6, -156.7 20.9, -156.6 21
Oahu:
    -157.8 21.5, -157.6 21.3, -158.1 21.3, -158.3 21.6, -158 21.7
Kauai:
    -159.4 21.9, -159.6 21.9, -159.8 22, -159.6 22.2, -159.4 22.2
Kodiak:
    -152 59, -153 57, -155 57
Unimak:
    -163.5 55, -163.1 54.7, -164.8 54.4, -164.5 54.9
Unalaska:
    -166.6 53.9, -166.2 53.9, -166.4 53.7, -167.8 53.3, -166.8 53.6, -167 53.9
Umnak:
    -168 53.3, -169.1 52.8, -168.3 53.5, -167.8 53.5
Atka:
    -175.3 52, -174 52.3, -174.1 52.1
Adak:
    -176.5 51.7, -177 51.6, -176.7 52
Tanaga:
    -178.1 51.7, -178.2 51.9, -177.6 51.8
Amchitka:
    179.5 51.4, 179.3 51.4, 178.6 51.6, 178.9 51.6
Kiska:
    177.3 51.9, 177.7 52.1, 177.6 51.9
Attu:
    172.8 53, 173.4 52.9, 172.9 52.8, 172.5 52.9
"""


def outlines(text: str = LAND) -> dict[str, np.ndarray]:
    """LAND text -> {name: (n, 2) lon, lat}, checked for the usual typos."""
    words = {}
    for line in text.strip().splitlines():
        if line.endswith(":"):
            name = line[:-1]
            words[name] = []
        else:
            words[name] += line.replace(",", " ").split()
    out = {}
    for name, w in words.items():
        p = np.array(w, float).reshape(-1, 2)  # fails on an odd count: a lost number
        step = np.diff(p, axis=0)
        km = np.hypot(step[:, 0] * np.cos(np.radians(p[1:, 1])), step[:, 1]) * 111
        if np.abs(p[:, 1]).max() > 90 or km.max() > 2500:
            raise ValueError(f"LAND {name!r}: a point is far off, check the digits")
        out[name] = p
    return out


def _cross(a, b, q):
    """> 0 where a -> b -> q turns left; q may be (n, 2)."""
    return (b[0] - a[0]) * (q[..., 1] - a[1]) - (b[1] - a[1]) * (q[..., 0] - a[0])


def _ear_clip(p):
    """Triangulate a simple polygon (n, 2) -> list of (i, j, k) vertex indices.

    Copes with repeated points and points in line.
    """
    idx = list(range(len(p)))
    if np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) < 0:
        idx.reverse()  # make it counter-clockwise
    tris, m, miss = [], 0, 0
    while len(idx) > 3:
        n = len(idx)
        m %= n
        i, j, k = idx[m - 1], idx[m], idx[(m + 1) % n]
        a, b, c = p[i], p[j], p[k]
        turn = _cross(a, b, c)
        if turn > 0 or (turn == 0 and miss > n):  # convex (or flat, once stuck)
            q = p[idx]
            inside = (
                (_cross(a, b, q) >= 0) & (_cross(b, c, q) >= 0) & (_cross(c, a, q) >= 0)
            )
            corner = (q == a).all(1) | (q == b).all(1) | (q == c).all(1)
            if not (inside & ~corner).any():
                tris.append((i, j, k))
                del idx[m]
                miss = 0
                continue
        m, miss = m + 1, miss + 1
        if miss > 2 * n:
            raise ValueError("a LAND outline crosses itself; cannot fill it")
    return tris + [tuple(idx)]


def _length(a, b):
    """Rough great-circle length, in degrees, between two (lon, lat) points."""
    return np.hypot((b[0] - a[0]) * np.cos(np.radians((a[1] + b[1]) / 2)), b[1] - a[1])


def _split(t, step):
    """Bisect a (3, 2) triangle across its longest edge until all edges <= step."""
    m = int(np.argmax([_length(t[m - 1], t[m]) for m in range(3)]))
    a, b, c = t[m - 1], t[m], t[m - 2]  # a-b is the longest edge
    if _length(a, b) <= step:
        return [t]
    mid = (a + b) / 2
    return _split(np.array([a, mid, c]), step) + _split(np.array([mid, b, c]), step)


@lru_cache
def land_trace() -> go.Mesh3d:
    """Filled LAND outlines as a non-hoverable Mesh3d, corners at altitude 0.

    Each outline is triangulated in lon/lat, then the triangles are split to
    at most 4 degrees so they follow the curved surface. Their middles sag up to
    about 5 km below it, so earth_trace draws the ocean 6 km down: land never
    covers anything at altitude 0.
    """
    tris = []
    for p in outlines().values():
        for t in _ear_clip(p):
            tris += _split(p[list(t)], 4.0)
    lonlat, corner = np.unique(np.vstack(tris), axis=0, return_inverse=True)
    lon, lat = lonlat.T
    x, y, z = geodetic_to_ecf(lat, lon)
    i, j, k = corner.reshape(-1, 3).T
    return go.Mesh3d(
        x=np.round(x, 1),
        y=np.round(y, 1),
        z=np.round(z, 1),
        i=i,
        j=j,
        k=k,
        color="#8fcf8f",
        hoverinfo="skip",
        name="Land",
        showlegend=False,
        **LIGHTING,
    )
