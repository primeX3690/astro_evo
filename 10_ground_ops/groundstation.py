# 10_ground_ops/groundstation.py
"""Ground station geometry: geodetic<->ECEF, ECI->topocentric az/el/range (WGS-84 + GMST)."""
import numpy as np
from datetime import datetime

WGS84_A = 6378.137
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2


def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])


def gmst_rad(dt_utc):
    jd = (
        367 * dt_utc.year
        - int(7 * (dt_utc.year + int((dt_utc.month + 9) / 12)) / 4)
        + int(275 * dt_utc.month / 9)
        + dt_utc.day + 1721013.5
        + (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24
    )
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600 * 3600 + 8640184.812866) * T
        + 0.093104 * T ** 2
        - 6.2e-6 * T ** 3
    )
    gmst_deg = (gmst_sec % 86400.0) / 240.0
    return np.radians(gmst_deg % 360.0)


def eci_to_ecef(r_eci_km, dt_utc):
    theta = gmst_rad(dt_utc)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return R @ np.asarray(r_eci_km)


class GroundStation:
    def __init__(self, name, lat_deg, lon_deg, alt_km=0.0):
        self.name = name
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.alt_km = alt_km
        self.r_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_km)

    def look_angles(self, r_sat_eci_km, dt_utc: datetime):
        r_sat_ecef = eci_to_ecef(r_sat_eci_km, dt_utc)
        rho_ecef = r_sat_ecef - self.r_ecef
        lat, lon = np.radians(self.lat_deg), np.radians(self.lon_deg)
        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ])
        east, north, up = R @ rho_ecef
        rng = np.linalg.norm(rho_ecef)
        el = np.degrees(np.arcsin(up / rng))
        az = (np.degrees(np.arctan2(east, north))) % 360.0
        return az, el, rng
