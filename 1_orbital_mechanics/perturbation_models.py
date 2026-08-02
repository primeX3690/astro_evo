# 1_orbital_mechanics/perturbation_models.py
"""
J2 (Earth oblateness), solar radiation pressure (SRP), and atmospheric
drag perturbations. HONEST NOTE: drag uses a simplified exponential
density model (Vallado-style reference bands), NOT real NRLMSISE-00.
"""
import numpy as np

J2 = 1.08263e-3
R_EARTH_EQ = 6378.137
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

P_SR = 4.56e-6  # N/m^2, solar radiation pressure at 1 AU
AU_KM = 1.495978707e8

ATMOSPHERE_BANDS = [
    (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00),
]


def j2_acceleration(r_vec, mu, j2=J2, r_eq=R_EARTH_EQ):
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    factor = -1.5 * j2 * mu * r_eq ** 2 / r ** 5
    z2_r2 = 5.0 * z ** 2 / r ** 2
    ax = factor * x * (1 - z2_r2)
    ay = factor * y * (1 - z2_r2)
    az = factor * z * (3 - z2_r2)
    return np.array([ax, ay, az])


def atmospheric_density(altitude_km):
    if altitude_km < 200:
        altitude_km = 200
    if altitude_km > 1000:
        return 0.0
    band = ATMOSPHERE_BANDS[0]
    for b in ATMOSPHERE_BANDS:
        if b[0] <= altitude_km:
            band = b
    h0, rho0, H = band
    return rho0 * np.exp(-(altitude_km - h0) / H)


def drag_acceleration(r_vec, v_vec, cd, area_to_mass_m2_kg, r_earth=R_EARTH_EQ,
                       earth_rotation_rate=EARTH_ROTATION_RATE):
    altitude_km = np.linalg.norm(r_vec) - r_earth
    rho = atmospheric_density(altitude_km)
    if rho == 0.0:
        return np.zeros(3)
    omega_vec = np.array([0, 0, earth_rotation_rate])
    v_atm = np.cross(omega_vec, r_vec)
    v_rel = v_vec - v_atm
    v_rel_ms = v_rel * 1000.0
    v_rel_mag_ms = np.linalg.norm(v_rel_ms)
    if v_rel_mag_ms < 1e-9:
        return np.zeros(3)
    a_drag_ms2 = -0.5 * cd * area_to_mass_m2_kg * rho * v_rel_mag_ms ** 2 * (v_rel_ms / v_rel_mag_ms)
    return a_drag_ms2 / 1000.0


def srp_acceleration(r_vec, sun_direction_unit, cr, area_to_mass_m2_kg,
                      p_sr=P_SR, r_earth=R_EARTH_EQ):
    sun_dir = sun_direction_unit / np.linalg.norm(sun_direction_unit)
    proj = np.dot(r_vec, sun_dir)
    perp = r_vec - proj * sun_dir
    perp_dist = np.linalg.norm(perp)
    in_shadow = (proj < 0) and (perp_dist < r_earth)
    if in_shadow:
        return np.zeros(3)
    a_srp_ms2 = -cr * p_sr * area_to_mass_m2_kg * sun_dir
    return a_srp_ms2 / 1000.0


def two_body_j2_eom(t, state, mu, j2=J2, r_eq=R_EARTH_EQ):
    r, v = state[:3], state[3:]
    a_twobody = -mu * r / np.linalg.norm(r) ** 3
    a_j2 = j2_acceleration(r, mu, j2, r_eq)
    return np.concatenate([v, a_twobody + a_j2])


def full_perturbed_eom(t, state, mu, cd, area_to_mass_drag, cr, area_to_mass_srp,
                        sun_direction_unit=np.array([1.0, 0.0, 0.0]),
                        j2=J2, r_eq=R_EARTH_EQ, include_drag=True, include_srp=True):
    r, v = state[:3], state[3:]
    a = -mu * r / np.linalg.norm(r) ** 3
    a = a + j2_acceleration(r, mu, j2, r_eq)
    if include_drag:
        a = a + drag_acceleration(r, v, cd, area_to_mass_drag, r_eq)
    if include_srp:
        a = a + srp_acceleration(r, sun_direction_unit, cr, area_to_mass_srp, r_earth=r_eq)
    return np.concatenate([v, a])


def analytic_nodal_regression_rate(a, e, i_deg, mu, j2=J2, r_eq=R_EARTH_EQ):
    n = np.sqrt(mu / a ** 3)
    p = a * (1 - e ** 2)
    i = np.radians(i_deg)
    return -1.5 * n * j2 * (r_eq / p) ** 2 * np.cos(i)