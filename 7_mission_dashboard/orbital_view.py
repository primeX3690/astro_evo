# 7_mission_dashboard/orbital_view.py
"""Pure data-generation functions for 2D/3D orbit rendering - Streamlit-independent, testable standalone."""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "1_orbital_mechanics"))
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH  # noqa: E402

EARTH_RADIUS_KM = 6371.0


def get_orbit_points_3d(a, e, i, raan, argp, n_points=200, mu=MU_EARTH):
    nus = np.linspace(0, 360, n_points, endpoint=False)
    points = np.zeros((n_points, 3))
    for idx, nu in enumerate(nus):
        r_vec, _ = keplerian_to_state_vector(a, e, i, raan, argp, nu, mu)
        points[idx] = r_vec
    return points


def get_earth_sphere(radius_km=EARTH_RADIUS_KM, resolution=30):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def summarize_mission(mission_spec, constraint_report):
    return {
        "Mission": mission_spec.name,
        "Origin radius (km)": round(mission_spec.r1, 1),
        "Destination radius (km)": round(mission_spec.r2, 1),
        "Delta-V required (km/s)": round(constraint_report["dv_total_kms"], 4),
        "Transfer time (hr)": round(constraint_report["transfer_time_hr"], 2),
        "Propellant mass fraction": round(constraint_report["propellant_mass_fraction"], 4),
        "Feasible": constraint_report["feasible"],
    }


def get_rotating_earth_sphere(rotation_deg, radius_km=EARTH_RADIUS_KM, resolution=30):
    """Earth sphere mesh genuinely rotated by rotation_deg about the polar (z) axis."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))

    theta = np.radians(rotation_deg)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    lon = np.degrees(np.arctan2(y, x))
    stripes = np.sin(np.radians(lon) * 6)
    return x_rot, y_rot, z, stripes


def build_rotating_earth_figure(orbit_points_eci, n_frames=36, rotation_per_frame_deg=10.0):
    """Plotly Figure with animation frames: Earth rotates, ground track builds up progressively."""
    import plotly.graph_objects as go

    def frame_data(rotation_deg, n_track_points):
        x, y, z, stripes = get_rotating_earth_sphere(rotation_deg)
        earth_trace = go.Surface(x=x, y=y, z=z, surfacecolor=stripes,
                                  colorscale="Blues", showscale=False, opacity=0.85)
        track = orbit_points_eci[:n_track_points]
        track_trace = go.Scatter3d(x=track[:, 0], y=track[:, 1], z=track[:, 2],
                                    mode="lines", line=dict(color="red", width=5))
        return [earth_trace, track_trace]

    n_points_total = len(orbit_points_eci)
    frames = []
    for i in range(n_frames):
        rotation = i * rotation_per_frame_deg
        n_track = int(n_points_total * (i + 1) / n_frames)
        frames.append(go.Frame(data=frame_data(rotation, n_track), name=str(i)))

    fig = go.Figure(data=frame_data(0, max(1, n_points_total // n_frames)), frames=frames)
    fig.update_layout(
        scene=dict(aspectmode="data"), height=600,
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
    )
    return fig