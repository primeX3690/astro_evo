# 6_simulation_hub/orbit_visualizer.py
"""Plotting utilities - orbit trajectory, ground track, delta-V budget bar chart."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EARTH_RADIUS_KM = 6371.0


def plot_orbit_trajectory(trajectory, title="Orbit Trajectory", filepath="orbit.png"):
    x, y = trajectory[:, 0], trajectory[:, 1]
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(EARTH_RADIUS_KM * np.cos(theta), EARTH_RADIUS_KM * np.sin(theta),
            color="steelblue", alpha=0.4, label="Earth")
    ax.plot(x, y, color="orange", linewidth=1.5, label="Trajectory")
    ax.plot(x[0], y[0], "go", markersize=8, label="Start")
    ax.plot(x[-1], y[-1], "rs", markersize=8, label="End")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    return filepath


def plot_ground_track(trajectory, dt_s, earth_rotation_rate_deg_s=0.004178, filepath="ground_track.png"):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon_inertial = np.degrees(np.arctan2(y, x))

    t = np.arange(len(trajectory)) * dt_s
    earth_rotation_deg = earth_rotation_rate_deg_s * t
    lon = ((lon_inertial - earth_rotation_deg + 180) % 360) - 180

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(lon, lat, s=3, c=t, cmap="viridis")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground Track")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    return filepath


def plot_delta_v_budget(dv1, dv2, labels=("Departure burn", "Arrival burn"),
                         title="Delta-V Budget", filepath="delta_v_budget.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    values = [dv1, dv2]
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylabel("Delta-V (km/s)")
    ax.set_title(f"{title} (total = {dv1+dv2:.3f} km/s)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)
    return filepath