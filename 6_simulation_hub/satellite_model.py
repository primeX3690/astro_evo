# 6_simulation_hub/satellite_model.py
"""Satellite bus model: solar power generation vs subsystem draw, spin-stabilized attitude."""
import numpy as np

SOLAR_CONSTANT_W_M2 = 1361.0


class SatelliteModel:
    def __init__(self, solar_panel_area_m2, panel_efficiency, subsystem_power_draws_w, battery_capacity_wh):
        self.solar_panel_area = solar_panel_area_m2
        self.panel_efficiency = panel_efficiency
        self.subsystem_power_draws = dict(subsystem_power_draws_w)
        self.battery_capacity_wh = battery_capacity_wh

    def solar_power_generated_w(self, sun_incidence_angle_deg):
        angle = np.radians(sun_incidence_angle_deg)
        if sun_incidence_angle_deg >= 90.0:
            return 0.0
        return SOLAR_CONSTANT_W_M2 * self.solar_panel_area * self.panel_efficiency * np.cos(angle)

    def total_power_draw_w(self):
        return sum(self.subsystem_power_draws.values())

    def power_balance_w(self, sun_incidence_angle_deg):
        return self.solar_power_generated_w(sun_incidence_angle_deg) - self.total_power_draw_w()

    def simulate_orbit_power(self, orbit_period_s, eclipse_fraction, dt=60.0):
        n_steps = int(orbit_period_s // dt)
        eclipse_start = orbit_period_s * (1 - eclipse_fraction)
        battery_wh = self.battery_capacity_wh
        times, batt_trace = [], []
        t = 0.0
        for _ in range(n_steps):
            in_eclipse = t >= eclipse_start
            angle = 90.0 if in_eclipse else 0.0
            balance_w = self.power_balance_w(angle)
            battery_wh += balance_w * (dt / 3600.0)
            battery_wh = min(battery_wh, self.battery_capacity_wh)
            times.append(t)
            batt_trace.append(battery_wh)
            t += dt
        return np.array(times), np.array(batt_trace), min(batt_trace)


class SpinStabilizedAttitude:
    def __init__(self, spin_rate_deg_s, moment_of_inertia_kg_m2):
        self.spin_rate = np.radians(spin_rate_deg_s)
        self.moment_of_inertia = moment_of_inertia_kg_m2
        self.angular_momentum = self.moment_of_inertia * self.spin_rate

    def orientation_at(self, t_s, initial_angle_rad=0.0):
        return (initial_angle_rad + self.spin_rate * t_s) % (2 * np.pi)

    def desaturation_delta_v_check(self, reaction_wheel_max_momentum, current_momentum):
        return abs(current_momentum) / reaction_wheel_max_momentum