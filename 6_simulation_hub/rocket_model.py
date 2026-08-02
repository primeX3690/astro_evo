# 6_simulation_hub/rocket_model.py
"""Rocket vehicle model: dry mass, propellant, thrust, Isp - Tsiolkovsky equation math."""
import numpy as np

G0 = 9.80665e-3  # km/s^2


class RocketModel:
    def __init__(self, dry_mass_kg, propellant_mass_kg, isp_s, thrust_n):
        self.dry_mass = dry_mass_kg
        self.propellant_mass = propellant_mass_kg
        self.isp = isp_s
        self.thrust = thrust_n
        self.exhaust_velocity_kms = isp_s * G0

    @property
    def wet_mass(self):
        return self.dry_mass + self.propellant_mass

    @property
    def mass_flow_rate_kg_s(self):
        v_e_ms = self.exhaust_velocity_kms * 1000.0
        return self.thrust / v_e_ms

    def delta_v_available(self):
        return self.exhaust_velocity_kms * np.log(self.wet_mass / self.dry_mass)

    def burn_time_for_delta_v(self, delta_v_kms):
        if delta_v_kms > self.delta_v_available():
            raise ValueError(
                f"Requested delta-V {delta_v_kms:.4f} km/s exceeds available "
                f"{self.delta_v_available():.4f} km/s for this propellant load"
            )
        mass_ratio = np.exp(delta_v_kms / self.exhaust_velocity_kms)
        final_mass = self.wet_mass / mass_ratio
        propellant_used_kg = self.wet_mass - final_mass
        return propellant_used_kg / self.mass_flow_rate_kg_s

    def thrust_accel_kms2(self, current_mass_kg):
        thrust_kg_km_s2 = self.thrust / 1000.0
        return thrust_kg_km_s2 / current_mass_kg