# 13_realtime_telemetry/telemetry_server.py
"""
Real-time telemetry streaming: pushes live CCSDS Space Packets (module 9's
wire format) over UDP as a satellite's state is propagated - this is the
actual mechanism real ground stations and mission-control front-ends
ingest telemetry by (UDP/TCP socket streams of CCSDS packets), as opposed
to reading a static file after the fact.

HONEST NOTE: this is software-only real-time streaming (an orbit
propagator feeding a live socket at wall-clock-paced intervals) - there
is no physical spacecraft hardware or RF front-end involved, so it is
NOT genuine Hardware-in-the-Loop (HIL) testing. What it does prove is
the data-plane architecture: a real socket, real CCSDS-framed packets,
consumed by an independent process in real time - the piece HIL testing
would plug into once real hardware (or a hardware simulator box) exists.
"""
import socket
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "9_ccsds_interop"))
from ccsds_tm_packet import pack_telemetry_packet  # noqa: E402


class TelemetryUDPServer:
    def __init__(self, host="127.0.0.1", port=52001, apid=200):
        self.host = host
        self.port = port
        self.apid = apid
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq_count = 0

    def send_state(self, t_s, pos_km, vel_kms, battery_wh):
        packet = pack_telemetry_packet(
            apid=self.apid, seq_count=self.seq_count % 0x4000,
            t_s=t_s, pos_km=pos_km, vel_kms=vel_kms, battery_wh=battery_wh,
        )
        self.sock.sendto(packet, (self.host, self.port))
        self.seq_count += 1
        return packet

    def stream_trajectory(self, times_s, states, battery_series=None, rate_hz=10.0, verbose=True):
        """Streams one CCSDS TM packet per state, paced at rate_hz (real
        wall-clock pacing, not simulation-time pacing) - demonstrates an
        actual live feed, not just a fast dump."""
        period = 1.0 / rate_hz
        for i, (t, state) in enumerate(zip(times_s, states)):
            battery = battery_series[i] if battery_series is not None else 100.0
            self.send_state(t, state[:3], state[3:6], battery)
            if verbose and i % 10 == 0:
                print(f"[telemetry_server] sent packet seq={self.seq_count-1} t={t:.1f}s "
                      f"pos=({state[0]:.1f},{state[1]:.1f},{state[2]:.1f}) km")
            time.sleep(period)

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_orbital_mechanics"))
    from keplerian_orbit import keplerian_to_state_vector, MU_EARTH
    from perturbation_models import two_body_j2_eom
    from two_body_problem import rk4_step

    r0, v0 = keplerian_to_state_vector(a=6793.0, e=0.001, i=51.6, raan=0, argp=0, nu=0)
    state = np.concatenate([r0, v0])
    dt_s = 1.0
    n = 50
    times = np.arange(n) * dt_s
    states = np.zeros((n, 6))
    states[0] = state
    for k in range(1, n):
        state = rk4_step(lambda t, y: two_body_j2_eom(t, y, MU_EARTH), times[k - 1], state, dt_s)
        states[k] = state

    server = TelemetryUDPServer()
    print(f"Streaming {n} real-time CCSDS telemetry packets to "
          f"{server.host}:{server.port} ...")
    server.stream_trajectory(times, states, rate_hz=20.0)
    server.close()
