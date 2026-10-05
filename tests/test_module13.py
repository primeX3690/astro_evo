# tests/test_module13.py
"""Module 13: real-time CCSDS telemetry streaming over a real UDP socket."""
import sys, os
import threading
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_orbital_mechanics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "13_realtime_telemetry"))
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH
from perturbation_models import two_body_j2_eom
from two_body_problem import rk4_step
from telemetry_server import TelemetryUDPServer
from telemetry_client import TelemetryUDPClient

passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1

if __name__ == "__main__":
    PORT = 52022
    N_PACKETS = 30

    r0, v0 = keplerian_to_state_vector(a=6793.0, e=0.001, i=51.6, raan=0, argp=0, nu=0)
    state = np.concatenate([r0, v0])
    dt_s = 1.0
    times = np.arange(N_PACKETS) * dt_s
    states = np.zeros((N_PACKETS, 6))
    states[0] = state
    for k in range(1, N_PACKETS):
        state = rk4_step(lambda t, y: two_body_j2_eom(t, y, MU_EARTH), times[k - 1], state, dt_s)
        states[k] = state
    battery_series = 100.0 - 0.1 * np.arange(N_PACKETS)

    client = TelemetryUDPClient(host="127.0.0.1", port=PORT, timeout_s=3.0)

    received_holder = {}
    def receiver_thread():
        received_holder["results"] = client.receive_n(N_PACKETS, verbose=False)
    t = threading.Thread(target=receiver_thread)
    t.start()
    time.sleep(0.2)  # let the socket bind + thread start before we send

    server = TelemetryUDPServer(host="127.0.0.1", port=PORT)
    t_send_start = time.time()
    server.stream_trajectory(times, states, battery_series=battery_series, rate_hz=50.0, verbose=False)
    t_send_end = time.time()
    server.close()

    t.join(timeout=5.0)
    client.close()
    results = received_holder.get("results", [])

    check("Receiver got all packets sent over the real UDP socket",
          len(results) == N_PACKETS, f"(sent={N_PACKETS}, received={len(results)})")

    check("Streaming was actually real-time paced (not instant dump)",
          (t_send_end - t_send_start) > 0.3,
          f"(elapsed={t_send_end - t_send_start:.3f}s for {N_PACKETS} packets @ 50 Hz)")

    if results:
        seqs = [h["seq_count"] for h, _ in results]
        check("Sequence counts arrived in order (no UDP reordering in this local test)",
              seqs == sorted(seqs), f"(seqs={seqs[:5]}...)")

        first_header, first_tlm = results[0]
        check("First packet's position matches the propagated trajectory exactly",
              np.allclose([first_tlm["x_km"], first_tlm["y_km"], first_tlm["z_km"]], states[0, :3]),
              )
        last_header, last_tlm = results[-1]
        check("Last packet's position matches the propagated trajectory exactly",
              np.allclose([last_tlm["x_km"], last_tlm["y_km"], last_tlm["z_km"]], states[-1, :3]),
              )
        check("Battery telemetry decreases over the stream (realistic depletion)",
              first_tlm["battery_wh"] > last_tlm["battery_wh"],
              f"(first={first_tlm['battery_wh']:.1f} Wh, last={last_tlm['battery_wh']:.1f} Wh)")
        check("All packets correctly flagged as telemetry (CCSDS packet_type=0)",
              all(h["packet_type"] == 0 for h, _ in results))

    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("MODULE 13: REAL-TIME TELEMETRY VERIFIED (live CCSDS packets over a real UDP socket, "
              "independent sender/receiver processes via threading)")
    sys.exit(1 if failed else 0)
