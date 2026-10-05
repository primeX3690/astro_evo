import sys, os
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_orbital_mechanics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "9_ccsds_interop"))
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH
from two_body_problem import propagate_orbit
from ccsds_oem import write_oem, read_oem
from ccsds_tm_packet import pack_primary_header, unpack_primary_header, pack_telemetry_packet, unpack_telemetry_packet
passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1
if __name__ == "__main__":
    r0, v0 = keplerian_to_state_vector(a=6793.0, e=0.001, i=51.6, raan=120, argp=40, nu=0)
    traj = propagate_orbit(r0, v0, duration_s=3600, dt=60.0)
    times_s = np.arange(len(traj)) * 60.0
    epoch = datetime(2026, 9, 16, 0, 0, 0)
    out_path = "/tmp/astroevo_test.oem"
    oem_text = write_oem(times_s, traj, epoch, "TESTSAT", "2026-001A", output_path=out_path)
    check("OEM text has required CCSDS header keyword", "CCSDS_OEM_VERS" in oem_text)
    check("OEM text has META_START/META_STOP block", "META_START" in oem_text and "META_STOP" in oem_text)
    meta, points = read_oem(out_path)
    check("OEM round-trip: same number of ephemeris points", len(points) == len(traj), f"(wrote {len(traj)}, read {len(points)})")
    max_pos_err = max(np.linalg.norm(np.array(s[:3]) - traj[i, :3]) for i, (t, s) in enumerate(points))
    check("OEM round-trip: position preserved to <1 mm", max_pos_err < 1e-6, f"(max diff={max_pos_err:.2e} km)")
    check("OEM metadata parsed", meta.get("OBJECT_NAME") == "TESTSAT" and meta.get("REF_FRAME") == "EME2000")
    header = pack_primary_header(apid=100, seq_count=42, data_length_octets=64)
    check("Primary header is exactly 6 octets", len(header) == 6)
    decoded = unpack_primary_header(header)
    check("Header round-trip: APID preserved", decoded["apid"] == 100)
    check("Header round-trip: sequence count preserved", decoded["seq_count"] == 42)
    check("Header round-trip: data length decodes correctly", decoded["data_length_octets"] == 64)
    check("Header flagged as telemetry", decoded["packet_type"] == 0)
    packet = pack_telemetry_packet(apid=200, seq_count=7, t_s=1234.5, pos_km=traj[10, :3], vel_kms=traj[10, 3:], battery_wh=74.2)
    hdr, tlm = unpack_telemetry_packet(packet)
    check("TM packet round-trip: position matches", np.allclose([tlm["x_km"], tlm["y_km"], tlm["z_km"]], traj[10, :3]))
    check("TM packet round-trip: battery matches", abs(tlm["battery_wh"] - 74.2) < 1e-9)
    check("TM packet APID/seq survive wire format", hdr["apid"] == 200 and hdr["seq_count"] == 7)
    print(f"\n{passed} passed, {failed} failed")
    if failed == 0: print("MODULE 9: CCSDS/ECSS INTEROP VERIFIED")
    sys.exit(1 if failed else 0)
