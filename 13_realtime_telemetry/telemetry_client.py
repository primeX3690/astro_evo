# 13_realtime_telemetry/telemetry_client.py
"""
Real-time telemetry receiver: a ground-side listener that ingests live
CCSDS Space Packets off a UDP socket and unpacks them - the receive-side
counterpart to telemetry_server.py, modeling what a mission-control
front-end actually does with an incoming telemetry stream.
"""
import socket
import sys
import os
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "9_ccsds_interop"))
from ccsds_tm_packet import unpack_telemetry_packet  # noqa: E402

PACKET_SIZE = 6 + struct.calcsize(">8d")  # 6-octet header + 8 float64 payload


class TelemetryUDPClient:
    def __init__(self, host="127.0.0.1", port=52001, timeout_s=2.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(timeout_s)
        self.received = []

    def receive_one(self):
        """Blocks (up to timeout_s) for one packet; returns (header, telemetry) or None on timeout."""
        try:
            data, _addr = self.sock.recvfrom(PACKET_SIZE)
        except socket.timeout:
            return None
        if len(data) != PACKET_SIZE:
            return None  # malformed/truncated packet - real ground software would log+drop
        header, telemetry = unpack_telemetry_packet(data)
        self.received.append((header, telemetry))
        return header, telemetry

    def receive_n(self, n, verbose=True):
        results = []
        for _ in range(n):
            result = self.receive_one()
            if result is None:
                break
            header, telemetry = result
            results.append(result)
            if verbose:
                print(f"[telemetry_client] rx seq={header['seq_count']} apid={header['apid']} "
                      f"t={telemetry['t_s']:.1f}s pos=({telemetry['x_km']:.1f},"
                      f"{telemetry['y_km']:.1f},{telemetry['z_km']:.1f}) km "
                      f"battery={telemetry['battery_wh']:.1f} Wh")
        return results

    def close(self):
        self.sock.close()
