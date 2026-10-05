# 9_ccsds_interop/ccsds_tm_packet.py
"""CCSDS 133.0-B-2 Space Packet Protocol - primary header pack/unpack + TM payload."""
import struct

PACKET_TYPE_TM = 0
PACKET_TYPE_TC = 1
_TM_PAYLOAD_FMT = ">8d"
_TM_FIELDS = ["t_s", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms", "battery_wh"]


def pack_primary_header(apid, seq_count, data_length_octets,
                         packet_type=PACKET_TYPE_TM, secondary_header_flag=0,
                         seq_flags=0b11, version=0):
    if not (0 <= apid <= 0x7FF):
        raise ValueError("APID must fit in 11 bits (0-2047)")
    if not (0 <= seq_count <= 0x3FFF):
        raise ValueError("sequence count must fit in 14 bits (0-16383)")
    if data_length_octets < 1:
        raise ValueError("data field must contain at least 1 octet")
    word0 = (version & 0x7) << 13
    word0 |= (packet_type & 0x1) << 12
    word0 |= (secondary_header_flag & 0x1) << 11
    word0 |= apid & 0x7FF
    word1 = (seq_flags & 0x3) << 14
    word1 |= seq_count & 0x3FFF
    packet_length_field = data_length_octets - 1
    return struct.pack(">HHH", word0, word1, packet_length_field)


def unpack_primary_header(header_bytes):
    if len(header_bytes) != 6:
        raise ValueError("CCSDS primary header must be exactly 6 octets")
    word0, word1, packet_length_field = struct.unpack(">HHH", header_bytes)
    return {
        "version": (word0 >> 13) & 0x7,
        "packet_type": (word0 >> 12) & 0x1,
        "secondary_header_flag": (word0 >> 11) & 0x1,
        "apid": word0 & 0x7FF,
        "seq_flags": (word1 >> 14) & 0x3,
        "seq_count": word1 & 0x3FFF,
        "data_length_octets": packet_length_field + 1,
    }


def pack_telemetry_packet(apid, seq_count, t_s, pos_km, vel_kms, battery_wh):
    x, y, z = pos_km
    vx, vy, vz = vel_kms
    payload = struct.pack(_TM_PAYLOAD_FMT, t_s, x, y, z, vx, vy, vz, battery_wh)
    header = pack_primary_header(apid, seq_count, len(payload), packet_type=PACKET_TYPE_TM)
    return header + payload


def unpack_telemetry_packet(packet_bytes):
    header = unpack_primary_header(packet_bytes[:6])
    payload = packet_bytes[6:6 + struct.calcsize(_TM_PAYLOAD_FMT)]
    values = struct.unpack(_TM_PAYLOAD_FMT, payload)
    telemetry = dict(zip(_TM_FIELDS, values))
    return header, telemetry
