# 9_ccsds_interop/ccsds_oem.py
"""
CCSDS 502.0-B-3 Orbit Ephemeris Message (OEM), plain-text (KVN) format.

Standard mission-ops teams use to exchange propagated ephemeris between
ground systems, flight dynamics tools, and conjunction-assessment
services. Writing/reading real OEM means AstroEvo's propagated
trajectories can be handed to, or received from, an actual ops team's
toolchain without a custom adapter.

HONEST NOTE: implements the KVN (Keyword-Value Notation) OEM variant
and the mandatory header/metadata/data blocks; the binary XML variant
and optional blocks (covariance, maneuvers) are not implemented here.
"""
from datetime import datetime, timedelta, timezone

CCSDS_OEM_VERSION = "3.0"


def _fmt_epoch(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write_oem(times_s, states_km, epoch, object_name, object_id,
              center_name="EARTH", ref_frame="EME2000", output_path=None,
              originator="ASTROEVO"):
    """
    times_s   : 1D array, seconds since `epoch`
    states_km : Nx6 array, [x,y,z,vx,vy,vz] in km, km/s
    epoch     : datetime (UTC) of times_s[0]
    Returns the OEM text; also writes to output_path if given.
    """
    lines = []
    lines.append(f"CCSDS_OEM_VERS = {CCSDS_OEM_VERSION}")
    lines.append(f"CREATION_DATE  = {_fmt_epoch(datetime.now(timezone.utc).replace(tzinfo=None))}")
    lines.append(f"ORIGINATOR     = {originator}")
    lines.append("")
    lines.append("META_START")
    lines.append(f"OBJECT_NAME          = {object_name}")
    lines.append(f"OBJECT_ID            = {object_id}")
    lines.append(f"CENTER_NAME          = {center_name}")
    lines.append(f"REF_FRAME            = {ref_frame}")
    lines.append("TIME_SYSTEM          = UTC")
    lines.append(f"START_TIME           = {_fmt_epoch(epoch + timedelta(seconds=float(times_s[0])))}")
    lines.append(f"STOP_TIME            = {_fmt_epoch(epoch + timedelta(seconds=float(times_s[-1])))}")
    lines.append("META_STOP")
    lines.append("")

    for t, state in zip(times_s, states_km):
        ts = _fmt_epoch(epoch + timedelta(seconds=float(t)))
        x, y, z, vx, vy, vz = state
        lines.append(f"{ts} {x:.6f} {y:.6f} {z:.6f} {vx:.9f} {vy:.9f} {vz:.9f}")

    text = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    return text


def read_oem(path_or_text, is_path=True):
    """
    Parses an OEM (KVN) file/string back into metadata + a list of
    (datetime, [x,y,z,vx,vy,vz]) ephemeris points. Round-trips with
    write_oem() above.
    """
    if is_path:
        with open(path_or_text) as f:
            text = f.read()
    else:
        text = path_or_text

    meta = {}
    points = []
    in_meta = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line == "META_START":
            in_meta = True
            continue
        if line == "META_STOP":
            in_meta = False
            continue
        if in_meta and "=" in line:
            key, val = line.split("=", 1)
            meta[key.strip()] = val.strip()
            continue
        if "=" in line and not in_meta and line.split("=")[0].strip().isupper():
            key, val = line.split("=", 1)
            meta[key.strip()] = val.strip()
            continue

        parts = line.split()
        if len(parts) == 7:
            ts_str = parts[0].rstrip("Z")
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            state = [float(p) for p in parts[1:]]
            points.append((ts, state))

    return meta, points
