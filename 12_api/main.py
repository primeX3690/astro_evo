# 12_api/main.py
"""
FastAPI service layer over AstroEvo's core capabilities - trajectory
planning, real-world TLE validation, and ground-station pass + link
budget analysis - as REST endpoints a B2B client can hit over HTTP
instead of running local Python scripts.

Run locally:
    uvicorn main:app --reload --port 8000
Then see interactive docs at http://localhost:8000/docs (auto-generated
by FastAPI from the Pydantic schemas below - this IS the API contract).
"""
import sys
import os
import datetime as dt

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..")
for sub in ["1_orbital_mechanics", "2_symbolic_mission_planner",
            "8_realworld_validation", "9_ccsds_interop", "10_ground_ops"]:
    sys.path.insert(0, os.path.join(_ROOT, sub))

import numpy as np  # noqa: E402
sys.path.insert(0, os.path.join(_ROOT, "3_evolutionary_optimizer"))  # noqa: E402
from keplerian_orbit import keplerian_to_state_vector, MU_EARTH  # noqa: E402
from perturbation_models import two_body_j2_eom  # noqa: E402
from two_body_problem import rk4_step  # noqa: E402
from mission_spec_parser import ORBIT_LIBRARY, MissionSpec  # noqa: E402
from genetic_engine import GeneticEngine  # noqa: E402
from constraint_solver import hohmann_delta_v  # noqa: E402
from validate_against_tle import validate as validate_tle_impl  # noqa: E402
from groundstation import GroundStation  # noqa: E402
from pass_predictor import predict_passes  # noqa: E402
from link_budget import evaluate_link  # noqa: E402

app = FastAPI(
    title="AstroEvo Mission Planning API",
    description="Trajectory optimization, real-world validation, and ground-ops "
                 "analysis for autonomous space mission planning.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# /trajectory - orbit transfer planning (wraps modules 1-3)
# ---------------------------------------------------------------------------
class TrajectoryRequest(BaseModel):
    origin: str = Field(..., examples=["LEO"], description="One of: " + ", ".join(ORBIT_LIBRARY.keys()))
    destination: str = Field(..., examples=["GEO"])
    max_delta_v_kms: float = Field(6.0, gt=0)
    max_time_hours: float = Field(12.0, gt=0)


class TrajectoryResponse(BaseModel):
    mission_name: str
    feasible: bool
    delta_v_kms: float
    hohmann_optimum_kms: float
    time_of_flight_hours: float
    transfer_angle_deg: float


@app.post("/trajectory/plan", response_model=TrajectoryResponse)
def plan_trajectory(req: TrajectoryRequest):
    if req.origin not in ORBIT_LIBRARY or req.destination not in ORBIT_LIBRARY:
        raise HTTPException(status_code=400,
                             detail=f"Unknown orbit(s). Known: {list(ORBIT_LIBRARY.keys())}")
    spec = MissionSpec(
        name=f"{req.origin}_to_{req.destination}",
        r1_km=ORBIT_LIBRARY[req.origin], r2_km=ORBIT_LIBRARY[req.destination],
        max_delta_v_kms=req.max_delta_v_kms, max_time_s=req.max_time_hours * 3600,
    )
    ga = GeneticEngine(spec, population_size=40, generations=40,
                        mutation_rate=0.3, elite_count=2, seed=42)
    best_gene, best_fitness, best_info = ga.run()
    _, _, hohmann_dv = hohmann_delta_v(spec.r1, spec.r2)

    return TrajectoryResponse(
        mission_name=spec.name,
        feasible=best_info["feasible"],
        delta_v_kms=best_info["dv_total_kms"],
        hohmann_optimum_kms=hohmann_dv,
        time_of_flight_hours=best_gene[1],
        transfer_angle_deg=best_gene[0],
    )


# ---------------------------------------------------------------------------
# /validation - real-world TLE cross-check (wraps module 8)
# ---------------------------------------------------------------------------
class TleValidationRequest(BaseModel):
    norad_id: int = Field(25544, description="NORAD catalog ID, e.g. 25544 = ISS")
    orbits: int = Field(3, gt=0, le=20)


class TleValidationResponse(BaseModel):
    object_name: str
    norad_id: int
    semi_major_axis_km: float
    period_min: float
    max_error_km: float
    mean_error_km: float


@app.post("/validation/tle", response_model=TleValidationResponse)
def validate_tle(req: TleValidationRequest):
    result = validate_tle_impl(norad_id=req.norad_id, duration_orbits=req.orbits)
    return TleValidationResponse(
        object_name=result["object_name"],
        norad_id=result["norad_id"],
        semi_major_axis_km=result["semi_major_axis_km"],
        period_min=result["period_min"],
        max_error_km=result["max_error_km"],
        mean_error_km=result["mean_error_km"],
    )


# ---------------------------------------------------------------------------
# /ground-ops - pass prediction + link budget (wraps module 10)
# ---------------------------------------------------------------------------
class GroundPassRequest(BaseModel):
    orbit: str = Field("LEO", description="One of: " + ", ".join(ORBIT_LIBRARY.keys()))
    inclination_deg: float = 51.6
    duration_hours: float = Field(24.0, gt=0, le=72,
                                   description="Longer windows are recommended (>=24h) since pass "
                                               "timing depends on the current Earth-rotation alignment "
                                               "with the chosen orbital plane (RAAN)")
    station_name: str = "Ground Station"
    station_lat_deg: float
    station_lon_deg: float
    station_alt_km: float = 0.0
    min_elevation_deg: float = 10.0
    tx_power_w: float = 1.0
    freq_hz: float = 437e6
    rx_antenna_gain_dbi: float = 15.0
    rx_system_noise_temp_k: float = 600.0
    data_rate_bps: float = 9600
    required_eb_n0_db: float = 10.0


class PassResult(BaseModel):
    aos_utc: str
    los_utc: str
    duration_s: float
    max_elevation_deg: float
    min_range_km: float
    link_margin_db: float
    link_closes: bool


class GroundPassResponse(BaseModel):
    station: str
    passes_found: int
    passes: list[PassResult]


@app.post("/ground-ops/passes", response_model=GroundPassResponse)
def ground_passes(req: GroundPassRequest):
    if req.orbit not in ORBIT_LIBRARY:
        raise HTTPException(status_code=400, detail=f"Unknown orbit '{req.orbit}'. Known: {list(ORBIT_LIBRARY.keys())}")

    a = ORBIT_LIBRARY[req.orbit]
    r0, v0 = keplerian_to_state_vector(a=a, e=0.001, i=req.inclination_deg, raan=0, argp=0, nu=0)
    state = np.concatenate([r0, v0])
    dt_s = 60.0
    n_steps = int(req.duration_hours * 3600 / dt_s)
    times_s = np.arange(n_steps) * dt_s
    states = np.zeros((n_steps, 6))
    states[0] = state
    for k in range(1, n_steps):
        state = rk4_step(lambda t, y: two_body_j2_eom(t, y, MU_EARTH), times_s[k - 1], state, dt_s)
        states[k] = state

    epoch = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    gs = GroundStation(req.station_name, req.station_lat_deg, req.station_lon_deg, req.station_alt_km)
    passes, geom = predict_passes(times_s, states, epoch, gs, min_elevation_deg=req.min_elevation_deg)

    results = []
    for p in passes:
        min_range = min(geom["range_km"][i] for i in range(len(times_s))
                         if p["aos_s"] <= times_s[i] <= p["los_s"])
        link = evaluate_link(
            tx_power_w=req.tx_power_w, tx_antenna_gain_dbi=0.0, tx_line_loss_db=0.5,
            range_km=min_range, freq_hz=req.freq_hz,
            rx_antenna_gain_dbi=req.rx_antenna_gain_dbi,
            rx_system_noise_temp_k=req.rx_system_noise_temp_k, rx_line_loss_db=0.5,
            data_rate_bps=req.data_rate_bps, required_eb_n0_db=req.required_eb_n0_db,
        )
        results.append(PassResult(
            aos_utc=p["aos_time"].isoformat(), los_utc=p["los_time"].isoformat(),
            duration_s=p["duration_s"], max_elevation_deg=p["max_elevation_deg"],
            min_range_km=min_range, link_margin_db=link["margin_db"], link_closes=link["link_closes"],
        ))

    return GroundPassResponse(station=req.station_name, passes_found=len(results), passes=results)


@app.get("/health")
def health():
    return {"status": "ok", "service": "astroevo-api", "version": "1.0.0"}
