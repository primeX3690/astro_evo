# AstroEvo — Benchmarks

All numbers below were actually measured by running each module's test
suite on this project's dev machine (CPU-only, no GPU). None of these
are estimates or projections.

| Module | What was measured | Result |
|---|---|---|
| 1 — Orbital Mechanics | RK4 propagation, 1 full LEO orbit | Energy drift: 0.00000000% |
| 1 — J2 perturbation | Numeric RAAN drift vs. analytic formula (15 orbits) | -4.97 deg/day vs -4.95 deg/day analytic (0.46% diff) |
| 1 — SRP + drag | Order-of-magnitude + 2-day decay simulation | SRP ~1.19e-7 m/s², drag ~2.4e-6 m/s² at 400km; altitude 293→253km over 2 days |
| 2 — Mission Planner | Full LEO→GEO transfer simulation | 1136 steps in 27.9 ms; radius error 0.05% |
| 3 — Evolutionary Optimizer | GA search, 40×40 (1600 Lambert solves) | 0.94s; delta-V within 0.00% of analytic Hohmann optimum |
| 3 — Multi-rev Lambert | M=1 both branches, physics conservation check | Momentum/energy match to machine precision; M=1 needs 2.5hr min vs M=0's 0.5hr |
| 4 — Neural Controller (2D) | PPO training, full lunar-lander task, 10,000 episodes | 183.4s; **85.5% landing success rate** |
| 4 — Neural Controller (3D) | PPO training, full 3D lander task, 10,000 episodes | 78s; **93.2% landing success rate** |
| 4 — CW Docking | RK4 vs. closed-form STM; rendezvous targeting accuracy | Diff = 7.1e-14; rendezvous position error = 1.76e-13 km |
| 5 — Self-Improve Loop | GA meta-optimizer, 5 trials | Config within 0.00-0.01% of Hohmann optimum |
| 6 — Simulation Hub | RK4 powered-burn vs. analytic Tsiolkovsky | Match to 6 decimal places |
| 7 — Mission Dashboard | Streamlit `AppTest`, GA + PDF button clicks | Zero exceptions; GA result matches module 3 standalone; PDF verified |
| 8 — Real-World TLE Validation | Own J2 propagator vs. SGP4 + real ISS TLE (NORAD 25544), 3 orbits | Max position error **0.58 km**, mean 0.18 km over ~4.6 hours — see `VALIDATION_REPORT.md` |
| 9 — CCSDS OEM Ephemeris | Write + read back a CCSDS 502.0-B OEM file | Round-trips 61 points to <1mm error |
| 9 — CCSDS Space Packet | Pack/unpack a real CCSDS 133.0-B-2 TM packet | 6-octet header + payload round-trips exactly |
| 10 — Ground Ops (geometry) | WGS-84 station position, overhead-satellite elevation check | Equatorial radius exact to 6378.137 km; overhead elevation = 90.0000° |
| 10 — Ground Ops (passes + link budget) | 51.6° LEO orbit, 24h window, 437 MHz UHF downlink | 2 passes found; FSPL = 145.25 dB (textbook match); link closes with positive margin |
| 11 — Safety Guardrail | Deterministic suicide-burn fallback wrapping a policy that never fires any thruster | **30/30 hard crashes → 0/30 hard crashes** across 30 episodes, purely from physics, zero learning involved |
| 11 — Safety Guardrail (non-interference) | Same guardrail wrapping an independently-conservative scripted policy | 0% override rate — guardrail only engages when genuinely needed |
| 12 — REST API | FastAPI endpoints for trajectory planning, TLE validation, ground-ops, tested via `TestClient` | 12/12 checks pass; GA trajectory result matches module 3 standalone to <0.01% |
| 13 — Real-Time Telemetry | 30 CCSDS TM packets streamed over a real UDP socket (threaded sender/receiver), 50 Hz | All 30 received intact, in order, real wall-clock pacing confirmed (0.6s elapsed, not instant) |
| 14 — Collision Avoidance (screening) | CW relative-motion close-approach scenario (reuses module 4's docking math) | 5.4 m miss distance detected, correctly flagged as a real conjunction |
| 14 — Collision Avoidance (maneuver) | Minimum-delta-V cross-track avoidance burn via CW rendezvous solver | 8.75 m/s burn pushes miss distance from 5.4 m to 5.0 km safe separation |
| 15 — Edge Inference | Trained PPO weights exported to pure C (no Python/numpy), compiled with gcc, cross-checked vs. Python on 10 test states | Action probabilities match to **5e-08** (float32 precision); 0/10 action mismatches |

## Reproducing these numbers
\`\`\`
cd tests
python test_module1.py    # J2, SRP, drag checks
python test_module2.py
python test_module3.py    # multi-rev Lambert
python test_module4.py    # 2D lander + CW docking, ~2-3 min
python test_module4_3d.py # 3D lander, ~1.5 min
python test_module5.py
python test_module6.py
python test_module7.py    # PDF report check
python test_module8.py    # real TLE validation vs SGP4 (needs: pip install sgp4)
python test_module9.py    # CCSDS OEM + Space Packet round-trip
python test_module10.py   # ground station passes + link budget
python test_module11.py   # deterministic safety guardrail
python test_module12.py   # REST API (needs: pip install fastapi uvicorn)
python test_module13.py   # real-time UDP telemetry streaming
python test_module14.py   # collision screening + avoidance maneuver
python test_module15.py   # edge inference (needs: gcc; run export_weights.py first)
\`\`\`

## Honest caveats
- Landing rates (85.5% 2D, 93.2% 3D) are seed-dependent (seed=2) — same ballpark, not guaranteed.
- Drag model is simplified exponential atmosphere, NOT real NRLMSISE-00.
- CW docking is translational-only (3-DOF), not full 6-DOF with attitude.
- All timings are single-threaded CPU. No GPU used anywhere.
- Module 8 validates against SGP4 (the real operational NORAD-catalog
  propagator), not GMAT/Orekit — see `VALIDATION_REPORT.md` for exactly
  why those specific tools aren't viably installable here, and why SGP4
  is a legitimate industry baseline in its own right.
- Module 9 implements the KVN (text) OEM variant and a fixed telemetry
  payload layout — not the full CCSDS/ECSS standard suite (no
  covariance/maneuver blocks, no binary XML OEM, no full ECSS-E-70 PUS
  service model).
- Module 10's pass geometry uses GMST-only Earth rotation (no polar
  motion/nutation) — fine for elevation-mask scheduling, not arcsecond
  pointing.
- Module 11's guardrail is deterministic physics (suicide-burn stopping
  distance), not formally verified with a proof assistant — "formal
  verification" in the strict sense (model checking / theorem proving)
  is a further step beyond what's implemented here.
- Module 12's API is containerized (see `Dockerfile`) but the image is
  NOT build-tested in this environment (no Docker daemon available in
  the dev sandbox) — Dockerfile syntax was manually reviewed, not
  `docker build`-verified. Build and verify it yourself before relying
  on it for a demo: `docker build -t astroevo-api . && docker run -p 8000:8000 astroevo-api`.
- Module 13 is real-time software telemetry streaming (a real UDP
  socket, real CCSDS framing, real wall-clock pacing) — it is NOT
  genuine Hardware-in-the-Loop (HIL) testing, since there is no
  physical spacecraft hardware or RF front-end involved. It's the
  data-plane piece HIL testing would plug into.
- Module 14's collision screening uses a fixed hard-body-radius +
  safety-buffer criterion (standard for a first-pass screen), not a
  full covariance-based probability-of-collision (Pc) calculation,
  which needs position-uncertainty data this repo doesn't model.
- Module 15 exports and runs on **x86 natively via gcc** to prove
  numeric correctness (Python vs. C match to 5e-08) — it is **not**
  cross-compiled for ARM, **not** quantized to int8, and has **not**
  run on real embedded/OBC hardware. Those are the concrete next
  steps the proof-of-correctness here sets up, not yet done.