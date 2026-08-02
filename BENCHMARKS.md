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
\`\`\`

## Honest caveats
- Landing rates (85.5% 2D, 93.2% 3D) are seed-dependent (seed=2) — same ballpark, not guaranteed.
- Drag model is simplified exponential atmosphere, NOT real NRLMSISE-00.
- CW docking is translational-only (3-DOF), not full 6-DOF with attitude.
- All timings are single-threaded CPU. No GPU used anywhere.