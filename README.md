## AstroEvo

A from-scratch autonomous mission-planning and control stack - orbital
mechanics with real perturbations (J2, drag, SRP), mission planning,
evolutionary trajectory optimization with multi-revolution Lambert
solving, reinforcement-learning landing control (2D and 3D), orbital
rendezvous/docking (Clohessy-Wiltshire), Kalman-filtered state
estimation, a self-improving hyperparameter search loop, physics
simulation + visualization, and an interactive dashboard with PDF
reporting. Built to run on a CPU-only, 8GB RAM laptop.

Every number in this README was produced by actually running the code
(see BENCHMARKS.md for the full measured table). One README, one
requirements.txt, one place to look.

## Quick start

\`\`\`bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

\`\`\`bash
python run_mission_ai.py --mode evolve --origin LEO --destination GEO
python run_mission_ai.py --mode lander --episodes 10000
python run_mission_ai.py --mode dashboard
\`\`\`

---

## Module 1 - 1_orbital_mechanics/

Files: keplerian_orbit.py, two_body_problem.py, utils.py,
perturbation_models.py, fast_propagator.py

- Keplerian elements <-> state vector conversion, RK4 two-body propagation
- J2 (Earth oblateness): numeric RAAN drift matches the analytic
  formula to 0.46% (-4.97 vs -4.95 deg/day)
- SRP + atmospheric drag: simplified exponential density model
  (honest note - NOT real NRLMSISE-00, which needs decades of empirical
  satellite data to build). Verified: SRP ~1.19e-7 m/s^2, drag ~2.4e-6
  m/s^2 at 400km altitude, both matching known real-world orders of
  magnitude; a low-orbit cubesat decays realistically over days
- fast_propagator.py: Numba-JIT version of the RK4 propagator -
  94x faster (0.098ms vs 9.2ms per call), verified identical to
  machine precision (diff ~1e-13)

\`\`\`bash
cd tests && python test_module1.py    # ~10s
\`\`\`

---

## Module 2 - 2_symbolic_mission_planner/

Files: mission_spec_parser.py, constraint_solver.py, temporal_planner.py

- Parses goals like "LEO to GEO transfer" into a MissionSpec
- Hohmann transfer delta-V/time, propellant mass fraction (Tsiolkovsky),
  feasibility checking against delta-V/time budgets
- temporal_planner.py actually propagates the full transfer through
  module 1's RK4 integrator - reaches the target orbit to 0.05% error

\`\`\`bash
cd tests && python test_module2.py    # ~5s
\`\`\`

---

## Module 3 - 3_evolutionary_optimizer/

Files: lambert_solver.py, trajectory_encoding.py, fitness_functions.py, genetic_engine.py

- From-scratch universal-variable Lambert solver (Stumpff functions),
  verified via angular-momentum/energy conservation to machine precision
- Multi-revolution Lambert (M>=1): finds both solution branches per
  revolution count, verified physically consistent; M=1 needs 2.5hr
  minimum vs M=0's 0.5hr (correctly reflects the extra revolution)
- Own genetic algorithm (no DEAP) - converges to within 0.00% of the
  analytic Hohmann optimum, independently rediscovering known physics
  through search

\`\`\`bash
cd tests && python test_module3.py    # ~5s
\`\`\`

---

## Module 4 - 4_neural_controller/

Files: lunar_lander_env.py (2D), lunar_lander_env_3d.py (3D),
reward_shaping.py, ppo_agent.py, docking_env.py

The most heavily debugged module - documented honestly rather than
cleaned up:

1. REINFORCE (vanilla policy gradient) - 3 real bugs found and fixed
   (missing state normalization, a gradient-averaging bug crushing the
   learning rate ~1000x, wrong entropy-gradient formula) but still 0%
   landing on the full task even after 10,000+ episodes
2. Actor-critic (learned value baseline) - still 0% after 6,000 episodes
3. Curriculum learning - real but partial success (~25% at reduced
   difficulty, decaying back to 0% at full difficulty)
4. Real clipped-PPO (Schulman et al. 2017, with GAE) - the actual
   fix: clipped surrogate objective + multiple epochs per batch prevents
   the catastrophic forgetting seen in 1-3
5. One more bug even after switching to PPO: the agent found a
   "hover-and-burn-all-fuel" exploit (sat at altitude 50 for 190 steps,
   then free-fell uncontrolled). Fixed with a direct per-step altitude
   cost (a potential-based version doesn't work - it telescopes to a
   path-independent total and doesn't discourage lingering)

Verified results:
- 2D lander: 85.5% landing success (10,000 episodes, ~183s)
- 3D lander: 93.2% landing success (10,000 episodes, ~78s) - needed
  a higher fuel budget (160 vs 100) since 3D requires correcting both x
  AND y, not just x
- CW docking (docking_env.py): Clohessy-Wiltshire relative motion
  (translational 3-DOF, not full 6-DOF attitude). RK4 matches the
  closed-form analytic solution to 7e-14; the rendezvous targeting
  solver reaches its target position to 1.76e-13 km error

\`\`\`bash
cd tests
python test_module4.py       # ~2-3 min (2D PPO + CW docking)
python test_module4_3d.py    # ~1.5 min (3D PPO)
\`\`\`

---

## Module 5 - 5_self_improve_loop/

Files: mission_generator.py, curriculum_trainer.py, meta_optimizer.py, experiment_logger.py

- Random mission/difficulty generator (feeds modules 2/3/4)
- Curriculum trainer (difficulty ramp for module 4's lander)
- Random-search hyperparameter tuning for module 3's GA and module 4's
  PPO, every trial logged to JSON with get_best()/summary()
- Honest note: the test suite's PPO search trials are short (600
  episodes, for test speed) - 0% landing rate in those trials is
  expected given module 4's own convergence curve, not a bug. Use
  episodes_per_trial=8000+ for a real search.

\`\`\`bash
cd tests && python test_module5.py    # ~1 min
\`\`\`

---

## Module 6 - 6_simulation_hub/

Files: rocket_model.py, physics_engine.py, satellite_model.py, kalman_filter.py, orbit_visualizer.py

- rocket_model.py: Tsiolkovsky rocket equation (delta-V available,
  burn time, live thrust acceleration as mass depletes)
- physics_engine.py: continuous-thrust powered flight, reusing module
  1's RK4 stepper - matches Tsiolkovsky to 6 decimal places (gravity
  isolated); with real gravity, correctly shows lower achieved delta-V
  than the impulsive ideal (gravity losses during a slow burn - a real
  physical effect)
- satellite_model.py: solar power generation (cosine law, matches hand
  calc exactly), subsystem power draw, single-orbit battery simulation,
  spin-stabilized attitude
- kalman_filter.py: Extended Kalman Filter for orbit determination
  - a real satellite never knows its exact position, it filters noisy
  measurements. Verified: reduces RMS position error by 75.1%
  compared to raw noisy measurements (3.37km -> 0.85km)
- orbit_visualizer.py: real matplotlib plots (orbit trajectory, ground
  track, delta-V bar chart)

\`\`\`bash
cd tests && python test_module6.py    # ~5s
\`\`\`

---

## Module 7 - 7_mission_dashboard/

Files: app.py, orbital_view.py, report_generator.py

- Streamlit dashboard: mission setup, feasibility summary, delta-V bar
  chart, interactive 3D orbit view, live-rotating Earth with a
  ground track that builds up progressively (genuine rotation, verified
  - coordinates actually change frame to frame), live GA optimizer run
  with a real-time fitness chart
- PDF report generation (report_generator.py, via reportlab) - one
  click produces a formatted mission report with the summary table and
  GA results
- Tested with Streamlit's AppTest framework (actually executes the
  script, not just checks the server starts) - zero exceptions on
  initial load, GA button click, and PDF button click. The live GA run
  matches module 3's standalone result (-3.8465) almost exactly,
  confirming the dashboard isn't showing mocked numbers.

\`\`\`bash
cd tests && python test_module7.py    # ~1 min
\`\`\`

---

## Project structure

\`\`\`
AstroEvo/
├── 1_orbital_mechanics/
├── 2_symbolic_mission_planner/
├── 3_evolutionary_optimizer/
├── 4_neural_controller/
├── 5_self_improve_loop/
├── 6_simulation_hub/
├── 7_mission_dashboard/
├── config/
│   └── mission_config.yaml
├── tests/                    (test_module1.py .. test_module7.py, test_module4_3d.py)
├── run_mission_ai.py          (master CLI switch)
├── requirements.txt           (single file, all dependencies)
├── BENCHMARKS.md
├── README.md                  (this file)
├── LICENSE                    (MIT)
└── .gitignore
\`\`\`

## How this was built

Every module was built, tested, and verified one at a time - never
shipped without running the code. Module 4's debugging history above is
kept honest rather than cleaned up, because it's genuinely useful
context for extending this project.