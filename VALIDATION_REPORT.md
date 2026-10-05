# AstroEvo — Physics Validation Report

This document exists to answer one question honestly: **how do we know
AstroEvo's from-scratch propagator is physically correct, rather than
just internally self-consistent?**

## What "industry-standard baseline" validation normally means

Mission-ops teams validate a new propagator against an already-trusted
one — GMAT (NASA, open-source), Orekit (CS Group, open-source Java),
STK/Ansys (commercial), or poliastro/Skyfield (Python). We attempted
this directly.

## What we tried, and why it doesn't work out-of-the-box here

| Tool | Result |
|---|---|
| **poliastro** (`pip install poliastro`) | **Fails to install** on current Python (3.12) — it pins an old `astropy` that imports the standard-library `imp` module, which was removed in Python 3.12. This is a known, unresolved dependency-rot issue in the poliastro project, not something fixable from our side without pinning to an old Python + old astropy stack. |
| **Orekit** | Java library with a Python wrapper (`orekit` via `pip`) that requires a JVM (via `orekit-jvm`/`jpype`) and a multi-hundred-MB data package (`orekit-data.zip`) fetched from a separate host. Not installable in a constrained/offline-first environment without that extra infrastructure. |
| **GMAT** | A full desktop application (NASA GSFC), not a Python package — there is no `pip install gmat`. Cross-validating against it means running the GMAT GUI/scripting engine separately and comparing exported ephemeris, which is a manual, non-automatable step outside this repo's test suite. |
| **SGP4** (`pip install sgp4`) | **Installs and runs cleanly.** This is the actual industry-standard propagator used operationally for every cataloged object in the public NORAD catalog — it is arguably a *more* universally "industry standard" baseline than GMAT/Orekit for LEO objects, since it's what Space-Track, Celestrak, and most commercial conjunction-screening services run. |

**Conclusion: we use SGP4 + real NORAD TLE data as the practical,
installable, industry-grade cross-validation baseline (module 8),
since poliastro and Orekit are not viably installable in this
environment and GMAT has no programmatic interface to automate
against.** This is a real limitation, documented rather than hidden.

## What we actually validated, and the result

See `8_realworld_validation/` and `tests/test_module8.py`.

- Fetched a real TLE for the ISS (NORAD 25544)
- Seeded our own two-body + J2 RK4 propagator from SGP4's own initial
  state vector (isolating "does our force model + integrator track a
  real orbit" from "did we parse the TLE's mean elements correctly" —
  mean vs. osculating elements is a known TLE subtlety)
- Propagated both independently for 3 full orbits (~4.6 hours) and
  compared position at every 10-second step

**Result: 0.58 km max position error, 0.18 km mean error, over 3 full
orbits of a real operational satellite.**

### Why this is not, and should not be, a bit-exact match

SGP4 and our propagator are genuinely different things:

| | SGP4 | AstroEvo |
|---|---|---|
| Frame | TEME (True Equator, Mean Equinox) | ECI-like (equivalent to TEME at the precision we operate at, but not frame-corrected) |
| Force model | Analytic mean-element theory with empirical drag term (B*) | Numerical two-body + J2 RK4 integration |
| Drag | Built into the SGP4 theory itself | Separately modeled (module 1), not active in this cross-check |

Sub-kilometer agreement over 3 orbits despite these differences is a
meaningful result — it shows the two independent approaches agree on
the orbit's shape, period, and altitude regime to a precision far
tighter than either the TLE's own accuracy (TLEs are generally only
good to ~1-3 km even for SGP4 itself, a few days from epoch) or
anything that would matter for mission-design-level planning.

## What is still NOT validated against an external baseline

Being honest about the boundary of what module 8 covers:

- **J2 secular drift** (RAAN precession rate) is checked against the
  closed-form analytic formula in `tests/test_module1.py` (0.46%
  agreement), not against a numerical reference implementation.
- **SRP and drag models** are checked against order-of-magnitude
  physical expectations (`test_module1.py`), not against a reference
  atmosphere model (NRLMSISE-00) or solar-pressure model.
- **No EGM96 or higher-order gravity field** is implemented — J2 only.
  A full EGM96 comparison would require either implementing the
  spherical-harmonic expansion ourselves (significant scope) or a
  working Orekit/GMAT install (see above).
- Multi-revolution trajectories beyond LEO (MEO/GEO transfers) are
  cross-checked against the **analytic Hohmann-transfer optimum**
  (module 2/3 tests), not against SGP4 (SGP4 doesn't model transfer
  trajectories, only catalog objects in stable orbits).

## Bottom line

AstroEvo's propagator is validated against **real NORAD TLE data via
SGP4** to sub-km accuracy over multiple real orbits — a genuine,
reproducible, automated cross-check against production space-industry
data, not synthetic self-consistency checks alone. It is **not** yet
cross-validated against GMAT/Orekit/EGM96 specifically, for the
concrete, documented tooling reasons above, not for lack of trying.
