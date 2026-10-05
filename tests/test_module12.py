# tests/test_module12.py
"""Module 12: FastAPI service layer - tested via FastAPI's TestClient (no running server needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "12_api"))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
passed, failed = 0, 0
def check(name, cond, detail=""):
    global passed, failed
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if cond: passed += 1
    else: failed += 1

if __name__ == "__main__":
    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200, f"(status={r.status_code})")
    check("GET /health reports ok status", r.json().get("status") == "ok")

    r = client.post("/trajectory/plan", json={"origin": "LEO", "destination": "GEO"})
    check("POST /trajectory/plan returns 200", r.status_code == 200, f"(status={r.status_code}, body={r.text[:200]})")
    body = r.json()
    check("Trajectory plan delta-V close to Hohmann optimum (<1% off)",
          abs(body["delta_v_kms"] - body["hohmann_optimum_kms"]) / body["hohmann_optimum_kms"] < 0.01,
          f"(GA={body['delta_v_kms']:.4f}, Hohmann={body['hohmann_optimum_kms']:.4f})")
    check("Trajectory plan marked feasible for default budget", body["feasible"] is True)

    r = client.post("/trajectory/plan", json={"origin": "LEO", "destination": "MARS"})
    check("Unknown destination orbit returns 400, not a crash", r.status_code == 400, f"(status={r.status_code})")

    r = client.post("/validation/tle", json={"norad_id": 25544, "orbits": 2})
    check("POST /validation/tle returns 200", r.status_code == 200, f"(status={r.status_code})")
    tle_body = r.json()
    check("TLE validation via API matches real ISS orbit regime",
          6600 <= tle_body["semi_major_axis_km"] <= 7000 and tle_body["max_error_km"] < 2.0,
          f"(a={tle_body['semi_major_axis_km']:.1f}, max_err={tle_body['max_error_km']:.3f})")

    r = client.post("/ground-ops/passes", json={
        "orbit": "LEO", "duration_hours": 24,
        "station_name": "Gorakhpur", "station_lat_deg": 26.7606, "station_lon_deg": 83.3732,
    })
    check("POST /ground-ops/passes returns 200", r.status_code == 200, f"(status={r.status_code}, body={r.text[:200]})")
    gp_body = r.json()
    check("Ground pass API finds at least one pass over 24 hours",
          gp_body["passes_found"] >= 1, f"(found={gp_body['passes_found']})")
    if gp_body["passes_found"] >= 1:
        p0 = gp_body["passes"][0]
        check("Pass result includes link margin + close/no-close verdict",
              "link_margin_db" in p0 and "link_closes" in p0,
              f"(margin={p0['link_margin_db']:.2f} dB, closes={p0['link_closes']})")

    r = client.get("/docs")
    check("Auto-generated OpenAPI docs (/docs) are served", r.status_code == 200)

    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("MODULE 12: REST API VERIFIED (trajectory, TLE validation, ground-ops all live over HTTP)")
    sys.exit(1 if failed else 0)
