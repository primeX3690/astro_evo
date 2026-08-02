# 7_mission_dashboard/app.py
"""Streamlit mission dashboard: mission planning, 3D orbit view (with live-rotating Earth),
live GA optimization run, and PDF report generation. Run with: streamlit run app.py"""
import sys
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "2_symbolic_mission_planner"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "3_evolutionary_optimizer"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "7_mission_dashboard"))

from mission_spec_parser import ORBIT_LIBRARY, MissionSpec  # noqa: E402
from constraint_solver import solve_constraints, hohmann_delta_v  # noqa: E402
from genetic_engine import GeneticEngine  # noqa: E402
from orbital_view import get_orbit_points_3d, get_earth_sphere, summarize_mission, build_rotating_earth_figure  # noqa: E402
from report_generator import generate_mission_report  # noqa: E402

st.set_page_config(page_title="AstroEvo Mission Dashboard", layout="wide")
st.title("AstroEvo — Mission Dashboard")

st.sidebar.header("Mission Setup")
orbit_names = list(ORBIT_LIBRARY.keys())
origin = st.sidebar.selectbox("Origin orbit", orbit_names, index=orbit_names.index("LEO"))
dest = st.sidebar.selectbox("Destination orbit", orbit_names, index=orbit_names.index("GEO"))
max_dv = st.sidebar.slider("Max delta-V budget (km/s)", 1.0, 10.0, 6.0, 0.1)
max_time_hr = st.sidebar.slider("Max transfer time (hr)", 1.0, 24.0, 12.0, 0.5)

spec = MissionSpec(
    name=f"{origin}_to_{dest}", r1_km=ORBIT_LIBRARY[origin], r2_km=ORBIT_LIBRARY[dest],
    max_delta_v_kms=max_dv, max_time_s=max_time_hr * 3600,
)

if origin == dest:
    st.warning("Origin and destination are the same orbit - pick two different orbits.")
    st.stop()

report = solve_constraints(spec)
summary = summarize_mission(spec, report)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Mission Summary")
    st.table({k: str(v) for k, v in summary.items()})
    if report["feasible"]:
        st.success("Mission is feasible within the given budget.")
    else:
        st.error(f"Mission is NOT feasible: {', '.join(report['violations'])}")

with col2:
    st.subheader("Delta-V Budget")
    dv1, dv2, dv_total = hohmann_delta_v(spec.r1, spec.r2)
    fig_dv = go.Figure(data=[go.Bar(x=["Departure burn", "Arrival burn"], y=[dv1, dv2],
                                     marker_color=["#4C72B0", "#DD8452"])])
    fig_dv.update_layout(yaxis_title="Delta-V (km/s)", title=f"Total: {dv_total:.3f} km/s", height=350)
    st.plotly_chart(fig_dv, width='stretch')

st.subheader("3D Orbit View")
origin_pts = get_orbit_points_3d(a=spec.r1, e=0.001, i=0, raan=0, argp=0)
dest_pts = get_orbit_points_3d(a=spec.r2, e=0.001, i=0, raan=0, argp=0)
ex, ey, ez = get_earth_sphere(resolution=25)

fig_3d = go.Figure()
fig_3d.add_trace(go.Surface(x=ex, y=ey, z=ez, colorscale="Blues", showscale=False, opacity=0.6, name="Earth"))
fig_3d.add_trace(go.Scatter3d(x=origin_pts[:, 0], y=origin_pts[:, 1], z=origin_pts[:, 2],
                               mode="lines", line=dict(color="green", width=4), name=origin))
fig_3d.add_trace(go.Scatter3d(x=dest_pts[:, 0], y=dest_pts[:, 1], z=dest_pts[:, 2],
                               mode="lines", line=dict(color="red", width=4), name=dest))
fig_3d.update_layout(height=600, scene=dict(aspectmode="data"))
st.plotly_chart(fig_3d, width='stretch')

st.subheader("Live Earth View (rotating, with ground track)")
fig_rotating = build_rotating_earth_figure(origin_pts, n_frames=36, rotation_per_frame_deg=10.0)
st.plotly_chart(fig_rotating, width='stretch')
st.caption("Click Play to rotate Earth and watch the ground track build up.")

st.subheader("Trajectory Optimization (Genetic Algorithm)")
if st.button("Run GA optimizer"):
    ga = GeneticEngine(spec, population_size=40, generations=40, seed=42)
    progress = st.progress(0)
    fitness_chart = st.empty()

    population = ga._init_population()
    best_fitness_history = []
    best_gene_overall, best_fitness_overall, best_info_overall = None, -np.inf, None
    for gen in range(ga.generations):
        fitnesses = np.array([
            __import__("fitness_functions").evaluate_trajectory(g, ga.mission_spec)[0]
            for g in population
        ])
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness_overall:
            best_gene_overall = population[gen_best_idx].copy()
            best_fitness_overall, best_info_overall = __import__("fitness_functions").evaluate_trajectory(
                best_gene_overall, ga.mission_spec
            )
        best_fitness_history.append(best_fitness_overall)

        elite_idx = np.argsort(fitnesses)[-ga.elite_count:]
        new_population = [population[i].copy() for i in elite_idx]
        while len(new_population) < ga.pop_size:
            pa = ga._tournament_select(population, fitnesses)
            pb = ga._tournament_select(population, fitnesses)
            child = ga._mutate(ga._crossover(pa, pb))
            new_population.append(child)
        population = new_population

        progress.progress((gen + 1) / ga.generations)
        fig_conv = go.Figure(data=[go.Scatter(y=best_fitness_history, mode="lines+markers")])
        fig_conv.update_layout(title="Best fitness (higher = lower delta-V) per generation",
                                xaxis_title="Generation", yaxis_title="Fitness", height=350)
        fitness_chart.plotly_chart(fig_conv, width='stretch')

    st.success(f"GA finished. Best fitness: {best_fitness_history[-1]:.4f}")
    st.session_state["ga_result"] = {
        "best_gene": best_gene_overall, "best_fitness": best_fitness_overall, "best_info": best_info_overall,
    }
    st.session_state["ga_spec"] = spec
    st.session_state["ga_report"] = report

if "ga_result" in st.session_state:
    st.subheader("Mission Report")
    if st.button("Generate PDF report"):
        pdf_path = os.path.join(_THIS_DIR, "_mission_report.pdf")
        generate_mission_report(
            st.session_state["ga_spec"], st.session_state["ga_report"],
            st.session_state["ga_result"], pdf_path,
        )
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF report", f, file_name="astroevo_mission_report.pdf",
                                mime="application/pdf")