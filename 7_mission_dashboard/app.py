import os
import sys
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "2_symbolic_mission_planner"))
sys.path.insert(0, os.path.join(ROOT, "3_evolutionary_optimizer"))

from mission_spec_parser import ORBIT_LIBRARY, MissionSpec
from genetic_engine import GeneticEngine
from constraint_solver import hohmann_delta_v

st.set_page_config(
    page_title="AstroEvo Mission Dashboard",
    page_icon="🛰️",
    layout="wide"
)

st.title("AstroEvo — Mission Planning Dashboard")
st.caption(
    "Orbital mission optimization simulation | Research prototype"
)

with st.sidebar:
    st.header("Mission Configuration")

    origin = st.selectbox(
        "Origin Orbit",
        list(ORBIT_LIBRARY.keys()),
        index=0
    )

    destinations = [
        orbit for orbit in ORBIT_LIBRARY
        if orbit != origin
    ]

    destination = st.selectbox(
        "Destination Orbit",
        destinations,
        index=destinations.index("GEO")
        if "GEO" in destinations else 0
    )

    max_dv = st.number_input(
        "Delta-V Budget (km/s)",
        min_value=0.1,
        value=5.0,
        step=0.1
    )

    max_time_hr = st.number_input(
        "Time Budget (hours)",
        min_value=0.5,
        value=24.0,
        step=0.5
    )

    population = st.slider(
        "Optimizer Population",
        min_value=20,
        max_value=200,
        value=80,
        step=10
    )

    generations = st.slider(
        "Optimizer Generations",
        min_value=10,
        max_value=200,
        value=60,
        step=10
    )

    run = st.button(
        "Run Mission Optimizer",
        type="primary"
    )

r1 = ORBIT_LIBRARY[origin]
r2 = ORBIT_LIBRARY[destination]

_, _, reference_dv = hohmann_delta_v(r1, r2)

col1, col2, col3 = st.columns(3)

col1.metric("Origin Radius", f"{r1:,.0f} km")
col2.metric("Destination Radius", f"{r2:,.0f} km")
col3.metric("Hohmann Reference Delta-V", f"{reference_dv:.4f} km/s")

st.info(
    "Configure the mission in the sidebar and run the optimizer "
    "to calculate a candidate orbital transfer."
)

if run:
    with st.spinner("Running evolutionary optimization..."):
        spec = MissionSpec(
            name=f"{origin}_to_{destination}",
            r1_km=r1,
            r2_km=r2,
            max_delta_v_kms=float(max_dv),
            max_time_s=float(max_time_hr) * 3600
        )

        ga = GeneticEngine(
            spec,
            population_size=int(population),
            generations=int(generations),
            mutation_rate=0.15,
            elite_count=4,
            seed=42
        )

        best_gene, best_fitness, best_info = ga.run()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Optimizer Delta-V",
        f"{best_info['dv_total_kms']:.4f} km/s"
    )

    col2.metric(
        "Transfer Time",
        f"{best_gene[1]:.2f} hours"
    )

    col3.metric(
        "Within Configured Constraints",
        "Yes" if best_info["feasible"] else "No"
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Evolutionary Solution", "Hohmann Reference"],
                y=[
                    best_info["dv_total_kms"],
                    reference_dv
                ]
            )
        ]
    )

    fig.update_layout(
        title="Orbital Transfer Delta-V Comparison",
        xaxis_title="Method",
        yaxis_title="Delta-V (km/s)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Optimization Results")

    st.json({
        "mission": spec.name,
        "transfer_angle_deg": round(float(best_gene[0]), 3),
        "transfer_time_hours": round(float(best_gene[1]), 3),
        "delta_v_kms": round(
            float(best_info["dv_total_kms"]), 6
        ),
        "fitness": float(best_fitness),
        "feasible": bool(best_info["feasible"])
    })

st.divider()

st.caption(
    "AstroEvo is an experimental software simulation. "
    "Results require independent scientific validation before "
    "any operational spaceflight use."
)
