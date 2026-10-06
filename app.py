"""ComfortFlex dashboard: streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from comfortflex.comfort import ComfortInputs, adaptive_band, pmv_band, pmv_ppd
from comfortflex.control import MPCWeights
from comfortflex.pipeline import Config, run

st.set_page_config(page_title="ComfortFlex", page_icon="🌡️", layout="wide")

BASE_C, MPC_C, BAND_C, DR_C = "#9aa5b1", "#0f766e", "rgba(15,118,110,0.10)", "rgba(220,38,38,0.22)"


@st.cache_data(show_spinner="Identifying the building model and simulating a week of closed-loop control...")
def cached_run(season, pmv_limit, dr_price, carbon_price, comfort_w):
    w = MPCWeights(carbon_price=carbon_price, dr_price=dr_price, comfort_occupied=comfort_w)
    r = run(Config(season=season, pmv_limit=pmv_limit, weights=w))
    f = r["ident"]["fit"]
    return {
        "band": r["band"], "baseline": r["baseline"], "mpc": r["mpc"],
        "kb": r["kpi_baseline"], "km": r["kpi_mpc"], "guard": r["guard_log"],
        "params": f.model.__dict__, "ua": f.model.ua, "tau": f.model.time_constant_h,
        "rmse_train": f.rmse_train, "metrics": r["ident"]["metrics"],
        "train": r["ident"]["train"], "valid": r["ident"]["valid"], "scenario": r["scenario"].name,
    }


# Sidebar -----------------------------------------------------------------------
st.sidebar.title("ComfortFlex")
st.sidebar.caption("Physics-informed comfort & grid demand-response control")
season = st.sidebar.radio("Scenario", ["summer", "winter"],
                          format_func=lambda s: {"summer": "Summer heatwave (cooling)",
                                                 "winter": "Winter cold snap (heating)"}[s])
cat = st.sidebar.select_slider("ISO 7730 comfort category", ["A (|PMV|≤0.2)", "B (|PMV|≤0.5)", "C (|PMV|≤0.7)"],
                               value="B (|PMV|≤0.5)")
pmv_limit = {"A": 0.2, "B": 0.5, "C": 0.7}[cat[0]]
dr_price = st.sidebar.slider("Flexibility event value (£/kWh avoided)", 0.0, 6.0, 3.0, 0.5)
carbon_price = st.sidebar.slider("Carbon shadow price (£/kgCO₂)", 0.0, 0.5, 0.10, 0.05)
comfort_w = st.sidebar.slider("Comfort penalty (£/K·h)", 0.5, 20.0, 4.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.caption("All data are synthetic. The plant is a 3-node building the controller never sees; "
                   "the controller uses a 2-node model identified from noisy sensor history.")

R = cached_run(season, pmv_limit, dr_price, carbon_price, comfort_w)
kb, km = R["kb"], R["km"]

st.title("ComfortFlex: comfort-aware HVAC control with grid demand response")
st.caption(f"{R['scenario']} · one office zone · 15-minute control step · 24 h MPC horizon")

tabs = st.tabs(["Overview", "1 · Comfort model", "2 · Building physics model", "3 · Control & demand response",
                "4 · Safety & deployment"])


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * (a - b) / b


# Overview ----------------------------------------------------------------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Electricity cost", f"£{km['cost_gbp']:.2f}", f"{pct(km['cost_gbp'], kb['cost_gbp']):+.0f}% vs thermostat",
                delta_color="inverse")
    c[1].metric("Grid CO₂", f"{km['co2_kg']:.1f} kg", f"{pct(km['co2_kg'], kb['co2_kg']):+.0f}%", delta_color="inverse")
    c[2].metric("Load during flexibility events", f"{km['dr_energy_kwh']:.1f} kWh",
                f"{pct(km['dr_energy_kwh'], kb['dr_energy_kwh']):+.0f}%", delta_color="inverse")
    c[3].metric("Occupied time within PMV band", f"{km['pct_occ_hours_in_band']:.0f}%",
                f"{km['pct_occ_hours_in_band'] - kb['pct_occ_hours_in_band']:+.0f} pts")
    c[4].metric("Model 7-day prediction error", f"{R['metrics']['rmse_K']:.2f} K RMSE")

    st.markdown("""
**What this demonstrates** — the full loop a KTP Associate would take from building physics to a controller running in
a live building:

1. **Human comfort** — ISO 7730 PMV/PPD (validated against the standard's reference table) and the EN 16798-1 adaptive
   model, inverted into a temperature band the controller can track.
2. **Building physics** — a grey-box RC thermal model identified from noisy 15-minute sensor history, validated on a
   held-out week (ASHRAE Guideline 14 style metrics), with a disturbance-augmented Kalman filter estimating the unmeasured thermal-mass state and heat gains.
3. **Grid-aware optimisation** — a linear-programme MPC that pre-heats / pre-cools the thermal mass to move load out of
   peak-price and high-carbon periods and out of demand-flexibility events, while holding the comfort band.
4. **Deployment** — a safety guard between optimiser and plant: sensor plausibility and staleness checks, seasonal
   changeover, power and ramp limits, automatic fallback to the BMS schedule, and an audit log.
""")
    kdf = pd.DataFrame({"Thermostat (BMS schedule)": kb, "ComfortFlex MPC": km}).round(2)
    kdf.index = ["Energy (kWh)", "Cost (£)", "CO₂ (kg)", "Energy in DR events (kWh)", "Mean load in DR events (kW)",
                 "Peak electrical load (kW)", "Occupied hours", "Discomfort (K·h outside band)",
                 "Occupied time in PMV band (%)", "Mean PPD occupied (%)", "Fallback steps",
                 "Flexibility payment vs baseline (£)"]
    st.table(kdf.style.format("{:.2f}"))

# Comfort -------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Thermal comfort: from PMV/PPD to a controllable band")
    c1, c2 = st.columns([1, 2])
    with c1:
        clo = st.slider("Clothing (clo)", 0.3, 1.5, 0.5 if season == "summer" else 1.0, 0.1)
        met = st.slider("Metabolic rate (met)", 0.8, 2.0, 1.2, 0.1)
        rh = st.slider("Relative humidity (%)", 20, 80, 55 if season == "summer" else 40, 5)
        vel = st.slider("Air speed (m/s)", 0.05, 0.8, 0.15, 0.05)
        p = ComfortInputs(rh=rh, vel=vel, met=met, clo=clo)
        lo, hi = pmv_band(p, pmv_limit)
        st.metric(f"Comfort band (|PMV| ≤ {pmv_limit})", f"{lo:.1f} – {hi:.1f} °C")
        trm = st.slider("Running-mean outdoor temp (°C), EN 16798-1", 10.0, 30.0, 18.0, 0.5)
        alo, ahi = adaptive_band(trm, "II")
        st.caption(f"Adaptive comfort, Category II: {alo:.1f} – {ahi:.1f} °C (free-running / mixed-mode spaces)")
    with c2:
        t = np.linspace(16, 32, 161)
        pv = np.array([pmv_ppd(x, x, vel, rh, met, clo) for x in t])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_vrect(x0=lo, x1=hi, fillcolor=BAND_C, line_width=0, annotation_text="comfort band")
        fig.add_trace(go.Scatter(x=t, y=pv[:, 0], name="PMV", line=dict(color=MPC_C, width=3)))
        fig.add_trace(go.Scatter(x=t, y=pv[:, 1], name="PPD %", line=dict(color="#b45309", dash="dot")),
                      secondary_y=True)
        fig.update_layout(height=420, xaxis_title="Operative temperature (°C)", margin=dict(t=30),
                          legend=dict(orientation="h"))
        fig.update_yaxes(title_text="PMV", secondary_y=False)
        fig.update_yaxes(title_text="PPD (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("PMV implementation matches ten ISO 7730:2005 Annex D reference cases to ±0.02 (see tests/test_comfort.py).")

# Building model --------------------------------------------------------------------
with tabs[2]:
    st.subheader("Grey-box thermal model identified from sensor data")
    c1, c2 = st.columns([1.3, 2])
    with c1:
        st.markdown("**2R2C model** — zone air node + lumped envelope/structure mass")
        st.latex(r"C_i\dot T_i=\tfrac{T_m-T_i}{R_{im}}+\tfrac{T_o-T_i}{R_{io}}+\dot Q_{hvac}+\dot Q_{int}+a_iS+d")
        st.latex(r"C_m\dot T_m=\tfrac{T_i-T_m}{R_{im}}+\tfrac{T_o-T_m}{R_{mo}}+a_mS")
        st.caption("d: unmeasured heat gain, estimated online by the Kalman filter (offset-free MPC)")
        prm = pd.Series(R["params"]).round(3)
        prm.index = ["Cᵢ (kWh/K)", "Cₘ (kWh/K)", "Rᵢₘ (K/kW)", "Rᵢₒ (K/kW)", "Rₘₒ (K/kW)", "aᵢ (m²)", "aₘ (m²)"]
        st.table(prm.rename("identified").to_frame().style.format("{:.3f}"))
        st.write(f"Heat-loss coefficient UA = **{R['ua']:.2f} kW/K**, dominant time constant = **{R['tau']:.0f} h**")
        m = R["metrics"]
        st.write(f"Held-out week, open-loop: RMSE **{m['rmse_K']:.2f} K**, MAE {m['mae_K']:.2f} K, "
                 f"CV(RMSE) {m['cv_rmse_pct']:.1f}%, NMBE {m['nmbe_pct']:+.2f}%")
        st.caption("The simulated 'true' building has three thermal nodes, so the fitted model is deliberately "
                   "mismatched, as with a real building.")
    with c2:
        v = R["valid"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=v.index, y=v["ti_meas"], name="Measured (held-out week)",
                                 line=dict(color=BASE_C)), 1, 1)
        fig.add_trace(go.Scatter(x=v.index, y=v["ti_pred"], name="Model, 7-day open-loop prediction",
                                 line=dict(color=MPC_C, width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=v.index, y=v["t_out"], name="Outdoor", line=dict(color="#64748b", dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=v.index, y=v["q_hvac"], name="HVAC thermal power (kW)", fill="tozeroy",
                                 line=dict(color="#b45309")), 2, 1)
        fig.update_layout(height=520, margin=dict(t=30), legend=dict(orientation="h"))
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="kW", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

# Control ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Closed-loop week: thermostat vs comfort-aware MPC")
    b, mres = R["baseline"], R["mpc"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.04)
    ev =mres["dr_event"].astype(int).diff().fillna(0)
    starts, ends = mres.index[ev == 1], mres.index[ev == -1]
    ymax_kw = 1.05 * max(b["p_elec"].max(), mres["p_elec"].max())
    for i, (s0, s1) in enumerate(zip(starts, ends)):
        for r_, (y0, y1) in ((1, (21.0, 28.5) if season == "summer" else (15.5, 24.5)), (2, (0.0, ymax_kw))):
            fig.add_trace(go.Scatter(x=[s0, s1, s1, s0, s0], y=[y0, y0, y1, y1, y0], fill="toself",
                                     fillcolor=DR_C, line=dict(width=0), mode="lines", hoverinfo="skip",
                                     name="Flexibility event", legendgroup="dr",
                                     showlegend=(i == 0 and r_ == 1)), r_, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["lo"], fill="tonexty", fillcolor=BAND_C, line=dict(width=0),
                             name="Comfort band (PMV when occupied)"), 1, 1)
    fig.add_trace(go.Scatter(x=b.index, y=b["ti"], name="Thermostat", line=dict(color=BASE_C)), 1, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["ti"], name="ComfortFlex MPC", line=dict(color=MPC_C, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=b.index, y=b["p_elec"], name="Thermostat kW", line=dict(color=BASE_C)), 2, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["p_elec"], name="MPC kW", line=dict(color=MPC_C)), 2, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["price"], name="Tariff £/kWh", line=dict(color="#b45309")), 3, 1)
    fig.add_trace(go.Scatter(x=mres.index, y=mres["carbon"], name="Grid kgCO₂/kWh", line=dict(color="#475569", dash="dot")), 3, 1)
    fig.update_layout(height=680, margin=dict(t=30), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Zone °C", row=1, col=1)
    fig.update_yaxes(title_text="Electric kW", row=2, col=1)
    fig.update_yaxes(title_text="Signals", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bands are demand-flexibility events (e.g. a DFS / DNO flexibility call). The MPC pre-conditions "
               "the thermal mass beforehand and coasts through the event inside the comfort band.")
    flex = mres["flex_h"].dropna()
    if len(flex):
        st.write("**Flexibility available at event start** (hours the HVAC could stay off before leaving the band): "
                 + ", ".join(f"{t:%a %H:%M} → {h:.2f} h" for t, h in flex.items()))

# Safety ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("From optimiser to live plant")
    st.markdown("""
```
LoRaWAN sensors ──► plausibility / staleness checks ──► Kalman state estimator ──► MPC (LP, HiGHS)
                         │ invalid / stale                                          │ infeasible / timeout
                         ▼                                                          ▼
                 BMS schedule fallback ◄──────────────── Safety guard: seasonal mode, power & ramp limits
                                                                     │
                                                       BMS / HVAC command + audit log
```
""")
    mres = R["mpc"]
    fb = mres[mres["mode"] == "fallback"]
    st.write(f"Fallback steps this week: **{len(fb)}** (a one-hour wireless sensor outage was injected on Wednesday).")
    log = pd.DataFrame(R["guard"], columns=["time", "event", "detail"])
    st.table(log.assign(time=log["time"].astype(str)).head(15))
    st.markdown("""
**Next steps toward live deployment**
- Use mean radiant temperature (globe or surface sensors) for operative temperature instead of air temperature.
- Calibrate PMV inputs (clothing, metabolic rate) and the comfort band from occupant feedback.
- Re-identify the model per zone on a rolling window, with drift detection.
- Move to multi-zone coupled models and a hybrid physics-ML residual model.
- Replay against measured data from a Cosysense site before a supervised pilot.
""")
