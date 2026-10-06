"""End-to-end pipeline: history -> identification -> closed-loop control -> KPIs."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import scenario as scn
from .comfort import ComfortInputs, pmv_band, pmv_ppd
from .control import (ComfortMPC, HeatPump, KalmanEstimator, MPCWeights, SafetyGuard,
                      ThermostatController, flexibility_hours)
from .identify import fit_rc2, validation_metrics
from .thermal import RC2, RC3, simulate

SEASON_COMFORT = {
    "summer": ComfortInputs(rh=55.0, vel=0.15, met=1.2, clo=0.5),
    "winter": ComfortInputs(rh=40.0, vel=0.10, met=1.2, clo=1.0),
}
# Unoccupied limits: frost/overheat protection plus a pre-cool / pre-heat limit
SETBACK_BAND = {"summer": (21.0, 28.0), "winter": (16.0, 23.0)}


@dataclass
class Config:
    season: str = "summer"
    pmv_limit: float = 0.5
    weights: MPCWeights | None = None
    horizon_h: float = 24.0
    sensor_fault: bool = True         # inject a wireless sensor outage on Wednesday
    forecast_noise: bool = True
    offset_free: bool = True          # disturbance-augmented Kalman filter
    margin_k: float = 0.5


def add_bands(df: pd.DataFrame, season: str, pmv_limit: float) -> tuple[pd.DataFrame, tuple]:
    lo_c, hi_c = pmv_band(SEASON_COMFORT[season], pmv_limit)
    df = df.copy()
    lo_s, hi_s = SETBACK_BAND[season]
    df["lo"] = np.where(df["occupied"], lo_c, lo_s)
    df["hi"] = np.where(df["occupied"], hi_c, hi_s)
    return df, (lo_c, hi_c)


def initial_state(plant: RC3, row, ti: float) -> np.ndarray:
    """Plant steady state with the zone held at `ti` (building already conditioned)."""
    q_lo, q_hi = -50.0, 50.0
    for _ in range(60):                       # bisection on the holding power
        q = 0.5 * (q_lo + q_hi)
        x = plant.steady_state(np.array([row.t_out, q, row.q_int, 0.0]))
        q_lo, q_hi = (q, q_hi) if x[0] < ti else (q_lo, q)
    return x


def _measure(rng, ti: float) -> float:
    return round(ti + rng.normal(0, 0.08), 1)    # 0.1 C sensor resolution


def generate_history(season: str, plant: RC3, hp: HeatPump, seed: int, start: str) -> pd.DataFrame:
    """A week of 'historical' BMS data with occasional manual setpoint changes."""
    base = scn.SCENARIOS[season]
    sc = replace(base, start=start, seed=seed, dr_days=())
    df = scn.build(sc)
    rng = np.random.default_rng(seed)
    thermo = ThermostatController(season, hp)
    A, B = plant.discrete(sc.dt_h)
    x = initial_state(plant, df.iloc[0], 26.0 if season == "summer" else 18.0)
    offset, ti_meas, q_log = 0.0, [], []
    for k, (ts, row) in enumerate(df.iterrows()):
        if k % 16 == 0 and rng.random() < 0.35:        # occupants nudge the thermostat
            offset = rng.choice([-1.5, -1.0, 0.0, 1.0, 1.5])
        y = _measure(rng, x[0])
        ti_meas.append(y)
        thermo.occupied_sp = (24.0 if season == "summer" else 21.0) + offset
        q = thermo.act(ts, y)
        q_log.append(q)
        x = A @ x + B @ np.array([row.t_out, q, row.q_int, row.solar])
    df["ti_meas"] = ti_meas
    df["q_hvac"] = q_log
    return df


def identify(season: str, plant: RC3, hp: HeatPump) -> dict:
    """Fit on one historical week, validate open-loop on a different week."""
    base = scn.SCENARIOS[season]
    t0 = pd.Timestamp(base.start)
    train = generate_history(season, plant, hp, seed=101, start=str((t0 - pd.Timedelta("14D")).date()))
    valid = generate_history(season, plant, hp, seed=202, start=str((t0 - pd.Timedelta("7D")).date()))
    dt = base.dt_h

    U_tr = scn.inputs(train, train["q_hvac"].to_numpy())
    fit = fit_rc2(train["ti_meas"].to_numpy(), U_tr[:-1], dt)

    # Open-loop 7-day validation: initial mass state from a short warm-up of the filter
    U_va = scn.inputs(valid, valid["q_hvac"].to_numpy())
    est = KalmanEstimator(fit.model, dt)
    y = valid["ti_meas"].to_numpy()
    x = est.update(y[0], None)
    warm = 16
    for k in range(1, warm + 1):
        x = est.update(y[k], U_va[k - 1])
    A, B = fit.model.discrete(dt)
    X = simulate(A, B, x, U_va[warm:-1])
    pred = X[:, 0]
    metrics = validation_metrics(pred, y[warm:])
    valid = valid.iloc[warm:].copy()
    valid["ti_pred"] = pred
    return {"fit": fit, "train": train, "valid": valid, "metrics": metrics}


def run_closed_loop(df: pd.DataFrame, n_steps: int, controller: str, plant: RC3, hp: HeatPump,
                    model: RC2 | None, cfg: Config, seed: int = 11) -> pd.DataFrame:
    """Simulate the true building under `controller` ("baseline" | "mpc")."""
    dt = (df.index[1] - df.index[0]) / pd.Timedelta("1h")
    rng = np.random.default_rng(seed)
    A3, B3 = plant.discrete(dt)
    x = initial_state(plant, df.iloc[0], 26.0 if cfg.season == "summer" else 18.0)
    thermo = ThermostatController(cfg.season, hp)
    N = int(cfg.horizon_h / dt)

    if controller == "mpc":
        mpc = ComfortMPC(model, hp, dt, horizon=N, weights=cfg.weights or MPCWeights(),
                         margin_k=cfg.margin_k)
        est = KalmanEstimator(model, dt, disturbance=cfg.offset_free)
        guard = SafetyGuard(hp)
        fc_noise = rng.normal(0, 1, (len(df), 2)) if cfg.forecast_noise else np.zeros((len(df), 2))

    rows, u_prev = [], None
    for k in range(n_steps):
        ts = df.index[k]
        row = df.iloc[k]
        y = _measure(rng, x[0])
        y_cosy = y
        if controller == "mpc" and cfg.sensor_fault and ts.dayofweek == 2 and 10 <= ts.hour < 11:
            y_cosy = None                                  # LoRaWAN packets lost
        flex_h = np.nan
        if controller == "baseline":
            q, mode = thermo.act(ts, y), "baseline"
        else:
            y_ok = guard.check_sensor(ts, y_cosy)
            xh = est.update(y_ok, u_prev)
            q_opt = None
            try:
                fc = df.iloc[k:k + N].copy()
                h = np.arange(len(fc)) * dt
                fc["t_out"] = fc["t_out"] + 0.5 * np.sqrt(h) * fc_noise[k:k + N, 0] * 0.3
                fc["q_int"] = np.clip(fc["q_int"] + 0.3 * fc_noise[k:k + N, 1], 0, None)
                q_opt = float(mpc.plan(xh, fc, q_prev=guard._last_cmd, d=est.d)["q"][0])
                if row.dr_event and (k == 0 or not df.dr_event.iloc[k - 1]):
                    flex_h = flexibility_hours(model, dt, xh, fc)
            except RuntimeError:
                q_opt = None
            q, mode = guard.command(ts, q_opt, thermo.act(ts, y))
        u = np.array([row.t_out, q, row.q_int, row.solar])
        pmv, ppd = pmv_ppd(x[0], x[0], *_comfort_args(cfg.season))
        rows.append({
            "ts": ts, "ti": x[0], "ti_meas": y_cosy if controller == "mpc" else y, "q_hvac": q,
            "p_elec": float(hp.electric_kw(q, row.t_out)), "mode": mode,
            "pmv": pmv, "ppd": ppd, "flex_h": flex_h,
        })
        x = A3 @ x + B3 @ u
        u_prev = u
    out = pd.DataFrame(rows).set_index("ts")
    out = out.join(df[["t_out", "solar", "occ", "occupied", "price", "carbon", "dr_event", "lo", "hi"]])
    if controller == "mpc":
        out.attrs["guard_log"] = guard.log
    return out


def _comfort_args(season: str):
    p = SEASON_COMFORT[season]
    return p.vel, p.rh, p.met, p.clo


def kpis(res: pd.DataFrame, pmv_limit: float) -> dict[str, float]:
    dt = (res.index[1] - res.index[0]) / pd.Timedelta("1h")
    e = res["p_elec"] * dt
    occ = res["occupied"]
    dr = res["dr_event"]
    viol = np.clip(res["lo"] - res["ti"], 0, None) + np.clip(res["ti"] - res["hi"], 0, None)
    return {
        "energy_kwh": float(e.sum()),
        "cost_gbp": float((e * res["price"]).sum()),
        "co2_kg": float((e * res["carbon"]).sum()),
        "dr_energy_kwh": float(e[dr].sum()),
        "dr_mean_kw": float(res.loc[dr, "p_elec"].mean()) if dr.any() else 0.0,
        "peak_kw": float(res["p_elec"].max()),
        "occupied_hours": float(occ.sum() * dt),
        "discomfort_kh": float((viol[occ] * dt).sum()),
        "pct_occ_hours_in_band": float(100.0 * (res.loc[occ, "pmv"].abs() <= pmv_limit + 1e-9).mean()),
        "mean_ppd_occ": float(res.loc[occ, "ppd"].mean()),
        "fallback_steps": int((res["mode"] == "fallback").sum()),
    }


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    base = scn.SCENARIOS[cfg.season]
    plant, hp = RC3(), HeatPump().for_season(cfg.season)
    ident = identify(cfg.season, plant, hp)

    sc_ext = replace(base, days=base.days + 2)          # extra days so the horizon never runs out
    df, band = add_bands(scn.build(sc_ext), cfg.season, cfg.pmv_limit)
    n = int(base.days * 24 / base.dt_h)
    res_b = run_closed_loop(df, n, "baseline", plant, hp, None, cfg)
    res_m = run_closed_loop(df, n, "mpc", plant, hp, ident["fit"].model, cfg)
    k_b, k_m = kpis(res_b, cfg.pmv_limit), kpis(res_m, cfg.pmv_limit)
    return {"config": cfg, "scenario": base, "band": band, "ident": ident,
            "baseline": res_b, "mpc": res_m, "kpi_baseline": k_b, "kpi_mpc": k_m,
            "guard_log": res_m.attrs.get("guard_log", [])}
