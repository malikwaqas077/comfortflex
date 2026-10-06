import numpy as np
import pandas as pd
import pytest

from comfortflex import scenario as scn
from comfortflex.control import (ComfortMPC, HeatPump, KalmanEstimator, MPCWeights,
                                 SafetyGuard, flexibility_hours)
from comfortflex.pipeline import Config, add_bands, run
from comfortflex.thermal import RC2


@pytest.fixture(scope="module")
def winter_fc():
    df, _ = add_bands(scn.build(scn.WINTER), "winter", 0.5)
    return df


def test_mpc_plan_is_comfortable_executable_and_seasonal(winter_fc):
    mpc = ComfortMPC(RC2(), HeatPump().for_season("winter"), 0.25, horizon=96)
    fc = winter_fc.iloc[24:120]                   # Monday 06:00 -> Tuesday 06:00
    out = mpc.plan(np.array([19.0, 18.0]), fc)
    occ_next = fc["occupied"].to_numpy()[1:]
    lo_next = fc["lo"].to_numpy()[1:]
    assert np.all(out["ti_pred"][:-1][occ_next] >= lo_next[occ_next] - 1e-6)
    assert np.all(np.abs(np.diff(np.r_[0.0, out["q"]])) <= mpc.max_ramp_kw + 1e-6)
    assert out["q"].min() >= -1e-9               # heating-only in winter


def test_mpc_shifts_load_out_of_dr_event(winter_fc):
    hp = HeatPump().for_season("winter")
    fc = winter_fc.iloc[2 * 96 + 32: 2 * 96 + 128].copy()   # Wednesday 08:00 + 24 h
    assert fc["dr_event"].any()
    x0 = np.array([21.0, 20.0])
    with_dr = ComfortMPC(RC2(), hp, 0.25, weights=MPCWeights(dr_price=3.0)).plan(x0, fc)["q"]
    no_dr = ComfortMPC(RC2(), hp, 0.25, weights=MPCWeights(dr_price=0.0)).plan(x0, fc)["q"]
    ev = fc["dr_event"].to_numpy()
    assert with_dr[ev].sum() < no_dr[ev].sum()


def test_kalman_recovers_hidden_mass_temperature():
    m = RC2()
    A, B = m.discrete(0.25)
    est = KalmanEstimator(m, 0.25)
    x = np.array([20.0, 14.0])                   # mass much colder than air
    u = np.array([5.0, 4.0, 1.0, 0.0])
    est.update(x[0], None)
    for _ in range(300):
        x = A @ x + B @ u
        xh = est.update(x[0], u)
    assert xh[1] == pytest.approx(x[1], abs=0.1)


def test_flexibility_hours_positive_inside_band(winter_fc):
    fc = winter_fc.iloc[40:80]
    h = flexibility_hours(RC2(), 0.25, np.array([22.0, 21.0]), fc)
    assert 0 < h <= 8


def test_guard_falls_back_on_bad_sensor_and_limits_ramp():
    hp = HeatPump()
    g = SafetyGuard(hp)
    ts = pd.Timestamp("2026-01-14 10:00")
    assert g.check_sensor(ts, 21.0) == 21.0
    assert g.check_sensor(ts, 45.0) is None              # implausible
    assert g.check_sensor(ts, None) is None              # now stale
    q, mode = g.command(ts, 10.0, q_fallback=2.0)
    assert mode == "fallback" and q == 2.0

    g2 = SafetyGuard(hp)
    q, mode = g2.command(ts, 14.0, q_fallback=0.0)
    assert mode == "mpc" and q == g2.max_ramp_kw         # ramp-limited from 0 kW
    g2.command(ts, float("nan"), q_fallback=1.0)
    assert any(e[1] == "optimiser_unavailable" for e in g2.log)


@pytest.mark.slow
def test_end_to_end_winter_mpc_beats_thermostat():
    r = run(Config(season="winter"))
    b, m = r["kpi_baseline"], r["kpi_mpc"]
    assert m["cost_gbp"] < b["cost_gbp"]
    assert m["dr_energy_kwh"] < 0.2 * b["dr_energy_kwh"]
    assert m["discomfort_kh"] <= b["discomfort_kh"]
    assert m["fallback_steps"] > 0                       # sensor outage was handled
    assert r["mpc"].loc[~r["mpc"]["occupied"], "ti"].min() > 15.0   # frost/setback protection held
