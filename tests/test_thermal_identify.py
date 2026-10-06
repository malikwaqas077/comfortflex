import numpy as np
import pytest

from comfortflex import scenario as scn
from comfortflex.control import HeatPump
from comfortflex.identify import fit_rc2, validation_metrics
from comfortflex.pipeline import identify
from comfortflex.thermal import RC2, RC3, discretise, simulate


def test_discretisation_converges_to_steady_state():
    m = RC2()
    u = np.array([5.0, 3.0, 1.0, 0.0])
    A, B = m.discrete(0.25)
    X = simulate(A, B, np.array([20.0, 20.0]), np.tile(u, (2000, 1)))
    assert X[-1] == pytest.approx(m.steady_state(u), abs=1e-6)


def test_discretise_matches_euler_for_tiny_step():
    Ac = np.array([[-1.0, 0.5], [0.2, -0.3]])
    Bc = np.array([[1.0], [0.0]])
    A, _ = discretise(Ac, Bc, 1e-5)
    assert A == pytest.approx(np.eye(2) + Ac * 1e-5, abs=1e-9)


def test_rc2_physics_properties():
    m = RC2()
    assert m.ua == pytest.approx(1 / 6 + 1 / 4)
    assert m.time_constant_h > 10


def test_fit_recovers_known_rc2_parameters():
    true = RC2(Ci=1.0, Cm=15.0, Rim=0.8, Rio=7.0, Rmo=2.5, ai=0.7, am=1.5)
    df = scn.build(scn.WINTER)
    rng = np.random.default_rng(0)
    q = np.clip(rng.normal(5, 3, len(df)), 0, 10)      # persistently exciting input
    U = scn.inputs(df, q)
    A, B = true.discrete(0.25)
    X = simulate(A, B, np.array([19.0, 17.0]), U)
    fit = fit_rc2(X[:, 0], U, 0.25)
    assert fit.rmse_train < 0.01
    assert fit.model.ua == pytest.approx(true.ua, rel=0.03)


def test_identified_model_generalises_to_unseen_week():
    res = identify("winter", RC3(), HeatPump().for_season("winter"))
    m = res["metrics"]
    assert m["rmse_K"] < 0.6          # 7-day open-loop prediction on held-out data
    assert abs(m["nmbe_pct"]) < 2.0


def test_validation_metrics():
    m = validation_metrics(np.array([21.0, 22.0]), np.array([21.0, 21.0]))
    assert m["max_abs_K"] == 1.0
    assert m["nmbe_pct"] == pytest.approx(100 * 1 / (2 * 21))
