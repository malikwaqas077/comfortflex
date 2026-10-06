"""Parameter identification of the 2R2C grey-box model from sensor data.

The fit minimises multi-step (simulation) error rather than one-step-ahead error,
which is what matters when the model is used for a 24 h MPC horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .thermal import RC2, simulate

# Physical bounds keep the optimiser inside plausible building physics.
LOWER = np.array([0.05, 1.0, 0.05, 0.5, 0.2, 1e-3, 1e-3])
UPPER = np.array([5.0, 80.0, 10.0, 60.0, 30.0, 5.0, 10.0])


@dataclass
class FitResult:
    model: RC2
    tm0: float
    rmse_train: float
    cost: float
    nfev: int


def _simulate_ti(theta: np.ndarray, U: np.ndarray, ti0: float, dt: float) -> np.ndarray:
    model = RC2.from_vector(np.exp(theta[:-1]))
    A, B = model.discrete(dt)
    X = simulate(A, B, np.array([ti0, theta[-1]]), U)
    return X[1:, 0]


def fit_rc2(ti_meas: np.ndarray, U: np.ndarray, dt: float,
            initial: RC2 | None = None) -> FitResult:
    """Fit RC2 parameters (and the unobserved initial mass temperature).

    ti_meas: measured zone temperature, length N+1 (state at each step boundary)
    U: inputs, shape (N, 4)
    """
    init = initial or RC2()
    tm0_guess = float(np.mean(ti_meas[:96]))
    theta0 = np.r_[np.log(np.clip(init.as_vector(), LOWER * 1.01, UPPER * 0.99)), tm0_guess]
    lb = np.r_[np.log(LOWER), -10.0]
    ub = np.r_[np.log(UPPER), 50.0]

    def resid(theta):
        return _simulate_ti(theta, U, ti_meas[0], dt) - ti_meas[1:]

    sol = least_squares(resid, theta0, bounds=(lb, ub), x_scale="jac", max_nfev=400)
    model = RC2.from_vector(np.exp(sol.x[:-1]))
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    return FitResult(model, float(sol.x[-1]), rmse, float(sol.cost), int(sol.nfev))


def validation_metrics(pred: np.ndarray, meas: np.ndarray) -> dict[str, float]:
    """RMSE / MAE in K plus ASHRAE Guideline 14 style CV(RMSE) and NMBE (%)."""
    e = pred - meas
    mean = float(np.mean(meas))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {
        "rmse_K": rmse,
        "mae_K": float(np.mean(np.abs(e))),
        "max_abs_K": float(np.max(np.abs(e))),
        "cv_rmse_pct": 100.0 * rmse / mean,
        "nmbe_pct": 100.0 * float(np.sum(e)) / (len(e) * mean),
    }
