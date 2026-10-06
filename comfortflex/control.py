"""Controllers, state estimation and the production safety guard.

* ThermostatController - the incumbent BMS behaviour (scheduled setpoints, P-control).
* ComfortMPC - linear-programme MPC on the identified RC model that trades off
  electricity cost, grid carbon, demand-flexibility events and PMV-derived comfort.
* KalmanEstimator - recovers the unmeasured thermal-mass state from the air sensor.
* SafetyGuard - the layer between the optimiser and live HVAC plant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

from .thermal import RC2


# HVAC plant -----------------------------------------------------------------

@dataclass
class HeatPump:
    """Reversible air-source heat pump / VRF serving the zone."""

    cap_heat_kw: float = 14.0    # thermal
    cap_cool_kw: float = 12.0

    def for_season(self, season: str) -> "HeatPump":
        """Seasonal changeover, as on real VRF/AHU plant: cooling-only in summer,
        heating-only in winter."""
        if season == "summer":
            return HeatPump(cap_heat_kw=0.0, cap_cool_kw=self.cap_cool_kw)
        return HeatPump(cap_heat_kw=self.cap_heat_kw, cap_cool_kw=0.0)

    def cop_heat(self, t_out):
        return np.clip(3.0 + 0.08 * (np.asarray(t_out) - 7.0), 1.8, 5.0)

    def cop_cool(self, t_out):
        return np.clip(4.2 - 0.09 * (np.asarray(t_out) - 25.0), 2.0, 6.0)

    def electric_kw(self, q_thermal, t_out):
        q = np.asarray(q_thermal, dtype=float)
        return np.where(q >= 0, q / self.cop_heat(t_out), -q / self.cop_cool(t_out))


# Incumbent controller ---------------------------------------------------------

@dataclass
class ThermostatController:
    season: str
    hp: HeatPump
    kp: float = 3.0                 # kW per K
    ki: float = 1.5                 # kW per K-hour
    dt: float = 0.25
    occupied_sp: float | None = None
    setback_sp: float | None = None
    start_hour: float = 7.0
    stop_hour: float = 18.0
    _integ: float = 0.0

    def __post_init__(self):
        if self.occupied_sp is None:
            self.occupied_sp = 24.0 if self.season == "summer" else 21.0
        if self.setback_sp is None:
            self.setback_sp = 28.0 if self.season == "summer" else 16.0

    def setpoint(self, ts) -> float:
        h = ts.hour + ts.minute / 60.0
        on = ts.dayofweek < 5 and self.start_hour <= h < self.stop_hour
        return self.occupied_sp if on else self.setback_sp

    def act(self, ts, ti: float) -> float:
        """PI loop with conditional-integration anti-windup, as in a typical BMS."""
        sp = self.setpoint(ts)
        err = (ti - sp) if self.season == "summer" else (sp - ti)
        cap = self.hp.cap_cool_kw if self.season == "summer" else self.hp.cap_heat_kw
        raw = self.kp * err + self.ki * self._integ
        if 0.0 < raw < cap or (raw <= 0.0 and err > 0) or (raw >= cap and err < 0):
            self._integ = float(np.clip(self._integ + err * self.dt, 0.0, cap / self.ki))
        q = float(np.clip(self.kp * err + self.ki * self._integ, 0.0, cap))
        return -q if self.season == "summer" else q


# State estimation -------------------------------------------------------------

class KalmanEstimator:
    """Steady-state Kalman filter for the 2-node model with an air-temperature sensor.

    With `disturbance=True` the state is augmented with an unmeasured heat gain d (kW,
    random walk) acting on the air node. It absorbs model mismatch and unmodelled
    gains, and passing it to the MPC gives offset-free tracking of the comfort band.
    """

    def __init__(self, model: RC2, dt: float, q_proc=(0.02, 0.005), r_meas=0.05 ** 2,
                 disturbance: bool = False, q_dist: float = 0.01):
        A, B = model.discrete(dt)
        self.B = B
        self.nd = 1 if disturbance else 0
        n = 2 + self.nd
        self.A = np.eye(n)
        self.A[:2, :2] = A
        if disturbance:
            self.A[:2, 2] = B[:, 1]            # d enters like HVAC power on the air node
        C = np.zeros((1, n))
        C[0, 0] = 1.0
        Q = np.diag(list(q_proc) + ([q_dist] if disturbance else []))
        P = solve_discrete_are(self.A.T, C.T, Q, np.array([[r_meas]]))
        self.K = (P @ C.T @ np.linalg.inv(C @ P @ C.T + r_meas)).ravel()
        self.C = C.ravel()
        self.z: np.ndarray | None = None

    @property
    def d(self) -> float:
        return float(self.z[2]) if self.nd and self.z is not None else 0.0

    def update(self, y: float | None, u_prev: np.ndarray | None) -> np.ndarray:
        """Advance one step and return the estimated [Ti, Tm]."""
        if self.z is None:
            if y is None:
                raise ValueError("estimator needs a valid first reading")
            self.z = np.r_[y, y, np.zeros(self.nd)].astype(float)
            return self.z[:2].copy()
        self.z = self.A @ self.z
        self.z[:2] += self.B @ u_prev
        if y is not None:                 # missing packet -> predict only
            self.z = self.z + self.K * (y - self.C @ self.z)
        return self.z[:2].copy()


# Model predictive control -----------------------------------------------------

@dataclass
class MPCWeights:
    carbon_price: float = 0.10       # GBP per kgCO2 (shadow price)
    dr_price: float = 3.00           # GBP per kWh avoided during a flexibility event
    comfort_occupied: float = 4.0    # GBP per K-hour outside the PMV band
    comfort_unoccupied: float = 1.0  # setback limits are protection limits
    move: float = 0.02               # GBP per kW of command change (compressor wear, chatter)


@dataclass
class ComfortMPC:
    model: RC2
    hp: HeatPump
    dt: float
    horizon: int = 96
    weights: MPCWeights = field(default_factory=MPCWeights)
    margin_k: float = 0.5            # constraint back-off for model/forecast error
    max_ramp_kw: float = 5.0         # matches SafetyGuard so plans are executable

    def __post_init__(self):
        A, B = self.model.discrete(self.dt)
        N = self.horizon
        self.A, self.B = A, B
        # Powers of A and the (constant) lower-triangular response of Ti to Q_hvac
        self.Apow = np.empty((N + 1, 2, 2))
        self.Apow[0] = np.eye(2)
        for k in range(1, N + 1):
            self.Apow[k] = A @ self.Apow[k - 1]
        g = np.array([(self.Apow[k] @ B[:, 1])[0] for k in range(N)])  # impulse response
        self.Gamma = np.zeros((N, N))
        for i in range(N):
            self.Gamma[i, : i + 1] = g[i::-1]
        self.last_status = "not run"

    def free_response(self, x0: np.ndarray, D: np.ndarray, d: float = 0.0) -> np.ndarray:
        """Ti trajectory (steps 1..N) with zero HVAC power. D columns: T_out, Q_int, solar;
        d is the estimated unmeasured heat gain, held constant over the horizon."""
        Bd = self.B[:, [0, 2, 3]]
        bq = self.B[:, 1] * d
        x = x0.copy()
        out = np.empty(self.horizon)
        for k in range(self.horizon):
            x = self.A @ x + Bd @ D[k] + bq
            out[k] = x[0]
        return out

    def plan(self, x0: np.ndarray, fc, q_prev: float = 0.0, d: float = 0.0) -> dict:
        """Solve one horizon. `fc` is a DataFrame slice (length N) with forecasts plus
        `lo`/`hi` comfort bounds. Returns the optimal power sequence and prediction."""
        N, dt, w = self.horizon, self.dt, self.weights
        D = fc[["t_out", "q_int", "solar"]].to_numpy()
        free = self.free_response(x0, D, d)
        G = self.Gamma
        # ti_pred[k] is the state at the end of step k, so it is judged against row k+1
        def nxt(col):
            a = fc[col].to_numpy()
            return np.r_[a[1:], a[-1]]

        occ_next = nxt("occupied").astype(bool)
        lo = nxt("lo") + np.where(occ_next, self.margin_k, 0.0)
        hi = nxt("hi") - np.where(occ_next, self.margin_k, 0.0)

        t_out = fc["t_out"].to_numpy()
        energy_price = fc["price"].to_numpy() + w.carbon_price * fc["carbon"].to_numpy() \
            + w.dr_price * fc["dr_event"].to_numpy()
        c_qh = dt * energy_price / self.hp.cop_heat(t_out)
        c_qc = dt * energy_price / self.hp.cop_cool(t_out)
        wc = dt * np.where(occ_next, w.comfort_occupied, w.comfort_unoccupied)
        c = np.r_[c_qh, c_qc, wc, wc, np.full(N, w.move)]

        I = np.eye(N)
        Z = np.zeros((N, N))
        # Comfort (soft, via slacks):
        #   free + G(qh-qc) + slo >= lo   ->  -G qh + G qc - slo <= free - lo
        #   free + G(qh-qc) - shi <= hi   ->   G qh - G qc - shi <= hi - free
        # Moves: du_k >= |q_k - q_{k-1}| with du_k <= max_ramp (ramp limit, move penalty)
        Dm = I - np.eye(N, k=-1)
        r0 = np.zeros(N)
        r0[0] = q_prev
        A_ub = np.block([[-G, G, -I, Z, Z], [G, -G, Z, -I, Z], [Dm, -Dm, Z, Z, -I], [-Dm, Dm, Z, Z, -I]])
        b_ub = np.r_[free - lo, hi - free, r0, -r0]
        bounds = ([(0, self.hp.cap_heat_kw)] * N + [(0, self.hp.cap_cool_kw)] * N + [(0, None)] * (2 * N)
                  + [(0, self.max_ramp_kw)] * N)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        self.last_status = res.message
        if not res.success:
            raise RuntimeError(f"MPC infeasible: {res.message}")
        qh, qc = res.x[:N], res.x[N:2 * N]
        q = qh - qc
        return {"q": q, "ti_pred": free + G @ q, "cost": float(res.fun)}


def flexibility_hours(model: RC2, dt: float, x0: np.ndarray, fc, max_steps: int = 32) -> float:
    """How long (h) the HVAC can be switched off from state x0 before the zone
    leaves its comfort band - the quantity an aggregator needs to bid flexibility."""
    A, B = model.discrete(dt)
    x = x0.copy()
    for k in range(min(max_steps, len(fc))):
        row = fc.iloc[k]
        x = A @ x + B @ np.array([row.t_out, 0.0, row.q_int, row.solar])
        if not (row.lo - 1e-6 <= x[0] <= row.hi + 1e-6):
            return k * dt
    return min(max_steps, len(fc)) * dt


# Safety guard -------------------------------------------------------------------

@dataclass
class SafetyGuard:
    """Sits between the optimiser and the live plant.

    * rejects implausible or stale sensor data and hands control to the BMS fallback,
    * clamps commands to plant limits and rate-limits changes (compressor protection),
    * falls back if the optimiser fails or its plan breaches hard limits,
    * records every intervention for audit.
    """

    hp: HeatPump
    t_min: float = 8.0
    t_max: float = 38.0
    max_jump_k: float = 2.0            # per step
    max_ramp_kw: float = 5.0           # per step
    stale_steps: int = 2
    log: list = field(default_factory=list)
    _last_good: float | None = None
    _missing: int = 0
    _last_cmd: float = 0.0

    def check_sensor(self, ts, y: float | None) -> float | None:
        if y is None or not np.isfinite(y):
            self._missing += 1
            if self._missing >= self.stale_steps:
                self.log.append((ts, "stale_sensor", "fallback to BMS schedule"))
            return None
        if not (self.t_min <= y <= self.t_max) or (
                self._last_good is not None and abs(y - self._last_good) > self.max_jump_k):
            self._missing += 1
            self.log.append((ts, "implausible_reading", f"{y:.1f} C rejected"))
            return None
        self._missing = 0
        self._last_good = y
        return y

    @property
    def sensor_ok(self) -> bool:
        return self._missing < self.stale_steps

    def command(self, ts, q_opt: float | None, q_fallback: float) -> tuple[float, str]:
        if q_opt is None or not np.isfinite(q_opt):
            mode, q = "fallback", q_fallback
            self.log.append((ts, "optimiser_unavailable", "fallback to BMS schedule"))
        elif not self.sensor_ok:
            mode, q = "fallback", q_fallback
        else:
            mode, q = "mpc", q_opt
        q_clamped = float(np.clip(q, -self.hp.cap_cool_kw, self.hp.cap_heat_kw))
        q_ramped = float(np.clip(q_clamped, self._last_cmd - self.max_ramp_kw,
                                 self._last_cmd + self.max_ramp_kw))
        if abs(q_ramped - q) > 1e-6:
            self.log.append((ts, "rate_or_limit_clamp", f"{q:.2f} -> {q_ramped:.2f} kW"))
        self._last_cmd = q_ramped
        return q_ramped, mode
