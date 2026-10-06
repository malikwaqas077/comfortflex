"""Grey-box (resistance-capacitance) thermal models of a building zone.

Units throughout: temperature C, power kW, capacitance kWh/K, resistance K/kW,
time step in hours. Inputs u = [T_out, Q_hvac, Q_internal, solar].
Q_hvac > 0 heats, < 0 cools (thermal power delivered to the zone air).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.linalg import expm


def discretise(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via the augmented matrix exponential."""
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


@dataclass
class RC2:
    """Two-node model: zone air (Ci) coupled to lumped envelope/structure mass (Cm).

    Ci dTi/dt = (Tm-Ti)/Rim + (To-Ti)/Rio + Q_hvac + Q_int + ai*S
    Cm dTm/dt = (Ti-Tm)/Rim + (To-Tm)/Rmo + am*S
    """

    Ci: float = 0.8
    Cm: float = 12.0
    Rim: float = 1.0
    Rio: float = 6.0
    Rmo: float = 3.0
    ai: float = 0.6      # solar aperture to air, kW per (kW/m2)
    am: float = 1.2      # solar aperture to mass

    def continuous(self) -> tuple[np.ndarray, np.ndarray]:
        Ci, Cm, Rim, Rio, Rmo = self.Ci, self.Cm, self.Rim, self.Rio, self.Rmo
        Ac = np.array([
            [-(1 / Rim + 1 / Rio) / Ci, 1 / (Rim * Ci)],
            [1 / (Rim * Cm), -(1 / Rim + 1 / Rmo) / Cm],
        ])
        Bc = np.array([
            [1 / (Rio * Ci), 1 / Ci, 1 / Ci, self.ai / Ci],
            [1 / (Rmo * Cm), 0.0, 0.0, self.am / Cm],
        ])
        return Ac, Bc

    def discrete(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return discretise(*self.continuous(), dt)

    def steady_state(self, u: np.ndarray) -> np.ndarray:
        Ac, Bc = self.continuous()
        return np.linalg.solve(Ac, -Bc @ u)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)])

    @classmethod
    def from_vector(cls, v) -> "RC2":
        return cls(*[float(x) for x in v])

    @property
    def ua(self) -> float:
        """Overall steady-state heat-loss coefficient, kW/K."""
        return 1 / self.Rio + 1 / (self.Rim + self.Rmo)

    @property
    def time_constant_h(self) -> float:
        """Slowest time constant of the zone, hours."""
        Ac, _ = self.continuous()
        return float(-1 / np.max(np.real(np.linalg.eigvals(Ac))))


@dataclass
class RC3:
    """Three-node "true" building used as the simulated plant.

    Adds a fast internal-mass node (furniture, partitions, slab surface) so the
    controller's 2-node model is deliberately structurally mismatched, as it
    would be with a real building.
    """

    Ci: float = 0.6
    Cf: float = 2.5      # internal fast mass
    Cm: float = 14.0     # envelope / slab
    Rif: float = 0.35
    Rim: float = 1.2
    Rio: float = 5.5
    Rmo: float = 2.8
    ai: float = 0.5
    af: float = 0.5
    am: float = 1.0

    def continuous(self) -> tuple[np.ndarray, np.ndarray]:
        Ci, Cf, Cm = self.Ci, self.Cf, self.Cm
        Rif, Rim, Rio, Rmo = self.Rif, self.Rim, self.Rio, self.Rmo
        Ac = np.array([
            [-(1 / Rif + 1 / Rim + 1 / Rio) / Ci, 1 / (Rif * Ci), 1 / (Rim * Ci)],
            [1 / (Rif * Cf), -1 / (Rif * Cf), 0.0],
            [1 / (Rim * Cm), 0.0, -(1 / Rim + 1 / Rmo) / Cm],
        ])
        Bc = np.array([
            [1 / (Rio * Ci), 1 / Ci, 0.6 / Ci, self.ai / Ci],
            [0.0, 0.0, 0.4 / Cf, self.af / Cf],
            [1 / (Rmo * Cm), 0.0, 0.0, self.am / Cm],
        ])
        return Ac, Bc

    def discrete(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return discretise(*self.continuous(), dt)

    def steady_state(self, u: np.ndarray) -> np.ndarray:
        Ac, Bc = self.continuous()
        return np.linalg.solve(Ac, -Bc @ u)


def simulate(A: np.ndarray, B: np.ndarray, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Open-loop simulation. U has shape (N, 4); returns states of shape (N+1, n)."""
    X = np.empty((len(U) + 1, len(x0)))
    X[0] = x0
    for k, u in enumerate(U):
        X[k + 1] = A @ X[k] + B @ u
    return X
