"""Human thermal comfort models.

* PMV / PPD after ISO 7730:2005 (Fanger), the steady-state model used for
  mechanically conditioned offices.
* Adaptive comfort after EN 16798-1:2019, for free-running / mixed-mode spaces.
* Inversion helpers that turn a comfort criterion (|PMV| <= limit) into an
  operative-temperature band a controller can track.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class ComfortInputs:
    """Personal and environmental parameters other than temperature."""

    rh: float = 50.0      # relative humidity, %
    vel: float = 0.1      # relative air speed, m/s
    met: float = 1.2      # metabolic rate, met (1.2 = sedentary office work)
    clo: float = 0.7      # clothing insulation, clo


def pmv_ppd(ta: float, tr: float, vel: float, rh: float, met: float, clo: float,
            wme: float = 0.0) -> tuple[float, float]:
    """Predicted Mean Vote and Predicted Percentage Dissatisfied (ISO 7730 Annex D).

    ta: air temperature [C], tr: mean radiant temperature [C], vel: air speed [m/s],
    rh: relative humidity [%], met: metabolic rate [met], clo: clothing [clo],
    wme: external work [met].
    """
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))  # vapour pressure, Pa
    icl = 0.155 * clo                    # clothing insulation, m2K/W
    m = met * 58.15                      # metabolic rate, W/m2
    w = wme * 58.15
    mw = m - w
    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(vel)          # forced convection coefficient
    taa = ta + 273.0
    tra = tr + 273.0

    # Iterative solution for clothing surface temperature
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = tcla / 50.0
    hc = hcf
    for _ in range(200):
        if abs(xn - xf) <= 0.00015:
            break
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
    else:  # pragma: no cover - only for pathological inputs
        raise ArithmeticError("PMV clothing temperature did not converge")
    tcl = 100.0 * xn - 273.0

    # Heat-loss components
    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)          # skin diffusion
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0   # sweating
    hl3 = 1.7e-5 * m * (5867.0 - pa)                   # latent respiration
    hl4 = 0.0014 * m * (34.0 - ta)                     # dry respiration
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)  # radiation
    hl6 = fcl * hc * (tcl - ta)                        # convection

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return pmv, ppd


def pmv_at(t_op: float, p: ComfortInputs) -> float:
    """PMV when air and mean radiant temperature both equal t_op."""
    return pmv_ppd(t_op, t_op, p.vel, p.rh, p.met, p.clo)[0]


def temperature_for_pmv(target: float, p: ComfortInputs) -> float:
    """Operative temperature at which PMV equals `target` (solved with Brent's method)."""
    return brentq(lambda t: pmv_at(t, p) - target, 5.0, 40.0, xtol=1e-4)


def pmv_band(p: ComfortInputs, limit: float = 0.5) -> tuple[float, float]:
    """Operative-temperature band satisfying |PMV| <= limit.

    limit 0.2 / 0.5 / 0.7 correspond to ISO 7730 categories A / B / C.
    """
    return temperature_for_pmv(-limit, p), temperature_for_pmv(limit, p)


# EN 16798-1 adaptive model -------------------------------------------------

ADAPTIVE_LIMITS = {"I": (3.0, 2.0), "II": (4.0, 3.0), "III": (5.0, 4.0)}  # (below, above)


def running_mean_outdoor(daily_means: list[float], alpha: float = 0.8) -> float:
    """Exponentially weighted running mean of daily outdoor temperature (EN 16798-1).

    `daily_means` is ordered oldest -> most recent (yesterday last).
    """
    trm = daily_means[0]
    for t in daily_means[1:]:
        trm = (1.0 - alpha) * t + alpha * trm
    return trm


def adaptive_band(t_rm: float, category: str = "II") -> tuple[float, float]:
    """Adaptive comfort band (operative temperature) for a running-mean outdoor temp."""
    t_comf = 0.33 * t_rm + 18.8
    below, above = ADAPTIVE_LIMITS[category]
    return t_comf - below, t_comf + above
