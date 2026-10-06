"""Synthetic but realistic operating scenarios for a UK commercial office zone.

Weather, occupancy, internal gains, a time-of-use tariff, a grid carbon-intensity
profile and demand-flexibility events, all at a fixed time step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    season: str                 # "summer" | "winter"
    start: str = "2026-07-13"   # a Monday
    days: int = 7
    dt_h: float = 0.25          # 15-minute steps (typical LoRaWAN sensor cadence)
    t_mean: float = 22.0        # mean outdoor temperature, C
    t_swing: float = 6.0        # half daily range, C
    solar_peak: float = 0.75    # kW/m2 global horizontal at noon
    dr_window: tuple[int, int] = (16, 19)   # demand-flexibility event, local hours
    dr_days: tuple[int, ...] = (2, 3)       # event days (0 = Monday)
    seed: int = 7


SUMMER = Scenario("Summer heatwave week", "summer", "2026-07-13", t_mean=24.0, t_swing=7.0,
                  solar_peak=0.8, dr_window=(15, 18))
WINTER = Scenario("Winter cold-snap week", "winter", "2026-01-12", t_mean=2.0, t_swing=4.0,
                  solar_peak=0.25, dr_window=(16, 19))
SCENARIOS = {"summer": SUMMER, "winter": WINTER}


def build(sc: Scenario) -> pd.DataFrame:
    """Return one row per time step with every exogenous signal the controllers need."""
    rng = np.random.default_rng(sc.seed)
    n = int(sc.days * 24 / sc.dt_h)
    idx = pd.date_range(sc.start, periods=n, freq=f"{int(sc.dt_h * 60)}min")
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek.to_numpy()
    day = ((idx - idx[0]) / pd.Timedelta("1D")).to_numpy().astype(int)

    # Outdoor temperature: daily sinusoid (min ~05:00, max ~15:00) + synoptic drift + noise
    drift = np.interp(np.arange(n), np.linspace(0, n, sc.days + 1),
                      rng.normal(0, 1.5, sc.days + 1))
    t_out = (sc.t_mean + drift
             + sc.t_swing * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
             + rng.normal(0, 0.3, n))

    # Global horizontal irradiance with day-to-day cloudiness
    cloud = np.clip(rng.normal(0.8, 0.15, sc.days), 0.3, 1.0)[day]
    solar = np.clip(np.sin(np.pi * (hour - 6.0) / 13.0), 0, None) ** 1.5 * sc.solar_peak * cloud

    # Occupancy (fraction of design headcount) - weekdays 08:00-18:00 with lunch dip
    weekday = dow < 5
    occ = np.where(weekday & (hour >= 8) & (hour < 18), 0.85, 0.0)
    occ = np.where(weekday & (hour >= 12) & (hour < 13.5), 0.55, occ)
    occ = np.where(weekday & (((hour >= 7.5) & (hour < 8)) | ((hour >= 18) & (hour < 19))), 0.25, occ)
    occupied = occ > 0.5
    q_int = 0.4 + 4.0 * occ   # people + plug + lighting, kW

    # Time-of-use tariff (GBP/kWh), Agile-style with a 16:00-19:00 peak
    price = np.full(n, 0.24)
    price[(hour >= 0) & (hour < 7)] = 0.13
    price[(hour >= 16) & (hour < 19)] = 0.38
    price = price + rng.normal(0, 0.01, n)

    # Grid carbon intensity (kgCO2/kWh): evening peak, midday solar dip, windy nights
    carbon = (0.17 + 0.07 * np.exp(-((hour - 18.0) ** 2) / 4.0)
              - 0.04 * np.exp(-((hour - 13.0) ** 2) / 6.0)
              - 0.03 * np.exp(-((hour - 3.0) ** 2) / 8.0)
              + rng.normal(0, 0.005, n))

    dr = np.isin(dow, sc.dr_days) & (hour >= sc.dr_window[0]) & (hour < sc.dr_window[1])

    return pd.DataFrame({
        "t_out": t_out, "solar": solar, "occ": occ, "occupied": occupied, "q_int": q_int,
        "price": price, "carbon": carbon, "dr_event": dr,
    }, index=idx)


def inputs(df: pd.DataFrame, q_hvac: np.ndarray | float = 0.0) -> np.ndarray:
    """Stack model inputs U = [T_out, Q_hvac, Q_int, solar]."""
    q = np.broadcast_to(np.asarray(q_hvac, dtype=float), (len(df),))
    return np.column_stack([df["t_out"].to_numpy(), q, df["q_int"].to_numpy(), df["solar"].to_numpy()])
