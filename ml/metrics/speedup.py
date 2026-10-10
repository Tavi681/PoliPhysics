"""Speed-up and break-even cost metrics (duration-normalized).

Simulator wall times come from Stage B on c2-standard-30 (when wall_time_s is
present). Surrogate times include model forward + KinematicSimulation u-eval +
integration overhead on the surrogate hardware (MPS / GPU VM).
"""
from __future__ import annotations

import math
from typing import Sequence


def sim_seconds(exit_time: float | None, t_end: float = 60.0) -> float:
    """Simulated duration for a run (exit_time if finite and >0, else t_end)."""
    if exit_time is None:
        return float(t_end)
    et = float(exit_time)
    if not math.isfinite(et) or et <= 0.0:
        return float(t_end)
    return et


def wall_per_sim_second(wall_time_s: float, simulated_s: float) -> float:
    if not math.isfinite(wall_time_s) or wall_time_s < 0:
        return float("nan")
    if not math.isfinite(simulated_s) or simulated_s <= 0:
        return float("nan")
    return float(wall_time_s) / float(simulated_s)


def speedup_per_run(
    sim_wall_s: float,
    sim_duration_s: float,
    surr_wall_s: float,
    surr_duration_s: float,
) -> float:
    """(sim wall / sim second) / (surr wall / surr second)."""
    r_sim = wall_per_sim_second(sim_wall_s, sim_duration_s)
    r_surr = wall_per_sim_second(surr_wall_s, surr_duration_s)
    if not math.isfinite(r_sim) or not math.isfinite(r_surr) or r_surr <= 0:
        return float("nan")
    return r_sim / r_surr


def mean_speedup(
    sim_walls: Sequence[float],
    sim_durs: Sequence[float],
    surr_walls: Sequence[float],
    surr_durs: Sequence[float],
) -> float:
    """Mean per-run speed-up over pairs with finite simulator wall times."""
    vals = []
    for sw, sd, uw, ud in zip(sim_walls, sim_durs, surr_walls, surr_durs):
        if not math.isfinite(float(sw)):
            continue
        s = speedup_per_run(float(sw), float(sd), float(uw), float(ud))
        if math.isfinite(s):
            vals.append(s)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def q_star(
    C_data: float,
    C_train: float,
    t_sim_per_run: float,
    t_surr_per_run: float,
) -> float:
    """Break-even query count on full-length runs.

    Q* = (C_data + C_train) / (t_sim_per_run - t_surr_per_run)
    """
    denom = float(t_sim_per_run) - float(t_surr_per_run)
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    num = float(C_data) + float(C_train)
    if not math.isfinite(num):
        return float("nan")
    return num / denom


def scale_surr_to_full_length(
    surr_wall_s: float,
    surr_sim_s: float,
    full_sim_s: float,
) -> float:
    """Extrapolate surrogate wall time to a full-length simulated duration."""
    rate = wall_per_sim_second(surr_wall_s, surr_sim_s)
    if not math.isfinite(rate):
        return float("nan")
    return rate * float(full_sim_s)
