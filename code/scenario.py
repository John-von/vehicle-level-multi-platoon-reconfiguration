"""Scenario generator and parameters.  The link qualities are synthetic and are not measured data."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    vehicles_per_source: int = 8
    dt: float = .4
    horizon: int = 20
    sweeps: int = 2
    cruise: float = 16.
    lane_width: float = 3.6
    stage_gap: float = 14.
    final_gap: float = 14.
    stage_time: float = 65.
    crossing_time: float = 16.
    compact_time: float = 45.
    ax_max: float = 2.
    ay_max: float = .65
    jx_max: float = 1.2
    jy_max: float = .6
    v_min: float = 7.
    v_max: float = 25.
    vy_max: float = .9
    heading_max: float = .12
    length: float = 4.8
    width: float = 1.9
    safety_dx: float = 8.
    safety_dy: float = 2.8
    motion_scale: float = .5
    speed_scale: float = 2.
    link_scale: float = .2231435513142097  # -log(.8)
    weight_speed: float = .15
    weight_link: float = .40
    tube_half_width: float = .15
    max_phase_wait: float = 12.
    input_rate_weight_x: float = .9
    input_rate_weight_y: float = 2.
    coupling_position_weight: float = .4
    coupling_speed_weight: float = .15
    qp_max_iter: int = 4000
    qp_objective_scale: float = 1.
    qp_solver: str = 'osqp'


def make_scenario(seed=0, cfg=Config()):
    rng = np.random.default_rng(seed)
    n = 3 * cfg.vehicles_per_source
    m = cfg.vehicles_per_source
    source = np.repeat(np.arange(3), m)
    labels = [f'{chr(65 + s)}{k + 1}' for s in range(3) for k in range(m)]
    initial = np.zeros((n, 4))  # relative x, y, relative vx, vy
    initial[:, 0] = np.tile(-20. * np.arange(m), 3) + np.repeat([5., 0., -5.], m)
    initial[:, 0] += rng.uniform(-.6, .6, n)
    initial[:, 1] = source * cfg.lane_width
    initial[:, 2] = rng.uniform(-1.2, 1.2, n)
    # Individual route tasks, balanced within each source; no fragment membership.
    route = np.concatenate([rng.permutation(np.tile([0, 1], m // 2)) for _ in range(3)])
    radio_feature = rng.uniform(-1, 1, n)
    pair_noise = rng.uniform(0, .08, (n, n))
    pair_noise = (pair_noise + pair_noise.T) / 2
    quality = (.98 - .22 * abs(radio_feature[:, None] - radio_feature[None, :]) / 2 - pair_noise)
    quality *= np.exp(-cfg.final_gap / 220.)
    np.fill_diagonal(quality, 1.)
    center = initial[:, 0].mean()
    stage_x = center + (n - 1) * cfg.stage_gap / 2 - np.arange(n) * cfg.stage_gap
    target_x = center + (n // 2 - 1) * cfg.final_gap / 2 - np.arange(n // 2) * cfg.final_gap
    return dict(seed=seed, labels=labels, initial=initial, source=source, route=route,
                quality=quality, radio_feature=radio_feature, stage_x=stage_x,
                target_x=target_x, config=asdict(cfg))


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
