"""Phase 3: moving obstacles, noisy sensing, Kalman tracking and the safety brake (all headless)."""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from smart_nav.environment import CityGrid
from smart_nav.perception import (
    BRAKE,
    CRUISE,
    KalmanFilter2D,
    MultiObjectTracker,
    NoisySensor,
    ObstacleField,
    PerceptionSystem,
    Track,
    TrackingStats,
    assess_threats,
    box_overlap,
)
from smart_nav.simulation import Simulation, SimulationConfig
from smart_nav.vehicle import Vehicle

DT = 1 / 60


# ---------------------------------------------------------------- vehicle kinematics


def test_vehicle_direction_matches_motion():
    v = Vehicle([(2, c) for c in range(10)], speed=4.0)
    v.update(0.1)
    assert np.allclose(v.direction, [0.0, 1.0], atol=1e-6)
    down = Vehicle([(r, 3) for r in range(10)], speed=4.0)
    down.update(0.1)
    assert np.allclose(down.direction, [1.0, 0.0], atol=1e-6)


def test_vehicle_trajectory_follows_route_around_corner():
    v = Vehicle([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)], speed=4.0)
    pos, dirs = v.trajectory([0.0, 1.5, 3.0, 10.0])
    assert np.allclose(pos, [[0, 0], [0, 1.5], [1, 2], [2, 2]])
    assert np.allclose(dirs[1], [0, 1]) and np.allclose(dirs[2], [1, 0])


def test_vehicle_respects_braking_and_acceleration_limits():
    v = Vehicle([(0, c) for c in range(100)], speed=6.0, max_accel=4.0, max_decel=12.0)
    v.set_target_speed(0.0)
    v.update(0.25)
    assert v.speed == pytest.approx(3.0)
    assert v.braking
    v.update(0.5)
    assert v.speed == 0.0
    v.set_target_speed(10.0)  # clamped to cruise speed
    assert v.target_speed == 6.0
    v.update(0.5)
    assert v.speed == pytest.approx(2.0)


# ---------------------------------------------------------------- obstacles & sensor


def test_obstacles_stay_on_road_sidewalks_and_cross():
    city = CityGrid(seed=3)
    field = ObstacleField(city, count=20, seed=3)
    crossed = set()
    for _ in range(int(60 / DT)):
        field.update(DT)
        for ob in field:
            r, c = ob.position
            assert -1.0 <= r <= city.rows and -1.0 <= c <= city.cols
            assert ob.line % city.block_size == 0  # always attached to a road line
            assert abs(ob.offset) <= 0.7 + 1e-9
            if ob.state == "cross":
                crossed.add(ob.id)
    assert len(crossed) >= 3


def test_sensor_adds_xy_noise_respects_range_and_rate():
    city = CityGrid(seed=1)
    field = ObstacleField(city, count=0, seed=1)
    near = field.make("h", 4, 5.0, 1, speed=0.0)
    far = field.make("h", 4, 30.0, 1, speed=0.0)
    sensor = NoisySensor(range_=8.0, noise_std=0.35, rate_hz=20.0, dropout=0.0, seed=1)
    errors, t, scans = [], 0.0, 0
    for _ in range(int(30 / DT)):
        t += DT
        dets = sensor.scan(t, (4.0, 5.0), field)
        scans += bool(dets)
        assert far.id not in {d.obstacle_id for d in dets}
        errors.extend(d.z - d.truth for d in dets if d.obstacle_id == near.id)
    errors = np.array(errors)
    assert scans == pytest.approx(20 * 30, rel=0.05)
    assert errors.std(axis=0) == pytest.approx([0.35, 0.35], rel=0.1)  # noise in both X and Y
    assert abs(errors.mean()) < 0.05


# ---------------------------------------------------------------- Kalman filter


def _track_straight_walker(seed: int, noise: float = 0.35, seconds: float = 12.0):
    rng = np.random.default_rng(seed)
    vel = np.array([0.6, 1.1])
    kf = None
    raw, est, vel_err = [], [], []
    t, next_scan = 0.0, 0.0
    while t < seconds:
        t += DT
        truth = np.array([2.0, 1.0]) + vel * t
        if kf is not None:
            kf.predict(DT)
        if t >= next_scan:
            next_scan += 0.05
            z = truth + rng.normal(0, noise, 2)
            if kf is None:
                kf = KalmanFilter2D(z, noise)
                continue
            kf.update(z)
            if t > 1.5:  # after convergence
                raw.append(np.hypot(*(z - truth)))
                est.append(np.hypot(*(kf.position - truth)))
                vel_err.append(np.hypot(*(kf.velocity - vel)))
    rmse = lambda e: math.sqrt(np.mean(np.square(e)))  # noqa: E731
    return rmse(raw), rmse(est), rmse(vel_err), kf


@pytest.mark.parametrize("seed", range(5))
def test_kalman_filter_greatly_reduces_error_on_constant_velocity_target(seed):
    raw, est, vel_err, kf = _track_straight_walker(seed)
    assert est < 0.5 * raw, (raw, est)
    assert vel_err < 0.3
    assert np.allclose(kf.velocity, [0.6, 1.1], atol=0.35)


def test_kalman_filter_beats_raw_sensor_on_random_pedestrians_and_cyclists():
    """Realistic pedestrians/cyclists that turn, stop and reverse - error measured on every track."""
    for seed in range(3):
        city = CityGrid(seed=seed)
        field = ObstacleField(city, count=12, seed=seed)
        sensor = NoisySensor(range_=1e9, noise_std=0.35, rate_hz=20.0, dropout=0.05, seed=seed)
        tracker = MultiObjectTracker(meas_std=0.35)
        stats = TrackingStats()
        t = 0.0
        for _ in range(int(40 / DT)):
            t += DT
            field.update(DT)
            dets = sensor.scan(t, (0.0, 0.0), field)
            tracker.step(t, DT, dets)
            truth = {ob.id: ob for ob in field}
            for d in dets:
                tr = tracker.tracks[d.obstacle_id]
                if tr.hits >= tracker.min_hits:
                    stats.record(
                        np.hypot(*(d.z - d.truth)),
                        np.hypot(*(tr.position - d.truth)),
                        np.hypot(*(tr.velocity - truth[d.obstacle_id].velocity)),
                    )
        assert stats.kf_rmse < 0.6 * stats.raw_rmse, (seed, stats.raw_rmse, stats.kf_rmse)


@pytest.mark.parametrize("seed", range(5))
def test_kalman_velocity_reconverges_after_pedestrian_turns(seed):
    rng = np.random.default_rng(seed)
    kf = KalmanFilter2D([0.0, 0.0], 0.35)
    pos = np.zeros(2)
    late = []
    for i in range(1, 161):  # 3 s heading +col, then 5 s heading +row (20 Hz)
        vel = np.array([0.0, 1.2]) if i <= 60 else np.array([1.2, 0.0])
        pos = pos + vel * 0.05
        kf.predict(0.05)
        kf.update(pos + rng.normal(0, 0.35, 2))
        if i > 60 + 40:  # 2 s after the turn
            late.append(np.hypot(*(kf.velocity - vel)))
    assert max(late) < 0.5
    assert math.sqrt(np.mean(np.square(late))) < 0.25


def test_tracker_drops_stale_tracks_and_confirms_after_min_hits():
    tracker = MultiObjectTracker(0.3, max_age=0.5, min_hits=2)

    class Det:
        def __init__(self, z):
            self.obstacle_id, self.kind, self.z = 7, "pedestrian", np.array(z, float)

    tracker.step(0.0, DT, [Det([1, 1])])
    assert tracker.confirmed() == []
    tracker.step(0.05, 0.05, [Det([1, 1.05])])
    assert [t.id for t in tracker.confirmed()] == [7]
    tracker.step(1.0, 0.95, [])
    assert tracker.tracks == {}


# ---------------------------------------------------------------- threat assessment


def _track(pos, vel, tid=1):
    kf = KalmanFilter2D(pos, 0.05)
    kf.x[2:] = vel
    kf.P[:] = np.eye(4) * 1e-4
    return Track(tid, "pedestrian", kf, 0.0)


def test_box_overlap_geometry():
    assert box_overlap((0, 0), (0, 1), (0.2, 0.5), 0.6, 0.3)
    assert not box_overlap((0, 0), (0, 1), (0.5, 0.2), 0.6, 0.3)
    assert box_overlap((0, 0), (1, 0), (0.5, 0.2), 0.6, 0.3)


def test_crossing_pedestrian_ahead_triggers_brake():
    v = Vehicle([(2, c) for c in range(30)], speed=6.0)
    crossing = assess_threats(v, [_track((3.0, 4.0), (-1.2, 0.0))], horizon=1.5)
    assert crossing.state != CRUISE and crossing.allowed_speed <= 0.5 * 6.0
    assert crossing.track_id == 1
    imminent = assess_threats(v, [_track((2.0, 0.9), (-1.0, 0.0))], horizon=1.5)
    assert imminent.state == BRAKE and imminent.allowed_speed == 0.0


@pytest.mark.parametrize(
    "pos,vel",
    [
        ((2.7, 3.0), (0.0, 1.0)),  # walking along the sidewalk, parallel to the car
        ((2.0, -3.0), (0.0, -1.0)),  # behind the car, moving away
        ((4.5, 4.0), (1.0, 0.0)),  # crossing elsewhere, moving away from the road
        ((3.0, 6.0), (-1.2, 0.0)),  # already past the car's path by the time it arrives
    ],
)
def test_harmless_obstacles_do_not_brake(pos, vel):
    v = Vehicle([(2, c) for c in range(30)], speed=6.0)
    if pos == (3.0, 6.0):
        pos = (1.0, 2.5)  # crossed the lane before the car gets there
    assert assess_threats(v, [_track(pos, vel)], horizon=1.5).state == CRUISE


# ---------------------------------------------------------------- closed-loop safety brake


def _find_crossing_spot(path, city, ahead=5):
    """A path index on a straight interior road stretch, at least ``ahead`` cells from the start."""
    for k in range(ahead, len(path) - 2):
        rows = {path[i][0] for i in range(k - ahead, k + 2)}
        cols = {path[i][1] for i in range(k - ahead, k + 2)}
        r, c = path[k]
        if len(rows) == 1 and c % city.block_size:
            return k, "h", r, c, (-1 if r == city.rows - 1 else 1)
        if len(cols) == 1 and r % city.block_size:
            return k, "v", c, r, (-1 if c == city.cols - 1 else 1)
    raise AssertionError("no straight stretch on route")


def _scripted_crossing(safety_brake: bool, seed: int = 11, trigger: float = 4.0):
    cfg = SimulationConfig(
        seed=seed, v2x=False, closure_rate=0.0, num_obstacles=0, safety_brake=safety_brake,
        sensor_dropout=0.0, random_endpoints=True,
    )
    sim = Simulation(cfg, headless=True)
    path = list(sim.vehicle.path)
    k, axis, line, along, side = _find_crossing_spot(path, sim.city)
    spot = np.array(path[k], dtype=float)
    ped = None
    for _ in range(int(60 / DT)):
        if ped is None and np.hypot(*(sim.vehicle.position - spot)) <= trigger and sim.vehicle.segment < k:
            # A distracted pedestrian stepping out without looking.
            ped = sim.perception.obstacles.make(axis, line, float(along), side, speed=1.2, yields=False)
            ped.can_cross = True
            ped.cross_rate = 0.0
            ped.start_crossing()
        sim.step(DT)
        if sim.vehicle.finished and ped is not None and ped.state == "walk":
            break
    result = sim.result()
    logs = [e.text for e in sim.log_entries]
    sim.close()
    return result, ped, logs


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_scripted_crossing_collides_without_safety_brake(seed):
    result, ped, logs = _scripted_crossing(safety_brake=False, seed=seed)
    assert ped is not None
    assert result.collisions >= 1
    assert any("COLLISION" in t for t in logs)


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_safety_brake_stops_for_crossing_pedestrian_and_resumes(seed):
    result, ped, logs = _scripted_crossing(safety_brake=True, seed=seed)
    assert ped is not None
    assert result.collisions == 0
    assert result.brake_events >= 1
    assert result.min_speed < 0.25 * 6.0  # it really braked (cruise speed is 6)
    assert result.reached_goal  # and drove on once the pedestrian cleared
    assert any("SAFETY BRAKE" in t for t in logs)
    assert any("released" in t for t in logs)


@pytest.mark.parametrize("seed", range(6))
def test_random_traffic_full_runs_are_collision_free(seed):
    cfg = SimulationConfig(seed=seed, random_endpoints=True, num_obstacles=16, incident_rate=0.3)
    sim = Simulation(cfg, headless=True)
    result = sim.run(max_frames=int(120 / DT), exit_on_arrival=True, render=False)
    sim.close()
    assert result.reached_goal
    assert result.collisions == 0
    assert result.safety_violations == 0
    assert result.kf_rmse < result.sensor_rmse


def test_perception_overlays_render_headless(tmp_path):
    cfg = SimulationConfig(seed=7, num_obstacles=12)
    sim = Simulation(cfg, headless=True)
    out = tmp_path / "p3.png"
    result = sim.run(max_frames=90, screenshot_path=str(out))
    sim.close()
    assert out.exists() and out.stat().st_size > 1000
    assert sim.perception.tracker.confirmed()
    assert result.frames == 90


def test_perception_can_be_disabled():
    sim = Simulation(SimulationConfig(seed=7, perception=False), headless=True)
    result = sim.run(max_frames=int(60 / DT), exit_on_arrival=True, render=False)
    sim.close()
    assert sim.perception is None and result.reached_goal and result.brake_events == 0


def test_perception_system_reports_tracking_stats():
    city = CityGrid(seed=2)
    ps = PerceptionSystem(city, num_obstacles=10, sensor_range=50, seed=2)
    v = Vehicle([city.start], speed=1.0)
    for i in range(600):
        ps.step(i * DT, DT, v)
    assert ps.stats.raw_rmse > 0.25 and ps.stats.kf_rmse < 0.6 * ps.stats.raw_rmse
