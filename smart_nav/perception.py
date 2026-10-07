"""Perception: moving obstacles, noisy sensing, Kalman-filter tracking and threat assessment.

Pipeline per frame:
    ObstacleField (ground truth) -> NoisySensor (range-limited, noisy, ~20 Hz)
    -> MultiObjectTracker (one constant-velocity KalmanFilter2D per obstacle)
    -> assess_threats (predicted obstacle motion vs. the vehicle's footprint along its route)
    -> allowed speed for the vehicle (cruise / slow / safety brake).

Coordinates are (row, col) in grid cells, matching the rest of the project.
Sensor detections carry the obstacle id, i.e. data association is assumed solved.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .environment import CityGrid
from .vehicle import VEHICLE_HALF_LENGTH, VEHICLE_HALF_WIDTH

SIDEWALK_OFFSET = 0.7
OBSTACLE_RADIUS = 0.12
KIND_PARAMS = {
    "pedestrian": {"speed": (0.8, 1.3), "cross_rate": 0.12},
    "cyclist": {"speed": (1.8, 2.6), "cross_rate": 0.05},
}


def box_overlap(center, direction, point, half_length: float, half_width: float) -> bool:
    """Does ``point`` fall inside the oriented box at ``center`` facing ``direction`` (unit, row/col)?"""
    d = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    lon = d[0] * direction[0] + d[1] * direction[1]
    lat = -d[0] * direction[1] + d[1] * direction[0]
    return abs(lon) < half_length and abs(lat) < half_width


def vehicle_contact(vehicle, point, margin: float = 0.0) -> bool:
    return box_overlap(
        vehicle.position,
        vehicle.direction,
        point,
        VEHICLE_HALF_LENGTH + OBSTACLE_RADIUS + margin,
        VEHICLE_HALF_WIDTH + OBSTACLE_RADIUS + margin,
    )


class MovingObstacle:
    """A pedestrian/cyclist walking a sidewalk next to a road line, occasionally crossing it.

    ``axis`` is "h" (road along row ``line``) or "v" (road along column ``line``); ``along`` is the
    coordinate along the road and ``offset`` the lateral offset from the road centre line.
    Walking along a sidewalk naturally crosses the perpendicular roads at intersections.
    """

    def __init__(
        self,
        id: int,
        kind: str,
        axis: str,
        line: int,
        along: float,
        side: int,
        direction: int,
        speed: float,
        bounds: Tuple[float, float],
        block_size: int,
        can_cross: bool = True,
        cross_rate: float = 0.1,
        crossing: bool = False,
        yields: bool = True,
    ) -> None:
        self.id = id
        self.yields = yields
        self.kind = kind
        self.axis = axis
        self.line = line
        self.along = float(along)
        self.side = side
        self.offset = side * SIDEWALK_OFFSET
        self.direction = direction
        self.speed = speed
        self.bounds = bounds
        self.block_size = block_size
        self.can_cross = can_cross
        self.cross_rate = cross_rate
        self.state = "walk"
        self.cross_target = self.offset
        self.wait_time = 0.0
        self.velocity = np.zeros(2)
        if crossing:
            self.start_crossing()

    @property
    def position(self) -> np.ndarray:
        return self._pos(self.along, self.offset)

    def _pos(self, along: float, offset: float) -> np.ndarray:
        if self.axis == "h":
            return np.array([self.line + offset, along])
        return np.array([along, self.line + offset])

    def start_crossing(self) -> bool:
        if not self.can_cross or self.state == "cross":
            return False
        self.state = "cross"
        self.cross_target = -self.side * SIDEWALK_OFFSET
        return True

    def _near_intersection(self) -> bool:
        m = self.along % self.block_size
        return m < 0.8 or m > self.block_size - 0.8

    def update(self, dt: float, rng: random.Random, blocked: Optional[Callable[[np.ndarray], bool]] = None) -> None:
        old = self.position
        along, offset = self.along, self.offset
        if self.state == "walk":
            along += self.direction * self.speed * dt
            lo, hi = self.bounds
            if along < lo or along > hi:
                along = min(hi, max(lo, along))
                self.direction *= -1
        else:
            step = self.speed * dt
            delta = self.cross_target - offset
            offset = self.cross_target if abs(delta) <= step else offset + math.copysign(step, delta)

        new = self._pos(along, offset)
        if blocked is not None and blocked(new):
            self.velocity = np.zeros(2)
            self.wait_time += dt
            if self.wait_time > 1.5:
                self.wait_time = 0.0
                if self.state == "cross":
                    self.cross_target = self.side * SIDEWALK_OFFSET
                else:
                    self.direction *= -1
            return

        self.wait_time = 0.0
        self.along, self.offset = along, offset
        if self.state == "cross" and offset == self.cross_target:
            self.side = 1 if offset > 0 else -1
            self.state = "walk"
            if rng.random() < 0.5:
                self.direction *= -1
        elif self.state == "walk" and not self._near_intersection() and rng.random() < self.cross_rate * dt:
            self.start_crossing()
        self.velocity = (self.position - old) / dt if dt > 0 else np.zeros(2)


class ObstacleField:
    def __init__(
        self,
        city: CityGrid,
        count: int = 8,
        seed: Optional[int] = None,
        exclude: Sequence[Sequence[float]] = (),
        exclude_radius: float = 4.0,
        cyclist_ratio: float = 0.3,
    ) -> None:
        self.city = city
        self.rng = random.Random(seed)
        self.exclude = [np.asarray(p, dtype=float) for p in exclude]
        self.exclude_radius = exclude_radius
        self.cyclist_ratio = cyclist_ratio
        self.obstacles: List[MovingObstacle] = []
        self._next_id = 1
        for _ in range(count):
            self.spawn_random()

    def __iter__(self):
        return iter(self.obstacles)

    def __len__(self) -> int:
        return len(self.obstacles)

    def make(
        self, axis: str, line: int, along: float, side: int, direction: int = 1, kind: str = "pedestrian",
        speed: Optional[float] = None, crossing: bool = False, yields: bool = True,
    ) -> MovingObstacle:
        city = self.city
        last_line = (city.rows if axis == "h" else city.cols) - 1
        hi = (city.cols if axis == "h" else city.rows) - 1
        params = KIND_PARAMS[kind]
        ob = MovingObstacle(
            id=self._next_id,
            kind=kind,
            axis=axis,
            line=line,
            along=along,
            side=side,
            direction=direction,
            speed=speed if speed is not None else self.rng.uniform(*params["speed"]),
            bounds=(0.0, float(hi)),
            block_size=city.block_size,
            can_cross=0 < line < last_line,
            cross_rate=params["cross_rate"],
            crossing=crossing,
            yields=yields,
        )
        self._next_id += 1
        self.obstacles.append(ob)
        return ob

    def place_stationary(self, position: Sequence[float], kind: str = "pedestrian") -> MovingObstacle:
        """An obstacle that stands still at ``position`` (e.g. someone stopped on the goal marker)."""
        r, c = float(position[0]), float(position[1])
        ob = self.make("h", int(round(r)), c, side=1, kind=kind, speed=0.0)
        ob.offset = ob.cross_target = r - ob.line
        ob.can_cross = False
        return ob

    def remove(self, obstacle_id: int) -> bool:
        before = len(self.obstacles)
        self.obstacles = [ob for ob in self.obstacles if ob.id != obstacle_id]
        return len(self.obstacles) < before

    def spawn_random(self) -> MovingObstacle:
        city, bs = self.city, self.city.block_size
        for _ in range(100):
            axis = self.rng.choice("hv")
            n_lines = (city.rows if axis == "h" else city.cols) // bs + 1
            line = self.rng.randrange(n_lines) * bs
            last_line = (city.rows if axis == "h" else city.cols) - 1
            side = 1 if line == 0 else -1 if line == last_line else self.rng.choice((-1, 1))
            hi = (city.cols if axis == "h" else city.rows) - 1
            along = self.rng.uniform(0, hi)
            pos = np.array([line + side * SIDEWALK_OFFSET, along]) if axis == "h" else np.array([along, line + side * SIDEWALK_OFFSET])
            if all(np.hypot(*(pos - e)) >= self.exclude_radius for e in self.exclude):
                break
        kind = "cyclist" if self.rng.random() < self.cyclist_ratio else "pedestrian"
        return self.make(axis, line, along, side, self.rng.choice((-1, 1)), kind)

    def update(self, dt: float, vehicle=None) -> None:
        in_path = None
        if vehicle is not None and vehicle.speed >= 0.3 and not vehicle.finished:
            in_path = self._swept_corridor(vehicle)
        for ob in self.obstacles:
            if vehicle is None:
                ob.update(dt, self.rng)
                continue
            # Nobody walks into the side of a car (they can't dodge one driving into them, though).
            # Moves that get them out of contact are always allowed.
            old_gap = float(np.hypot(*(ob.position - vehicle.position)))
            gate = in_path if in_path is not None and ob.yields and not in_path(ob.position) else None

            def blocked(p: np.ndarray, old_gap=old_gap, gate=gate) -> bool:
                if vehicle_contact(vehicle, p, margin=0.05) and np.hypot(*(p - vehicle.position)) < old_gap:
                    return True
                # Gap acceptance: nobody steps right in front of a car that is about to arrive. Once
                # in the road they keep going; avoiding them is then up to the car's perception.
                return gate is not None and gate(p)

            ob.update(dt, self.rng, blocked)

    @staticmethod
    def _swept_corridor(vehicle, gap: float = 0.6, margin: float = 0.15) -> Callable[[np.ndarray], bool]:
        ts = np.linspace(0.0, gap, 7)
        pos, dirs = vehicle.trajectory(vehicle.speed * ts)
        half_len = VEHICLE_HALF_LENGTH + OBSTACLE_RADIUS + margin
        half_wid = VEHICLE_HALF_WIDTH + OBSTACLE_RADIUS + margin

        def inside(p: np.ndarray) -> bool:
            d = np.asarray(p, dtype=float) - pos
            lon = d[:, 0] * dirs[:, 0] + d[:, 1] * dirs[:, 1]
            lat = -d[:, 0] * dirs[:, 1] + d[:, 1] * dirs[:, 0]
            return bool(np.any((np.abs(lon) < half_len) & (np.abs(lat) < half_wid)))

        return inside


@dataclass
class Detection:
    obstacle_id: int
    kind: str
    z: np.ndarray
    time: float
    truth: np.ndarray  # ground truth, for evaluation only


class NoisySensor:
    """Range-limited camera/LiDAR-like sensor with Gaussian X/Y noise and random dropouts."""

    def __init__(
        self, range_: float = 7.0, noise_std: float = 0.35, rate_hz: float = 20.0, dropout: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.range = range_
        self.noise_std = noise_std
        self.period = 1.0 / rate_hz
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self._last_scan = -math.inf

    def scan(self, now: float, origin: Sequence[float], obstacles: Iterable[MovingObstacle]) -> List[Detection]:
        if now - self._last_scan < self.period - 1e-9:
            return []
        self._last_scan = now
        origin = np.asarray(origin, dtype=float)
        out = []
        for ob in obstacles:
            truth = ob.position
            if np.hypot(*(truth - origin)) > self.range or self.rng.random() < self.dropout:
                continue
            out.append(Detection(ob.id, ob.kind, truth + self.rng.normal(0.0, self.noise_std, 2), now, truth.copy()))
        return out


class KalmanFilter2D:
    """Constant-velocity Kalman filter. State [r, c, vr, vc]; measures position only."""

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def __init__(
        self,
        z0: Sequence[float],
        meas_std: float,
        accel_std: float = 1.0,
        init_vel_std: float = 2.0,
        maneuver_gate: Optional[float] = 4.6,
    ) -> None:
        self.x = np.array([z0[0], z0[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([meas_std**2, meas_std**2, init_vel_std**2, init_vel_std**2])
        self.R = np.eye(2) * meas_std**2
        self.q = accel_std**2
        self.maneuver_gate = maneuver_gate
        self.maneuver_vel_var = 1.0
        self.maneuver_streak = 3
        self.maneuvers = 0
        self._streak = 0
        self.nis = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def position_std(self) -> float:
        return float(math.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]))))

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        Q = self.q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: Sequence[float]) -> None:
        z = np.asarray(z, dtype=float)
        H = self.H
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        self.nis = float(y @ np.linalg.solve(S, y))
        self._streak = self._streak + 1 if self.maneuver_gate and self.nis > self.maneuver_gate else 0
        if self._streak >= self.maneuver_streak:
            # Several innovations in a row that constant-velocity motion can't explain: the target turned,
            # stopped or reversed. Re-open the velocity uncertainty so the filter re-converges quickly.
            self.P[2:, 2:] += np.eye(2) * self.maneuver_vel_var
            self.maneuvers += 1
            self._streak = 0
            S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def predict_position(self, t: float) -> np.ndarray:
        return self.x[:2] + self.x[2:] * t


@dataclass
class Track:
    id: int
    kind: str
    kf: KalmanFilter2D
    last_update: float
    hits: int = 1

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity


class MultiObjectTracker:
    def __init__(
        self,
        meas_std: float,
        accel_std: float = 1.0,
        max_age: float = 1.0,
        min_hits: int = 2,
        maneuver_gate: Optional[float] = 4.6,
    ) -> None:
        self.meas_std = meas_std
        self.accel_std = accel_std
        self.maneuver_gate = maneuver_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}

    def step(self, now: float, dt: float, detections: Sequence[Detection]) -> List[Track]:
        for tr in self.tracks.values():
            tr.kf.predict(dt)
        updated = []
        for det in detections:
            tr = self.tracks.get(det.obstacle_id)
            if tr is None:
                tr = Track(det.obstacle_id, det.kind, KalmanFilter2D(det.z, self.meas_std, self.accel_std, maneuver_gate=self.maneuver_gate), now)
                self.tracks[det.obstacle_id] = tr
            else:
                tr.kf.update(det.z)
                tr.hits += 1
                tr.last_update = now
            updated.append(tr)
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_update > self.max_age]:
            del self.tracks[tid]
        return updated

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]


@dataclass
class TrackingStats:
    """Position error of raw detections vs. the filtered estimate at the same instants."""

    raw_sq: float = 0.0
    kf_sq: float = 0.0
    vel_sq: float = 0.0
    n: int = 0

    def record(self, raw_err: float, kf_err: float, vel_err: float) -> None:
        self.raw_sq += raw_err**2
        self.kf_sq += kf_err**2
        self.vel_sq += vel_err**2
        self.n += 1

    @property
    def raw_rmse(self) -> float:
        return math.sqrt(self.raw_sq / self.n) if self.n else 0.0

    @property
    def kf_rmse(self) -> float:
        return math.sqrt(self.kf_sq / self.n) if self.n else 0.0

    @property
    def velocity_rmse(self) -> float:
        return math.sqrt(self.vel_sq / self.n) if self.n else 0.0


CRUISE, SLOW, BRAKE = "CRUISE", "SLOW", "BRAKE"


@dataclass
class ThreatAssessment:
    state: str
    allowed_speed: float
    track_id: Optional[int] = None
    time_to_conflict: Optional[float] = None
    distance_to_conflict: Optional[float] = None


def assess_threats(
    vehicle,
    tracks: Sequence[Track],
    horizon: float = 1.5,
    step: float = 0.05,
    margin: float = 0.08,
    standoff: float = 0.5,
    caution: float = 0.5,
    creep_speed: float = 1.0,
    drift: float = 0.25,
    max_drift: float = 0.2,
) -> ThreatAssessment:
    """Predict each track forward (constant velocity) and test it against the vehicle's footprint
    along its route. The route is sampled under two speed hypotheses (cruise and current speed) so
    the decision holds whether the vehicle keeps braking or resumes. On a predicted conflict the
    allowed speed is capped to ``caution * cruise`` and to the speed from which the vehicle can still
    stop ``standoff`` cells short of the conflict point.
    """
    cruise = vehicle.cruise_speed
    if vehicle.finished or not tracks:
        return ThreatAssessment(CRUISE, cruise)

    ts = np.arange(0.0, horizon + 1e-9, step)
    best: Optional[Tuple[float, float, int]] = None

    # Occupancy check: anything already standing or walking in the lane ahead must be stopped for,
    # whatever its estimated velocity (velocity estimates lag right after a pedestrian turns).
    path_dist = np.arange(0.0, cruise * horizon + 1e-9, step * creep_speed)
    path_pos, path_dirs = vehicle.trajectory(path_dist)
    occupied: Optional[Tuple[float, int]] = None
    for tr in tracks:
        grow = margin + min(0.1, tr.kf.position_std)
        d = tr.position[None, :] - path_pos
        lon = d[:, 0] * path_dirs[:, 0] + d[:, 1] * path_dirs[:, 1]
        lat = -d[:, 0] * path_dirs[:, 1] + d[:, 1] * path_dirs[:, 0]
        hit = (np.abs(lon) < VEHICLE_HALF_LENGTH + OBSTACLE_RADIUS + grow) & (
            np.abs(lat) < VEHICLE_HALF_WIDTH + OBSTACLE_RADIUS + grow
        )
        if hit.any():
            s_hit = float(path_dist[int(np.argmax(hit))])
            if occupied is None or s_hit < occupied[0]:
                occupied = (s_hit, tr.id)

    for v_hyp in sorted({cruise, max(vehicle.speed, creep_speed)}):
        dist = v_hyp * ts
        pos, dirs = vehicle.trajectory(dist)
        for tr in tracks:
            # The safety envelope grows with look-ahead time: estimation error plus the chance that a
            # pedestrian changes course (bounded by ``drift`` cells/s, capped at ``max_drift``).
            grow = margin + min(0.1, tr.kf.position_std) + np.minimum(drift * ts, max_drift)
            half_len = VEHICLE_HALF_LENGTH + OBSTACLE_RADIUS + grow
            half_wid = VEHICLE_HALF_WIDTH + OBSTACLE_RADIUS + grow
            obs = tr.position[None, :] + ts[:, None] * tr.velocity[None, :]
            d = obs - pos
            lon = d[:, 0] * dirs[:, 0] + d[:, 1] * dirs[:, 1]
            lat = -d[:, 0] * dirs[:, 1] + d[:, 1] * dirs[:, 0]
            hit = (np.abs(lon) < half_len) & (np.abs(lat) < half_wid)
            if hit.any():
                k = int(np.argmax(hit))
                cand = (float(dist[k]), float(ts[k]), tr.id)
                if best is None or cand[0] < best[0]:
                    best = cand

    def stop_speed(s: float) -> float:
        return math.sqrt(max(0.0, 2.0 * 0.8 * vehicle.max_decel * (s - standoff)))

    result = ThreatAssessment(CRUISE, cruise)
    if best is not None:
        s, t, tid = best
        allowed = min(caution * cruise, stop_speed(s))
        result = ThreatAssessment(BRAKE if allowed < 0.1 * cruise else SLOW, allowed, tid, t, s)
    if occupied is not None:
        s, tid = occupied
        allowed = stop_speed(s)
        if allowed < min(cruise, result.allowed_speed):
            t = s / max(vehicle.speed, 1e-6)
            result = ThreatAssessment(BRAKE if allowed < 0.1 * cruise else SLOW, allowed, tid, min(t, 99.0), s)
    return result


class PerceptionSystem:
    """Owns the obstacle world, sensor, tracker and evaluation stats; one ``step`` per frame."""

    def __init__(
        self,
        city: CityGrid,
        num_obstacles: int = 8,
        sensor_range: float = 7.0,
        sensor_noise: float = 0.35,
        sensor_rate: float = 20.0,
        sensor_dropout: float = 0.05,
        accel_std: float = 1.0,
        horizon: float = 1.5,
        seed: Optional[int] = None,
        exclude: Sequence[Sequence[float]] = (),
    ) -> None:
        self.obstacles = ObstacleField(city, num_obstacles, seed, exclude)
        self.sensor = NoisySensor(sensor_range, sensor_noise, sensor_rate, sensor_dropout, seed)
        self.tracker = MultiObjectTracker(sensor_noise, accel_std)
        self.horizon = horizon
        self.stats = TrackingStats()
        self.assessment = ThreatAssessment(CRUISE, float("inf"))
        self.trails: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}
        self.frame_errors: List[Tuple[float, float]] = []

    def step(self, now: float, dt: float, vehicle) -> ThreatAssessment:
        self.obstacles.update(dt, vehicle)
        detections = self.sensor.scan(now, vehicle.position, self.obstacles)
        self.tracker.step(now, dt, detections)
        truth_by_id = {ob.id: ob for ob in self.obstacles}
        self.frame_errors = []
        for det in detections:
            self.trails.setdefault(det.obstacle_id, deque(maxlen=8)).append((now, det.z))
            tr = self.tracker.tracks[det.obstacle_id]
            if tr.hits >= self.tracker.min_hits:
                truth = truth_by_id[det.obstacle_id]
                raw_err = float(np.hypot(*(det.z - det.truth)))
                kf_err = float(np.hypot(*(tr.position - det.truth)))
                self.stats.record(raw_err, kf_err, float(np.hypot(*(tr.velocity - truth.velocity))))
                self.frame_errors.append((raw_err, kf_err))
        self.assessment = assess_threats(vehicle, self.tracker.confirmed(), self.horizon)
        return self.assessment

    def recent_measurements(self, now: float, max_age: float = 0.4) -> List[Tuple[np.ndarray, float]]:
        out = []
        for trail in self.trails.values():
            out.extend((z, now - t) for t, z in trail if now - t <= max_age)
        return out

    def collisions(self, vehicle) -> List[MovingObstacle]:
        return [ob for ob in self.obstacles if vehicle_contact(vehicle, ob.position)]
