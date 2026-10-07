"""Ties the environment, A* router, V2X network and vehicle together in a Pygame loop."""

from __future__ import annotations

import logging
import os
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import pygame

from .environment import CityGrid, CityRenderer
from .perception import BRAKE, CRUISE, SLOW, PerceptionSystem
from .routing import RouteResult, find_route
from .v2x_network import CLEARED, INCIDENT, V2XMessage, V2XNetwork
from .vehicle import Vehicle

log = logging.getLogger("smart_nav")

BANNER_SECONDS = 2.5
GHOST_SECONDS = 3.0


@dataclass
class SimulationConfig:
    blocks_x: int = 8
    blocks_y: int = 6
    block_size: int = 4
    cell_size: int = 20
    closure_rate: float = 0.2
    seed: Optional[int] = None
    speed: float = 6.0
    fps: int = 60
    random_endpoints: bool = False
    show_explored: bool = True
    v2x: bool = True
    incident_rate: float = 0.3
    incident_duration: Optional[Tuple[float, float]] = (12.0, 25.0)
    path_bias: float = 0.6
    max_incidents: int = 6
    perception: bool = True
    num_obstacles: int = 8
    sensor_range: float = 7.0
    sensor_noise: float = 0.35
    sensor_rate: float = 20.0
    sensor_dropout: float = 0.05
    safety_brake: bool = True
    brake_horizon: float = 1.5


@dataclass
class SimulationResult:
    frames: int
    sim_time: float
    reached_goal: bool
    path_length: int
    path_cost: float
    explored_nodes: int
    closed_segments: int
    incidents: int = 0
    reroutes: int = 0
    safety_violations: int = 0
    collisions: int = 0
    brake_events: int = 0
    min_speed: float = 0.0
    sensor_rmse: float = 0.0
    kf_rmse: float = 0.0
    screenshot: Optional[str] = None


@dataclass
class LogEntry:
    time: float
    text: str
    level: str


class Simulation:
    def __init__(self, config: Optional[SimulationConfig] = None, headless: bool = False) -> None:
        self.config = config or SimulationConfig()
        self.headless = headless
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.display.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None
        self.paused = False
        self.new_scenario(self.config.seed)

    def new_scenario(self, seed: Optional[int] = None) -> None:
        cfg = self.config
        self.city = CityGrid(cfg.blocks_x, cfg.blocks_y, cfg.block_size, cfg.closure_rate, seed)
        if cfg.random_endpoints:
            self.city.randomize_endpoints()
        self.route: RouteResult = find_route(self.city.passable, self.city.start, self.city.goal)
        if not self.route.found:
            raise RuntimeError(f"No route from {self.city.start} to {self.city.goal}")
        self.initial_route = self.route
        self.vehicle = Vehicle(self.route.path, cfg.speed)
        self.time = 0.0
        self.reroutes = 0
        self.safety_violations = 0
        self.log_entries: Deque[LogEntry] = deque(maxlen=200)
        self.ghost_path: List[Tuple[int, int]] = []
        self.ghost_time = -1e9
        self.banner_until = -1e9
        self._arrival_logged = False
        self.collisions = 0
        self.brake_events = 0
        self.min_speed = self.vehicle.speed
        self.drive_state = CRUISE
        self._episode_braked = False
        self._in_contact: set = set()

        self.perception: Optional[PerceptionSystem] = None
        if cfg.perception:
            self.perception = PerceptionSystem(
                self.city,
                num_obstacles=cfg.num_obstacles,
                sensor_range=cfg.sensor_range,
                sensor_noise=cfg.sensor_noise,
                sensor_rate=cfg.sensor_rate,
                sensor_dropout=cfg.sensor_dropout,
                horizon=cfg.brake_horizon,
                seed=None if seed is None else seed + 1000,
                exclude=[self.city.start],
            )

        self.network = V2XNetwork(
            self.city,
            rate=cfg.incident_rate if cfg.v2x else 0.0,
            duration=cfg.incident_duration,
            path_bias=cfg.path_bias,
            max_active=cfg.max_incidents,
            seed=seed,
        )
        self.network.subscribe(self._on_v2x)

        self.renderer = CityRenderer(self.city, cfg.cell_size)
        if self.screen is None or self.screen.get_size() != self.renderer.size:
            self.screen = pygame.display.set_mode(self.renderer.size)
        pygame.display.set_caption("Smart Navigation - A* Route Optimization + V2X")
        self.log(
            f"Route planned {self.city.start} -> {self.city.goal}: {int(self.route.cost)} steps, "
            f"{len(self.route.explored)} nodes explored. V2X {'on' if cfg.v2x else 'off'}, "
            f"{cfg.num_obstacles if cfg.perception else 0} moving obstacles."
        )

    def log(self, text: str, level: str = "info") -> None:
        self.log_entries.append(LogEntry(self.time, text, level))
        log.info("[t=%6.2fs] %s", self.time, text)

    def _on_v2x(self, msg: V2XMessage) -> None:
        inc = msg.incident
        if msg.type == INCIDENT:
            on_route = not self.vehicle.finished and inc.cell in self.vehicle.cells_ahead()
            self.log(
                f"V2X ALERT: {inc.kind} #{inc.id} at {inc.cell}" + (" - ON ROUTE" if on_route else ""),
                "alert",
            )
            if on_route:
                self.replan(f"{inc.kind} at {inc.cell}", force=True)
        elif msg.type == CLEARED:
            self.log(f"V2X CLEAR: {inc.kind} #{inc.id} at {inc.cell} reopened", "clear")
            if not self.vehicle.finished:
                self.replan(f"{inc.cell} reopened", force=False)

    def remaining_steps(self) -> int:
        return len(self.vehicle.path) - 1 - min(self.vehicle.segment + 1, len(self.vehicle.path) - 1)

    def replan(self, reason: str, force: bool) -> bool:
        """Mid-route A* from the vehicle's anchor cell to the goal.

        ``force`` swaps in the new route unconditionally (current one is blocked); otherwise it is
        only adopted if strictly shorter. Returns True if the route changed.
        """
        if self.vehicle.finished:
            return False
        anchor = self.vehicle.anchor
        started = _time.perf_counter()
        result = find_route(self.city.passable, anchor, self.city.goal)
        elapsed_ms = (_time.perf_counter() - started) * 1000
        old_steps = self.remaining_steps()

        if not result.found:
            self.vehicle.reroute([anchor])
            self.log(f"NO ROUTE from {anchor} to goal - holding position", "alert")
            return True
        if not force and result.cost >= old_steps:
            return False

        abandoned = self.vehicle.reroute(result.path)
        self.route = result
        self.reroutes += 1
        self.ghost_path, self.ghost_time = abandoned, self.time
        self.banner_until = self.time + BANNER_SECONDS
        verb = "RE-ROUTE" if force else "OPTIMIZED"
        self.log(
            f"{verb} #{self.reroutes}: {reason} -> A* from {anchor}: {int(result.cost)} steps "
            f"({int(result.cost) - old_steps:+d}) in {elapsed_ms:.1f} ms",
            "reroute",
        )
        return True

    def inject_incident_on_route(self, kind: str = "accident") -> bool:
        """Report an incident on the vehicle's upcoming route (demo key / tests)."""
        ahead = self.vehicle.cells_ahead()[self.network.min_lead : -1]
        for cell in ahead[len(ahead) // 3 :] + ahead[: len(ahead) // 3]:
            if self.network.report(cell, self.time, kind=kind, vehicle=self.vehicle):
                return True
        return False

    def hud_lines(self):
        if self.vehicle.finished:
            status = "ARRIVED"
        elif self.paused:
            status = "PAUSED"
        else:
            status = {CRUISE: "DRIVING", SLOW: "SLOWING", BRAKE: "BRAKING"}[self.drive_state]
        v = self.vehicle
        line1 = (
            f"{status} {v.completion:4.0%} | v {v.speed:3.1f}/{v.cruise_speed:g} | {self.remaining_steps()} left | "
            f"reroutes {self.reroutes} | V2X {len(self.network.active)} | t={self.time:5.1f}s"
        )
        line2 = "[R] map [I] incident [Space] pause [Esc] quit"
        if self.perception is not None:
            st = self.perception.stats
            line2 += (
                f" | tracks {len(self.perception.tracker.confirmed())} | "
                f"err KF {st.kf_rmse:.2f} vs sensor {st.raw_rmse:.2f}"
            )
        return [line1, line2]

    def _update_perception(self, dt: float) -> None:
        if self.perception is None:
            return
        assessment = self.perception.step(self.time, dt, self.vehicle)
        if not self.config.safety_brake or self.vehicle.finished:
            self.vehicle.set_target_speed(self.vehicle.cruise_speed)
            return
        self.vehicle.set_target_speed(assessment.allowed_speed)
        prev = self.drive_state
        if assessment.state == prev:
            return
        if assessment.state != CRUISE:
            tr = self.perception.tracker.tracks.get(assessment.track_id)
            who = f"{tr.kind if tr else 'obstacle'} #{assessment.track_id}"
            when = f"in {assessment.time_to_conflict:.2f}s, {assessment.distance_to_conflict:.1f} cells ahead"
        if assessment.state == BRAKE and not self._episode_braked:
            # One SAFETY BRAKE per episode, however often the limit flickers between brake and creep.
            self.brake_events += 1
            self._episode_braked = True
            self.log(f"SAFETY BRAKE: Kalman track {who} will cross our path {when}", "alert")
        elif assessment.state == SLOW and prev == CRUISE:
            self.log(f"CAUTION: slowing for {who} predicted to cross {when}", "alert")
        elif assessment.state == CRUISE:
            self._episode_braked = False
            self.log("Path clear - brake released, resuming cruise speed", "clear")
        self.drive_state = assessment.state

    def _check_collisions(self) -> None:
        if self.perception is None:
            return
        contact = {ob.id for ob in self.perception.collisions(self.vehicle)}
        for oid in contact - self._in_contact:
            self.collisions += 1
            self.log(f"COLLISION with obstacle #{oid} at speed {self.vehicle.speed:.1f}", "alert")
        self._in_contact = contact

    def step(self, dt: float) -> None:
        if self.paused:
            return
        self.time += dt
        self.network.update(self.time, dt, self.vehicle)
        self._update_perception(dt)
        self.vehicle.update(dt)
        self._check_collisions()
        if self._arrival_logged is False and self.vehicle.speed < self.min_speed:
            self.min_speed = self.vehicle.speed
        if any(not self.city.passable[c] for c in self.vehicle.committed_cells):
            self.safety_violations += 1
            self.log(f"SAFETY: vehicle entered blocked cell near {self.vehicle.anchor}", "alert")
        if self.vehicle.finished and not self._arrival_logged:
            self._arrival_logged = True
            self.log(f"Arrived at goal {self.city.goal} after {self.reroutes} re-route(s)", "clear")

    def render(self) -> None:
        explored = self.route.explored if self.config.show_explored else ()
        ghost_strength = 1.0 - (self.time - self.ghost_time) / GHOST_SECONDS
        banners = []
        if self.time < self.banner_until:
            banners.append((f"V2X: DYNAMIC RE-ROUTE #{self.reroutes}", "reroute"))
        if self.perception is not None and self.config.safety_brake and not self.vehicle.finished:
            if self.drive_state == BRAKE:
                banners.append(("SAFETY BRAKE", "brake"))
            elif self.drive_state == SLOW:
                banners.append(("CAUTION: SLOWING", "slow"))
        self.renderer.draw(
            self.screen,
            self.vehicle.path,
            explored,
            self.vehicle,
            self.hud_lines(),
            incidents=list(self.network.active.values()),
            now=self.time,
            ghost_path=self.ghost_path,
            ghost_strength=ghost_strength,
            banners=banners,
            log_entries=[(f"[{e.time:5.1f}s] {e.text}", e.level) for e in self.log_entries],
            perception=self.perception,
        )
        pygame.display.flip()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.new_scenario(None)
                elif event.key == pygame.K_i:
                    self.inject_incident_on_route()
        return True

    def run(
        self,
        max_frames: Optional[int] = None,
        exit_on_arrival: bool = False,
        screenshot_path: Optional[str] = None,
        render: bool = True,
    ) -> SimulationResult:
        """Run the loop. Headless runs use a fixed timestep and don't sleep.

        ``render=False`` skips drawing (headless only) for fast batch runs.
        """
        frames = 0
        fixed_dt = 1.0 / self.config.fps
        running = True
        while running:
            running = self._handle_events()
            dt = fixed_dt if self.headless else self.clock.tick(self.config.fps) / 1000.0
            self.step(dt)
            if render or not self.headless:
                self.render()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            if exit_on_arrival and self.vehicle.finished:
                break

        if screenshot_path:
            if not render:
                self.render()
            self.save_screenshot(screenshot_path)
        return self.result(frames, screenshot_path)

    def save_screenshot(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pygame.image.save(self.screen, path)

    def result(self, frames: int = 0, screenshot: Optional[str] = None) -> SimulationResult:
        return SimulationResult(
            frames=frames,
            sim_time=self.time,
            reached_goal=self.vehicle.finished and self.vehicle.position == tuple(map(float, self.city.goal)),
            path_length=len(self.vehicle.path),
            path_cost=float(len(self.vehicle.path) - 1),
            explored_nodes=len(self.route.explored),
            closed_segments=len(self.city.closed_segments),
            incidents=sum(1 for m in self.network.history if m.type == INCIDENT),
            reroutes=self.reroutes,
            safety_violations=self.safety_violations,
            collisions=self.collisions,
            brake_events=self.brake_events,
            min_speed=self.min_speed,
            sensor_rmse=self.perception.stats.raw_rmse if self.perception else 0.0,
            kf_rmse=self.perception.stats.kf_rmse if self.perception else 0.0,
            screenshot=screenshot,
        )

    def close(self) -> None:
        pygame.quit()
