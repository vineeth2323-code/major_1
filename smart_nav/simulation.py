"""Ties the environment, A* router, V2X network and vehicle together in a Pygame loop."""

from __future__ import annotations

import logging
import math
import os
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

from .analytics import RunAnalytics
from .environment import CityGrid, CityRenderer
from .perception import BRAKE, CRUISE, SLOW, PerceptionSystem
from .routing import RouteResult, find_route
from .v2x_network import CLEARED, INCIDENT, V2XMessage, V2XNetwork
from .vehicle import Vehicle

log = logging.getLogger("smart_nav")

BANNER_SECONDS = 2.5
GHOST_SECONDS = 3.0
STOPPED = 0.05
CREEP = 1.0

ARRIVED, SAFE_HALT, IN_PROGRESS = "ARRIVED", "SAFE_HALT", "IN_PROGRESS"
GOAL_ISOLATED, GOAL_OCCUPIED, PATH_OBSTRUCTED, WATCHDOG = (
    "goal isolated", "goal occupied", "path obstructed", "watchdog timeout")


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
    max_sim_time: float = 300.0
    halt_patience: float = 15.0
    blocked_patience: float = 5.0


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
    outcome: str = IN_PROGRESS
    halt_reason: Optional[str] = None
    safe_halts: int = 0
    avg_frame_ms: float = 0.0
    peak_frame_ms: float = 0.0
    energy_wh: float = 0.0
    efficiency_pct: float = 0.0
    screenshot: Optional[str] = None
    report_paths: Optional[Tuple[str, str]] = None


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
        self.halt_reason: Optional[str] = None
        self.halted_at = 0.0
        self.safe_halts = 0
        self.watchdog_tripped = False
        self._blocked_time = 0.0
        self._clear_time = 0.0
        self._retry_at = 0.0
        self.analytics = RunAnalytics(cfg.fps, cruise_speed=cfg.speed)

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
        self.scenario_seed = seed

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

    @property
    def arrived(self) -> bool:
        return self.vehicle.finished and self.vehicle.position == tuple(map(float, self.city.goal))

    @property
    def outcome(self) -> str:
        if self.arrived:
            return ARRIVED
        return SAFE_HALT if self.halt_reason else IN_PROGRESS

    @property
    def done(self) -> bool:
        """The mission is over: arrived, or held in a safe halt for ``halt_patience`` seconds."""
        if self.arrived:
            return True
        if self.watchdog_tripped:
            return self.vehicle.speed < 1e-6
        return self.halt_reason is not None and self.time - self.halted_at >= self.config.halt_patience

    def _enter_safe_halt(self, reason: str, detail: str) -> None:
        if self.halt_reason is not None:
            return
        self.halt_reason, self.halted_at = reason, self.time
        self.safe_halts += 1
        self.vehicle.pulled_over = True
        self.drive_state = BRAKE
        self._clear_time = 0.0
        self._retry_at = self.time + 1.0
        where = self.vehicle.path[-1] if reason == GOAL_ISOLATED else self.vehicle.anchor
        self.log(f"CRITICAL: {detail}", "critical")
        self.log(f"SAFE HALT ({reason}): pulled over at {where}, holding until clear", "critical")
        self.analytics.event(self.time, "halt", reason)

    def _leave_safe_halt(self, detail: str) -> None:
        reason = self.halt_reason
        self.halt_reason = None
        self.vehicle.pulled_over = False
        self._blocked_time = 0.0
        self.drive_state = CRUISE
        self._episode_braked = False
        self.log(f"RECOVERED from safe halt ({reason}): {detail}", "clear")
        self.analytics.event(self.time, "resume", reason or "")

    def _try_resume_route(self) -> bool:
        """After a goal-isolation halt: re-plan from where the vehicle stands (or will stop)."""
        v = self.vehicle
        start = v.path[-1] if v.finished else v.anchor
        result = find_route(self.city.passable, start, self.city.goal)
        if not result.found:
            return False
        if v.finished:
            v.resume(result.path)
        else:
            v.reroute(result.path)
        self.route = result
        self._leave_safe_halt(f"A* found a {int(result.cost)}-step route from {start} to the goal")
        return True

    def isolate_goal(self, duration: float = math.inf) -> int:
        """Close every road into the goal via V2X, bypassing the reachability guard (demo/tests).

        Closures are permanent unless ``duration`` (seconds) is given."""
        gr, gc = self.city.goal
        closed = 0
        for cell in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
            if 0 <= cell[0] < self.city.rows and 0 <= cell[1] < self.city.cols and self.city.passable[cell]:
                if self.network.report(cell, self.time, kind="road closure", duration=duration,
                                       vehicle=self.vehicle, allow_isolation=True):
                    closed += 1
        return closed

    def occupy_goal(self, kind: str = "pedestrian"):
        """Put a stationary obstacle on the goal marker (demo/tests)."""
        if self.perception is None:
            raise RuntimeError("perception is disabled")
        ob = self.perception.obstacles.place_stationary(self.city.goal, kind)
        self.log(f"Obstacle #{ob.id} ({kind}) stopped on the goal marker {self.city.goal}", "alert")
        return ob

    def _on_v2x(self, msg: V2XMessage) -> None:
        inc = msg.incident
        if msg.type == INCIDENT:
            self.analytics.event(self.time, "incident", f"{inc.kind} at {inc.cell}")
            on_route = not self.vehicle.finished and inc.cell in self.vehicle.cells_ahead()
            self.log(
                f"V2X ALERT: {inc.kind} #{inc.id} at {inc.cell}" + (" - ON ROUTE" if on_route else ""),
                "alert",
            )
            if on_route:
                self.replan(f"{inc.kind} at {inc.cell}", force=True)
        elif msg.type == CLEARED:
            self.log(f"V2X CLEAR: {inc.kind} #{inc.id} at {inc.cell} reopened", "clear")
            if self.halt_reason == GOAL_ISOLATED:
                self._try_resume_route()
            elif not self.vehicle.finished:
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
            abandoned = self.vehicle.reroute([anchor])
            self.ghost_path, self.ghost_time = abandoned, self.time
            self._enter_safe_halt(
                GOAL_ISOLATED,
                f"NO ROUTE to goal {self.city.goal}: V2X closures isolate it",
            )
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
        self.analytics.event(self.time, "reroute", reason)
        return True

    def inject_incident_on_route(self, kind: str = "accident") -> bool:
        """Report an incident on the vehicle's upcoming route (demo key / tests)."""
        ahead = self.vehicle.cells_ahead()[self.network.min_lead : -1]
        for cell in ahead[len(ahead) // 3 :] + ahead[: len(ahead) // 3]:
            if self.network.report(cell, self.time, kind=kind, vehicle=self.vehicle):
                return True
        return False

    def hud_lines(self):
        if self.halt_reason:
            status = "SAFE HALT"
        elif self.arrived:
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
        line2 = "[R] map [I] incident [X/O] edge [Space] pause [Esc] quit"
        if self.perception is not None:
            st = self.perception.stats
            line2 += (
                f" | tracks {len(self.perception.tracker.confirmed())} | "
                f"err KF {st.kf_rmse:.2f} vs sensor {st.raw_rmse:.2f}"
            )
        return [line1, line2]

    def _govern_speed(self, allowed: float) -> None:
        """Final speed command: perception limit, a smooth stop at the end of the path, watchdog."""
        v = self.vehicle
        limit = min(allowed, math.sqrt(2.0 * 0.8 * v.max_decel * v.remaining_distance) + 0.5)
        if self.watchdog_tripped:
            limit = 0.0
        v.set_target_speed(limit)

    def _check_watchdog(self) -> None:
        if self.watchdog_tripped or self.arrived or self.time < self.config.max_sim_time:
            return
        self.watchdog_tripped = True
        self.halt_reason = None
        self._enter_safe_halt(
            WATCHDOG, f"mission exceeded max_sim_time={self.config.max_sim_time:g}s without reaching the goal"
        )

    def _update_halt(self, dt: float, assessment) -> None:
        """Detect a vehicle stuck behind a stationary obstacle, and recover from safe halts."""
        v = self.vehicle
        if self.halt_reason == GOAL_ISOLATED:
            if self.time >= self._retry_at:
                self._retry_at = self.time + 1.0
                self._try_resume_route()
            return
        if self.halt_reason in (GOAL_OCCUPIED, PATH_OBSTRUCTED):
            self._clear_time = self._clear_time + dt if assessment is not None and assessment.state == CRUISE else 0.0
            if self._clear_time >= 1.0:
                self._leave_safe_halt("obstruction cleared, resuming to the goal")
            return
        if self.halt_reason is not None or assessment is None or v.finished:
            return
        # Stopped, or only creeping towards the standoff point, behind an obstacle.
        stuck = assessment.state != CRUISE and v.speed < CREEP
        self._blocked_time = self._blocked_time + dt if stuck else 0.0
        if self._blocked_time < self.config.blocked_patience:
            return
        if self.perception is None:
            return
        goal = np.asarray(self.city.goal, dtype=float)
        stationary = [t for t in self.perception.tracker.confirmed() if float(np.hypot(*t.velocity)) < 0.3]
        on_goal = [t for t in stationary if float(np.hypot(*(t.position - goal))) < 1.0]
        if on_goal and v.remaining_distance < 4.0:
            tr = on_goal[0]
            self._enter_safe_halt(GOAL_OCCUPIED, f"goal {self.city.goal} occupied by stationary {tr.kind} "
                                  f"#{tr.id} for {self._blocked_time:.0f}s")
            return
        tr = self.perception.tracker.tracks.get(assessment.track_id)
        if tr is not None and tr in stationary:
            self._enter_safe_halt(PATH_OBSTRUCTED, f"stationary {tr.kind} #{tr.id} has blocked the lane "
                                  f"for {self._blocked_time:.0f}s")

    def _update_perception(self, dt: float):
        """Run perception and return (allowed speed, assessment)."""
        cruise = self.vehicle.cruise_speed
        if self.perception is None:
            return cruise, None
        assessment = self.perception.step(self.time, dt, self.vehicle)
        if not self.config.safety_brake or self.vehicle.finished:
            return cruise, assessment
        if self.halt_reason is not None:
            # Hold the safe halt until the obstruction has been clear for a moment.
            hold = self.halt_reason in (GOAL_OCCUPIED, PATH_OBSTRUCTED)
            return (0.0 if hold else assessment.allowed_speed), assessment
        prev = self.drive_state
        if assessment.state == prev:
            return assessment.allowed_speed, assessment
        if assessment.state != CRUISE:
            tr = self.perception.tracker.tracks.get(assessment.track_id)
            who = f"{tr.kind if tr else 'obstacle'} #{assessment.track_id}"
            when = f"in {assessment.time_to_conflict:.2f}s, {assessment.distance_to_conflict:.1f} cells ahead"
        if assessment.state == BRAKE and not self._episode_braked:
            # One SAFETY BRAKE per episode, however often the limit flickers between brake and creep.
            self.brake_events += 1
            self._episode_braked = True
            self.log(f"SAFETY BRAKE: Kalman track {who} will cross our path {when}", "alert")
            self.analytics.event(self.time, "brake", who)
        elif assessment.state == SLOW and prev == CRUISE:
            self.log(f"CAUTION: slowing for {who} predicted to cross {when}", "alert")
        elif assessment.state == CRUISE:
            self._episode_braked = False
            self.log("Path clear - brake released, resuming cruise speed", "clear")
        self.drive_state = assessment.state
        return assessment.allowed_speed, assessment

    def _check_collisions(self) -> None:
        if self.perception is None:
            return
        contact = {ob.id for ob in self.perception.collisions(self.vehicle)}
        for oid in contact - self._in_contact:
            self.collisions += 1
            self.log(f"COLLISION with obstacle #{oid} at speed {self.vehicle.speed:.1f}", "alert")
            self.analytics.event(self.time, "collision", f"obstacle #{oid}")
        self._in_contact = contact

    def step(self, dt: float) -> None:
        if self.paused:
            return
        started = _time.perf_counter()
        self.time += dt
        self.network.update(self.time, dt, self.vehicle)
        allowed, assessment = self._update_perception(dt)
        self._check_watchdog()
        self._update_halt(dt, assessment)
        self._govern_speed(allowed)
        self.vehicle.update(dt)
        self._check_collisions()
        if self._arrival_logged is False and self.vehicle.speed < self.min_speed:
            self.min_speed = self.vehicle.speed
        if any(not self.city.passable[c] for c in self.vehicle.committed_cells):
            self.safety_violations += 1
            self.log(f"SAFETY: vehicle entered blocked cell near {self.vehicle.anchor}", "alert")
        if self.arrived and not self._arrival_logged:
            self._arrival_logged = True
            self.log(f"Arrived at goal {self.city.goal} after {self.reroutes} re-route(s)", "clear")
            self.analytics.event(self.time, "arrived", str(self.city.goal))
        self.analytics.record(
            self.time,
            dt,
            self.vehicle.speed,
            (_time.perf_counter() - started) * 1000,
            self.perception.frame_errors if self.perception else (),
            "CRUISE" if self.vehicle.finished else self.drive_state,
            self.vehicle.target_speed,
            active=not self.done,
        )

    def render(self) -> None:
        started = _time.perf_counter()
        explored = self.route.explored if self.config.show_explored else ()
        ghost_strength = 1.0 - (self.time - self.ghost_time) / GHOST_SECONDS
        banners = []
        if self.time < self.banner_until:
            banners.append((f"V2X: DYNAMIC RE-ROUTE #{self.reroutes}", "reroute"))
        if self.halt_reason:
            banners.append((f"SAFE HALT: {self.halt_reason.upper()}", "brake"))
        elif self.perception is not None and self.config.safety_brake and not self.vehicle.finished:
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
        self.analytics.add_render_time((_time.perf_counter() - started) * 1000)

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
                elif event.key == pygame.K_x:
                    self.isolate_goal(duration=10.0)
                elif event.key == pygame.K_o and self.perception is not None:
                    self.occupy_goal()
        return True

    def run(
        self,
        max_frames: Optional[int] = None,
        exit_on_arrival: bool = False,
        screenshot_path: Optional[str] = None,
        render: bool = True,
        report_dir: Optional[str] = None,
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
            if exit_on_arrival and self.done:
                break

        if screenshot_path:
            if not render:
                self.render()
            self.save_screenshot(screenshot_path)
        result = self.result(frames, screenshot_path)
        if report_dir:
            result.report_paths = self.export_report(report_dir, result)
        return result

    def report_meta(self, result: Optional[SimulationResult] = None) -> Dict[str, object]:
        r = result or self.result(self.analytics.frames)
        cfg = self.config
        meta: Dict[str, object] = {
            "Outcome": f"**{r.outcome}**" + (f" ({r.halt_reason})" if r.halt_reason else ""),
            "Seed": self.scenario_seed,
            "Map": f"{self.city.rows}x{self.city.cols} cells, {len(self.city.closed_segments)} road-work closures",
            "Route": f"{self.city.start} -> {self.city.goal}, final path {r.path_length - 1} steps",
            "Simulated time": f"{r.sim_time:.2f} s ({r.frames} frames at {cfg.fps} FPS)",
            "V2X": f"{'on' if cfg.v2x else 'off'}, rate {cfg.incident_rate}/s",
            "Perception": (f"{cfg.num_obstacles} obstacles, noise {cfg.sensor_noise} cells, range {cfg.sensor_range}, "
                           f"safety brake {'on' if cfg.safety_brake else 'off'}") if cfg.perception else "off",
            "Safety violations / collisions": f"{r.safety_violations} / {r.collisions}",
        }
        return meta

    def export_report(self, out_dir: str, result: Optional[SimulationResult] = None) -> Tuple[str, str]:
        """Write ``performance_report.md`` and ``analytics_summary.png`` into ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        chart = self.analytics.plot(
            os.path.join(out_dir, "analytics_summary.png"),
            title=f"Smart Navigation run analytics (seed {self.scenario_seed}, {self.outcome})",
        )
        report = self.analytics.write_report(
            os.path.join(out_dir, "performance_report.md"), self.report_meta(result), chart
        )
        return report, chart

    def save_screenshot(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pygame.image.save(self.screen, path)

    def result(self, frames: Optional[int] = None, screenshot: Optional[str] = None) -> SimulationResult:
        lat, energy = self.analytics.latency(), self.analytics.energy_summary()
        return SimulationResult(
            frames=self.analytics.frames if frames is None else frames,
            sim_time=self.time,
            reached_goal=self.arrived,
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
            outcome=self.outcome,
            halt_reason=self.halt_reason,
            safe_halts=self.safe_halts,
            avg_frame_ms=lat["avg"],
            peak_frame_ms=lat["peak"],
            energy_wh=energy["total_wh"],
            efficiency_pct=energy["efficiency_pct"],
            screenshot=screenshot,
        )

    def close(self) -> None:
        pygame.quit()
