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
            f"{len(self.route.explored)} nodes explored. V2X {'online' if cfg.v2x else 'offline'}."
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
        status = "ARRIVED" if self.vehicle.finished else ("PAUSED" if self.paused else "DRIVING")
        return [
            f"{status} {self.vehicle.completion:4.0%} | {self.remaining_steps()} steps left | "
            f"reroutes {self.reroutes} | V2X incidents {len(self.network.active)} | t={self.time:5.1f}s",
            "[R] new map  [I] inject incident  [Space] pause  [Esc] quit",
        ]

    def step(self, dt: float) -> None:
        if self.paused:
            return
        self.time += dt
        self.network.update(self.time, dt, self.vehicle)
        self.vehicle.update(dt)
        if any(not self.city.passable[c] for c in self.vehicle.committed_cells):
            self.safety_violations += 1
            self.log(f"SAFETY: vehicle entered blocked cell near {self.vehicle.anchor}", "alert")
        if self.vehicle.finished and not self._arrival_logged:
            self._arrival_logged = True
            self.log(f"Arrived at goal {self.city.goal} after {self.reroutes} re-route(s)", "clear")

    def render(self) -> None:
        explored = self.route.explored if self.config.show_explored else ()
        ghost_strength = 1.0 - (self.time - self.ghost_time) / GHOST_SECONDS
        banner = f"V2X: DYNAMIC RE-ROUTE #{self.reroutes}" if self.time < self.banner_until else None
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
            banner=banner,
            log_entries=[(f"[{e.time:5.1f}s] {e.text}", e.level) for e in self.log_entries],
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
    ) -> SimulationResult:
        """Run the loop. Headless runs use a fixed timestep and don't sleep."""
        frames = 0
        fixed_dt = 1.0 / self.config.fps
        running = True
        while running:
            running = self._handle_events()
            dt = fixed_dt if self.headless else self.clock.tick(self.config.fps) / 1000.0
            self.step(dt)
            self.render()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            if exit_on_arrival and self.vehicle.finished:
                break

        if screenshot_path:
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
            screenshot=screenshot,
        )

    def close(self) -> None:
        pygame.quit()
