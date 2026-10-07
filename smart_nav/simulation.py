"""Ties the environment, A* router and vehicle together in a Pygame loop."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pygame

from .environment import CityGrid, CityRenderer
from .routing import RouteResult, find_route
from .vehicle import Vehicle


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


@dataclass
class SimulationResult:
    frames: int
    reached_goal: bool
    path_length: int
    path_cost: float
    explored_nodes: int
    closed_segments: int
    screenshot: Optional[str] = None


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
        self.vehicle = Vehicle(self.route.path, cfg.speed)
        self.renderer = CityRenderer(self.city, cfg.cell_size)
        if self.screen is None or self.screen.get_size() != self.renderer.size:
            self.screen = pygame.display.set_mode(self.renderer.size)
        pygame.display.set_caption("Smart Navigation - A* Route Optimization")

    def hud_lines(self):
        status = "ARRIVED" if self.vehicle.finished else ("PAUSED" if self.paused else "DRIVING")
        return [
            f"A* route: {len(self.route.path) - 1} steps | explored {len(self.route.explored)} nodes | "
            f"{len(self.city.closed_segments)} closures | {status} {self.vehicle.completion:4.0%}",
            "[R] new map   [Space] pause   [Esc] quit",
        ]

    def step(self, dt: float) -> None:
        if not self.paused:
            self.vehicle.update(dt)

    def render(self) -> None:
        explored = self.route.explored if self.config.show_explored else ()
        self.renderer.draw(self.screen, self.route.path, explored, self.vehicle, self.hud_lines())
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
            os.makedirs(os.path.dirname(os.path.abspath(screenshot_path)), exist_ok=True)
            pygame.image.save(self.screen, screenshot_path)

        return SimulationResult(
            frames=frames,
            reached_goal=self.vehicle.finished and self.vehicle.position == tuple(map(float, self.city.goal)),
            path_length=len(self.route.path),
            path_cost=self.route.cost,
            explored_nodes=len(self.route.explored),
            closed_segments=len(self.city.closed_segments),
            screenshot=screenshot_path,
        )

    def close(self) -> None:
        pygame.quit()
