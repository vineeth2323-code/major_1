"""Virtual autonomous vehicle that follows a planned route."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import pygame

Cell = Tuple[int, int]


class Vehicle:
    def __init__(self, path: Sequence[Cell], speed: float = 6.0) -> None:
        """``speed`` is in grid cells per second."""
        if not path:
            raise ValueError("Vehicle needs a non-empty path")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.path: List[Cell] = [tuple(p) for p in path]
        self.speed = speed
        self.segment = 0
        self.progress = 0.0
        self.distance_travelled = 0.0
        self.position: Tuple[float, float] = (float(self.path[0][0]), float(self.path[0][1]))
        self.heading = self._segment_heading(0) if len(self.path) > 1 else 0.0

    @property
    def finished(self) -> bool:
        return self.segment >= len(self.path) - 1

    @property
    def total_distance(self) -> float:
        return sum(self._segment_length(i) for i in range(len(self.path) - 1))

    @property
    def completion(self) -> float:
        total = self.total_distance
        return 1.0 if total == 0 else min(1.0, self.distance_travelled / total)

    def update(self, dt: float) -> None:
        remaining = self.speed * dt
        while remaining > 1e-12 and not self.finished:
            length = self._segment_length(self.segment)
            step = min(remaining, length - self.progress)
            self.progress += step
            self.distance_travelled += step
            remaining -= step
            if self.progress >= length - 1e-12:
                self.segment += 1
                self.progress = 0.0
        self._sync_pose()

    def _sync_pose(self) -> None:
        if self.finished:
            self.position = (float(self.path[-1][0]), float(self.path[-1][1]))
            return
        a, b = self.path[self.segment], self.path[self.segment + 1]
        t = self.progress / self._segment_length(self.segment)
        self.position = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        self.heading = self._segment_heading(self.segment)

    def _segment_length(self, i: int) -> float:
        a, b = self.path[i], self.path[i + 1]
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _segment_heading(self, i: int) -> float:
        """Heading in degrees, screen convention: 0 = east, 90 = south."""
        a, b = self.path[i], self.path[i + 1]
        return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))

    def draw(self, surface: pygame.Surface, renderer) -> None:
        cs = renderer.cell_size
        length, width = int(cs * 1.1), int(cs * 0.65)
        car = pygame.Surface((length, width), pygame.SRCALPHA)
        pygame.draw.rect(car, (30, 144, 255), car.get_rect(), border_radius=max(2, cs // 6))
        pygame.draw.rect(car, (200, 230, 255), pygame.Rect(int(length * 0.6), 2, int(length * 0.22), width - 4), border_radius=2)
        pygame.draw.rect(car, (255, 255, 200), pygame.Rect(length - 3, 1, 3, 3))
        pygame.draw.rect(car, (255, 255, 200), pygame.Rect(length - 3, width - 4, 3, 3))
        rotated = pygame.transform.rotate(car, -self.heading)
        surface.blit(rotated, rotated.get_rect(center=renderer.to_pixel(self.position)))
