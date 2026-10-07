"""Virtual autonomous vehicle that follows a planned route."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pygame

Cell = Tuple[int, int]

VEHICLE_HALF_LENGTH = 0.55
VEHICLE_HALF_WIDTH = 0.3


class Vehicle:
    def __init__(
        self, path: Sequence[Cell], speed: float = 6.0, max_accel: float = 4.0, max_decel: float = 12.0
    ) -> None:
        """``speed`` is the cruise speed in grid cells per second; accel/decel in cells/s^2."""
        if not path:
            raise ValueError("Vehicle needs a non-empty path")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.path: List[Cell] = [tuple(p) for p in path]
        self.cruise_speed = speed
        self.speed = speed
        self.target_speed = speed
        self.max_accel = max_accel
        self.max_decel = max_decel
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

    @property
    def braking(self) -> bool:
        return self.target_speed < self.speed - 1e-6

    @property
    def direction(self) -> np.ndarray:
        """Unit heading vector in (row, col)."""
        h = math.radians(self.heading)
        return np.array([math.sin(h), math.cos(h)])

    def set_target_speed(self, speed: float) -> None:
        self.target_speed = min(self.cruise_speed, max(0.0, speed))

    def trajectory(self, distances: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Positions and unit headings (row, col) at the given distances ahead along the route.

        Distances past the goal clamp to the goal.
        """
        distances = np.asarray(distances, dtype=float)
        pts = np.array([self.position] + [tuple(map(float, p)) for p in self.path[self.segment + 1 :]])
        if len(pts) < 2:
            return np.repeat(pts, len(distances), axis=0), np.repeat(self.direction[None], len(distances), axis=0)
        seg = np.diff(pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        keep = seg_len > 1e-9
        if not keep.any():
            return np.repeat(pts[:1], len(distances), axis=0), np.repeat(self.direction[None], len(distances), axis=0)
        pts = np.vstack([pts[:1], pts[1:][keep]])
        seg, seg_len = seg[keep], seg_len[keep]
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        s = np.clip(distances, 0.0, cum[-1])
        idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
        frac = (s - cum[idx]) / seg_len[idx]
        positions = pts[idx] + seg[idx] * frac[:, None]
        return positions, seg[idx] / seg_len[idx][:, None]

    @property
    def anchor(self) -> Cell:
        """Cell the vehicle is committed to reaching next; re-planning starts here."""
        return self.path[min(self.segment + 1, len(self.path) - 1)]

    @property
    def committed_cells(self) -> Tuple[Cell, ...]:
        """Cells the vehicle occupies or can no longer avoid (current segment)."""
        return tuple(self.path[self.segment : self.segment + 2])

    def cells_ahead(self) -> List[Cell]:
        """Planned cells beyond the anchor, in driving order."""
        return self.path[self.segment + 2 :]

    def reroute(self, new_tail: Sequence[Cell]) -> List[Cell]:
        """Replace the route after the anchor with ``new_tail`` (which must start at the anchor).

        Returns the abandoned part of the old route.
        """
        if self.finished:
            raise RuntimeError("vehicle already arrived")
        anchor_idx = self.segment + 1
        new_tail = [tuple(p) for p in new_tail]
        if not new_tail or new_tail[0] != self.path[anchor_idx]:
            raise ValueError(f"new route must start at the anchor {self.path[anchor_idx]}")
        abandoned = self.path[anchor_idx:]
        self.path = self.path[:anchor_idx] + new_tail
        return abandoned

    def update(self, dt: float) -> None:
        if self.target_speed < self.speed:
            self.speed = max(self.target_speed, self.speed - self.max_decel * dt)
        else:
            self.speed = min(self.target_speed, self.speed + self.max_accel * dt)
        if self.finished:
            self.speed = 0.0
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
        length, width = int(cs * 2 * VEHICLE_HALF_LENGTH), max(6, int(cs * 2 * VEHICLE_HALF_WIDTH))
        car = pygame.Surface((length, width), pygame.SRCALPHA)
        radius = max(2, cs // 6)
        pygame.draw.rect(car, (240, 242, 246), car.get_rect(), border_radius=radius)
        pygame.draw.rect(car, (30, 34, 44), car.get_rect(), 1, border_radius=radius)
        pygame.draw.rect(car, (60, 80, 110), pygame.Rect(int(length * 0.55), 2, int(length * 0.22), width - 4), border_radius=2)
        pygame.draw.rect(car, (255, 240, 150), pygame.Rect(length - 3, 1, 2, 3))
        pygame.draw.rect(car, (255, 240, 150), pygame.Rect(length - 3, width - 4, 2, 3))
        tail = (255, 30, 30) if self.braking or (self.speed < 0.05 and not self.finished) else (120, 20, 20)
        pygame.draw.rect(car, tail, pygame.Rect(0, 1, 3, 3))
        pygame.draw.rect(car, tail, pygame.Rect(0, width - 4, 3, 3))
        rotated = pygame.transform.rotate(car, -self.heading)
        surface.blit(rotated, rotated.get_rect(center=renderer.to_pixel(self.position)))
