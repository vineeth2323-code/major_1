"""City grid environment and its Pygame renderer.

The city is a coordinate grid of ``rows x cols`` cells indexed ``(row, col)``.
Roads run along every ``block_size``-th row and column; where they cross is an
intersection node. Some road segments between intersections are randomly closed
(road works) so the router has to plan around them, while every intersection
is guaranteed to stay reachable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pygame

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Segment:
    """A road stretch between two adjacent intersections (endpoints excluded from ``cells``)."""

    a: Cell
    b: Cell
    cells: Tuple[Cell, ...]


class CityGrid:
    def __init__(
        self,
        blocks_x: int = 8,
        blocks_y: int = 6,
        block_size: int = 4,
        closure_rate: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        if blocks_x < 1 or blocks_y < 1:
            raise ValueError("blocks_x and blocks_y must be >= 1")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if not 0.0 <= closure_rate < 1.0:
            raise ValueError("closure_rate must be in [0, 1)")

        self.blocks_x = blocks_x
        self.blocks_y = blocks_y
        self.block_size = block_size
        self.rows = blocks_y * block_size + 1
        self.cols = blocks_x * block_size + 1
        self.seed = seed
        self.rng = random.Random(seed)

        self.passable = np.zeros((self.rows, self.cols), dtype=bool)
        self.passable[::block_size, :] = True
        self.passable[:, ::block_size] = True

        self.intersections: List[Cell] = [
            (r, c) for r in range(0, self.rows, block_size) for c in range(0, self.cols, block_size)
        ]
        self.closed_segments: List[Segment] = []
        self._close_random_segments(closure_rate)

        self.start: Cell = self.intersections[0]
        self.goal: Cell = self.intersections[-1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def is_intersection(self, cell: Cell) -> bool:
        return cell[0] % self.block_size == 0 and cell[1] % self.block_size == 0

    def is_road(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols and bool(self.passable[cell])

    def segments(self) -> Iterator[Segment]:
        bs = self.block_size
        for r in range(0, self.rows, bs):
            for c in range(0, self.cols - 1, bs):
                yield Segment((r, c), (r, c + bs), tuple((r, c + i) for i in range(1, bs)))
        for c in range(0, self.cols, bs):
            for r in range(0, self.rows - 1, bs):
                yield Segment((r, c), (r + bs, c), tuple((r + i, c) for i in range(1, bs)))

    def set_endpoints(self, start: Cell, goal: Cell) -> None:
        for name, cell in (("start", start), ("goal", goal)):
            if not self.is_road(cell):
                raise ValueError(f"{name} {cell} is not on a road")
        self.start, self.goal = tuple(start), tuple(goal)

    def randomize_endpoints(self, min_distance: Optional[int] = None) -> None:
        """Pick two distinct intersections at least ``min_distance`` apart (Manhattan)."""
        if min_distance is None:
            min_distance = (self.rows + self.cols) // 2
        candidates = list(self.intersections)
        for _ in range(500):
            a, b = self.rng.sample(candidates, 2)
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= min_distance:
                self.set_endpoints(a, b)
                return
        self.set_endpoints(candidates[0], candidates[-1])

    def _close_random_segments(self, rate: float) -> None:
        segments = list(self.segments())
        self.rng.shuffle(segments)
        target = int(len(segments) * rate)
        for seg in segments:
            if len(self.closed_segments) >= target:
                break
            for cell in seg.cells:
                self.passable[cell] = False
            if self._intersections_connected():
                self.closed_segments.append(seg)
            else:
                for cell in seg.cells:
                    self.passable[cell] = True

    def _intersections_connected(self) -> bool:
        origin = self.intersections[0]
        seen = {origin}
        queue = deque([origin])
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nxt = (nr, nc)
                if nxt not in seen and self.is_road(nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        return all(i in seen for i in self.intersections)


COLORS = {
    "background": (46, 64, 54),
    "asphalt": (58, 60, 66),
    "lane": (205, 205, 190),
    "intersection": (78, 80, 88),
    "node": (170, 175, 185),
    "barrier": (220, 60, 50),
    "barrier_alt": (240, 240, 240),
    "explored": (80, 160, 255, 70),
    "path": (255, 210, 60),
    "path_done": (90, 220, 120),
    "start": (60, 200, 90),
    "goal": (230, 70, 70),
    "hud_bg": (22, 24, 30),
    "hud_text": (230, 232, 240),
}

BUILDING_PALETTE = [
    (120, 110, 100), (104, 116, 128), (136, 120, 96), (96, 112, 104),
    (128, 104, 112), (112, 124, 140), (140, 132, 116),
]


class CityRenderer:
    """Draws a :class:`CityGrid` (plus route and vehicle overlays) onto a Pygame surface."""

    def __init__(self, city: CityGrid, cell_size: int = 20, hud_height: int = 44) -> None:
        self.city = city
        self.cell_size = cell_size
        self.hud_height = hud_height
        self.width = city.cols * cell_size
        self.height = city.rows * cell_size + hud_height
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.Font(None, 22)
        self.label_font = pygame.font.Font(None, max(14, int(cell_size * 0.9)))
        self._static: Optional[pygame.Surface] = None

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def cell_rect(self, cell: Cell) -> pygame.Rect:
        r, c = cell
        cs = self.cell_size
        return pygame.Rect(c * cs, self.hud_height + r * cs, cs, cs)

    def to_pixel(self, pos: Sequence[float]) -> Tuple[float, float]:
        """Map a (row, col) position (floats allowed) to the pixel at that cell's centre."""
        r, c = pos
        cs = self.cell_size
        return (c + 0.5) * cs, self.hud_height + (r + 0.5) * cs

    def static_layer(self) -> pygame.Surface:
        if self._static is None:
            self._static = self._build_static_layer()
        return self._static

    def _build_static_layer(self) -> pygame.Surface:
        city, cs = self.city, self.cell_size
        surf = pygame.Surface(self.size)
        surf.fill(COLORS["background"])
        rng = random.Random(city.seed)

        bs = city.block_size
        inset = max(2, cs // 6)
        for by in range(city.blocks_y):
            for bx in range(city.blocks_x):
                top_left = self.cell_rect((by * bs + 1, bx * bs + 1))
                block = pygame.Rect(top_left.x, top_left.y, (bs - 1) * cs, (bs - 1) * cs)
                pygame.draw.rect(surf, rng.choice(BUILDING_PALETTE), block.inflate(-2 * inset, -2 * inset), border_radius=4)

        closed_cells = {cell for seg in city.closed_segments for cell in seg.cells}
        for r in range(city.rows):
            for c in range(city.cols):
                cell = (r, c)
                if city.passable[cell] or cell in closed_cells:
                    pygame.draw.rect(surf, COLORS["asphalt"], self.cell_rect(cell))

        for r in range(city.rows):
            for c in range(city.cols):
                cell = (r, c)
                if not city.passable[cell] or city.is_intersection(cell):
                    continue
                cx, cy = self.to_pixel(cell)
                dash = cs * 0.25
                if r % bs == 0:
                    pygame.draw.line(surf, COLORS["lane"], (cx - dash, cy), (cx + dash, cy), 2)
                else:
                    pygame.draw.line(surf, COLORS["lane"], (cx, cy - dash), (cx, cy + dash), 2)

        for cell in closed_cells:
            rect = self.cell_rect(cell).inflate(-cs // 5, -cs // 5)
            pygame.draw.rect(surf, COLORS["barrier_alt"], rect)
            pygame.draw.line(surf, COLORS["barrier"], rect.topleft, rect.bottomright, 3)
            pygame.draw.line(surf, COLORS["barrier"], rect.topright, rect.bottomleft, 3)

        for node in city.intersections:
            pygame.draw.rect(surf, COLORS["intersection"], self.cell_rect(node))
            pygame.draw.circle(surf, COLORS["node"], self.to_pixel(node), max(2, cs // 5))
        return surf

    def draw(
        self,
        surface: pygame.Surface,
        path: Sequence[Cell] = (),
        explored: Sequence[Cell] = (),
        vehicle=None,
        hud_lines: Sequence[str] = (),
    ) -> None:
        surface.blit(self.static_layer(), (0, 0))

        if explored:
            overlay = pygame.Surface(self.size, pygame.SRCALPHA)
            for cell in explored:
                overlay.fill(COLORS["explored"], self.cell_rect(cell))
            surface.blit(overlay, (0, 0))

        if len(path) >= 2:
            done_upto = vehicle.segment + 1 if vehicle is not None else 0
            remaining = [self.to_pixel(p) for p in path[max(0, done_upto - 1):]]
            if vehicle is not None and not vehicle.finished:
                remaining[0] = self.to_pixel(vehicle.position)
            if len(remaining) >= 2:
                pygame.draw.lines(surface, COLORS["path"], False, remaining, max(3, self.cell_size // 5))
            travelled = [self.to_pixel(p) for p in path[:done_upto]]
            if vehicle is not None:
                travelled.append(self.to_pixel(vehicle.position))
            if len(travelled) >= 2:
                pygame.draw.lines(surface, COLORS["path_done"], False, travelled, max(3, self.cell_size // 5))

        self._draw_marker(surface, self.city.start, COLORS["start"], "S")
        self._draw_marker(surface, self.city.goal, COLORS["goal"], "G")

        if vehicle is not None:
            vehicle.draw(surface, self)

        self._draw_hud(surface, hud_lines)

    def _draw_marker(self, surface: pygame.Surface, cell: Cell, color, label: str) -> None:
        center = self.to_pixel(cell)
        radius = int(self.cell_size * 0.6)
        pygame.draw.circle(surface, (255, 255, 255), center, radius + 2)
        pygame.draw.circle(surface, color, center, radius)
        text = self.label_font.render(label, True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=center))

    def _draw_hud(self, surface: pygame.Surface, lines: Sequence[str]) -> None:
        pygame.draw.rect(surface, COLORS["hud_bg"], pygame.Rect(0, 0, self.width, self.hud_height))
        y = 4
        for line in lines[:2]:
            text = self.font.render(line, True, COLORS["hud_text"])
            surface.blit(text, (8, y))
            y += 18
