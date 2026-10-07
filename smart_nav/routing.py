"""A* (A-Star) routing engine.

The core :func:`a_star` works on any graph described by a ``neighbors``
function, a ``heuristic`` and an edge ``cost``. :func:`find_route` wraps it
for 4-connected occupancy grids such as :class:`smart_nav.environment.CityGrid`.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

GRID_MOVES: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class RouteResult:
    """Outcome of a search: the path (empty if unreachable), its cost and the expansion order."""

    path: List[Hashable]
    cost: float
    explored: List[Hashable] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.path)


def manhattan(a: Cell, b: Cell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit_cost(_a: Hashable, _b: Hashable) -> float:
    return 1.0


def a_star(
    start: Hashable,
    goal: Hashable,
    neighbors: Callable[[Hashable], Iterable[Hashable]],
    heuristic: Callable[[Hashable, Hashable], float],
    cost: Callable[[Hashable, Hashable], float] = unit_cost,
) -> RouteResult:
    """Return the lowest-cost path from ``start`` to ``goal``.

    Optimal as long as ``heuristic`` is admissible (never overestimates).
    Ties on f-score are broken towards lower h (closer to the goal), then FIFO.
    """
    counter = itertools.count()
    open_heap: List[Tuple[float, float, int, Hashable]] = []
    h0 = heuristic(start, goal)
    heapq.heappush(open_heap, (h0, h0, next(counter), start))
    came_from: Dict[Hashable, Hashable] = {}
    g_score: Dict[Hashable, float] = {start: 0.0}
    closed: set = set()
    explored: List[Hashable] = []

    while open_heap:
        _f, _h, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)

        if current == goal:
            return RouteResult(_reconstruct(came_from, current), g_score[current], explored)

        for nxt in neighbors(current):
            if nxt in closed:
                continue
            step = cost(current, nxt)
            if step < 0:
                raise ValueError("A* requires non-negative edge costs")
            tentative = g_score[current] + step
            if tentative < g_score.get(nxt, math.inf):
                came_from[nxt] = current
                g_score[nxt] = tentative
                h = heuristic(nxt, goal)
                heapq.heappush(open_heap, (tentative + h, h, next(counter), nxt))

    return RouteResult([], math.inf, explored)


def _reconstruct(came_from: Dict[Hashable, Hashable], node: Hashable) -> List[Hashable]:
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def grid_neighbors(passable: np.ndarray) -> Callable[[Cell], Iterator[Cell]]:
    rows, cols = passable.shape

    def neighbors(cell: Cell) -> Iterator[Cell]:
        r, c = cell
        for dr, dc in GRID_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc]:
                yield (nr, nc)

    return neighbors


def find_route(passable: np.ndarray, start: Cell, goal: Cell) -> RouteResult:
    """A* over a boolean grid (``True`` = drivable), 4-connected, unit step cost."""
    passable = np.asarray(passable, dtype=bool)
    for name, cell in (("start", start), ("goal", goal)):
        if not _in_bounds(passable, cell):
            raise ValueError(f"{name} {cell} is outside the grid {passable.shape}")
        if not passable[cell]:
            raise ValueError(f"{name} {cell} is not on a drivable cell")
    return a_star(tuple(start), tuple(goal), grid_neighbors(passable), manhattan)


def _in_bounds(grid: np.ndarray, cell: Optional[Cell]) -> bool:
    return (
        cell is not None
        and len(cell) == 2
        and 0 <= cell[0] < grid.shape[0]
        and 0 <= cell[1] < grid.shape[1]
    )
