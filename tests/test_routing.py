import heapq
import math
import random
from collections import deque

import numpy as np
import pytest

from smart_nav.routing import a_star, find_route, manhattan


def bfs_distance(grid, start, goal):
    rows, cols = grid.shape
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return None


def assert_valid_path(grid, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1
    assert all(grid[p] for p in path)


def test_straight_line():
    grid = np.ones((1, 5), dtype=bool)
    route = find_route(grid, (0, 0), (0, 4))
    assert route.path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert route.cost == 4


def test_detours_around_wall():
    grid = np.ones((5, 5), dtype=bool)
    grid[0:4, 2] = False
    route = find_route(grid, (0, 0), (0, 4))
    assert_valid_path(grid, route.path, (0, 0), (0, 4))
    assert route.cost == 12


def test_start_equals_goal():
    grid = np.ones((3, 3), dtype=bool)
    route = find_route(grid, (1, 1), (1, 1))
    assert route.path == [(1, 1)] and route.cost == 0


def test_unreachable_returns_empty():
    grid = np.ones((3, 3), dtype=bool)
    grid[:, 1] = False
    route = find_route(grid, (0, 0), (0, 2))
    assert not route.found and route.cost == math.inf


@pytest.mark.parametrize("cell", [(-1, 0), (0, 9), (1, 1)])
def test_invalid_endpoints_raise(cell):
    grid = np.ones((3, 3), dtype=bool)
    grid[1, 1] = False
    with pytest.raises(ValueError):
        find_route(grid, cell, (0, 0))


@pytest.mark.parametrize("seed", range(25))
def test_matches_bfs_on_random_grids(seed):
    rng = np.random.default_rng(seed)
    grid = rng.random((15, 20)) > 0.3
    free = list(zip(*np.nonzero(grid)))
    start, goal = (tuple(int(v) for v in free[i]) for i in rng.choice(len(free), 2, replace=False))
    expected = bfs_distance(grid, start, goal)
    route = find_route(grid, start, goal)
    if expected is None:
        assert not route.found
    else:
        assert route.cost == expected
        assert_valid_path(grid, route.path, start, goal)


def test_weighted_graph_matches_dijkstra():
    rng = random.Random(7)
    nodes = [(rng.randint(0, 50), rng.randint(0, 50)) for _ in range(60)]
    nodes = list(dict.fromkeys(nodes))
    weights = {}
    for a in nodes:
        for b in sorted(nodes, key=lambda n: manhattan(a, n))[1:5]:
            if (a, b) not in weights:
                weights[(a, b)] = weights[(b, a)] = manhattan(a, b) * rng.uniform(1.0, 2.0)
    edges = {n: [] for n in nodes}
    for (a, b), w in weights.items():
        edges[a].append((b, w))

    def dijkstra(src, dst):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                return d
            if d > dist[u]:
                continue
            for v, w in edges[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        return math.inf

    for _ in range(20):
        s, g = rng.sample(nodes, 2)
        route = a_star(s, g, lambda n: (v for v, _ in edges[n]), manhattan, lambda a, b: weights[(a, b)])
        assert route.cost == pytest.approx(dijkstra(s, g))
        if route.found:
            assert sum(weights[(a, b)] for a, b in zip(route.path, route.path[1:])) == pytest.approx(route.cost)
