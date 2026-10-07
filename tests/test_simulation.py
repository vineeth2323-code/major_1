import numpy as np
import pygame
import pytest

from smart_nav.environment import CityGrid, CityRenderer
from smart_nav.routing import find_route
from smart_nav.simulation import Simulation, SimulationConfig
from smart_nav.vehicle import Vehicle


@pytest.mark.parametrize("seed", range(10))
def test_city_grid_layout(seed):
    city = CityGrid(blocks_x=6, blocks_y=4, block_size=4, closure_rate=0.3, seed=seed)
    assert city.shape == (17, 25)
    assert len(city.intersections) == 5 * 7
    assert all(city.is_road(i) for i in city.intersections)
    assert not city.passable[1, 1]
    assert city.closed_segments
    for seg in city.closed_segments:
        assert not any(city.passable[c] for c in seg.cells)
    for node in city.intersections:
        assert find_route(city.passable, city.start, node).found


def test_city_grid_is_deterministic_per_seed():
    a, b = CityGrid(seed=42), CityGrid(seed=42)
    assert np.array_equal(a.passable, b.passable)


def test_random_endpoints_are_far_apart_intersections():
    city = CityGrid(seed=3)
    city.randomize_endpoints()
    assert city.start != city.goal
    assert city.is_intersection(city.start) and city.is_intersection(city.goal)


def test_vehicle_follows_path_to_goal():
    path = [(0, 0), (0, 1), (0, 2), (1, 2)]
    v = Vehicle(path, speed=2.0)
    v.update(0.25)
    assert v.position == pytest.approx((0.0, 0.5))
    assert v.heading == pytest.approx(0.0)
    v.update(1.0)
    assert v.position == pytest.approx((0.5, 2.0))
    assert v.heading == pytest.approx(90.0)
    v.update(10)
    assert v.finished and v.position == (1.0, 2.0) and v.completion == 1.0


def test_renderer_draws_headless():
    pygame.display.init()
    pygame.font.init()
    city = CityGrid(seed=1)
    renderer = CityRenderer(city, cell_size=16)
    surface = pygame.Surface(renderer.size)
    route = find_route(city.passable, city.start, city.goal)
    renderer.draw(surface, route.path, route.explored, Vehicle(route.path), ["hud"])
    pixels = pygame.surfarray.array3d(surface)
    assert len(np.unique(pixels.reshape(-1, 3), axis=0)) > 10
    pygame.quit()


def test_full_simulation_headless(tmp_path):
    shot = tmp_path / "frame.png"
    sim = Simulation(SimulationConfig(seed=5, speed=30.0), headless=True)
    try:
        assert pygame.display.get_driver() == "dummy"
        assert sim.screen.get_size() == sim.renderer.size
        result = sim.run(max_frames=5000, exit_on_arrival=True, screenshot_path=str(shot))
    finally:
        sim.close()
    assert result.reached_goal
    assert result.frames < 5000
    assert result.path_cost == result.path_length - 1
    assert shot.exists() and shot.stat().st_size > 0
