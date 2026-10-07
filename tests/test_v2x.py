import pygame
import pytest

from smart_nav.environment import CityGrid
from smart_nav.routing import find_route
from smart_nav.simulation import Simulation, SimulationConfig
from smart_nav.v2x_network import CLEARED, INCIDENT, V2XNetwork
from smart_nav.vehicle import Vehicle


def make_vehicle(city):
    return Vehicle(find_route(city.passable, city.start, city.goal).path, speed=6.0)


def test_vehicle_reroute_splices_after_anchor():
    v = Vehicle([(0, 0), (0, 1), (0, 2), (0, 3)], speed=1.0)
    v.update(0.5)
    assert v.anchor == (0, 1) and v.committed_cells == ((0, 0), (0, 1))
    with pytest.raises(ValueError):
        v.reroute([(1, 1), (1, 2)])
    abandoned = v.reroute([(0, 1), (1, 1), (1, 2)])
    assert abandoned == [(0, 1), (0, 2), (0, 3)]
    assert v.path == [(0, 0), (0, 1), (1, 1), (1, 2)]
    v.update(10)
    assert v.finished and v.position == (1.0, 2.0)


def test_report_rejects_unsafe_cells():
    city = CityGrid(seed=2)
    net = V2XNetwork(city, seed=2)
    v = make_vehicle(city)
    assert net.report(city.goal, 0.0, vehicle=v) is None
    assert net.report(city.start, 0.0, vehicle=v) is None
    assert net.report(v.anchor, 0.0, vehicle=v) is None
    assert net.report((1, 1), 0.0, vehicle=v) is None  # building, not road
    assert not net.active


def test_incident_broadcast_and_expiry():
    city = CityGrid(seed=4)
    net = V2XNetwork(city, rate=0.0, seed=4)
    received = []
    net.subscribe(received.append)
    v = make_vehicle(city)
    cell = v.cells_ahead()[5]
    inc = net.report(cell, 1.0, kind="construction", duration=2.0, vehicle=v)
    assert inc is not None and not city.passable[cell]
    assert [m.type for m in received] == [INCIDENT]
    net.update(2.5, 0.1, v)
    assert not city.passable[cell]
    net.update(3.0, 0.1, v)
    assert city.passable[cell] and not net.active
    assert [m.type for m in received] == [INCIDENT, CLEARED]


@pytest.mark.parametrize("seed", range(5))
def test_spontaneous_incidents_keep_goal_reachable(seed):
    city = CityGrid(seed=seed, closure_rate=0.3)
    net = V2XNetwork(city, rate=50.0, duration=None, path_bias=0.5, max_active=15, seed=seed)
    v = make_vehicle(city)
    for i in range(200):
        net.update(i * 0.05, 0.05, v)
    assert len(net.active) == 15
    for inc in net.active.values():
        assert inc.cell not in (city.start, city.goal)
        assert inc.cell not in v.committed_cells
    assert find_route(city.passable, v.anchor, city.goal).found


def test_mid_route_blockage_reroutes_and_arrives(tmp_path):
    """A blockage injected on the remaining route triggers mid-route A* and the vehicle still arrives."""
    sim = Simulation(SimulationConfig(seed=7, speed=8.0, v2x=False), headless=True)
    try:
        sim.run(max_frames=60)
        assert not sim.vehicle.finished
        original_tail = sim.vehicle.cells_ahead()

        assert sim.inject_incident_on_route()
        incident = next(iter(sim.network.active.values()))
        assert incident.cell in original_tail
        assert sim.reroutes == 1
        assert incident.cell not in sim.vehicle.path
        assert sim.remaining_steps() == find_route(sim.city.passable, sim.vehicle.anchor, sim.city.goal).cost
        assert any("RE-ROUTE #1" in e.text for e in sim.log_entries)

        sim.run(max_frames=1, screenshot_path=str(tmp_path / "reroute.png"))
        assert sim.time < sim.banner_until
        assert (tmp_path / "reroute.png").stat().st_size > 0

        result = sim.run(max_frames=10_000, exit_on_arrival=True)
    finally:
        sim.close()
    assert result.reached_goal
    assert result.reroutes == 1
    assert result.safety_violations == 0
    assert sim.vehicle.position == tuple(map(float, sim.city.goal))


def test_clearance_restores_optimal_route():
    sim = Simulation(SimulationConfig(seed=3, speed=6.0, v2x=False), headless=True)
    try:
        sim.run(max_frames=30)
        assert sim.inject_incident_on_route()
        incident = next(iter(sim.network.active.values()))
        sim.network.clear(incident.id, sim.time)
        assert sim.city.passable[incident.cell]
        assert sim.remaining_steps() == find_route(sim.city.passable, sim.vehicle.anchor, sim.city.goal).cost
        assert any("CLEAR" in e.text for e in sim.log_entries)
    finally:
        sim.close()


@pytest.mark.parametrize("seed", range(8))
def test_stress_random_incidents_headless(seed):
    cfg = SimulationConfig(
        seed=seed, speed=6.0, random_endpoints=True, incident_rate=1.5,
        incident_duration=(2.0, 6.0), path_bias=0.9, max_incidents=8,
    )
    sim = Simulation(cfg, headless=True)
    try:
        assert pygame.display.get_driver() == "dummy"
        result = sim.run(max_frames=20_000, exit_on_arrival=True)
    finally:
        sim.close()
    assert result.reached_goal
    assert result.safety_violations == 0
    assert result.incidents > 0
    assert result.reroutes >= 1
