"""Lock-up hardening: unreachable or occupied goal and the run watchdog end in a SAFE HALT."""

import numpy as np
import pytest

import main
from smart_nav.simulation import (
    GOAL_ISOLATED, GOAL_OCCUPIED, SAFE_HALT, WATCHDOG, Simulation, SimulationConfig,
)

DT = 1 / 60


def make_sim(**kw):
    cfg = dict(seed=7, incident_rate=0.0)
    cfg.update(kw)
    return Simulation(SimulationConfig(**cfg), headless=True)


def advance(sim, seconds):
    for _ in range(int(seconds / DT)):
        sim.step(DT)


def logs(sim, level=None):
    return [e.text for e in sim.log_entries if level is None or e.level == level]


def test_v2x_isolation_guard_and_override():
    sim = make_sim()
    try:
        gr, gc = sim.city.goal
        neighbours = [c for c in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1))
                      if 0 <= c[0] < sim.city.rows and 0 <= c[1] < sim.city.cols and sim.city.passable[c]]
        for c in neighbours[:-1]:
            assert sim.network.report(c, 0.0, vehicle=sim.vehicle)
        assert not sim.network.can_block(neighbours[-1], sim.vehicle)
        assert sim.network.can_block(neighbours[-1], sim.vehicle, allow_isolation=True)
    finally:
        sim.close()


def test_isolated_goal_ends_in_safe_halt_not_a_hang():
    sim = make_sim(halt_patience=3.0)
    try:
        advance(sim, 1.0)
        assert sim.isolate_goal() >= 1
        result = sim.run(max_frames=int(60 / DT), exit_on_arrival=True, render=False)
        assert result.outcome == SAFE_HALT and result.halt_reason == GOAL_ISOLATED
        assert not result.reached_goal and result.safe_halts == 1
        assert result.sim_time < 1.0 + 3.0 + 2.0  # halted promptly, then waited halt_patience
        assert sim.vehicle.speed == 0 and sim.vehicle.pulled_over
        assert result.safety_violations == 0 and result.collisions == 0
        crit = logs(sim, "critical")
        assert any("NO ROUTE" in t for t in crit) and any("SAFE HALT" in t for t in crit)
        assert sim.analytics.counts().get("halt") == 1
    finally:
        sim.close()


def test_isolated_goal_recovers_when_closures_clear():
    sim = make_sim(halt_patience=60.0)
    try:
        advance(sim, 1.0)
        sim.isolate_goal(duration=4.0)
        assert sim.halt_reason == GOAL_ISOLATED
        result = sim.run(max_frames=int(60 / DT), exit_on_arrival=True, render=False)
        assert result.outcome == "ARRIVED" and result.reached_goal
        assert result.safe_halts == 1 and result.halt_reason is None
        assert any(t.startswith("RECOVERED") for t in logs(sim, "clear"))
        assert result.safety_violations == 0 and result.collisions == 0
    finally:
        sim.close()


def test_obstacle_on_goal_marker_causes_safe_halt_then_resume():
    sim = make_sim(halt_patience=1000.0)
    try:
        ob = sim.occupy_goal()
        for _ in range(int(60 / DT)):
            sim.step(DT)
            if sim.halt_reason:
                break
        assert sim.halt_reason == GOAL_OCCUPIED
        assert sim.collisions == 0 and not sim.arrived
        gap = np.hypot(*(np.asarray(sim.vehicle.position) - np.asarray(sim.city.goal)))
        assert 0.6 < gap < 3.0
        assert any("occupied by stationary pedestrian" in t for t in logs(sim, "critical"))

        advance(sim, 3.0)
        assert sim.halt_reason == GOAL_OCCUPIED and sim.vehicle.speed < 0.3 and sim.collisions == 0

        sim.perception.obstacles.remove(ob.id)
        for _ in range(int(30 / DT)):
            sim.step(DT)
            if sim.arrived:
                break
        assert sim.arrived and sim.halt_reason is None and sim.collisions == 0
        assert any("RECOVERED" in t for t in logs(sim, "clear"))
    finally:
        sim.close()


def test_obstacle_on_goal_run_terminates():
    sim = make_sim(halt_patience=2.0)
    try:
        sim.occupy_goal()
        result = sim.run(max_frames=int(90 / DT), exit_on_arrival=True, render=False)
        assert result.outcome == SAFE_HALT and result.halt_reason == GOAL_OCCUPIED
        assert result.frames < int(90 / DT) and result.collisions == 0
    finally:
        sim.close()


def test_watchdog_stops_runaway_mission():
    sim = make_sim(max_sim_time=2.0)
    try:
        result = sim.run(max_frames=int(30 / DT), exit_on_arrival=True, render=False)
        assert result.outcome == SAFE_HALT and result.halt_reason == WATCHDOG
        assert sim.vehicle.speed == 0 and result.sim_time < 4.0
        assert any("max_sim_time" in t for t in logs(sim, "critical"))
    finally:
        sim.close()


def test_safe_halt_renders_headless():
    sim = make_sim(halt_patience=1.0)
    try:
        advance(sim, 1.0)
        sim.isolate_goal()
        advance(sim, 0.5)
        sim.render()
        assert "SAFE HALT" in sim.hud_lines()[0]
    finally:
        sim.close()


@pytest.mark.parametrize("case,reason", [("isolate-goal", "goal isolated"), ("occupy-goal", "goal occupied")])
def test_cli_edge_cases_finish_gracefully(case, reason, capsys):
    assert main.main(["--headless", "--seed", "7", "--quiet", "--no-v2x", "--edge-case", case, "--max-time", "120"]) == 0
    out = capsys.readouterr().out
    assert "outcome=SAFE_HALT" in out and f"halt_reason='{reason}'" in out
