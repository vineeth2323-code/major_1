import math

import numpy as np
import pygame
import pytest

import main
from smart_nav.analytics import EnergyModel, RunAnalytics
from smart_nav.simulation import Simulation, SimulationConfig

DT = 1 / 60


def test_constant_cruise_energy_matches_road_load():
    a = RunAnalytics(fps=60, cruise_speed=6.0)
    for i in range(600):
        a.record((i + 1) * DT, DT, 6.0, 1.0)
    e = a.energy_summary()
    m = EnergyModel()
    assert e["distance_m"] == pytest.approx(6.0 * m.cell_m * 10.0)
    assert e["braking_loss_wh"] == 0 and e["idle_wh"] == 0
    assert e["traction_wh"] == pytest.approx(m.road_load(15.0, 150.0) / m.drivetrain_eff / 3600)
    assert e["efficiency_pct"] == pytest.approx(100.0)
    assert e["hard_brakes"] == 0 and e["stops"] == 0


def test_braking_and_idling_are_penalised():
    a = RunAnalytics(fps=60, cruise_speed=6.0)
    t, v = 0.0, 6.0
    while v > 0:
        v = max(0.0, v - 12.0 * DT)
        t += DT
        a.record(t, DT, v, 1.0)
    for _ in range(120):
        t += DT
        a.record(t, DT, 0.0, 1.0)
    e = a.energy_summary()
    kinetic_wh = 0.5 * 1500 * 15.0**2 / 3600
    assert kinetic_wh * 0.9 < e["braking_loss_wh"] < kinetic_wh
    assert e["idle_time"] == pytest.approx(2.0, abs=DT)
    assert e["idle_wh"] == pytest.approx(1200 * 2.0 / 3600, rel=0.02)
    assert e["stops"] == 1 and e["hard_brakes"] == 1
    assert e["efficiency_pct"] < 100


def test_no_idle_charged_after_mission_ends():
    a = RunAnalytics()
    for i in range(60):
        a.record(i * DT, DT, 0.0, 1.0, active=False)
    assert a.energy.idle_time == 0


def test_tracking_rmse_over_time():
    a = RunAnalytics()
    for i in range(600):
        errs = [(0.5, 0.2), (0.5, 0.2)] if i % 3 == 0 else []
        a.record((i + 1) * DT, DT, 6.0, 1.0, errors=errs)
    trk = a.tracking()
    assert trk["raw_rmse"] == pytest.approx(0.5) and trk["kf_rmse"] == pytest.approx(0.2)
    assert trk["reduction_pct"] == pytest.approx(60.0)
    t, raw, kf = a.rolling_rmse(1.0)
    assert len(t) == 600 and np.nanmax(np.abs(raw - 0.5)) < 1e-9 and np.nanmax(np.abs(kf - 0.2)) < 1e-9
    buckets = a.tracking_buckets(5.0)
    assert [round(b[0]) for b in buckets] == [0, 5] and all(b[3] > b[4] for b in buckets)


def test_latency_statistics():
    a = RunAnalytics(fps=60)
    for i, ms in enumerate([30.0] + [4.0] * 98 + [20.0]):
        a.record(i * DT, DT, 6.0, ms)
        a.add_render_time(1.0)
    lat = a.latency()
    assert lat["avg"] == pytest.approx((30 + 4 * 98 + 20) / 100 + 1.0)
    assert lat["peak"] == 31.0 and lat["peak_steady"] == 21.0
    assert lat["p50"] == pytest.approx(5.0)
    assert lat["over_budget_pct"] == pytest.approx(2.0)


@pytest.fixture(scope="module")
def reported_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    sim = Simulation(SimulationConfig(seed=7), headless=True)
    try:
        result = sim.run(exit_on_arrival=True, report_dir=str(out))
        yield sim, result, out
    finally:
        sim.close()


def test_run_exports_markdown_report_and_chart(reported_run):
    sim, result, out = reported_run
    assert result.reached_goal and result.outcome == "ARRIVED"
    report, chart = result.report_paths
    text = (out / "performance_report.md").read_text()
    assert report.endswith("performance_report.md") and chart.endswith("analytics_summary.png")
    for heading in ("Computational latency", "Kalman filter tracking accuracy", "Energy efficiency", "Events"):
        assert f"## {heading}" in text
    assert f"| V2X re-routes | {result.reroutes} |" in text
    assert f"| Safety-brake activations | {result.brake_events} |" in text
    assert "![Analytics summary](analytics_summary.png)" in text
    img = pygame.image.load(chart)
    assert img.get_width() >= 1000 and img.get_height() >= 800


def test_run_metrics_are_consistent(reported_run):
    sim, result, _ = reported_run
    a = sim.analytics
    assert a.frames == result.frames > 0
    counts = a.counts()
    assert counts.get("reroute", 0) == result.reroutes > 0
    assert counts.get("brake", 0) == result.brake_events
    assert counts.get("incident", 0) == result.incidents
    trk = a.tracking()
    assert trk["kf_rmse"] == pytest.approx(result.kf_rmse) and trk["kf_rmse"] < 0.75 * trk["raw_rmse"]
    lat = a.latency()
    # The simulation step itself (routing, V2X, perception, KF, control) is far inside the 60 FPS budget.
    assert lat["step_avg"] < a.budget_ms / 4
    assert lat["p50"] < a.budget_ms
    assert 0 < result.efficiency_pct <= 100 and result.energy_wh > 0
    assert len(a.speeds) == result.frames and max(a.speeds) <= sim.vehicle.cruise_speed + 1e-9


def test_cli_writes_report(tmp_path, capsys):
    assert main.main(["--headless", "--seed", "3", "--quiet", "--report-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "outcome=ARRIVED" in out and "report:" in out
    assert (tmp_path / "performance_report.md").exists() and (tmp_path / "analytics_summary.png").exists()
    assert math.isfinite(float(out.split("avg_frame_ms=")[1].split()[0]))
