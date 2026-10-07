"""Run-level performance analytics: frame latency, tracking accuracy, energy use and events.

The simulation calls :meth:`RunAnalytics.record` once per step (and :meth:`add_render_time` per
drawn frame). At the end of a run :meth:`RunAnalytics.write_report` exports a Markdown summary and
:meth:`RunAnalytics.plot` a consolidated matplotlib chart.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GRAVITY = 9.81
STOP_SPEED = 0.05  # cells/s; below this the vehicle counts as stationary


@dataclass
class EnergyModel:
    """Longitudinal vehicle energy model (per frame), in SI units.

    Positive tractive demand (kinetic energy gained + rolling resistance + aerodynamic drag) is drawn
    through the drivetrain; negative demand is dissipated by the brakes, of which ``regen_eff`` is
    recovered. Standing still before arrival burns ``idle_power`` (engine idle / auxiliaries).
    """

    cell_m: float = 2.5
    mass: float = 1500.0
    crr: float = 0.012
    cda: float = 0.65
    air_density: float = 1.2
    drivetrain_eff: float = 0.85
    regen_eff: float = 0.0
    idle_power: float = 1200.0

    def road_load(self, v: float, dist: float) -> float:
        """Energy (J) to overcome rolling resistance and drag over ``dist`` metres at ``v`` m/s."""
        return self.mass * GRAVITY * self.crr * dist + 0.5 * self.air_density * self.cda * v * v * dist


@dataclass
class EnergyTotals:
    traction_j: float = 0.0
    braking_loss_j: float = 0.0
    regen_j: float = 0.0
    idle_j: float = 0.0
    distance_m: float = 0.0
    idle_time: float = 0.0
    hard_brakes: int = 0
    stops: int = 0

    @property
    def total_j(self) -> float:
        return self.traction_j + self.idle_j - self.regen_j


@dataclass
class Event:
    time: float
    kind: str
    label: str = ""


@dataclass
class RunAnalytics:
    fps: int = 60
    energy_model: EnergyModel = field(default_factory=EnergyModel)
    cruise_speed: float = 6.0
    hard_brake_decel: float = 6.0  # cells/s^2

    def __post_init__(self) -> None:
        self.times: List[float] = []
        self.speeds: List[float] = []
        self.target_speeds: List[float] = []
        self.states: List[str] = []
        self.step_ms: List[float] = []
        self.render_ms: List[float] = []
        self.raw_sq: List[float] = []
        self.kf_sq: List[float] = []
        self.err_n: List[int] = []
        self.events: List[Event] = []
        self.energy = EnergyTotals()
        self._prev_speed: Optional[float] = None
        self._hard_braking = False
        self._stopped = False

    # ------------------------------------------------------------------ recording
    @property
    def budget_ms(self) -> float:
        return 1000.0 / self.fps

    def record(
        self,
        t: float,
        dt: float,
        speed: float,
        compute_ms: float,
        errors: Iterable[Tuple[float, float]] = (),
        state: str = "CRUISE",
        target_speed: Optional[float] = None,
        active: bool = True,
    ) -> None:
        """One simulation step. ``errors`` are (raw, filtered) position errors measured this frame;
        ``active`` is False once the mission is over (no idling is charged after arrival)."""
        errs = list(errors)
        self.times.append(t)
        self.speeds.append(speed)
        self.target_speeds.append(speed if target_speed is None else target_speed)
        self.states.append(state)
        self.step_ms.append(compute_ms)
        self.render_ms.append(0.0)
        self.raw_sq.append(sum(r * r for r, _ in errs))
        self.kf_sq.append(sum(k * k for _, k in errs))
        self.err_n.append(len(errs))
        prev = speed if self._prev_speed is None else self._prev_speed
        self._integrate_energy(prev, speed, dt, active)
        self._prev_speed = speed

    def add_render_time(self, ms: float) -> None:
        if self.render_ms:
            self.render_ms[-1] += ms

    def event(self, t: float, kind: str, label: str = "") -> None:
        self.events.append(Event(t, kind, label))

    def _integrate_energy(self, v0_cells: float, v1_cells: float, dt: float, active: bool) -> None:
        m, e = self.energy_model, self.energy
        v0, v1 = v0_cells * m.cell_m, v1_cells * m.cell_m
        v_avg = 0.5 * (v0 + v1)
        dist = v_avg * dt
        demand = 0.5 * m.mass * (v1 * v1 - v0 * v0) + m.road_load(v_avg, dist)
        if demand >= 0:
            e.traction_j += demand / m.drivetrain_eff
        else:
            e.braking_loss_j += -demand * (1 - m.regen_eff)
            e.regen_j += -demand * m.regen_eff
        e.distance_m += dist
        stopped = v1_cells < STOP_SPEED
        if active and stopped:
            e.idle_j += m.idle_power * dt
            e.idle_time += dt
        if active and stopped and not self._stopped and v0_cells >= STOP_SPEED:
            e.stops += 1
        self._stopped = stopped
        decel = (v0_cells - v1_cells) / dt if dt > 0 else 0.0
        hard = decel > self.hard_brake_decel
        if hard and not self._hard_braking:
            e.hard_brakes += 1
        self._hard_braking = hard

    # ------------------------------------------------------------------ summaries
    @property
    def frames(self) -> int:
        return len(self.times)

    def frame_ms(self) -> np.ndarray:
        return np.asarray(self.step_ms) + np.asarray(self.render_ms)

    def latency(self) -> Dict[str, float]:
        ms = self.frame_ms()
        if not len(ms):
            return {k: 0.0 for k in ("avg", "p50", "p95", "peak", "peak_steady", "over_budget_pct", "headroom")}
        steady = ms[1:] if len(ms) > 1 else ms
        avg = float(ms.mean())
        return {
            "avg": avg,
            "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)),
            "peak": float(ms.max()),
            "peak_steady": float(steady.max()),
            "over_budget_pct": float((ms > self.budget_ms).mean() * 100),
            "headroom": self.budget_ms / avg if avg > 0 else float("inf"),
            "step_avg": float(np.mean(self.step_ms)),
            "render_avg": float(np.mean(self.render_ms)),
        }

    def tracking(self) -> Dict[str, float]:
        n = int(np.sum(self.err_n))
        if not n:
            return {"samples": 0, "raw_rmse": 0.0, "kf_rmse": 0.0, "reduction_pct": 0.0}
        raw = math.sqrt(float(np.sum(self.raw_sq)) / n)
        kf = math.sqrt(float(np.sum(self.kf_sq)) / n)
        return {"samples": n, "raw_rmse": raw, "kf_rmse": kf, "reduction_pct": 100 * (1 - kf / raw) if raw else 0.0}

    def rolling_rmse(self, window: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(times, raw_rmse, kf_rmse) over a trailing ``window`` seconds (NaN where no samples)."""
        t = np.asarray(self.times)
        if not len(t):
            return t, t, t
        cs = [np.concatenate([[0.0], np.cumsum(a)]) for a in (self.raw_sq, self.kf_sq, self.err_n)]
        lo = np.searchsorted(t, t - window, side="right")
        hi = np.arange(1, len(t) + 1)
        raw_s, kf_s, n = (c[hi] - c[lo] for c in cs)
        with np.errstate(invalid="ignore", divide="ignore"):
            return t, np.where(n > 0, np.sqrt(raw_s / n), np.nan), np.where(n > 0, np.sqrt(kf_s / n), np.nan)

    def tracking_buckets(self, width: float = 5.0) -> List[Tuple[float, float, int, float, float]]:
        """Tracking accuracy per ``width``-second interval: (start, end, samples, raw, kf)."""
        out = []
        if not self.times:
            return out
        t = np.asarray(self.times)
        raw, kf, n = map(np.asarray, (self.raw_sq, self.kf_sq, self.err_n))
        for start in np.arange(0.0, t[-1] + 1e-9, width):
            sel = (t > start) & (t <= start + width)
            cnt = int(n[sel].sum())
            if cnt:
                out.append((start, min(start + width, t[-1]), cnt, math.sqrt(raw[sel].sum() / cnt), math.sqrt(kf[sel].sum() / cnt)))
        return out

    def energy_summary(self) -> Dict[str, float]:
        m, e = self.energy_model, self.energy
        km = e.distance_m / 1000.0
        cruise = self.cruise_speed * m.cell_m
        baseline = m.road_load(cruise, e.distance_m) / m.drivetrain_eff
        total = e.total_j
        return {
            "distance_m": e.distance_m,
            "total_wh": total / 3600.0,
            "traction_wh": e.traction_j / 3600.0,
            "braking_loss_wh": e.braking_loss_j / 3600.0,
            "idle_wh": e.idle_j / 3600.0,
            "baseline_wh": baseline / 3600.0,
            "wh_per_km": total / 3600.0 / km if km > 0 else 0.0,
            "le_per_100km": total / 3.6e6 / 8.9 / km * 100 if km > 0 else 0.0,
            "efficiency_pct": 100.0 * baseline / total if total > 0 else 100.0,
            "idle_time": e.idle_time,
            "hard_brakes": e.hard_brakes,
            "stops": e.stops,
        }

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in self.events:
            out[ev.kind] = out.get(ev.kind, 0) + 1
        return out

    # ------------------------------------------------------------------ export
    def write_report(self, path: str, meta: Optional[Dict[str, object]] = None, chart: Optional[str] = None) -> str:
        meta = dict(meta or {})
        lat, trk, en, cnt = self.latency(), self.tracking(), self.energy_summary(), self.counts()
        realtime = lat["p95"] <= self.budget_ms
        lines = ["# Smart Navigation - Performance Report", ""]
        if meta:
            lines += ["## Run", "", "| Parameter | Value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in meta.items()]
            lines.append("")
        lines += [
            "## Computational latency",
            "",
            f"Real-time budget at {self.fps} FPS: **{self.budget_ms:.2f} ms/frame**. "
            f"Status: **{'WITHIN BUDGET' if realtime else 'OVER BUDGET'}** (95th percentile).",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Frames | {self.frames} |",
            f"| Average frame (step + render) | {lat['avg']:.3f} ms |",
            f"| - simulation step | {lat.get('step_avg', 0):.3f} ms |",
            f"| - rendering | {lat.get('render_avg', 0):.3f} ms |",
            f"| Median | {lat['p50']:.3f} ms |",
            f"| 95th percentile | {lat['p95']:.3f} ms |",
            f"| Peak | {lat['peak']:.3f} ms |",
            f"| Peak excluding first (warm-up) frame | {lat['peak_steady']:.3f} ms |",
            f"| Frames over budget | {lat['over_budget_pct']:.2f} % |",
            f"| Headroom (budget / average) | {lat['headroom']:.1f}x |",
            "",
            "## Kalman filter tracking accuracy",
            "",
        ]
        if trk["samples"]:
            lines += [
                f"Over {trk['samples']} confirmed-track detections the filter reduced position error by "
                f"**{trk['reduction_pct']:.1f} %** relative to the raw sensor.",
                "",
                "| Metric | Raw sensor | Kalman filter |",
                "|---|---|---|",
                f"| Position RMSE (cells) | {trk['raw_rmse']:.3f} | {trk['kf_rmse']:.3f} |",
                "",
                "| Interval (s) | Samples | Raw RMSE | KF RMSE | Reduction |",
                "|---|---|---|---|---|",
            ]
            for a, b, n, r, k in self.tracking_buckets():
                lines.append(f"| {a:.0f}-{b:.1f} | {n} | {r:.3f} | {k:.3f} | {100 * (1 - k / r):.0f} % |")
        else:
            lines.append("No tracking samples (perception disabled or no obstacle came into sensor range).")
        m = self.energy_model
        lines += [
            "",
            "## Energy efficiency",
            "",
            f"Model: {m.mass:.0f} kg vehicle, 1 cell = {m.cell_m} m, Crr {m.crr}, CdA {m.cda} m^2, drivetrain "
            f"{m.drivetrain_eff:.0%}, regen {m.regen_eff:.0%}, idle {m.idle_power:.0f} W. Efficiency compares "
            "actual energy with an ideal constant-cruise drive over the same distance.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Distance | {en['distance_m']:.0f} m |",
            f"| Total energy | {en['total_wh']:.1f} Wh |",
            f"| - traction | {en['traction_wh']:.1f} Wh |",
            f"| - idling | {en['idle_wh']:.1f} Wh ({en['idle_time']:.1f} s) |",
            f"| Lost to braking | {en['braking_loss_wh']:.1f} Wh |",
            f"| Consumption | {en['wh_per_km']:.0f} Wh/km ({en['le_per_100km']:.2f} L-equivalent/100 km) |",
            f"| Ideal cruise energy | {en['baseline_wh']:.1f} Wh |",
            f"| **Efficiency score** | **{en['efficiency_pct']:.1f} %** |",
            f"| Hard-brake events | {en['hard_brakes']} |",
            f"| Full stops | {en['stops']} |",
            "",
            "## Events",
            "",
            "| Event | Count |",
            "|---|---|",
            f"| V2X incidents | {cnt.get('incident', 0)} |",
            f"| V2X re-routes | {cnt.get('reroute', 0)} |",
            f"| Safety-brake activations | {cnt.get('brake', 0)} |",
            f"| Safe halts | {cnt.get('halt', 0)} |",
            f"| Collisions | {cnt.get('collision', 0)} |",
        ]
        if self.events:
            lines += ["", "### Timeline", "", "| t (s) | Event | Detail |", "|---|---|---|"]
            lines += [f"| {e.time:.2f} | {e.kind} | {e.label} |" for e in self.events]
        if chart:
            lines += ["", "## Chart", "", f"![Analytics summary]({os.path.relpath(chart, os.path.dirname(os.path.abspath(path)))})"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def plot(self, path: str, title: str = "Smart Navigation - run analytics") -> str:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 9), dpi=100)
        FigureCanvasAgg(fig)
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        t, raw, kf = self.rolling_rmse()

        if np.isfinite(raw).any():
            trk = self.tracking()
            ax1.plot(t, raw, color="#e0413a", lw=1.4, label=f"raw sensor (RMSE {trk['raw_rmse']:.3f})")
            ax1.plot(t, kf, color="#1f9d55", lw=1.6, label=f"Kalman filter (RMSE {trk['kf_rmse']:.3f})")
            ax1.fill_between(t, kf, raw, where=raw >= kf, color="#1f9d55", alpha=0.15, interpolate=True,
                             label=f"KF error gap ({trk['reduction_pct']:.0f} % lower)")
            ax1.legend(loc="upper right", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "no tracking samples", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_ylabel("position error\n(cells, 1 s RMSE)")
        ax1.set_title("Tracking accuracy: Kalman filter vs raw sensor", fontsize=10)
        ax1.grid(alpha=0.3)

        self._shade_states(ax2, t)
        ax2.plot(t, self.speeds, color="#2563eb", lw=1.6, label="vehicle speed")
        ax2.plot(t, self.target_speeds, color="#64748b", lw=1.0, ls="--", label="commanded speed")
        styles = {"reroute": ("#f59e0b", "V2X re-route"), "halt": ("#7c3aed", "safe halt"), "collision": ("black", "collision")}
        seen = set()
        for ev in self.events:
            if ev.kind in styles:
                color, label = styles[ev.kind]
                ax2.axvline(ev.time, color=color, ls=":", lw=1.5, label=None if ev.kind in seen else label)
                seen.add(ev.kind)
        ax2.set_ylabel("speed (cells/s)")
        ax2.set_ylim(bottom=0)
        ax2.set_title("Vehicle speed profile (red = safety brake, orange = caution)", fontsize=10)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(alpha=0.3)

        ms = self.frame_ms()
        ax3.plot(t, ms, color="#0f766e", lw=0.8, label="frame compute time")
        ax3.axhline(self.budget_ms, color="#e0413a", ls="--", lw=1.2, label=f"real-time budget ({self.budget_ms:.1f} ms)")
        lat = self.latency()
        ax3.set_title(f"Per-frame latency: avg {lat['avg']:.2f} ms, p95 {lat['p95']:.2f} ms", fontsize=10)
        ax3.set_ylabel("ms / frame")
        ax3.set_xlabel("simulation time (s)")
        ax3.set_ylim(0, max(self.budget_ms * 1.3, float(np.percentile(ms, 99.5)) * 1.1 if len(ms) else 1))
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path)
        return path

    def _shade_states(self, ax, t: np.ndarray) -> None:
        colors = {"BRAKE": "#ef4444", "SLOW": "#f59e0b"}
        start = None
        states = self.states + ["END"]
        for i, s in enumerate(states):
            if start is not None and s != states[start]:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=colors[states[start]], alpha=0.18, lw=0)
                start = None
            if start is None and s in colors:
                start = i


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return (float(arr.mean()), float(arr.max())) if len(arr) else (0.0, 0.0)
