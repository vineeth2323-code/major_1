"""Simulated V2X (Vehicle-to-Everything) traffic broadcast network.

Roadside infrastructure "broadcasts" incident alerts (accidents, construction,
debris) that block a road coordinate for a while, then broadcasts a clearance.
Subscribers (e.g. the vehicle's navigation stack) receive every message
synchronously, so they can re-plan before the vehicle advances.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .environment import CityGrid
from .routing import find_route

Cell = Tuple[int, int]

INCIDENT_KINDS = ("accident", "construction", "debris")
INCIDENT = "INCIDENT"
CLEARED = "CLEARED"


@dataclass
class Incident:
    id: int
    kind: str
    cell: Cell
    reported_at: float
    expires_at: Optional[float] = None

    def age(self, now: float) -> float:
        return now - self.reported_at

    def describe(self) -> str:
        return f"{self.kind} #{self.id} at {self.cell}"


@dataclass(frozen=True)
class V2XMessage:
    type: str
    incident: Incident
    timestamp: float


Subscriber = Callable[[V2XMessage], None]


class V2XNetwork:
    """Generates incidents as a Poisson process and broadcasts them.

    ``rate`` is incidents per simulated second. ``path_bias`` is the probability
    that a spontaneous incident lands on the vehicle's remaining route (at least
    ``min_lead`` cells ahead); otherwise a random road cell is chosen.
    ``duration`` is a (min, max) lifetime in seconds, or ``None`` for permanent.
    """

    def __init__(
        self,
        city: CityGrid,
        rate: float = 0.3,
        duration: Optional[Tuple[float, float]] = (12.0, 25.0),
        path_bias: float = 0.6,
        max_active: int = 6,
        min_lead: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        if rate < 0:
            raise ValueError("rate must be >= 0")
        if not 0.0 <= path_bias <= 1.0:
            raise ValueError("path_bias must be in [0, 1]")
        self.city = city
        self.rate = rate
        self.duration = duration
        self.path_bias = path_bias
        self.max_active = max_active
        self.min_lead = min_lead
        self.rng = random.Random(seed)
        self.active: Dict[int, Incident] = {}
        self.history: List[V2XMessage] = []
        self._subscribers: List[Subscriber] = []
        self._ids = itertools.count(1)
        self._road_cells = [
            (int(r), int(c)) for r, c in zip(*city.passable.nonzero()) if not city.is_intersection((r, c))
        ]

    def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    @property
    def blocked_cells(self) -> Dict[Cell, Incident]:
        return {inc.cell: inc for inc in self.active.values()}

    def update(self, now: float, dt: float, vehicle=None) -> List[V2XMessage]:
        """Advance the network clock: expire old incidents, maybe spawn a new one."""
        messages = self._expire(now)
        if vehicle is not None and vehicle.finished:
            return messages
        if self.rate > 0 and len(self.active) < self.max_active:
            if self.rng.random() < 1.0 - math.exp(-self.rate * dt):
                for _ in range(20):
                    cell = self._pick_cell(vehicle)
                    if cell is None:
                        break
                    incident = self.report(cell, now, vehicle=vehicle)
                    if incident is not None:
                        messages.append(self.history[-1])
                        break
        return messages

    def report(
        self,
        cell: Cell,
        now: float,
        kind: Optional[str] = None,
        duration: Optional[float] = None,
        vehicle=None,
        allow_isolation: bool = False,
    ) -> Optional[Incident]:
        """Block ``cell`` and broadcast it. Returns ``None`` if the cell can't be blocked safely.

        ``allow_isolation`` skips the reachability check (an authority closing roads regardless), so
        the goal may become unreachable; the vehicle must then handle it with a safe halt.
        """
        cell = (int(cell[0]), int(cell[1]))
        if not self.can_block(cell, vehicle, allow_isolation):
            return None
        if duration is None and self.duration is not None:
            duration = self.rng.uniform(*self.duration)
        incident = Incident(
            id=next(self._ids),
            kind=kind or self.rng.choice(INCIDENT_KINDS),
            cell=cell,
            reported_at=now,
            expires_at=None if duration is None else now + duration,
        )
        self.city.block(cell)
        self.active[incident.id] = incident
        self._broadcast(V2XMessage(INCIDENT, incident, now))
        return incident

    def clear(self, incident_id: int, now: float) -> Optional[V2XMessage]:
        incident = self.active.pop(incident_id, None)
        if incident is None:
            return None
        self.city.unblock(incident.cell)
        msg = V2XMessage(CLEARED, incident, now)
        self._broadcast(msg)
        return msg

    def can_block(self, cell: Cell, vehicle=None, allow_isolation: bool = False) -> bool:
        """A cell is blockable if it's open road, not an endpoint, not under/just ahead of the
        vehicle (it can't brake instantly), and blocking it leaves the goal reachable."""
        city = self.city
        if not city.is_road(cell) or cell in (city.start, city.goal):
            return False
        if vehicle is None:
            return True
        if vehicle.finished or cell in vehicle.committed_cells:
            return False
        if allow_isolation:
            return True
        city.passable[cell] = False
        try:
            return find_route(city.passable, vehicle.anchor, city.goal).found
        finally:
            city.passable[cell] = True

    def _pick_cell(self, vehicle) -> Optional[Cell]:
        if vehicle is not None and self.rng.random() < self.path_bias:
            ahead = vehicle.cells_ahead()[self.min_lead : -1]
            if ahead:
                return self.rng.choice(ahead)
        return self.rng.choice(self._road_cells) if self._road_cells else None

    def _expire(self, now: float) -> List[V2XMessage]:
        due = [i for i, inc in self.active.items() if inc.expires_at is not None and inc.expires_at <= now]
        return [msg for msg in (self.clear(i, now) for i in due) if msg is not None]

    def _broadcast(self, msg: V2XMessage) -> None:
        self.history.append(msg)
        for callback in list(self._subscribers):
            callback(msg)


def incidents_on_path(path: Sequence[Cell], blocked: Dict[Cell, Incident]) -> List[Incident]:
    return [blocked[c] for c in path if c in blocked]
