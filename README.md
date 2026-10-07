# Smart Navigation and Route Optimization System

Autonomous-vehicle navigation simulator.

- **Phase 1:** a 2D city grid environment rendered with Pygame, an A* routing engine,
  and a virtual vehicle that drives the optimal route.
- **Phase 2:** a simulated V2X (Vehicle-to-Everything) network that broadcasts
  spontaneous incident alerts, and dynamic mid-route A* re-routing around them.

![Simulation](docs/simulation.png)

## Layout

```
main.py                  CLI entry point
smart_nav/
  environment.py         CityGrid (roads, intersections, road closures, start/goal) + CityRenderer (Pygame)
  routing.py             Generic A* search + find_route() for occupancy grids
  v2x_network.py         V2XNetwork: incident alerts / clearances broadcast to subscribers
  vehicle.py             Vehicle that interpolates along a path, can be re-routed, draws itself
  simulation.py          Pygame loop wiring environment + router + V2X + vehicle
  visualize.py           Static matplotlib route plot
tests/                   pytest suite (runs headless)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
python main.py                       # interactive window
python main.py --seed 7 --random-endpoints --closure-rate 0.3
```

Controls: `R` new random map, `I` inject an incident on the route, `Space` pause, `Esc` quit.

V2X options: `--no-v2x`, `--incident-rate 0.3` (per sim second), `--incident-duration 12 25`
(seconds, `0 0` = permanent), `--path-bias 0.6` (chance an incident lands on the route),
`--max-incidents 6`, `--quiet` (hide the event log on stdout).

Headless (no display, e.g. CI / cloud):

```bash
python main.py --headless --seed 7 --screenshot artifacts/final.png --plot artifacts/route.png
```

Headless runs use SDL's `dummy` video driver, a fixed timestep, and stop when the
vehicle reaches the goal (or after `--frames N`).

## Map model

- Grid cells are indexed `(row, col)`; roads run along every `block_size`-th row/column,
  intersections sit where they cross.
- `closure_rate` closes random road segments between intersections (red/white barriers)
  while keeping every intersection reachable, so the router has to plan detours.
- A* runs on the 4-connected drivable cells with unit step cost and a Manhattan
  heuristic (admissible, so the result is optimal). Blue cells in the window are the
  nodes A* expanded.

## V2X and dynamic re-routing

- `V2XNetwork.update()` runs every frame on simulated time: incidents (accident /
  construction / debris) spawn as a Poisson process, block one road cell, and are
  broadcast to subscribers; when they expire the cell reopens and a `CLEARED`
  message is broadcast.
- Safety rules: an incident never lands on the start/goal, on the cells the vehicle
  is occupying or already committed to entering (it can't brake instantly), or
  anywhere that would cut the vehicle off from the goal.
- The simulation subscribes to the feed. If an incident lands on the remaining route,
  it immediately runs A* from the vehicle's next cell (its *anchor*) to the goal and
  splices in the new route. On a clearance it re-plans and adopts the new route only
  if it is strictly shorter.
- Visuals: incident cells flash red/yellow with a broadcast pulse ring, the abandoned
  route fades out in red, a `V2X: DYNAMIC RE-ROUTE #n` banner flashes, and the log
  panel at the bottom (also printed to stdout) records alerts, re-routes, and clearances.

## Tests

```bash
python -m pytest
```

Covers A* correctness (vs. BFS on random grids and Dijkstra on weighted graphs),
map generation/connectivity, vehicle kinematics, headless rendering, end-to-end
headless runs, V2X broadcast/expiry/safety rules, a mid-route blockage that must
trigger re-routing and still reach the goal, and randomized high-incident stress runs
that assert the vehicle never enters a blocked cell.
