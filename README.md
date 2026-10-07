# Smart Navigation and Route Optimization System

Autonomous-vehicle navigation simulator. **Phase 1:** a 2D city grid environment
rendered with Pygame, an A* routing engine, and a virtual vehicle that drives
the optimal route.

![Simulation](docs/simulation.png)

## Layout

```
main.py                  CLI entry point
smart_nav/
  environment.py         CityGrid (roads, intersections, road closures, start/goal) + CityRenderer (Pygame)
  routing.py             Generic A* search + find_route() for occupancy grids
  vehicle.py             Vehicle that interpolates along a path and draws itself
  simulation.py          Pygame loop wiring environment + router + vehicle
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

Controls: `R` new random map, `Space` pause, `Esc` quit.

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

## Tests

```bash
python -m pytest
```

Covers A* correctness (vs. BFS on random grids and Dijkstra on weighted graphs),
map generation/connectivity, vehicle kinematics, headless rendering, and an
end-to-end headless simulation run.
