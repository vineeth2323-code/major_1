# Smart Navigation and Route Optimization System

Autonomous-vehicle navigation simulator.

- **Phase 1:** a 2D city grid environment rendered with Pygame, an A* routing engine,
  and a virtual vehicle that drives the optimal route.
- **Phase 2:** a simulated V2X (Vehicle-to-Everything) network that broadcasts
  spontaneous incident alerts, and dynamic mid-route A* re-routing around them.
- **Phase 3:** perception and sensor fusion: pedestrians and cyclists move on the
  sidewalks and cross the roads, a noisy range-limited sensor observes them, a 2D
  Kalman filter tracks each one's position and velocity, and a safety brake slows or
  stops the car when a tracked obstacle is predicted to cross its path.

![Simulation](docs/simulation.png)

## Layout

```
main.py                  CLI entry point
smart_nav/
  environment.py         CityGrid (roads, intersections, road closures, start/goal) + CityRenderer (Pygame)
  routing.py             Generic A* search + find_route() for occupancy grids
  v2x_network.py         V2XNetwork: incident alerts / clearances broadcast to subscribers
  perception.py          Moving obstacles, NoisySensor, KalmanFilter2D tracker, threat assessment
  vehicle.py             Vehicle with accel/brake limits; follows a path, can be re-routed, draws itself
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

Perception options: `--obstacles 8`, `--sensor-noise 0.35` (std, cells), `--sensor-range 7`,
`--no-brake` (track obstacles but never brake - for comparison), `--no-perception`.

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

## Perception, Kalman tracking and the safety brake

Each frame runs `ObstacleField -> NoisySensor -> MultiObjectTracker -> assess_threats -> Vehicle`:

- **Moving obstacles** (`ObstacleField`): pedestrians (0.8-1.3 cells/s) and cyclists
  (1.8-2.6 cells/s) walk the sidewalks 0.7 cells either side of a road centre line, so they
  cross the side streets at every intersection. Now and then one crosses the road
  mid-block. They obey basic gap acceptance: they won't step out right in front of a car
  that is less than ~0.6 s away, and won't walk into a stopped car. Once they're on the road
  they keep going, and avoiding them is the car's job.
- **Noisy sensor** (`NoisySensor`): a 20 Hz, 7-cell range camera/LiDAR model. Every reading
  gets independent Gaussian noise on X and Y (std 0.35 cells), and 5% of readings are dropped.
- **Kalman filter** (`KalmanFilter2D`): constant-velocity model, state `[r, c, vr, vc]`,
  position-only measurements, white-noise-acceleration process model. If several innovations
  in a row fall outside the chi-square gate (the target turned, stopped or reversed), the
  filter widens its velocity uncertainty so it re-converges quickly. `MultiObjectTracker` runs one
  filter per obstacle (sensor detections carry the obstacle id, so data association is assumed
  solved) and confirms tracks after 2 hits.
- **Threat assessment** (`assess_threats`): every confirmed track is projected forward with
  its filtered velocity over a 1.5 s horizon. It is checked against the car's oriented
  footprint at the same instants along its actual upcoming route, which includes corners. The
  safety envelope grows with look-ahead time. Any track already standing in the lane ahead also
  counts, whatever its estimated velocity. On a predicted conflict the car's speed is capped at
  half its cruise speed (`SLOW`) and at the speed from which it can still stop 0.5 cells short
  (`BRAKE`). It accelerates back to cruise once the path is clear.
- **Visuals:** blue dot = true obstacle position (dark centre = cyclist), red X's = the
  latest noisy readings (fading), green ring + arrow = Kalman estimate and velocity
  vector (red when it's the threat being braked for), faint circle = sensor range. The
  car shows brake lights, `SAFETY BRAKE` / `CAUTION` banners flash, the HUD shows live
  Kalman vs. raw-sensor RMSE, and the log records every brake and release.

![Safety brake](docs/safety_brake.png)

## Tests

```bash
python -m pytest
```

Covers A* correctness (vs. BFS on random grids and Dijkstra on weighted graphs),
map generation/connectivity, vehicle kinematics, headless rendering, end-to-end
headless runs, V2X broadcast/expiry/safety rules, a mid-route blockage that must
trigger re-routing and still reach the goal, and randomized high-incident stress runs
that assert the vehicle never enters a blocked cell.

Phase 3 tests (`tests/test_perception.py`) cover:
- Sensor noise statistics, range and rate.
- The Kalman filter keeping position error under half the raw sensor's on constant-velocity
  targets, accurate velocity estimates, and velocity re-converging after a turn.
- Lower error than the raw sensor on random pedestrians and cyclists that turn, stop and
  reverse.
- Threat geometry: crossing vs. parallel, behind, and moving-away obstacles.
- A scripted distracted pedestrian stepping out in front of the car, which **collides with the
  brake disabled and is avoided with it enabled**; the car then resumes and reaches the goal.
- Full random runs with 16 moving obstacles plus V2X incidents and zero collisions.
