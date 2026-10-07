# Smart Navigation - Performance Report

## Run

| Parameter | Value |
|---|---|
| Outcome | **ARRIVED** |
| Seed | 7 |
| Map | 25x33 cells, 22 road-work closures |
| Route | (0, 0) -> (24, 32), final path 64 steps |
| Simulated time | 14.72 s (883 frames at 60 FPS) |
| V2X | on, rate 0.3/s |
| Perception | 8 obstacles, noise 0.35 cells, range 7.0, safety brake on |
| Safety violations / collisions | 0 / 0 |

## Computational latency

Real-time budget at 60 FPS: **16.67 ms/frame**. Status: **WITHIN BUDGET** (95th percentile).

| Metric | Value |
|---|---|
| Frames | 883 |
| Average frame (step + render) | 3.307 ms |
| - simulation step | 0.899 ms |
| - rendering | 2.408 ms |
| Median | 2.657 ms |
| 95th percentile | 5.079 ms |
| Peak | 22.931 ms |
| Peak excluding first (warm-up) frame | 22.931 ms |
| Frames over budget | 0.57 % |
| Headroom (budget / average) | 5.0x |

## Kalman filter tracking accuracy

Over 302 confirmed-track detections the filter reduced position error by **46.2 %** relative to the raw sensor.

| Metric | Raw sensor | Kalman filter |
|---|---|---|
| Position RMSE (cells) | 0.483 | 0.260 |

| Interval (s) | Samples | Raw RMSE | KF RMSE | Reduction |
|---|---|---|---|---|
| 0-5.0 | 188 | 0.475 | 0.244 | 49 % |
| 5-10.0 | 87 | 0.511 | 0.289 | 43 % |
| 10-14.7 | 27 | 0.433 | 0.262 | 39 % |

## Energy efficiency

Model: 1500 kg vehicle, 1 cell = 2.5 m, Crr 0.012, CdA 0.65 m^2, drivetrain 85%, regen 0%, idle 1200 W. Efficiency compares actual energy with an ideal constant-cruise drive over the same distance.

| Metric | Value |
|---|---|
| Distance | 160 m |
| Total energy | 83.9 Wh |
| - traction | 83.1 Wh |
| - idling | 0.8 Wh (2.4 s) |
| Lost to braking | 103.6 Wh |
| Consumption | 524 Wh/km (5.89 L-equivalent/100 km) |
| Ideal cruise energy | 13.8 Wh |
| **Efficiency score** | **16.5 %** |
| Hard-brake events | 7 |
| Full stops | 1 |

## Events

| Event | Count |
|---|---|
| V2X incidents | 6 |
| V2X re-routes | 4 |
| Safety-brake activations | 1 |
| Safe halts | 0 |
| Collisions | 0 |

### Timeline

| t (s) | Event | Detail |
|---|---|---|
| 1.78 | incident | construction at (20, 16) |
| 1.78 | reroute | construction at (20, 16) |
| 2.13 | incident | debris at (12, 8) |
| 2.13 | reroute | debris at (12, 8) |
| 2.45 | brake | pedestrian #7 |
| 4.18 | incident | debris at (8, 18) |
| 6.90 | incident | debris at (24, 28) |
| 6.90 | reroute | debris at (24, 28) |
| 6.95 | incident | debris at (12, 16) |
| 6.95 | reroute | debris at (12, 16) |
| 7.68 | incident | accident at (24, 23) |
| 14.72 | arrived | (24, 32) |

## Chart

![Analytics summary](analytics_summary.png)
