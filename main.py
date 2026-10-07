"""Entry point: python main.py [--headless] ..."""

from __future__ import annotations

import argparse
import logging

from smart_nav.simulation import Simulation, SimulationConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Navigation - A* route simulation")
    p.add_argument("--headless", action="store_true", help="render off-screen (SDL dummy driver)")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--exit-on-arrival", action="store_true", help="stop when the vehicle reaches the goal")
    p.add_argument("--screenshot", default=None, help="save the final frame to this PNG")
    p.add_argument("--plot", default=None, help="save a matplotlib plot of the route to this PNG")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--blocks-x", type=int, default=8)
    p.add_argument("--blocks-y", type=int, default=6)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--cell-size", type=int, default=20)
    p.add_argument("--closure-rate", type=float, default=0.2)
    p.add_argument("--speed", type=float, default=6.0, help="vehicle speed in cells/second")
    p.add_argument("--random-endpoints", action="store_true")
    p.add_argument("--hide-explored", action="store_true")
    p.add_argument("--no-v2x", action="store_true", help="disable spontaneous V2X incidents")
    p.add_argument("--incident-rate", type=float, default=0.3, help="V2X incidents per simulated second")
    p.add_argument("--incident-duration", type=float, nargs=2, default=(12.0, 25.0), metavar=("MIN", "MAX"),
                   help="incident lifetime range in seconds (0 0 = permanent)")
    p.add_argument("--path-bias", type=float, default=0.6, help="probability an incident lands on the route")
    p.add_argument("--max-incidents", type=int, default=6)
    p.add_argument("--obstacles", type=int, default=8, help="number of moving pedestrians/cyclists")
    p.add_argument("--no-perception", action="store_true", help="disable moving obstacles and sensors")
    p.add_argument("--sensor-noise", type=float, default=0.35, help="sensor position noise std (cells)")
    p.add_argument("--sensor-range", type=float, default=7.0, help="sensor range (cells)")
    p.add_argument("--no-brake", action="store_true", help="disable the Kalman-based safety brake")
    p.add_argument("--quiet", action="store_true", help="don't print the event log")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = SimulationConfig(
        blocks_x=args.blocks_x,
        blocks_y=args.blocks_y,
        block_size=args.block_size,
        cell_size=args.cell_size,
        closure_rate=args.closure_rate,
        seed=args.seed,
        speed=args.speed,
        random_endpoints=args.random_endpoints,
        show_explored=not args.hide_explored,
        v2x=not args.no_v2x,
        incident_rate=args.incident_rate,
        incident_duration=None if max(args.incident_duration) <= 0 else tuple(args.incident_duration),
        path_bias=args.path_bias,
        max_incidents=args.max_incidents,
        perception=not args.no_perception,
        num_obstacles=args.obstacles,
        sensor_noise=args.sensor_noise,
        sensor_range=args.sensor_range,
        safety_brake=not args.no_brake,
    )
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    exit_on_arrival = args.exit_on_arrival or (args.headless and args.frames is None)
    sim = Simulation(config, headless=args.headless)
    try:
        if args.plot:
            from smart_nav.visualize import plot_route

            plot_route(sim.city, sim.route, args.plot)
        result = sim.run(args.frames, exit_on_arrival, args.screenshot)
    finally:
        sim.close()
    print(
        f"frames={result.frames} reached_goal={result.reached_goal} "
        f"path_steps={result.path_length - 1} cost={result.path_cost:g} "
        f"explored={result.explored_nodes} closures={result.closed_segments} "
        f"incidents={result.incidents} reroutes={result.reroutes} "
        f"safety_violations={result.safety_violations} collisions={result.collisions} "
        f"brake_events={result.brake_events} sensor_rmse={result.sensor_rmse:.3f} "
        f"kf_rmse={result.kf_rmse:.3f} sim_time={result.sim_time:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
