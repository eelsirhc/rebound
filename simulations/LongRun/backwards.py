#!/usr/bin/env python3
"""
backwards.py -- Integrate a REBOUND simulation backwards 5000 years from 2000-01-01.

Includes the Sun, 8 major planets, and a named comet loaded from JPL Horizons.
Steps 1 Earth year at a time, recording positions and orbital elements for all
bodies into an HDF5 file.  Saves a resumable .bin snapshot every P years
(P supplied on the command line), named by days before 2000-01-01.

Usage
-----
    python backwards.py 73P --period 100
    python backwards.py "C/1995 O1" --period 500
    python backwards.py 209P --period 200 --output-dir snapshots/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.time import Time
import rebound

PLANETS = [
    "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune",
]

DAYS_PER_YEAR = 365.25
TOTAL_YEARS = 5000
START_JD = Time("2000-01-01", format="iso", scale="tt").jd  # fixed reference epoch

ORBITAL_ELEMENTS = ["a", "e", "inc", "Omega", "omega", "f", "M"]


def jd_to_year(jd: float) -> int:
    """Approximate calendar year for a Julian Date (works for ancient dates)."""
    return int(2000.0 + (jd - 2451545.0) / DAYS_PER_YEAR)


def safe_name(name: str) -> str:
    """Filesystem- and HDF5-key-safe version of a body name."""
    return name.replace("/", "_").replace(" ", "_")


def build_sim(comet: str, start_jd: float) -> rebound.Simulation:
    date_str = f"JD{start_jd}"
    sim = rebound.Simulation()
    sim.units = ("AU", "day", "Msun")
    sim.t = start_jd

    sim.add("Sun", date=date_str)
    for planet in PLANETS:
        sim.add(planet, date=date_str)
    sim.add(comet, m=0, date=date_str)

    sim.move_to_com()
    sim.integrator = "ias15"
    sim.dt = -DAYS_PER_YEAR  # negative drives IAS15 backwards

    return sim


def record_step(step: int, sim: rebound.Simulation, body_names: list,
                times_jd: np.ndarray, years: np.ndarray,
                pos: np.ndarray, vel: np.ndarray, elem: np.ndarray) -> None:
    times_jd[step] = sim.t
    years[step] = jd_to_year(sim.t)

    sun = sim.particles[0]
    for i, _ in enumerate(body_names):
        p = sim.particles[i]
        pos[step, i] = [p.x, p.y, p.z]
        vel[step, i] = [p.vx, p.vy, p.vz]
        if i == 0:
            elem[step, i] = np.nan
        else:
            o = p.orbit(primary=sun)
            elem[step, i] = [o.a, o.e, o.inc, o.Omega, o.omega, o.f, o.M]


def write_hdf5(path: str, body_names: list, times_jd: np.ndarray,
               years: np.ndarray, pos: np.ndarray, vel: np.ndarray,
               elem: np.ndarray) -> None:
    n_steps, n_bodies, _ = pos.shape
    body_idx = np.tile(np.arange(n_bodies, dtype=np.int8), n_steps)

    df = pd.DataFrame({
        "time_jd": np.repeat(times_jd, n_bodies),
        "year":    np.repeat(years, n_bodies),
        "body":    body_idx,
        "x":  pos.reshape(-1, 3)[:, 0],
        "y":  pos.reshape(-1, 3)[:, 1],
        "z":  pos.reshape(-1, 3)[:, 2],
        "vx": vel.reshape(-1, 3)[:, 0],
        "vy": vel.reshape(-1, 3)[:, 1],
        "vz": vel.reshape(-1, 3)[:, 2],
        **{el: elem.reshape(-1, len(ORBITAL_ELEMENTS))[:, j]
           for j, el in enumerate(ORBITAL_ELEMENTS)},
    })

    df.to_hdf(path, key="data", mode="w", complevel=5, complib="blosc")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate a named comet + solar system backwards 5000 years.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "comet",
        help="Horizons small-body designation, e.g. '73P', 'C/1995 O1', '209P'.",
    )
    parser.add_argument(
        "--period", "-P",
        type=float,
        required=True,
        metavar="YEARS",
        help="Save a .bin snapshot every this many years of simulation time.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output files (default: current directory).",
    )
    args = parser.parse_args()

    start_jd = START_JD
    period_days = args.period * DAYS_PER_YEAR

    print(f"Comet      : {args.comet}")
    print(f"Start      : JD {start_jd:.4f}  ({Time(start_jd, format='jd', scale='tt').iso})")
    print(f"Save every : {args.period} yr  ({period_days:.1f} days)")
    print(f"Total      : {TOTAL_YEARS} yr")
    print()

    print("Adding bodies from Horizons...")
    try:
        sim = build_sim(args.comet, start_jd)
    except Exception as exc:
        print(f"ERROR building simulation: {exc}", file=sys.stderr)
        sys.exit(1)

    body_names = ["Sun"] + PLANETS + [args.comet]
    print(f"  {sim.N} particles: Sun + {len(PLANETS)} planets + comet")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = safe_name(args.comet)

    # Pre-allocate arrays: (timestep, body, component)
    n_steps = TOTAL_YEARS
    n_bodies = len(body_names)
    times_jd = np.zeros(n_steps)
    years    = np.zeros(n_steps, dtype=np.int32)
    pos      = np.zeros((n_steps, n_bodies, 3))
    vel      = np.zeros((n_steps, n_bodies, 3))
    elem     = np.full((n_steps, n_bodies, len(ORBITAL_ELEMENTS)), np.nan)

    next_bin_at_days = period_days
    bin_count = 0

    print("Integrating backwards (1 yr/step)...")
    for step in range(n_steps):
        try:
            sim.integrate(sim.t - DAYS_PER_YEAR)
        except rebound.Encounter as exc:
            print(f"WARNING: close encounter at t={sim.t:.2f}: {exc}", file=sys.stderr)

        record_step(step, sim, body_names, times_jd, years, pos, vel, elem)

        elapsed_days = (step + 1) * DAYS_PER_YEAR

        if elapsed_days >= next_bin_at_days - 0.5:
            days_before = int(round(elapsed_days))
            filename = os.path.join(args.output_dir, f"{prefix}_{days_before}.bin")
            sim.save_to_file(filename, delete_file=True)
            next_bin_at_days += period_days
            bin_count += 1
            elapsed_years = elapsed_days / DAYS_PER_YEAR
            print(f"  [{bin_count:4d}] {elapsed_years:8.1f} yr back  ({days_before:8d} days)  -> {os.path.basename(filename)}")

    h5_path = os.path.join(args.output_dir, f"{prefix}_backwards.h5")
    print(f"\nWriting HDF5 to '{h5_path}'...")
    write_hdf5(h5_path, body_names, times_jd, years, pos, vel, elem)

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"  {n_steps * n_bodies:,} rows ({n_steps} yr x {n_bodies} bodies)  ({size_mb:.1f} MB)")
    print(f"\nDone. {bin_count} .bin snapshots + 1 HDF5 file saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
