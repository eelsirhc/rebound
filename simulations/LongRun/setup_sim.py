#!/usr/bin/env python3
"""
setup_sim.py – Build a REBOUND initial-conditions file (Setup.bin) for the
MeteorStormsPR C simulation.

The simulation contains the Sun, inner planets (Mercury, Venus, Earth, Mars),
Jupiter, Saturn, and one massless comet, all placed at their JPL Horizons
positions for the requested start date.

Units: AU, day, Msun  (REBOUND ecliptic/heliocentric default)

Usage
-----
    python setup_sim.py 209P --date 2014-05-29
    python setup_sim.py "C/1995 O1" --date 2024-01-01
    python setup_sim.py 73P --jd 2459000.5
"""

import argparse
import sys
from astropy.time import Time
import rebound


INNER_PLANETS = ["Mercury", "Venus", "Earth", "Mars"]
OUTER_PLANETS = ["Jupiter", "Saturn", "Uranus", "Neptune"]


def date_to_jd(date_str: str) -> float:
    """Convert an ISO calendar date string (YYYY-MM-DD) to Julian Day number."""
    #print (date_str, Time(date_str, format="iso", scale="tt").jd)
    return Time(date_str, format="iso", scale="tt").jd


def build_sim(comet: str, jd: float, include_comet: bool = True) -> rebound.Simulation:
    """
    Construct a REBOUND simulation with the Sun, inner planets, Jupiter,
    Saturn, and optionally the named comet at the given Julian Date.

    All bodies are added using REBOUND's built-in Horizons interface so that
    positions are consistent with the DE ephemeris.  The comet is massless.

    Parameters
    ----------
    comet : str
        Horizons-compatible small-body designation, e.g. "209P", "73P",
        "C/1995 O1".
    jd : float
        Start epoch as a Julian Date (TT).
    include_comet : bool
        Whether to add the comet to the simulation (default: True).

    Returns
    -------
    rebound.Simulation
    """
    sim = rebound.Simulation()
    sim.units = ("AU", "day", "Msun")
    sim.t = jd  # simulation time = JD of epoch
    labels = [] 
    date_str = f"JD{jd}"
    
    sim.add("Sun", date=date_str)
    labels.append((len(sim.particles)-1,"Sun"))
    for planet in INNER_PLANETS:
        sim.add(planet, date=date_str)
        labels.append((len(sim.particles)-1,planet))
    for planet in OUTER_PLANETS:
        sim.add(planet, date=date_str)
        labels.append((len(sim.particles)-1,planet))

    if include_comet:
        # Comet is massless (m=0 is the REBOUND default for small bodies, but
        # being explicit keeps the code clear).
        sim.add(comet, m=0, date=date_str)
        labels.append((len(sim.particles)-1,comet))

    sim.move_to_com()
    return sim,labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate Setup.bin for the MeteorStormsPR REBOUND simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "comet",
        nargs="?",
        default=None,
        help="Horizons small-body designation, e.g. '209P', '73P', 'C/1995 O1'. Not needed with --reload.",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Start date as an ISO calendar date (TT scale). Default: today.",
    )
    date_group.add_argument(
        "--jd",
        type=float,
        default=None,
        metavar="JD",
        help="Start epoch as a Julian Date (overrides --date).",
    )
    parser.add_argument(
        "--output",
        default="Setup.bin",
        metavar="FILE",
        help="Output binary file name (default: Setup.bin).",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Generate setup without comet (Sun and planets only).",
    )
    parser.add_argument(
        "--reload",
        #metavar="FILE",
        default=None,
        help="Load existing Setup.bin file and add comet to it.",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.reload and args.no_comet:
        print("ERROR: --reload and --no-comet are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    
    if args.reload:
        if not args.comet:
            print("ERROR: Comet name required when using --reload.", file=sys.stderr)
            sys.exit(1)
    elif args.no_comet:
        pass  # No comet argument needed
    else:
        if not args.comet:
            print("ERROR: Comet name required (or use --no-comet or --reload).", file=sys.stderr)
            sys.exit(1)

    # Resolve start epoch
    if args.jd is not None:
        jd = args.jd
    elif args.date is not None:
        jd = date_to_jd(args.date)
    else:
        jd = Time.now().tt.jd
    # Handle reload case
    if args.reload:
        print(f"Loading existing setup from: {args.reload}")
        try:
            sim = rebound.Simulation(args.reload)
        except Exception as exc:
            print(f"ERROR loading {args.reload}: {exc}", file=sys.stderr)
            sys.exit(1)
        print(sim.N)
        # Get epoch from loaded simulation
        #jd = sim.t CANT DO THIS IT DOESNT WORK
        date_str = f"JD{jd}"
        
        print(f"Comet   : {args.comet}")
        print(f"Epoch   : JD {jd:.4f}  ({Time(jd, format='jd', scale='tt').iso})")
        print(f"Output  : {args.output}")
        
        # Add comet to loaded simulation
        try:
            sim.add(args.comet, m=0, date=date_str)
            labels = [(len(sim.particles)-1,args.comet)]
            sim.move_to_com()
        except Exception as exc:
            print(f"ERROR adding comet: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.no_comet:
            comet_label = "(none)"
            print(f"Comet   : {comet_label}")
        else:
            print(f"Comet   : {args.comet}")
        
        print(f"Epoch   : JD {jd:.4f}  ({Time(jd, format='jd', scale='tt').iso})")
        print(f"Output  : {args.output}")
        
        try:
            sim,labels = build_sim(args.comet if not args.no_comet else "dummy", jd, 
                           include_comet=not args.no_comet)
        except Exception as exc:
            print(f"ERROR building simulation: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\nParticles added ({sim.N} total):")
   
    #if not args.no_comet:
    #    labels.append(args.comet)
    
    for num, label in labels:
        p = sim.particles[num]
        r = (p.x**2 + p.y**2 + p.z**2) ** 0.5
        print(f"  [{num}] {label:<20s}  r = {r:.4f} AU  m = {p.m:.4g} Msun")

    sim.save_to_file(args.output, delete_file=True)
    print(f"\nSaved simulation to '{args.output}'.")


if __name__ == "__main__":
    main()
