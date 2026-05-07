#!/usr/bin/env python3
"""
prepare_comet_runs.py - Generate comet simulation configurations

Reads results/names.csv with comet orbital data, calculates orbital periods,
and produces a CSV file with comet names and simulation durations.

The output CSV is used to:
1. Run setup_sim.py to generate Setup.bin for each comet
2. Run the C simulation with the appropriate number of years

Output columns:
- comet_name: Horizons-compatible comet designation
- period_years: Orbital period calculated from semimajor axis
- sim_years: Suggested simulation duration (multiple of period)

Usage:
    python prepare_comet_runs.py
    python prepare_comet_runs.py --input ../../../../../results/names.csv --output comet_runs.csv
"""

import argparse
import pandas as pd
import numpy as np
from tidy.comets import safe_body_name

def calculate_period(semimajor_axis_au):
    """
    Calculate orbital period in years using Kepler's third law.
    
    P² = a³ (with P in years, a in AU, assuming solar mass)
    
    Parameters
    ----------
    semimajor_axis_au : float or array-like
        Semimajor axis in AU
        
    Returns
    -------
    float or array-like
        Orbital period in years
    """
    return np.sqrt(semimajor_axis_au**3)


def extract_comet_name(accepted_name):
    """
    Extract the primary comet designation from accepted_name.
    
    For most entries, this is just the accepted_name.
    Handles special cases and standardizes naming.
    
    Parameters
    ----------
    accepted_name : str
        The accepted_name column value
        
    Returns
    -------
    str
        Horizons-compatible comet designation
    """
    # Strip whitespace
    name = accepted_name.strip()
    
    # For numbered comets, return as-is (e.g., "90000030", "90000091")
    # These are JPL small-body SPK IDs
    if name.isdigit():
        return name
    
    # For standard designations, return as-is
    # e.g., "C/1995 O1", "P/1999 RO28", "2P/Encke"
    return name


def determine_sim_years(period_years, min_years=10, max_years=200, 
                       orbits_per_sim=3):
    """
    Determine simulation duration based on orbital period.
    
    Aims for multiple complete orbits per simulation, within reasonable bounds.
    
    Parameters
    ----------
    period_years : float
        Orbital period in years
    min_years : int
        Minimum simulation duration
    max_years : int
        Maximum simulation duration
    orbits_per_sim : int
        Target number of complete orbits per simulation
        
    Returns
    -------
    int
        Suggested simulation duration in years
    """
    # Calculate duration for target number of orbits
    suggested_years = orbits_per_sim * period_years
    
    # Clamp to reasonable bounds and round
    sim_years = int(np.clip(suggested_years, min_years, max_years))
    
    # For very short-period comets, round up to nearest 5
    if sim_years < 50:
        sim_years = max(min_years, int(np.ceil(sim_years / 5) * 5))
    else:
        # For longer periods, round to nearest 10
        sim_years = int(np.ceil(sim_years / 10) * 10)
    
    return sim_years


def main():
    parser = argparse.ArgumentParser(
        description="Prepare comet simulation configurations from orbital data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default="../../../../../results/names.csv",
        help="Input CSV file with comet data (default: ../../../../../results/names.csv)",
    )
    parser.add_argument(
        "--output",
        default="comet_runs.csv",
        help="Output CSV file for simulation runs (default: comet_runs.csv)",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=50,
        help="Minimum simulation duration in years (default: 50)",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=500,
        help="Maximum simulation duration in years (default: 500)",
    )
    parser.add_argument(
        "--orbits",
        type=int,
        default=10,
        help="Target number of orbits per simulation (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Reading comet data from: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"Found {len(df)} comets in input file")
    
    # Extract comet names
    df['comet_name'] = df['accepted_name'].apply(extract_comet_name)
    df["safe_name"] = df["comet_name"].apply(safe_body_name)
    # Calculate orbital periods
    df['period_years'] = calculate_period(df['semimajoraxis_au'])
    
    # Determine simulation durations
    df['sim_years'] = df['period_years'].apply(
        lambda p: determine_sim_years(
            p, 
            min_years=args.min_years,
            max_years=args.max_years,
            orbits_per_sim=args.orbits
        )
    )

    def _output_year(sim_years):
        """Calculate output start year as 75% of sim time, but at least 30 years before end."""
        return int(np.floor(min(sim_years * 0.75, sim_years - 30)))
        
    df["output_year"] = df["sim_years"].apply(_output_year)
    
    # Select output columns
    output_df = df[['comet_name', 'safe_name', 'period_years', 'sim_years', "output_year"]].copy()
    
    # Sort by period for convenience
    output_df = output_df.sort_values('period_years')
    
    # Save to CSV
    output_df.to_csv(args.output, index=False, float_format='%.2f')
    print(f"\nWrote {len(output_df)} comet configurations to: {args.output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Period range: {output_df['period_years'].min():.2f} - {output_df['period_years'].max():.2f} years")
    print(f"Simulation duration range: {output_df['sim_years'].min()} - {output_df['sim_years'].max()} years")
    print(f"Mean period: {output_df['period_years'].mean():.2f} years")
    print(f"Median period: {output_df['period_years'].median():.2f} years")
    
    # Show a few examples
    print("\n" + "="*60)
    print("Example entries:")
    print("="*60)
    print(output_df.head(10).to_string(index=False))
    
    # Usage instructions
    print("\n" + "="*60)
    print("Usage Instructions:")
    print("="*60)
    print(f"For each row in {args.output}:")
    print("  1. Generate setup:  python setup_sim.py <comet_name>")
    print("  2. Run simulation:  ./rebound <sim_years>")
    print("\nExample:")
    first_row = output_df.iloc[0]
    print(f"  python setup_sim.py '{first_row['comet_name']}'")
    print(f"  ./rebound {first_row['sim_years']}")


if __name__ == "__main__":
    main()
