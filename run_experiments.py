#!/usr/bin/env python3
"""
Experiment runner for GA benchmark.
Runs multiple seeds and methods, saves outputs to results/ folder.

Usage:
    python3 run_experiments.py [--problem g06] [--methods elite death debrules] [--seeds 5] [--gens 300] [--pop 80]

Defaults: problem=g06, methods=all 6, seeds=3, gens=300, pop=80
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from itertools import product

METHODS = ['static', 'dynamic', 'adaptive', 'elite', 'death', 'debrules']

def run_experiment(problem, method, seed, gens, pop, results_dir):
    """Run a single GA experiment and save output."""
    output_csv = results_dir / f"{problem}_{method}_s{seed}.csv"
    
    cmd = [
        './main',
        '--problem', problem,
        '--method', method,
        '--seed', str(seed),
        '--gen', str(gens),
        '--pop', str(pop),
        '--out', str(output_csv)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"    ✓ Output: {output_csv}")
            if result.stdout:
                print(f"    {result.stdout.strip()}")
            return True
        else:
            print(f"    ✗ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run GA benchmark experiments')
    parser.add_argument('--problem', default='g06', 
                        choices=['g01', 'g03', 'g06', 'g07'],
                        help='Problem to optimize')
    parser.add_argument('--methods', nargs='+', default=METHODS,
                        choices=METHODS,
                        help='Constraint handling methods')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds to run')
    parser.add_argument('--gens', type=int, default=300,
                        help='Number of generations')
    parser.add_argument('--pop', type=int, default=80,
                        help='Population size')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print(f"GA Benchmark Experiment Runner")
    print(f"Problem: {args.problem}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Seeds: {args.seeds} (1..{args.seeds})")
    print(f"Generations: {args.gens}, Population: {args.pop}")
    print(f"Results directory: {results_dir}")
    print()
    
    # Run experiments
    total = len(args.methods) * args.seeds
    completed = 0
    failed = 0
    
    for method, seed in product(args.methods, range(1, args.seeds + 1)):
        try:
            if run_experiment(args.problem, method, seed, args.gens, args.pop, results_dir):
                completed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            failed += 1
    
    print()
    print(f"Summary: {completed}/{total} completed, {failed} failed")
    print(f"CSV files saved to {results_dir}/")
    print()
    print(f"To visualize results:")
    print(f"  python3 visualize.py {results_dir}/*.csv")

if __name__ == '__main__':
    main()
