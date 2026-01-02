#!/usr/bin/env python3
"""
Visualization script for GA constraint-handling benchmark runs.
Reads CSV output from main and generates comprehensive plots.

Usage:
    python3 visualize.py g06_elite_s1.csv
    python3 visualize.py *.csv  # Plots multiple runs
"""

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_filename(filename):
    """Extract problem, method, seed from filename like 'g06_elite_s1.csv'"""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-1].startswith('s'):
        return parts[0], '_'.join(parts[1:-1]), parts[-1]
    return stem, "unknown", "unknown"

def plot_single_run(df, filename):
    """Create a 2x2 subplot for a single run."""
    problem, method, seed = parse_filename(filename)
    title = f"{problem} | {method} | {seed}"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    gen = df['gen']
    
    # Plot 1: Convergence (best_feasible_obj vs best_overall_obj)
    ax = axes[0, 0]
    ax.plot(gen, df['best_overall_obj'], label='best_overall_obj', alpha=0.7, linewidth=1.5)
    feasible = df['best_feasible_obj'].dropna()
    if not feasible.empty:
        ax.plot(feasible.index, feasible, label='best_feasible_obj', alpha=0.9, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective Value')
    ax.set_title('Convergence Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Feasible Ratio
    ax = axes[0, 1]
    ax.fill_between(gen, df['feasible_ratio'], alpha=0.4, color='green')
    ax.plot(gen, df['feasible_ratio'], color='darkgreen', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Feasible Ratio')
    ax.set_title('Population Feasibility')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Penalty Parameters (lambda and alpha)
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(gen, df['lambda'], label='lambda', color='tab:blue', linewidth=2)
    l2 = ax2.plot(gen, df['alpha'], label='alpha', color='tab:red', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Lambda', color='tab:blue')
    ax2.set_ylabel('Alpha', color='tab:red')
    ax.set_title('Penalty Parameters')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    
    # Plot 4: Timing (elapsed time and per-generation time)
    ax = axes[1, 1]
    if 'elapsed_sec' in df.columns and 'gen_time_sec' in df.columns:
        ax.plot(gen, df['elapsed_sec'], label='Cumulative Time', linewidth=2)
        ax_twin = ax.twinx()
        ax_twin.plot(gen, df['gen_time_sec'], label='Per-Gen Time', alpha=0.7, color='orange')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cumulative Time (sec)', color='tab:blue')
        ax_twin.set_ylabel('Per-Gen Time (sec)', color='orange')
        ax.set_title('Runtime Information')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax_twin.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3)
        
        # Legend
        lns = [plt.Line2D([0], [0], color='tab:blue', lw=2),
               plt.Line2D([0], [0], color='orange', lw=2)]
        ax.legend(lns, ['Cumulative Time', 'Per-Gen Time'], loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_comparison(files):
    """Create comparison plots for multiple runs."""
    if not files:
        return None
    
    data_dict = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            problem, method, seed = parse_filename(f)
            key = f"{problem}_{method}"
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append((seed, df))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    if not data_dict:
        return None
    
    # Create comparison plot: best_feasible_obj over generations for each method
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: All runs (one line per run)
    ax = axes[0]
    for key, runs in data_dict.items():
        for seed, df in runs:
            ax.plot(df['gen'], df['best_feasible_obj'], label=f"{key} ({seed})", alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Feasible Objective')
    ax.set_title('Convergence Comparison (All Runs)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean and std per method
    ax = axes[1]
    for key in sorted(data_dict.keys()):
        runs = data_dict[key]
        # Align all runs to same generation count
        min_len = min(len(df) for _, df in runs)
        objectives = np.array([df['best_feasible_obj'].iloc[:min_len].values for _, df in runs])
        gen = runs[0][1]['gen'].iloc[:min_len].values
        
        # Replace NaN with forward/backward fill, then mean
        filled_objs = []
        for obj_series in objectives:
            s = pd.Series(obj_series)
            s = s.fillna(method='ffill').fillna(method='bfill')
            filled_objs.append(s.values)
        objectives = np.array(filled_objs)
        
        mean_obj = np.nanmean(objectives, axis=0)
        std_obj = np.nanstd(objectives, axis=0)
        
        ax.plot(gen, mean_obj, label=key, linewidth=2, marker='o', markersize=4, 
                markevery=max(1, len(gen)//20))
        ax.fill_between(gen, mean_obj - std_obj, mean_obj + std_obj, alpha=0.2)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Feasible Objective')
    ax.set_title('Mean ± Std Dev (Multiple Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("No files provided. Usage: python3 visualize.py <file.csv> [<file.csv> ...]")
        sys.exit(1)
    
    # Collect all CSV files
    files = []
    for arg in sys.argv[1:]:
        if '*' in arg or '?' in arg:
            files.extend(glob.glob(arg))
        else:
            files.append(arg)
    
    files = [f for f in files if f.endswith('.csv')]
    
    if not files:
        print("No CSV files found.")
        sys.exit(1)
    
    print(f"Found {len(files)} CSV file(s).")
    
    # Plot individual runs
    for f in files:
        try:
            print(f"Plotting {f}...")
            df = pd.read_csv(f)
            fig = plot_single_run(df, f)
            output = f.replace('.csv', '_plot.png')
            fig.savefig(output, dpi=150, bbox_inches='tight')
            print(f"  -> Saved {output}")
            plt.close(fig)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Plot comparison if multiple files
    if len(files) > 1:
        try:
            print(f"Plotting comparison ({len(files)} runs)...")
            fig = plot_comparison(files)
            if fig:
                output = "comparison_plot.png"
                fig.savefig(output, dpi=150, bbox_inches='tight')
                print(f"  -> Saved {output}")
                plt.close(fig)
        except Exception as e:
            print(f"  Comparison error: {e}")
    
    print("Done!")

if __name__ == '__main__':
    main()
