# GA Constraint-Handling Benchmark

A C++ genetic algorithm implementation for solving constrained optimization problems with multiple constraint-handling methods. Includes automatic runtime logging and Python visualization tools.

## Overview

This project implements a **Genetic Algorithm (GA)** with six different constraint-handling methods applied to CEC 2006 benchmark problems:

- **static** — Fixed penalty term
- **dynamic** — Time-varying penalty (increases over generations)
- **adaptive** — Penalty adapts based on population feasibility ratio
- **elite** — Penalty + elite proximity (guides infeasible solutions toward feasible region)
- **death** — Extreme penalty for infeasible solutions
- **debrules** — Classic Deb's feasibility-first comparator rule

Each method optimizes four benchmark problems: **g01, g03, g06, g07** (from CEC 2006).

---

## Dependencies

### C++
- C++17 compiler (clang++ or g++)
- Standard library only (no external C++ dependencies)

### Python (for visualization & batch runs)
```bash
pip install pandas matplotlib
```

---

## Installation & Compilation

### Build the executable:

```bash
clang++ -O2 -std=c++17 main.cpp -o main
```

Or with g++:
```bash
g++ -O2 -std=c++17 main.cpp -o main
```

Verify:
```bash
./main --help
```

---

## Quick Start

### Single optimization run:

```bash
./main --problem g06 --method elite --seed 1 --gen 300 --pop 80 --out g06_elite_s1.csv
```

**Output (console):**
```
Problem=g06 Method=3 Seed=1
Best: feasible=1 obj=-6648.6348... violation=0
x=[14.231, 1.125]
Total Runtime: 0.002946 sec
```

**Output (CSV):** `g06_elite_s1.csv` — per-generation convergence and timing data

### Visualize a single run:

```bash
python3 visualize.py g06_elite_s1.csv
# → Creates: g06_elite_s1_plot.png
```

---

## Command-Line Options

```
Usage:
  ./main --problem g01|g03|g06|g07 --method static|dynamic|adaptive|elite|death|debrules [options]

Options:
  --problem PROB    Benchmark problem: g01, g03, g06, or g07 (REQUIRED)
  --method METHOD   Constraint-handling method (REQUIRED)
                    - static, dynamic, adaptive, elite, death, debrules
  --seed N          Random seed (default: 1)
  --pop N           Population size (default: 120)
  --gen N           Number of generations (default: 2000)
  --pc X            Crossover probability (default: 0.9)
  --eta_c X         SBX crossover distribution index (default: 15)
  --eta_m X         Polynomial mutation distribution index (default: 20)
  --pm_gene X       Per-gene mutation probability (default: auto = 1/dim)
  --out FILE        Output CSV file (optional; no logging if omitted)
  --help            Show this message
```

---

## Batch Experiments

### Run multiple seeds and methods automatically:

```bash
# Run all 6 methods, 3 seeds each, 300 gen, pop 80 (g06)
python3 run_experiments.py --problem g06 --seeds 3

# Custom: elite & debrules only, 5 seeds, 500 gen, pop 100 (g07)
python3 run_experiments.py --problem g07 --methods elite debrules --seeds 5 --gens 500 --pop 100

# Run all g01 experiments (5 seeds, default settings)
python3 run_experiments.py --problem g01 --seeds 5
```

**Output:** All CSVs saved to `results/` folder
- Naming: `{problem}_{method}_s{seed}.csv`
- Example: `g06_elite_s1.csv`, `g06_elite_s2.csv`, ...

---

## Visualization

### Visualize individual runs:

```bash
python3 visualize.py g06_elite_s1.csv
# → Single run dashboard: g06_elite_s1_plot.png
```

### Visualize all experiments:

```bash
python3 visualize.py results/*.csv
# → 6 individual plots + comparison_plot.png
```

### Selective visualization:

```bash
python3 visualize.py results/g06_*.csv        # Only g06
python3 visualize.py results/*_elite*.csv     # Only elite method
```

---

## Output Format

### CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `gen` | int | Generation number (0 to --gen) |
| `best_feasible_obj` | float | Best objective among **feasible** solutions (NaN if none exist) |
| `best_overall_obj` | float | Best objective ignoring feasibility |
| `feasible_ratio` | float | Fraction of population satisfying all constraints (0–1) |
| `mean_infeasible_violation` | float | Average total constraint violation for infeasible individuals |
| `lambda` | float | Current penalty multiplier (controls feasibility pressure) |
| `alpha` | float | Elite proximity weight (used by elite method) |
| `elapsed_sec` | float | **Total runtime since start (seconds)** |
| `gen_time_sec` | float | **Time spent in this generation (seconds)** |

### Plot Output

**Per-run dashboard (2×2 grid):**

1. **Top-left:** Convergence curves
   - `best_overall_obj` (dashed, ignores feasibility)
   - `best_feasible_obj` (solid, only feasible)
   - Shows quality of solutions found

2. **Top-right:** Feasible ratio over time
   - Green fill = proportion of feasible individuals
   - Shows when population becomes feasible

3. **Bottom-left:** Penalty parameters (dual axes)
   - Left: `lambda` (penalty weight schedule)
   - Right: `alpha` (elite proximity weight)
   - Shows how penalty parameters evolve

4. **Bottom-right:** Runtime information
   - Left: cumulative time (elapsed_sec)
   - Right: per-generation time (gen_time_sec)
   - Useful for identifying computational bottlenecks

**Comparison plot (1×2 grid):**

1. **Left:** All runs overlaid
   - One line per individual run
   - Shows variability across random seeds

2. **Right:** Mean ± Std Dev
   - Aggregated per method across seeds
   - Shows method robustness and convergence

---

## Constraint-Handling Methods Explained

### **static** (StaticPenalty)
Fixed penalty: `fitness = objective + lambda * violation`
- Simple, no adaptation
- Lambda = 100 (constant)
- Best when problem difficulty is known

### **dynamic** (DynamicPenalty)
Penalty increases over time: `lambda(t) = lambda0 * (1 + dyn_k * t)^dyn_p`
- Starts soft (allows infeasibility exploration)
- Becomes strict as generations progress
- Encourages feasibility in later stages

### **adaptive** (AdaptivePenalty)
Penalty responds to feasibility: `lambda ← lambda * exp(eta * (target − feasible_ratio))`
- If too many infeasible: increase lambda
- If too many feasible: decrease lambda
- Self-balancing

### **elite** (EliteProximityPenalty)
Penalty + proximity term: `fitness = objective + lambda * violation + alpha * distance_to_elite`
- Maintains archive of best feasible solutions
- Guides infeasible solutions toward feasible region
- Mixes dynamic & adaptive lambda updates
- **Generally best for constrained problems**

### **death** (DeathPenalty)
Extreme punishment: infeasible solutions get `obj + 1e12 * (1 + violation)`
- Very aggressive
- May lose diversity
- Works if constraints are simple

### **debrules** (DebRules)
Comparator-based (no penalty arithmetic):
1. Feasible beats infeasible
2. Among feasible: lower objective wins
3. Among infeasible: lower violation wins
- Deterministic, no parameters to tune
- Standard benchmark approach

---

## Benchmark Problems

### **g06** (2D, 2 constraints)
- Simple, low-dimensional
- Good for quick testing
- Global optimum: f* = -6666.515

### **g01** (13D, 9 linear constraints)
- Mixed: binary (0-1) + continuous (0-100)
- Linear constraints only
- Global optimum: f* = -15

### **g07** (10D, 8 nonlinear constraints)
- Quadratic objective & constraints
- Medium difficulty
- Global optimum: f* = 24.306

### **g03** (10D, 1 equality constraint)
- Product maximization
- Equality converted to inequality via tolerance
- Global optimum: f* = -1

---

## Example Workflows

### Workflow 1: Quick comparison (5 min)

```bash
# Run light experiments
python3 run_experiments.py --problem g06 --methods elite death debrules --seeds 2 --gens 100

# Visualize
python3 visualize.py results/*.csv

# View: comparison_plot.png to see which method performs best
```

### Workflow 2: Comprehensive benchmark (30 min)

```bash
# Run all methods, multiple seeds
python3 run_experiments.py --problem g06 --seeds 5 --gens 300

# Visualize all
python3 visualize.py results/*.csv

# Analyze: Individual plots show convergence paths
#          Comparison plot shows mean performance & robustness
```

### Workflow 3: Compare problems

```bash
# Run one method across all problems
for prob in g01 g03 g06 g07; do
  python3 run_experiments.py --problem $prob --methods elite --seeds 3 --gens 200
done

# Visualize by problem
for prob in g01 g03 g06 g07; do
  python3 visualize.py results/${prob}_*.csv
done
```

---

## Tips & Tricks

### Performance Tuning

- **Fast debugging:** `--gen 50 --pop 40` (quick validation)
- **Production runs:** `--gen 300 --pop 100` (better results, slower)
- **Population vs Generations:** Larger pop explores more; more gen refines better

### Timing Analysis

- Check `gen_time_sec` column to find computational bottlenecks
- Per-generation time ~constant? Algorithm scales well
- Per-generation time increasing? Population growing dynamically (shouldn't happen here)

### CSV Analysis (Beyond visualization)

```python
import pandas as pd
df = pd.read_csv("g06_elite_s1.csv")

# When did feasibility first appear?
first_feas = df[df['feasible_ratio'] > 0]['gen'].min()
print(f"First feasible solution at gen {first_feas}")

# Total time spent
total_time = df['elapsed_sec'].iloc[-1]
print(f"Total time: {total_time} sec")

# Best objective found
best_obj = df['best_feasible_obj'].dropna().min()
print(f"Best feasible objective: {best_obj}")
```

---

## Troubleshooting

### Compilation errors

**Error:** `unknown type name 'Vec'` or similar
- Make sure you're compiling `main.cpp`, not a different file
- Use: `clang++ -O2 -std=c++17 main.cpp -o main`

### Python import errors

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install pandas matplotlib
# Or if using venv:
/Users/muratalkaptan/Heuristic/.venv/bin/pip install pandas matplotlib
```

### Plots not generating

- Check that CSV files exist: `ls results/*.csv`
- Verify CSV format: `head -2 results/g06_elite_s1.csv`
- Run with explicit path: `python3 visualize.py /full/path/to/file.csv`

### Experiments hang

- Check if `./main` is built: `ls -l main`
- Recompile: `clang++ -O2 -std=c++17 main.cpp -o main`
- Run single test first: `./main --problem g06 --method elite --seed 1 --gen 10`

---

## Project Structure

```
Heuristic/
├── main.cpp                 # Main GA implementation
├── visualize.py            # Single/multi-run visualization
├── run_experiments.py       # Batch experiment runner
├── README.md               # This file
├── main                    # Compiled executable
├── results/                # Experiment outputs (auto-created)
│   ├── g06_elite_s1.csv
│   ├── g06_elite_s2.csv
│   └── ...
└── *.png                   # Generated plots
```

---

## References

- **Benchmark Problems:** Liang et al., "Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization"
- **GA Operators:** SBX crossover + polynomial mutation (standard in evolutionary algorithms)
- **Constraint Handling:** Multiple methods following Coello & Montes (2002) framework

---

## License

This project is provided as-is for research and educational purposes.

---

## Questions?

Review the code comments in `main.cpp` for implementation details of each method.
Check individual plot bottom-right panels for timing breakdown per run.
