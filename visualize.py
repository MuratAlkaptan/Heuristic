import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "ga_sweeps/summary.csv"
SAVE_DIR = "ga_sweeps/plots"
TOP_K = 20   # how many best solutions to inspect closely

import os
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# LOAD + CLEAN
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Normalize feasibility
df["feasible"] = df["feasible"].str.upper() == "YES"

# Drop broken rows (failed runs, nan fitness)
df = df.dropna(subset=["bestFitness", "quality", "timeDays", "costEUR"])

print(f"Loaded {len(df)} runs")

# -----------------------------
# 1) Fitness distribution (global picture)
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["bestFitness"], bins=40, kde=True)
plt.title("Distribution of Best Fitness Across All Runs")
plt.xlabel("Best Fitness")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fitness_distribution.png")
plt.close()

# -----------------------------
# 2) Feasible vs infeasible fitness
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="feasible", y="bestFitness")
plt.title("Fitness vs Feasibility")
plt.xlabel("Feasible")
plt.ylabel("Best Fitness")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fitness_vs_feasible.png")
plt.close()

# -----------------------------
# 3) Population size sensitivity
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="popSize", y="bestFitness")
plt.title("Effect of Population Size on Best Fitness")
plt.xlabel("Population Size")
plt.ylabel("Best Fitness")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fitness_vs_popsize.png")
plt.close()

# -----------------------------
# 4) Mutation rate sensitivity
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="pMut", y="bestFitness")
plt.title("Effect of Mutation Rate on Best Fitness")
plt.xlabel("Mutation Rate")
plt.ylabel("Best Fitness")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fitness_vs_mutation.png")
plt.close()

# -----------------------------
# 5) Penalty strength tradeoff
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="alphaTime",
    y="alphaBudget",
    size="bestFitness",
    hue="feasible",
    sizes=(20, 300),
    alpha=0.7
)
plt.title("Penalty Scaling Landscape")
plt.xlabel("alphaTime")
plt.ylabel("alphaBudget")
plt.legend(title="Feasible", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/penalty_landscape.png")
plt.close()

# -----------------------------
# 6) Best feasible solutions (Pareto-like view)
# -----------------------------
feas = df[df["feasible"]].copy()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=feas,
    x="timeDays",
    y="costEUR",
    size="quality",
    hue="quality",
    sizes=(30, 300),
    palette="viridis",
    alpha=0.8
)
plt.axvline(10.0, linestyle="--", color="red", alpha=0.5)
plt.axhline(500.0, linestyle="--", color="red", alpha=0.5)
plt.title("Feasible Solutions: Cost–Time–Quality Tradeoff")
plt.xlabel("Time (days)")
plt.ylabel("Cost (EUR)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feasible_tradeoff.png")
plt.close()

# -----------------------------
# 7) Top-K solutions table (printed)
# -----------------------------
top = df.sort_values("bestFitness", ascending=False).head(TOP_K)

print("\nTop solutions:")
print(
    top[
        ["bestFitness","quality","timeDays","costEUR",
         "feasible","popSize","pMut","alphaTime","alphaBudget","seed"]
    ].to_string(index=False)
)

# -----------------------------
# 8) Stability across seeds (important)
# -----------------------------
plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="seed", y="bestFitness")
plt.title("Fitness Stability Across Random Seeds")
plt.xlabel("Seed")
plt.ylabel("Best Fitness")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fitness_vs_seed.png")
plt.close()

print(f"\nPlots saved under: {SAVE_DIR}")
