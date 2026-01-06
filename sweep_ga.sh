#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
MAIN_CPP="main.cpp"
CITIES_CSV="world_cities.csv"
TRAVEL_CSV="world_travel_hours.csv"

# Compiler flags
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -I.}"

# Output
OUTDIR="ga_sweeps"
mkdir -p "$OUTDIR"
SUMMARY_CSV="$OUTDIR/summary.csv"

# Parameter sweeps (edit these freely)
POPS=(10 50 100 200)
GENS=(200)
TOURNK=(2 3)
PCROSS=(0.9)
PMUT=(0.20 0.35 0.50)

# Penalty scaling sweep (helps diagnose infeasible dominance)
ALPHA_TIME=(200 400 800 1500)
ALPHA_BUDGET=(120 240 600 1200)

# Hybrid methodParam (maxSwaps)
METHOD_PARAM=(50)

# Seeds for statistical robustness
SEEDS=(11 22 33 44 55)

# --- HELPERS ---
require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: Missing required file: $1" >&2
    exit 1
  fi
}

parse_output() {
  # Reads program output file, prints:
  # bestFitness, quality, timeDays, costEUR, feasible, itineraryLen
  local f="$1"

  local fitness quality time cost feasible itlen
  fitness="$(grep -E '^Fitness:' "$f" | awk '{print $2}' | tail -n1)"
  quality="$(grep -E '^Quality:' "$f" | awk '{print $2}' | tail -n1)"
  time="$(grep -E '^Time:' "$f" | awk '{print $2}' | tail -n1)"
  cost="$(grep -E '^Cost:' "$f" | awk '{print $2}' | tail -n1)"
  feasible="$(grep -E '^Feasible:' "$f" | awk '{print $2}' | tail -n1)"
  itlen="$(grep -E '^Itinerary \(' "$f" | sed -E 's/^Itinerary \(([0-9]+) cities\):/\1/' | tail -n1)"

  # Fallbacks if parsing fails
  fitness="${fitness:-nan}"
  quality="${quality:-nan}"
  time="${time:-nan}"
  cost="${cost:-nan}"
  feasible="${feasible:-UNKNOWN}"
  itlen="${itlen:-nan}"

  echo "$fitness,$quality,$time,$cost,$feasible,$itlen"
}

patch_and_build_and_run() {
  local run_id="$1"
  local pop="$2"
  local gens="$3"
  local tourn="$4"
  local pcross="$5"
  local pmut="$6"
  local atime="$7"
  local abud="$8"
  local mparam="$9"
  local seed="${10}"

  local work_cpp="$OUTDIR/main_run_${run_id}.cpp"
  local exe="$OUTDIR/ga_run_${run_id}"
  local log="$OUTDIR/run_${run_id}.log"

  cp "$MAIN_CPP" "$work_cpp"

  # Patch dataset paths (optional; keeps consistent if you move files)
  sed -i.bak \
    -e "s|string citiesPath = \".*\";|string citiesPath = \"$CITIES_CSV\";|g" \
    -e "s|string travelPath = \".*\";|string travelPath = \"$TRAVEL_CSV\";|g" \
    "$work_cpp"

  # Patch srand
  sed -i.bak -E "s/srand\\([0-9]+\\);/srand($seed);/g" "$work_cpp"

  # Patch problem params
  sed -i.bak -E "s/const double alphaTime = [0-9.]+;/const double alphaTime = $atime;/g" "$work_cpp"
  sed -i.bak -E "s/const double alphaBudget = [0-9.]+;/const double alphaBudget = $abud;/g" "$work_cpp"

  # Patch GA params
  sed -i.bak -E "s/const int popSize = [0-9]+;/const int popSize = $pop;/g" "$work_cpp"
  sed -i.bak -E "s/const int gens = [0-9]+;/const int gens = $gens;/g" "$work_cpp"
  sed -i.bak -E "s/const int tournK = [0-9]+;/const int tournK = $tourn;/g" "$work_cpp"
  sed -i.bak -E "s/const double pCross = [0-9.]+;/const double pCross = $pcross;/g" "$work_cpp"
  sed -i.bak -E "s/const double pMut = [0-9.]+;/const double pMut = $pmut;/g" "$work_cpp"

  # Patch methodParam (hybrid maxSwaps)
  sed -i.bak -E "s/const int methodParam = [0-9]+;/const int methodParam = $mparam;/g" "$work_cpp"

  # Build
  "$CXX" $CXXFLAGS "$work_cpp" -o "$exe"

  # Run
  "$exe" > "$log"

  echo "$log"
}

# --- MAIN ---
require_file "$MAIN_CPP"
require_file "$CITIES_CSV"
require_file "$TRAVEL_CSV"

# CSV header
echo "run_id,popSize,gens,tournK,pCross,pMut,alphaTime,alphaBudget,methodParam,seed,bestFitness,quality,timeDays,costEUR,feasible,itineraryLen,logfile" > "$SUMMARY_CSV"

run_id=0
for pop in "${POPS[@]}"; do
  for g in "${GENS[@]}"; do
    for t in "${TOURNK[@]}"; do
      for pc in "${PCROSS[@]}"; do
        for pm in "${PMUT[@]}"; do
          for at in "${ALPHA_TIME[@]}"; do
            for ab in "${ALPHA_BUDGET[@]}"; do
              for mp in "${METHOD_PARAM[@]}"; do
                for seed in "${SEEDS[@]}"; do
                  run_id=$((run_id+1))
                  echo "Running $run_id: pop=$pop gens=$g tourn=$t pC=$pc pM=$pm aT=$at aB=$ab maxSwaps=$mp seed=$seed"

                  log="$(patch_and_build_and_run "$run_id" "$pop" "$g" "$t" "$pc" "$pm" "$at" "$ab" "$mp" "$seed")"
                  metrics="$(parse_output "$log")"

                  echo "$run_id,$pop,$g,$t,$pc,$pm,$at,$ab,$mp,$seed,$metrics,$log" >> "$SUMMARY_CSV"
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Done. Summary: $SUMMARY_CSV"
