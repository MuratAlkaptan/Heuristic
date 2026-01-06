#pragma once
#include <algorithm>
#include <cstdlib>

// local rng for header
static inline int randi_local(int lo, int hi) { return lo + (std::rand() % (hi - lo + 1)); }

// Deb-style comparator inside header (keeps header independent)
static inline bool better_local(const Metrics& a, const Metrics& b) {
    if (a.feasible != b.feasible) return a.feasible;
    if (a.feasible) return a.fitness > b.fitness;
    return a.violation < b.violation;
}

// Hybrid assess: light repair via swap attempts that reduce violation / improve feasibility
// maxSwaps = methodParam
static inline Metrics assess(Genome& g,
                             const std::vector<City>& cities,
                             const std::vector<std::vector<double>>& travel,
                             double budget, double timeLimit,
                             double alphaTime, double alphaBudget,
                             int maxSwaps) {

    int N = (int)cities.size();
    g.len = std::max(1, std::min(N, g.len));

    // Evaluate current
    Metrics bestM = evaluate(cities, travel, g, budget, timeLimit, alphaTime, alphaBudget);

    // If already feasible, no need to repair aggressively
    if (bestM.feasible) return bestM;

    // Try to improve by swapping a city inside the prefix with one outside
    // (this tends to change cost/time characteristics)
    int L = g.len;

    for (int t = 0; t < maxSwaps; ++t) {
        if (L <= 0 || L >= N) break;

        int i = randi_local(0, L - 1);
        int j = randi_local(L, N - 1);

        std::swap(g.perm[i], g.perm[j]);
        Metrics candM = evaluate(cities, travel, g, budget, timeLimit, alphaTime, alphaBudget);

        if (better_local(candM, bestM)) {
            bestM = candM; // keep swap
            if (bestM.feasible) break;
        } else {
            // revert
            std::swap(g.perm[i], g.perm[j]);
        }
    }

    return bestM;
}
