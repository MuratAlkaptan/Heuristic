// hybrid: small random repair attempt, then evaluate with penalties
//O(N * maxSwaps) where N is the number of cities in the permutation
static int randi_local(int lo, int hi) { return lo + (rand() % (hi - lo + 1)); }

Metrics assess(Genome& g,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit,
               double alphaTime, double alphaBudget,
               int maxSwaps) {

    int N = cities.size();
    g.len = max(1, g.len);

    // quick repair attempt: swap a prefix city with a non-prefix city if it reduces violation
    Metrics cur = evaluate(cities, travel, g, budget, timeLimit, 0, 0);

    if (!cur.feasible && g.len < N && maxSwaps > 0) {
        for (int it = 0; it < maxSwaps; ++it) {
            int i = randi_local(0, g.len - 1);
            int j = randi_local(g.len, N - 1);

            swap(g.perm[i], g.perm[j]);

            Metrics cand = evaluate(cities, travel, g, budget, timeLimit, 0, 0);

            double curV = max(0.0, cur.timeDays - timeLimit) + max(0.0, cur.costEUR - budget);
            double canV = max(0.0, cand.timeDays - timeLimit) + max(0.0, cand.costEUR - budget);

            if (canV <= curV) {
                cur = cand;
                if (cur.feasible) break;
            } else {
                swap(g.perm[i], g.perm[j]); // revert
            }
        }
    }

    // final evaluation uses penalties
    return evaluate(cities, travel, g, budget, timeLimit, alphaTime, alphaBudget);
}
