// repair-only: modifies genome until feasible, then returns unpenalized fitness
//O(N^2) in the worst case where N is the number of cities
//worst case happens when we have to reduce length from N to 1
Metrics assess(Genome& g,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit,
               double /*alphaTime unused*/, double /*alphaBudget unused*/,
               int /*methodParam unused*/) {

    g.len = cities.size();

    Metrics m = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
    while (!m.feasible && g.len > 1) {
        g.len--;
        m = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
    }

    // For repair-only, fitness is just quality (no penalties)
    m.fitness = m.quality;
    return m;
}
