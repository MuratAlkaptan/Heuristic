// penalty-only: does not modify genome, just evaluates with penalties
//O(number_of_cities_in_permutation) Time
//O(1) space
Metrics assess(Genome& g,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit,
               double alphaTime, double alphaBudget,
               int methodParam) {
    //O(number_of_cities_in_permutation)
    return evaluate(cities, travel, g, budget, timeLimit, alphaTime, alphaBudget);
}
