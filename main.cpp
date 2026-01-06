#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <unordered_set>
#include <cstdint>

using namespace std;

// -----------------------------
// data structures
// -----------------------------
struct City {
    int sites;
    string name;
    string country;
    double satisfaction;
    double visitHours;
    double dailyStayEUR;
    double attractionEUR;
};

struct Genome {
    vector<int> perm;
    int len;
};

struct Metrics {
    double quality = 0;
    double timeDays = 0;
    double costEUR = 0;
    double fitness = std::numeric_limits<double>::lowest();
    bool feasible = false;

    // NEW: total constraint violation; 0 for feasible
    double violation = 0.0;
};

// -----------------------------
// random helpers
// -----------------------------
static int randi(int lo, int hi) { // inclusive
    return lo + (rand() % (hi - lo + 1));
}
static double rand01() {
    return rand() / (double)RAND_MAX;
}

// -----------------------------
// evaluate (objective + feasibility + penalty fitness)
// -----------------------------
// O(L) where L = g.len
Metrics evaluate(const vector<City>& cities, const vector<vector<double>>& travel,
                 const Genome& g, double budget, double timeLimit,
                 double alphaTime, double alphaBudget) {

    Metrics m;
    int L = max(1, min((int)g.perm.size(), g.len));

    double timeHours = 0.0, cost = 0.0, quality = 0.0;

    for (int i = 0; i < L; ++i) {
        const City& c = cities[g.perm[i]];

        // CHANGED: removed constant +5 per city
        quality += c.sites + c.satisfaction;

        timeHours += c.visitHours;
        cost += c.dailyStayEUR * ceil(c.visitHours / 8.0) + c.attractionEUR * c.sites;

        if (i + 1 < L) timeHours += travel[g.perm[i]][g.perm[i + 1]];
    }

    m.quality = quality;
    m.timeDays = timeHours / 8.0;
    m.costEUR = cost;

    double overTime = max(0.0, m.timeDays - timeLimit);
    double overBudget = max(0.0, m.costEUR - budget);

    m.feasible = (overTime <= 0.0 && overBudget <= 0.0);

    // NEW: Deb-rule violation metric (independent of penalty scaling)
    m.violation = overTime + (overBudget / 100.0);

    // Keep your penalty fitness (still useful for ranking feasible candidates)
    if (m.feasible) {
        m.fitness = quality;
    } else {
        m.fitness = quality
            - alphaTime * overTime * overTime
            - alphaBudget * (overBudget / 100.0) * (overBudget / 100.0);
    }

    return m;
}

// -----------------------------
// constraint method
// -----------------------------
#include "constraint_hybrid.h"

// -----------------------------
// Deb-style feasibility-first comparator (for GA selection/replacement)
// -----------------------------
static inline bool better(const Metrics& a, const Metrics& b) {
    if (a.feasible != b.feasible) return a.feasible;    // feasible always preferred
    if (a.feasible) return a.fitness > b.fitness;       // both feasible: higher fitness
    return a.violation < b.violation;                   // both infeasible: smaller violation
}

// -----------------------------
// GA components
// -----------------------------
struct GAResult {
    Genome best;
    Metrics bestM;
};

// tournament selection (uses Deb ordering)
// Time O(K), Space O(1)
int tournament(const vector<Metrics>& metrics, int tournK) {
    int besti = randi(0, (int)metrics.size() - 1);
    for (int i = 1; i < tournK; ++i) {
        int c = randi(0, (int)metrics.size() - 1);
        if (better(metrics[c], metrics[besti])) besti = c;
    }
    return besti;
}

// mutation: swap two cities, small chance to tweak length
void mutate(Genome& g) {
    int N = (int)g.perm.size();
    int i = randi(0, N - 1), j = randi(0, N - 1);
    swap(g.perm[i], g.perm[j]);

    if (rand01() < 0.35) {
        g.len += (rand01() < 0.5) ? -1 : 1;
        g.len = max(1, min(N, g.len));
    }
}

// NEW: heavy mutation to escape stagnation
void mutate_heavy(Genome& g) {
    int N = (int)g.perm.size();

    // segment reversal
    int l = randi(0, N - 1), r = randi(0, N - 1);
    if (l > r) swap(l, r);
    reverse(g.perm.begin() + l, g.perm.begin() + r + 1);

    // multiple random swaps
    int k = randi(2, 6);
    for (int t = 0; t < k; ++t) {
        int i = randi(0, N - 1), j = randi(0, N - 1);
        swap(g.perm[i], g.perm[j]);
    }

    // stronger length perturbation
    int delta = randi(-3, 3);
    g.len = max(1, min(N, g.len + delta));
}

// ordered crossover
Genome crossover(const Genome& p1, const Genome& p2) {
    int N = (int)p1.perm.size();
    int l = randi(0, N - 1), r = randi(0, N - 1);
    if (l > r) swap(l, r);

    Genome child;
    child.perm.assign(N, -1);
    vector<char> used(N, 0);

    for (int i = l; i <= r; ++i) {
        child.perm[i] = p1.perm[i];
        used[p1.perm[i]] = 1;
    }

    int idx = (r + 1) % N;
    for (int k = 0; k < N; ++k) {
        int v = p2.perm[(r + 1 + k) % N];
        if (!used[v]) {
            while (child.perm[idx] != -1) idx = (idx + 1) % N;
            child.perm[idx] = v;
        }
    }

    child.len = (rand01() < 0.5) ? p1.len : p2.len;
    child.len = max(1, min(N, child.len));
    return child;
}

void shuffle_vec(vector<int>& a) {
    for (int i = (int)a.size() - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        swap(a[i], a[j]);
    }
}

// NEW: hash of (len + first len cities) to suppress clones
static uint64_t hash_prefix(const Genome& g) {
    uint64_t h = 1469598103934665603ULL; // FNV-1a offset
    auto mix = [&](uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };

    int L = max(1, min((int)g.perm.size(), g.len));
    mix((uint64_t)L);
    for (int i = 0; i < L; ++i) mix((uint64_t)(g.perm[i] + 1));
    return h;
}

// -----------------------------
// main GA function
// -----------------------------
GAResult runGA(int popSize, int gens, int tournK, double pCross, double pMut,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit, double alphaTime, double alphaBudget,
               int methodParam /* hybrid uses it as maxSwaps */) {

    int N = (int)cities.size();
    vector<Genome> pop(popSize);
    vector<Metrics> metrics(popSize);

    // initialize population
    for (int i = 0; i < popSize; ++i) {
        pop[i].perm.resize(N);
        iota(pop[i].perm.begin(), pop[i].perm.end(), 0);
        shuffle_vec(pop[i].perm);
        pop[i].len = randi(3, min(18, N));

        Genome g = pop[i];
        metrics[i] = assess(g, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
        pop[i] = g;
    }

    GAResult best;
    best.best = pop[0];
    best.bestM = metrics[0];
    for (int i = 1; i < popSize; ++i) {
        if (better(metrics[i], best.bestM)) {
            best.best = pop[i];
            best.bestM = metrics[i];
        }
    }

    // Anti-early-convergence settings (simple, effective)
    const int ELITES = min(2, popSize);                 // reduced elitism pressure
    const int STAG_LIMIT = 25;                          // stagnation trigger
    const int IMMIGRANTS = max(1, popSize / 10);        // 10% random immigrants

    int stagn = 0;

    for (int gen = 0; gen < gens; ++gen) {
        vector<Genome> offspring;
        vector<Metrics> offm;
        offspring.reserve(popSize);
        offm.reserve(popSize);

        // adaptive mutation if stagnant
        double curPMut = (stagn >= STAG_LIMIT) ? min(1.0, pMut * 2.0) : pMut;

        while ((int)offspring.size() < popSize) {
            int a = tournament(metrics, tournK);
            int b = tournament(metrics, tournK);

            Genome child = (rand01() < pCross) ? crossover(pop[a], pop[b]) : pop[a];

            if (rand01() < curPMut) {
                if (stagn >= STAG_LIMIT && rand01() < 0.35) mutate_heavy(child);
                else mutate(child);
            }

            Metrics m = assess(child, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
            offspring.push_back(child);
            offm.push_back(m);
        }

        // -------- Replacement: keep few elites + best offspring (reduces takeover) --------
        vector<int> pidx(popSize);
        iota(pidx.begin(), pidx.end(), 0);
        sort(pidx.begin(), pidx.end(),
             [&](int i, int j) { return better(metrics[i], metrics[j]); });

        vector<int> oidx(popSize);
        iota(oidx.begin(), oidx.end(), 0);
        sort(oidx.begin(), oidx.end(),
             [&](int i, int j) { return better(offm[i], offm[j]); });

        vector<Genome> nextPop;
        vector<Metrics> nextMet;
        nextPop.reserve(popSize);
        nextMet.reserve(popSize);

        // elites from parents
        for (int e = 0; e < ELITES; ++e) {
            nextPop.push_back(pop[pidx[e]]);
            nextMet.push_back(metrics[pidx[e]]);
        }
        // fill from best offspring
        for (int i = 0; (int)nextPop.size() < popSize && i < popSize; ++i) {
            nextPop.push_back(offspring[oidx[i]]);
            nextMet.push_back(offm[oidx[i]]);
        }

        pop = std::move(nextPop);
        metrics = std::move(nextMet);

        // -------- Random immigrants (diversity injection) --------
        for (int k = 0; k < IMMIGRANTS; ++k) {
            int idxWorst = popSize - 1 - k;

            Genome g;
            g.perm.resize(N);
            iota(g.perm.begin(), g.perm.end(), 0);
            shuffle_vec(g.perm);
            g.len = randi(3, min(18, N));

            Metrics mm = assess(g, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
            pop[idxWorst] = g;
            metrics[idxWorst] = mm;
        }

        // -------- Duplicate suppression (breaks clone populations) --------
        unordered_set<uint64_t> seen;
        for (int i = 0; i < popSize; ++i) {
            uint64_t h = hash_prefix(pop[i]);
            if (seen.find(h) != seen.end()) {
                mutate_heavy(pop[i]);
                metrics[i] = assess(pop[i], cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
            } else {
                seen.insert(h);
            }
        }

        // -------- Update best + stagnation counter --------
        int bi = 0;
        for (int i = 1; i < popSize; ++i) if (better(metrics[i], metrics[bi])) bi = i;

        if (better(metrics[bi], best.bestM)) {
            best.best = pop[bi];
            best.bestM = metrics[bi];
            stagn = 0;
        } else {
            stagn++;
        }

        cout << "Gen " << gen
             << ": Best fitness = " << best.bestM.fitness
             << " feasible=" << (best.bestM.feasible ? "YES" : "NO")
             << " viol=" << best.bestM.violation
             << " stagn=" << stagn
             << "\n";
    }

    return best;
}

// -----------------------------
// dataset loading
// -----------------------------
vector<string> split(const string& s, char delim = ',') {
    vector<string> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) out.push_back(item);
    return out;
}

void load_dataset(const string& citiesPath, const string& travelPath,
                  vector<City>& cities, vector<vector<double>>& travelMatrix) {

    ifstream fin(citiesPath.c_str());
    string line;
    getline(fin, line); // header
    while (getline(fin, line)) {
        vector<string> c = split(line);
        City city;
        city.country = c[1];
        city.name = c[2];
        city.sites = stoi(c[5]);
        city.satisfaction = stod(c[6]);
        city.visitHours = stod(c[7]);
        city.dailyStayEUR = stod(c[8]);
        city.attractionEUR = stod(c[9]);
        cities.push_back(city);
    }
    fin.close();

    int N = (int)cities.size();
    travelMatrix.assign(N, vector<double>(N, 0.0));

    fin.open(travelPath.c_str());
    getline(fin, line); // header
    while (getline(fin, line)) {
        vector<string> t = split(line);
        int i = stoi(t[0]);
        int j = stoi(t[1]);
        double hours = stod(t[2]);
        travelMatrix[i][j] = hours;
        travelMatrix[j][i] = hours;
    }
    fin.close();
}

// -----------------------------
// main
// -----------------------------
int main(int argc, char** argv) {
    srand(12345);

    string citiesPath = "world_cities.csv";
    string travelPath = "world_travel_hours.csv";

    vector<City> cities;
    vector<vector<double>> travel;

    load_dataset(citiesPath, travelPath, cities, travel);

    cout << "Loaded " << cities.size() << " cities\n\n";

    // problem params
    const double budget = 500.0;
    const double timeLimit = 10.0;
    const double alphaTime = 200.0;
    const double alphaBudget = 120.0;

    // GA params
    const int popSize = 50;
    const int gens = 200;
    const int tournK = 2;
    const double pCross = 0.9;
    const double pMut = 0.35;

    // hybrid maxSwaps
    const int methodParam = 50;

    GAResult result = runGA(popSize, gens, tournK, pCross, pMut,
                            cities, travel, budget, timeLimit,
                            alphaTime, alphaBudget, methodParam);

    Genome final = result.best;
    Metrics m = assess(final, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);

    cout << "\n========== Best Solution ==========\n";
    cout << "Fitness: " << m.fitness << "\n";
    cout << "Quality: " << m.quality << "\n";
    cout << "Time: " << m.timeDays << " days\n";
    cout << "Cost: " << m.costEUR << " EUR\n";
    cout << "Feasible: " << (m.feasible ? "YES" : "NO") << "\n";
    cout << "Violation: " << m.violation << "\n\n";

    cout << "Itinerary (" << final.len << " cities):\n";
    for (int i = 0; i < final.len; ++i) {
        const City& c = cities[final.perm[i]];
        cout << "  " << (i + 1) << ". " << c.name << ", " << c.country
             << " (sites=" << c.sites << ", satisfaction=" << c.satisfaction << ")\n";
    }

    return 0;
}
