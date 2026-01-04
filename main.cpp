// g++ -O2 -std=c++17 main.cpp -o main
// ./main

#include "ga_baseline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::string;
using std::vector;

static constexpr double HOURS_PER_DAY = 8.0;

struct City {
    // Loaded attributes
    string country;
    string name;

    // Optional (from CSV; not used in optimization, but handy for future visualization)
    double x = 0.0;
    double y = 0.0;

    int sites = 0;                // number of attractions
    double satisfaction = 0.0;    // 0..100 (per city)
    double dailyStayEUR = 0.0;    // lodging cost per day
    double attractionEUR = 0.0;   // avg cost per attraction
    double visitHours = 0.0;      // time to "cover" the city attractions
};

struct Dataset {
    vector<City> cities;
    vector<vector<double>> travelHours; // symmetric NxN
};

enum class Method { PenaltyOnly, RepairOnly, Hybrid };

struct Genome {
    vector<int> perm; // permutation of [0..N-1]
    int len = 1;      // number of visited cities from the front
};

struct Metrics {
    double quality = 0.0;    // raw "quality"
    double timeDays = 0.0;   // total time in days (city + travel)
    double costEUR = 0.0;    // lodging + attractions
    bool feasible = false;   // timeDays <= T and costEUR <= B
    double fitness = -1e300; // what GA maximizes
};

static double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// ----------------------- CSV loading (simple, no quoted commas) -----------------------

static vector<string> splitCSVLine(const string& line) {
    vector<string> out;
    std::stringstream ss(line);
    string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

static Dataset loadDatasetFromCSVs(const string& citiesCsvPath, const string& travelCsvPath) {
    Dataset D;

    // Cities CSV header:
    // id,country,city,x,y,sites,satisfaction,visit_hours,daily_stay_eur,attraction_eur
    {
        std::ifstream fin(citiesCsvPath);
        if (!fin) throw std::runtime_error("Cannot open cities CSV: " + citiesCsvPath);

        string line;
        if (!std::getline(fin, line)) throw std::runtime_error("Cities CSV is empty: " + citiesCsvPath); // header

        // We load by id order (0..N-1). If file is already in id order, this is trivial.
        // If not, we still place each city at its id index.
        vector<City> byId;

        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            auto c = splitCSVLine(line);
            if (c.size() < 10) throw std::runtime_error("Bad cities row (expected 10 columns): " + line);

            int id = std::stoi(c[0]);
            if (id < 0) throw std::runtime_error("Negative city id: " + line);

            if ((int)byId.size() <= id) byId.resize(id + 1);

            City city;
            city.country       = c[1];
            city.name          = c[2];
            city.x             = std::stod(c[3]);
            city.y             = std::stod(c[4]);
            city.sites         = std::stoi(c[5]);
            city.satisfaction  = std::stod(c[6]);
            city.visitHours    = std::stod(c[7]);
            city.dailyStayEUR  = std::stod(c[8]);
            city.attractionEUR = std::stod(c[9]);

            byId[id] = city;
        }

        // Basic sanity check: ensure no "default-empty" city slots exist
        for (int i = 0; i < (int)byId.size(); ++i) {
            if (byId[i].name.empty()) {
                throw std::runtime_error("Missing city id " + std::to_string(i) + " in cities CSV.");
            }
        }

        D.cities = std::move(byId);
    }

    const int N = (int)D.cities.size();
    if (N == 0) throw std::runtime_error("No cities loaded from: " + citiesCsvPath);

    D.travelHours.assign(N, vector<double>(N, 0.0));

    // Travel CSV header:
    // from_id,to_id,travel_hours
    {
        std::ifstream fin(travelCsvPath);
        if (!fin) throw std::runtime_error("Cannot open travel CSV: " + travelCsvPath);

        string line;
        if (!std::getline(fin, line)) throw std::runtime_error("Travel CSV is empty: " + travelCsvPath); // header

        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            auto e = splitCSVLine(line);
            if (e.size() < 3) throw std::runtime_error("Bad travel row (expected 3 columns): " + line);

            int i = std::stoi(e[0]);
            int j = std::stoi(e[1]);
            double h = std::stod(e[2]);

            if (i < 0 || j < 0 || i >= N || j >= N || i == j) {
                throw std::runtime_error("Travel edge out of range: " + line);
            }
            D.travelHours[i][j] = h;
            D.travelHours[j][i] = h; // undirected
        }
    }

    return D;
}

// ----------------------- Objective helpers -----------------------

static double cityQuality(const City& c) {
    // weights: wCities=5 is applied per visited city externally
    return (double)c.sites + c.satisfaction;
}

static Metrics evaluatePlan(
    const Dataset& D,
    const Genome& G,
    Method method,
    double budgetEUR,
    double timeDaysLimit,
    double alphaTime,
    double alphaBudget,
    int hybridIters,
    std::mt19937& rng
);

static void enforcePermutation(Genome& g) {
    // safety: ensure perm is a permutation (assumes it already nearly is)
    // This function is intentionally light; GA operators should preserve perm validity.
    // (No-op in normal operation.)
    (void)g;
}

static Metrics computeMetricsNoRepair(
    const Dataset& D,
    const Genome& G,
    double budgetEUR,
    double timeDaysLimit
) {
    Metrics M;
    int N = (int)D.cities.size();
    int L = std::max(1, std::min(G.len, N));

    double quality = 0.0;
    double timeHours = 0.0;
    double cost = 0.0;

    // Start anywhere => no "arrival" cost/time for first city.
    for (int i = 0; i < L; ++i) {
        int c = G.perm[i];
        const City& city = D.cities[c];

        quality += 5.0 + cityQuality(city);
        timeHours += city.visitHours;

        // lodging days approximated as visitDays = ceil(visitHours / 8)
        double days = std::ceil(city.visitHours / HOURS_PER_DAY);
        cost += city.dailyStayEUR * days;
        cost += city.attractionEUR * (double)city.sites;

        if (i + 1 < L) {
            int n = G.perm[i + 1];
            timeHours += D.travelHours[c][n];
        }
    }

    M.quality = quality;
    M.timeDays = timeHours / HOURS_PER_DAY;
    M.costEUR = cost;
    M.feasible = (M.timeDays <= timeDaysLimit && M.costEUR <= budgetEUR);
    return M;
}

// Repair-only: drop cities until feasible (hard feasibility).
// Removal criterion: remove city with lowest (deltaQuality / deltaTimeDays) => "worst value density".
static void repairDropWorstRatio(
    const Dataset& D,
    Genome& G,
    double budgetEUR,
    double timeDaysLimit
) {
    int N = (int)D.cities.size();
    G.len = std::max(1, std::min(G.len, N));

    while (true) {
        Metrics M = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
        if (M.feasible) return;
        if (G.len <= 1) return;

        int L = G.len;

        int bestRemoveIdx = -1;
        double worstRatio = +1e300; // remove smallest ratio

        for (int i = 0; i < L; ++i) {
            int ci = G.perm[i];
            const City& C = D.cities[ci];

            double dQ = 5.0 + cityQuality(C);

            double dTHours = C.visitHours;
            if (L >= 2) {
                if (i == 0) {
                    int c1 = G.perm[1];
                    dTHours += D.travelHours[ci][c1];
                } else if (i == L - 1) {
                    int cp = G.perm[L - 2];
                    dTHours += D.travelHours[cp][ci];
                } else {
                    int cp = G.perm[i - 1];
                    int cn = G.perm[i + 1];
                    dTHours += D.travelHours[cp][ci] + D.travelHours[ci][cn] - D.travelHours[cp][cn];
                }
            }

            double dTDays = std::max(1e-6, dTHours / HOURS_PER_DAY);
            double ratio = dQ / dTDays;

            if (ratio < worstRatio) {
                worstRatio = ratio;
                bestRemoveIdx = i;
            }
        }

        if (bestRemoveIdx < 0) return;

        std::swap(G.perm[bestRemoveIdx], G.perm[G.len - 1]);
        G.len -= 1;
    }
}

// Hybrid: attempt limited swaps between "in-plan" and "out-of-plan" cities to reduce violation.
// If still infeasible after hybridIters, fall back to penalty.
static void hybridLightSwapRepair(
    const Dataset& D,
    Genome& G,
    double budgetEUR,
    double timeDaysLimit,
    int hybridIters,
    std::mt19937& rng
) {
    int N = (int)D.cities.size();
    G.len = std::max(1, std::min(G.len, N));

    auto violationScore = [&](const Metrics& M)->double {
        double overT = std::max(0.0, M.timeDays - timeDaysLimit);
        double overB = std::max(0.0, M.costEUR - budgetEUR);
        return overT + (overB / 100.0);
    };

    Metrics cur = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
    if (cur.feasible) return;

    std::uniform_int_distribution<int> inDist(0, std::max(0, G.len - 1));
    std::uniform_int_distribution<int> outDist(std::min(G.len, N - 1), N - 1);

    for (int it = 0; it < hybridIters; ++it) {
        if (G.len >= N) break;
        int i = inDist(rng);
        int j = outDist(rng);

        std::swap(G.perm[i], G.perm[j]);
        Metrics cand = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);

        if (violationScore(cand) < violationScore(cur)) {
            cur = cand;
            if (cur.feasible) return;
        } else {
            std::swap(G.perm[i], G.perm[j]);
        }
    }
}

static Metrics evaluatePlan(
    const Dataset& D,
    const Genome& Gin,
    Method method,
    double budgetEUR,
    double timeDaysLimit,
    double alphaTime,
    double alphaBudget,
    int hybridIters,
    std::mt19937& rng
) {
    Genome G = Gin;
    enforcePermutation(G);

    if (method == Method::RepairOnly) {
        repairDropWorstRatio(D, G, budgetEUR, timeDaysLimit);
        Metrics M = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
        M.fitness = M.quality;
        return M;
    }

    if (method == Method::Hybrid) {
        hybridLightSwapRepair(D, G, budgetEUR, timeDaysLimit, hybridIters, rng);
        Metrics M = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
        if (M.feasible) {
            M.fitness = M.quality;
        } else {
            double overT = std::max(0.0, M.timeDays - timeDaysLimit);
            double overB = std::max(0.0, M.costEUR - budgetEUR);
            double penT = alphaTime * overT * overT;
            double penB = alphaBudget * (overB / 100.0) * (overB / 100.0);
            M.fitness = M.quality - (penT + penB);
        }
        return M;
    }

    // Penalty-only
    Metrics M = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
    double overT = std::max(0.0, M.timeDays - timeDaysLimit);
    double overB = std::max(0.0, M.costEUR - budgetEUR);
    double penT = alphaTime * overT * overT;
    double penB = alphaBudget * (overB / 100.0) * (overB / 100.0);
    M.fitness = M.quality - (penT + penB);
    return M;
}

// --- Permutation crossover (Order Crossover, OX) ---
static vector<int> orderCrossoverOX(const vector<int>& p1, const vector<int>& p2, std::mt19937& rng) {
    int N = (int)p1.size();
    std::uniform_int_distribution<int> D(0, N - 1);
    int l = D(rng), r = D(rng);
    if (l > r) std::swap(l, r);

    vector<int> child(N, -1);
    vector<char> used(N, 0);

    for (int i = l; i <= r; ++i) {
        child[i] = p1[i];
        used[p1[i]] = 1;
    }

    int idx = (r + 1) % N;
    for (int k = 0; k < N; ++k) {
        int v = p2[(r + 1 + k) % N];
        if (!used[v]) {
            while (child[idx] != -1) idx = (idx + 1) % N;
            child[idx] = v;
            used[v] = 1;
        }
    }
    return child;
}

static string methodName(Method m) {
    switch (m) {
        case Method::PenaltyOnly: return "PenaltyOnly";
        case Method::RepairOnly:  return "RepairOnly";
        case Method::Hybrid:      return "Hybrid";
    }
    return "Unknown";
}

int main(int argc, char** argv) {
    // Default CSV filenames (put them next to the executable or run with explicit paths)
    string citiesPath = (argc > 1) ? argv[1] : "west_europe_cities.csv";
    string travelPath = (argc > 2) ? argv[2] : "west_europe_travel_hours.csv";

    Dataset D;
    try {
        D = loadDatasetFromCSVs(citiesPath, travelPath);
    } catch (const std::exception& e) {
        std::cerr << "Dataset load error: " << e.what() << "\n";
        std::cerr << "Usage: ./travel [cities.csv] [travel.csv]\n";
        return 1;
    }

    const int N = (int)D.cities.size();

    // Hypothetical user input (fixed for benchmark)
    const double budgetEUR = 500.0;
    const double timeDaysLimit = 4.0;

    // Penalty weights (tuned for this synthetic scale; adjust if you want)
    const double alphaTime = 200.0;   // penalty multiplier for time overrun^2
    const double alphaBudget = 120.0; // penalty multiplier for (budget overrun / 100)^2

    // Hybrid light repair iterations
    const int hybridIters = 50;

    GAParams P;
    P.popSize = 10;
    P.generations = 50;
    P.tournamentK = 3;
    P.elites = 4;
    P.pCrossover = 0.90;
    P.pMutation = 0.35;
    P.seed = 12345;

    auto makeRandom = [&](std::mt19937& rng)->Genome {
        Genome g;
        g.perm.resize(N);
        std::iota(g.perm.begin(), g.perm.end(), 0);
        std::shuffle(g.perm.begin(), g.perm.end(), rng);

        std::uniform_int_distribution<int> L(3, std::min(18, N));
        g.len = L(rng);
        return g;
    };

    auto crossover = [&](const Genome& a, const Genome& b, std::mt19937& rng)->Genome {
        Genome c;
        c.perm = orderCrossoverOX(a.perm, b.perm, rng);

        std::uniform_real_distribution<double> U(0.0, 1.0);
        c.len = (U(rng) < 0.5) ? a.len : b.len;

        c.len = std::max(1, std::min(c.len, N));
        return c;
    };

    auto mutate = [&](Genome& g, std::mt19937& rng) {
        std::uniform_int_distribution<int> Didx(0, N - 1);
        int i = Didx(rng), j = Didx(rng);
        std::swap(g.perm[i], g.perm[j]);

        std::uniform_real_distribution<double> U(0.0, 1.0);
        if (U(rng) < 0.35) {
            int delta = (U(rng) < 0.5) ? -1 : +1;
            g.len = std::max(1, std::min(N, g.len + delta));
        }
    };

    auto runOne = [&](Method method) {
        cout << "\n====================================================\n";
        cout << "Method: " << methodName(method) << "\n";
        cout << "Cities: " << N << " | Budget: " << budgetEUR << " | Time: " << timeDaysLimit << " days\n";
        cout << "Gen  BestFit   Quality   Time(d)  Cost(EUR)  Feasible  AvgFit\n";
        cout << "---- --------- --------- -------  ---------  --------  ---------\n";

        auto fitness = [&](const Genome& g, std::mt19937& rng)->double {
            Metrics M = evaluatePlan(D, g, method, budgetEUR, timeDaysLimit, alphaTime, alphaBudget, hybridIters, rng);
            return M.fitness;
        };

        auto onGen = [&](int gen, const Genome& best, double bestFit, double avgFit) {
            std::mt19937 tmp(999 + (unsigned)gen);
            Metrics M = evaluatePlan(D, best, method, budgetEUR, timeDaysLimit, alphaTime, alphaBudget, hybridIters, tmp);

            cout << std::setw(4) << gen << " "
                 << std::setw(9) << std::fixed << std::setprecision(2) << bestFit << " "
                 << std::setw(9) << M.quality << " "
                 << std::setw(7) << M.timeDays << " "
                 << std::setw(9) << M.costEUR << "  "
                 << std::setw(8) << (M.feasible ? "yes" : "no") << "  "
                 << std::setw(9) << avgFit
                 << "\n";
        };

        GAResult<Genome> R = runGA<Genome>(P, makeRandom, fitness, crossover, mutate, onGen);

        // Final report: print best itinerary
        std::mt19937 tmp(424242);
        Metrics M = evaluatePlan(D, R.bestGenome, method, budgetEUR, timeDaysLimit, alphaTime, alphaBudget, hybridIters, tmp);

        cout << "\nBest itinerary (" << methodName(method) << ")\n";
        cout << "Fitness: " << M.fitness << " | Quality: " << M.quality
             << " | Time: " << M.timeDays << " days"
             << " | Cost: " << M.costEUR << " EUR"
             << " | Feasible: " << (M.feasible ? "yes" : "no") << "\n";

        cout << "Cities (" << R.bestGenome.len << "):\n";
        for (int i = 0; i < R.bestGenome.len; ++i) {
            int id = R.bestGenome.perm[i];
            const auto& c = D.cities[id];
            cout << "  - " << c.name << ", " << c.country
                 << " | sites=" << c.sites
                 << " sat=" << std::fixed << std::setprecision(1) << c.satisfaction
                 << " visitH=" << std::fixed << std::setprecision(1) << c.visitHours
                 << " daily=" << std::fixed << std::setprecision(0) << c.dailyStayEUR
                 << " attr=" << std::fixed << std::setprecision(0) << c.attractionEUR
                 << "\n";
        }
    };

    runOne(Method::PenaltyOnly);
    runOne(Method::RepairOnly);
    runOne(Method::Hybrid);

    return 0;
}
