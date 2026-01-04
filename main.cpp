// travel_problem.cpp
// Travel itinerary optimization over a fixed "West Europe" synthetic dataset.
// Goal: maximize travel "quality" under soft constraints (time, budget).
// Compare three constraint handlers:
//  1) Penalty-only
//  2) Repair-only (drop lowest quality-per-time until feasible)
//  3) Hybrid (limited swap-in "repair-light"; if still infeasible -> penalty)
//
// Compile: g++ -O2 -std=c++17 travel_problem.cpp -o travel
// Run:     ./travel

#include "ga_baseline.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::string;
using std::vector;

static constexpr double HOURS_PER_DAY = 8.0;

struct City {
    string country;
    string name;

    // Synthetic attributes (fixed after generation)
    int sites = 0;                // number of attractions
    double satisfaction = 0.0;    // 0..100 (per city)
    double dailyStayEUR = 0.0;    // lodging cost per day
    double attractionEUR = 0.0;   // avg cost per attraction
    double visitHours = 0.0;      // time to "cover" the city attractions
};

// Synthetic 2D coordinate (km) for travel time graph generation
struct Pt { double x=0, y=0; };

struct Dataset {
    vector<City> cities;
    vector<Pt> pos;
    vector<vector<double>> travelHours; // symmetric
};

enum class Method { PenaltyOnly, RepairOnly, Hybrid };

struct Genome {
    vector<int> perm; // permutation of [0..N-1]
    int len = 1;      // number of visited cities from the front
};

struct Metrics {
    double quality = 0.0;   // raw "quality"
    double timeDays = 0.0;  // total time in days (city + travel)
    double costEUR = 0.0;   // lodging + attractions
    bool feasible = false;  // timeDays <= T and costEUR <= B
    double fitness = -1e300; // what GA maximizes
};

static double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

static Dataset buildWestEuropeDataset() {
    // Fixed city list + fixed synthetic positions (km) for reproducible travel times.
    // City attributes are generated deterministically from a fixed RNG seed.
    Dataset D;

    // Country anchors (rough synthetic map of West Europe)
    auto add = [&](const string& country, const string& name, double ax, double ay, double dx, double dy) {
        D.cities.push_back(City{country, name});
        D.pos.push_back(Pt{ax + dx, ay + dy});
    };

    // Anchors
    const Pt Portugal{  0,   0};
    const Pt Spain   {200, 120};
    const Pt France  {450, 260};
    const Pt Belgium {520, 420};
    const Pt Neth    {540, 470};
    const Pt Lux     {540, 405};
    const Pt UK      {360, 520};
    const Pt Ireland {250, 560};
    const Pt Swiss   {610, 290};
    const Pt Germany {680, 420};
    const Pt Austria {780, 380};
    const Pt Italy   {650, 170};

    // Portugal
    add("Portugal","Lisbon",  Portugal.x,Portugal.y,  10,  0);
    add("Portugal","Porto",   Portugal.x,Portugal.y,  30, 80);

    // Spain
    add("Spain","Madrid",     Spain.x,Spain.y,  10,  10);
    add("Spain","Barcelona",  Spain.x,Spain.y, 140,  60);
    add("Spain","Valencia",   Spain.x,Spain.y, 120,  10);
    add("Spain","Seville",    Spain.x,Spain.y,  40, -60);
    add("Spain","Granada",    Spain.x,Spain.y,  80, -70);
    add("Spain","Bilbao",     Spain.x,Spain.y,  60, 110);

    // France
    add("France","Paris",     France.x,France.y,  40, 160);
    add("France","Lyon",      France.x,France.y, 180,  10);
    add("France","Marseille", France.x,France.y, 220, -80);
    add("France","Nice",      France.x,France.y, 260, -60);
    add("France","Bordeaux",  France.x,France.y, -40,  20);
    add("France","Toulouse",  France.x,France.y,  10, -20);
    add("France","Nantes",    France.x,France.y, -70,  90);
    add("France","Strasbourg",France.x,France.y, 320,  80);
    add("France","Lille",     France.x,France.y,  60, 210);

    // Belgium
    add("Belgium","Brussels", Belgium.x,Belgium.y,  0,  0);
    add("Belgium","Bruges",   Belgium.x,Belgium.y, -40, -10);
    add("Belgium","Antwerp",  Belgium.x,Belgium.y,  10,  20);
    add("Belgium","Ghent",    Belgium.x,Belgium.y, -20,  10);

    // Netherlands
    add("Netherlands","Amsterdam", Neth.x,Neth.y,  0,  0);
    add("Netherlands","Rotterdam", Neth.x,Neth.y, -20, -30);
    add("Netherlands","The Hague", Neth.x,Neth.y, -30, -20);
    add("Netherlands","Utrecht",   Neth.x,Neth.y,  10, -10);

    // Luxembourg
    add("Luxembourg","Luxembourg City", Lux.x,Lux.y, 0, 0);

    // UK
    add("United Kingdom","London",     UK.x,UK.y,  30,  0);
    add("United Kingdom","Edinburgh",  UK.x,UK.y, -30, 140);
    add("United Kingdom","Manchester", UK.x,UK.y, -10,  70);
    add("United Kingdom","Liverpool",  UK.x,UK.y, -30,  60);
    add("United Kingdom","Bristol",    UK.x,UK.y,  10,  20);

    // Ireland
    add("Ireland","Dublin",  Ireland.x,Ireland.y,  0,  0);
    add("Ireland","Cork",    Ireland.x,Ireland.y, -30, -50);
    add("Ireland","Galway",  Ireland.x,Ireland.y, -60, -10);

    // Switzerland
    add("Switzerland","Zurich",   Swiss.x,Swiss.y,  20,  80);
    add("Switzerland","Geneva",   Swiss.x,Swiss.y, -60,  10);
    add("Switzerland","Basel",    Swiss.x,Swiss.y, -10,  90);
    add("Switzerland","Bern",     Swiss.x,Swiss.y, -20,  60);
    add("Switzerland","Lausanne", Swiss.x,Swiss.y, -40,  20);

    // Germany
    add("Germany","Frankfurt",  Germany.x,Germany.y, -40, -10);
    add("Germany","Cologne",    Germany.x,Germany.y, -70,  20);
    add("Germany","Dusseldorf", Germany.x,Germany.y, -80,  30);
    add("Germany","Stuttgart",  Germany.x,Germany.y, -10, -70);
    add("Germany","Munich",     Germany.x,Germany.y,  40, -90);
    add("Germany","Hamburg",    Germany.x,Germany.y,  20, 140);
    add("Germany","Berlin",     Germany.x,Germany.y, 220,  70);
    add("Germany","Heidelberg", Germany.x,Germany.y, -30, -40);

    // Austria
    add("Austria","Vienna",    Austria.x,Austria.y,  60,  10);
    add("Austria","Salzburg",  Austria.x,Austria.y,   0, -30);
    add("Austria","Innsbruck", Austria.x,Austria.y, -40, -50);

    // Italy (north/central emphasis)
    add("Italy","Milan",     Italy.x,Italy.y,   0,  60);
    add("Italy","Turin",     Italy.x,Italy.y, -60,  50);
    add("Italy","Venice",    Italy.x,Italy.y,  80,  60);
    add("Italy","Florence",  Italy.x,Italy.y,  20,  10);
    add("Italy","Bologna",   Italy.x,Italy.y,  50,  30);
    add("Italy","Rome",      Italy.x,Italy.y,  40, -40);

    // --- Generate synthetic attributes deterministically ---
    std::mt19937 rng(2026);
    auto ur = [&](double a, double b) {
        std::uniform_real_distribution<double> U(a, b);
        return U(rng);
    };
    auto ui = [&](int a, int b) {
        std::uniform_int_distribution<int> U(a, b);
        return U(rng);
    };

    auto countryDailyBase = [&](const string& c)->double {
        if (c == "Switzerland") return 170;
        if (c == "Luxembourg") return 155;
        if (c == "United Kingdom") return 145;
        if (c == "Ireland") return 135;
        if (c == "France") return 135;
        if (c == "Netherlands") return 125;
        if (c == "Belgium") return 120;
        if (c == "Germany") return 120;
        if (c == "Austria") return 115;
        if (c == "Italy") return 115;
        if (c == "Spain") return 100;
        if (c == "Portugal") return 95;
        return 120;
    };

    auto countryAttractionBase = [&](const string& c)->double {
        if (c == "Switzerland") return 22;
        if (c == "Luxembourg") return 20;
        if (c == "United Kingdom") return 20;
        if (c == "Ireland") return 18;
        if (c == "France") return 18;
        if (c == "Netherlands") return 16;
        if (c == "Belgium") return 15;
        if (c == "Germany") return 15;
        if (c == "Austria") return 14;
        if (c == "Italy") return 16;
        if (c == "Spain") return 12;
        if (c == "Portugal") return 11;
        return 15;
    };

    for (auto& city : D.cities) {
        // Sites: create variance; major capitals tend to have more sites
        bool capitalish = (city.name == "Paris" || city.name == "London" || city.name == "Amsterdam" ||
                           city.name == "Madrid" || city.name == "Rome" || city.name == "Berlin" ||
                           city.name == "Vienna" || city.name == "Dublin" || city.name == "Brussels" ||
                           city.name == "Lisbon" || city.name == "Zurich");

        city.sites = capitalish ? ui(14, 26) : ui(8, 20);

        // Satisfaction: capitals slightly higher; add randomness
        double baseSat = capitalish ? ur(78, 95) : ur(65, 90);
        city.satisfaction = clamp(baseSat, 40, 98);

        // Visit hours: proportional to sites with random "density"
        double hoursPerSite = ur(0.8, 1.6);
        city.visitHours = clamp(city.sites * hoursPerSite, 6.0, 28.0); // 0.75 to 3.5 days at 8h/day

        // Costs
        double dailyBase = countryDailyBase(city.country);
        double attractBase = countryAttractionBase(city.country);
        city.dailyStayEUR = clamp(dailyBase * ur(0.85, 1.20), 60.0, 220.0);
        city.attractionEUR = clamp(attractBase * ur(0.75, 1.25), 6.0, 35.0);
    }

    // --- Build travel time matrix (hours), symmetric, deterministic ---
    int N = (int)D.cities.size();
    D.travelHours.assign(N, vector<double>(N, 0.0));

    // Deterministic "noise" for each pair using seeded RNG, but stable.
    std::mt19937 rng2(777);
    std::uniform_real_distribution<double> noise(0.90, 1.30);

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = D.pos[i].x - D.pos[j].x;
            double dy = D.pos[i].y - D.pos[j].y;
            double distKm = std::sqrt(dx*dx + dy*dy);

            double base = 0.5 + (distKm / 120.0); // synthetic "effective" speed
            double t = base * noise(rng2);

            // clamp unrealistic extremes
            t = clamp(t, 0.4, 18.0);
            D.travelHours[i][j] = D.travelHours[j][i] = t;
        }
    }

    return D;
}

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
// Removal criterion: remove city with worst (deltaQuality / deltaTimeDays).
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

        // Identify removal that hurts quality least per unit time saved.
        // deltaTime includes: visitHours + travel edge adjustments.
        int bestRemoveIdx = -1;
        double worstRatio = +1e300; // we want smallest quality/time? Actually "lowest quality per time" => remove smallest ratio.
        // We'll compute ratio = deltaQuality / deltaTimeSaved; remove smallest ratio.

        for (int i = 0; i < L; ++i) {
            int ci = G.perm[i];
            const City& C = D.cities[ci];

            // quality contribution of city
            double dQ = 5.0 + cityQuality(C);

            // time saved: visitHours + travel edges adjustment
            double dTHours = C.visitHours;

            if (L >= 2) {
                if (i == 0) {
                    // removing first city removes travel from city0->city1
                    int c1 = G.perm[1];
                    dTHours += D.travelHours[ci][c1];
                } else if (i == L - 1) {
                    // removing last city removes travel from prev->last
                    int cp = G.perm[L - 2];
                    dTHours += D.travelHours[cp][ci];
                } else {
                    int cp = G.perm[i - 1];
                    int cn = G.perm[i + 1];
                    // remove cp->ci and ci->cn, add cp->cn
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

        // Remove by swapping that city with last visited and decreasing len
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
        // Normalize budget overshoot to "hundreds of euros" scale
        return overT + (overB / 100.0);
    };

    Metrics cur = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);
    if (cur.feasible) return;

    std::uniform_int_distribution<int> inDist(0, std::max(0, G.len - 1));
    std::uniform_int_distribution<int> outDist(std::min(G.len, N - 1), N - 1);

    for (int it = 0; it < hybridIters; ++it) {
        if (G.len >= N) break; // nowhere to swap from outside
        int i = inDist(rng);
        int j = outDist(rng);

        // Try swap a visited city with a non-visited one (changes set, keeps len).
        std::swap(G.perm[i], G.perm[j]);

        Metrics cand = computeMetricsNoRepair(D, G, budgetEUR, timeDaysLimit);

        // Greedy accept if it reduces violation; else revert.
        if (violationScore(cand) < violationScore(cur)) {
            cur = cand;
            if (cur.feasible) return;
        } else {
            std::swap(G.perm[i], G.perm[j]); // revert
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
        // Repair-only uses hard feasibility; fitness is just quality (feasible by construction in normal cases).
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
            // penalties (squared)
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

int main() {
    Dataset D = buildWestEuropeDataset();
    const int N = (int)D.cities.size();

    // Hypothetical user input (fixed for benchmark)
    const double budgetEUR = 1000.0;
    const double timeDaysLimit = 7.0;

    // Penalty weights (tuned for this synthetic scale; adjust if you want)
    const double alphaTime = 200.0;   // penalty multiplier for time overrun^2
    const double alphaBudget = 120.0; // penalty multiplier for (budget overrun / 100)^2

    // Hybrid light repair iterations
    const int hybridIters = 50;

    GAParams P;
    P.popSize = 220;
    P.generations = 250;
    P.tournamentK = 3;
    P.elites = 2;
    P.pCrossover = 0.90;
    P.pMutation = 0.35;
    P.seed = 12345;

    auto makeRandom = [&](std::mt19937& rng)->Genome {
        Genome g;
        g.perm.resize(N);
        std::iota(g.perm.begin(), g.perm.end(), 0);
        std::shuffle(g.perm.begin(), g.perm.end(), rng);

        // Favor realistic initial lengths for 7 days (still allow exploration)
        std::uniform_int_distribution<int> L(3, std::min(18, N));
        g.len = L(rng);
        return g;
    };

    auto crossover = [&](const Genome& a, const Genome& b, std::mt19937& rng)->Genome {
        Genome c;
        c.perm = orderCrossoverOX(a.perm, b.perm, rng);

        // inherit/average length
        std::uniform_real_distribution<double> U(0.0, 1.0);
        if (U(rng) < 0.5) c.len = a.len;
        else c.len = b.len;

        c.len = std::max(1, std::min(c.len, N));
        return c;
    };

    auto mutate = [&](Genome& g, std::mt19937& rng) {
        std::uniform_int_distribution<int> Didx(0, N - 1);
        int i = Didx(rng), j = Didx(rng);
        std::swap(g.perm[i], g.perm[j]);

        // occasionally adjust length
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
            // Recompute metrics for logging (using a deterministic RNG copy to avoid side effects)
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
