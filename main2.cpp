#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

constexpr double HOURS_PER_DAY = 8.0;

// -------------------- Data --------------------

struct City {
    int sites = 0;
    string name, country;
    double satisfaction = 0.0, visitHours = 0.0, dailyStayEUR = 0.0, attractionEUR = 0.0;
};

struct Genome {
    vector<int> perm; // permutation of city IDs
    int len = 1;      // how many from the front are visited
};

struct Metrics {
    double quality = 0.0, timeDays = 0.0, costEUR = 0.0, fitness = -1e300;
    bool feasible = false;
};

// -------------------- CSV --------------------

static vector<string> splitCSV(const string& s) {
    vector<string> out;
    string item;
    stringstream ss(s);
    while (getline(ss, item, ',')) out.push_back(item);
    return out;
}

static void load_dataset(const string& citiesPath, const string& travelPath,
                         vector<City>& cities, vector<vector<double>>& travel) {
    ifstream fin(citiesPath);
    if (!fin) throw runtime_error("Cannot open: " + citiesPath);

    string line;
    getline(fin, line); // header

    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto c = splitCSV(line);
        // id,country,city,x,y,sites,satisfaction,visit_hours,daily_stay_eur,attraction_eur
        if (c.size() < 10) throw runtime_error("Bad cities row");

        City city;
        city.country      = c[1];
        city.name         = c[2];
        city.sites        = stoi(c[5]);
        city.satisfaction = stod(c[6]);
        city.visitHours   = stod(c[7]);
        city.dailyStayEUR = stod(c[8]);
        city.attractionEUR= stod(c[9]);
        cities.push_back(city);
    }
    fin.close();

    int N = (int)cities.size();
    travel.assign(N, vector<double>(N, 0.0));

    fin.open(travelPath);
    if (!fin) throw runtime_error("Cannot open: " + travelPath);

    getline(fin, line); // header
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto t = splitCSV(line);
        if (t.size() < 3) throw runtime_error("Bad travel row");

        int i = stoi(t[0]);
        int j = stoi(t[1]);
        double h = stod(t[2]);
        travel[i][j] = travel[j][i] = h; // symmetric
    }
}

// -------------------- Penalty-only objective --------------------

static Metrics evaluatePenaltyOnly(const vector<City>& cities,
                                  const vector<vector<double>>& travel,
                                  const Genome& g,
                                  double budget, double timeLimit,
                                  double alphaTime, double alphaBudget) {
    Metrics m;
    int N = (int)cities.size();
    int L = max(1, min(g.len, N));

    double timeHours = 0.0, cost = 0.0, quality = 0.0;

    for (int i = 0; i < L; ++i) {
        const City& c = cities[g.perm[i]];
        quality += 5.0 + c.sites + c.satisfaction;
        timeHours += c.visitHours;

        cost += c.dailyStayEUR * ceil(c.visitHours / HOURS_PER_DAY);
        cost += c.attractionEUR * c.sites;

        if (i + 1 < L) timeHours += travel[g.perm[i]][g.perm[i + 1]];
    }

    m.quality = quality;
    m.timeDays = timeHours / HOURS_PER_DAY;
    m.costEUR = cost;
    m.feasible = (m.timeDays <= timeLimit && m.costEUR <= budget);

    double overT = max(0.0, m.timeDays - timeLimit);
    double overB = max(0.0, m.costEUR - budget);

    // penalty-only fitness (always includes penalties; if feasible, penalties are 0)
    m.fitness = quality
              - alphaTime   * overT * overT
              - alphaBudget * (overB / 100.0) * (overB / 100.0);

    return m;
}

// -------------------- GA operators (minimal) --------------------

static int tournamentSelect(const vector<double>& fit, int tournK, mt19937& rng) {
    uniform_int_distribution<int> D(0, (int)fit.size() - 1);
    int best = D(rng);
    for (int i = 1; i < tournK; ++i) {
        int j = D(rng);
        if (fit[j] > fit[best]) best = j;
    }
    return best;
}

static Genome orderCrossover(const Genome& p1, const Genome& p2, mt19937& rng) {
    int N = (int)p1.perm.size();
    uniform_int_distribution<int> D(0, N - 1);
    int l = D(rng), r = D(rng);
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
            used[v] = 1;
        }
    }

    uniform_real_distribution<double> U(0.0, 1.0);
    child.len = (U(rng) < 0.5) ? p1.len : p2.len;
    child.len = max(1, min(child.len, N));
    return child;
}

static void mutate(Genome& g, mt19937& rng) {
    int N = (int)g.perm.size();
    uniform_int_distribution<int> D(0, N - 1);
    swap(g.perm[D(rng)], g.perm[D(rng)]);

    uniform_real_distribution<double> U(0.0, 1.0);
    if (U(rng) < 0.35) {
        int delta = (U(rng) < 0.5) ? -1 : +1;
        g.len = max(1, min(N, g.len + delta));
    }
}

// -------------------- Main --------------------

int main(int argc, char** argv) {
    string citiesPath = (argc > 1) ? argv[1] : "west_europe_cities.csv";
    string travelPath = (argc > 2) ? argv[2] : "west_europe_travel_hours.csv";

    vector<City> cities;
    vector<vector<double>> travel;

    try {
        load_dataset(citiesPath, travelPath, cities, travel);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    const int N = (int)cities.size();
    cout << "Loaded " << N << " cities\n\n";

    // Problem parameters (penalty-only)
    const double budget = 500.0, timeLimit = 4.0;
    const double alphaTime = 200.0, alphaBudget = 120.0;

    // GA parameters (keep yours)
    const int popSize = 10, gens = 50, tournK = 3, elites = 3;
    const double pCross = 0.9, pMut = 0.35;
    const unsigned seed = 12345;

    mt19937 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);

    auto makeRandom = [&]() {
        Genome g;
        g.perm.resize(N);
        iota(g.perm.begin(), g.perm.end(), 0);
        shuffle(g.perm.begin(), g.perm.end(), rng);
        g.len = uniform_int_distribution<int>(3, min(18, N))(rng);
        return g;
    };

    vector<Genome> pop(popSize);
    vector<double> fit(popSize);

    for (int i = 0; i < popSize; ++i) {
        pop[i] = makeRandom();
        fit[i] = evaluatePenaltyOnly(cities, travel, pop[i], budget, timeLimit, alphaTime, alphaBudget).fitness;
    }

    Genome best = pop[0];
    double bestFit = fit[0];

    cout << "Penalty-only GA\n";
    cout << "Budget: " << budget << " EUR | Time: " << timeLimit << " days\n\n";

    for (int gen = 0; gen < gens; ++gen) {
        vector<int> idx(popSize);
        iota(idx.begin(), idx.end(), 0);
        partial_sort(idx.begin(), idx.begin() + elites, idx.end(),
                     [&](int a, int b) { return fit[a] > fit[b]; });

        vector<Genome> next;
        next.reserve(popSize);
        for (int e = 0; e < elites; ++e) next.push_back(pop[idx[e]]);

        while ((int)next.size() < popSize) {
            int pa = tournamentSelect(fit, tournK, rng);
            int pb = tournamentSelect(fit, tournK, rng);
            Genome child = (U(rng) < pCross) ? orderCrossover(pop[pa], pop[pb], rng) : pop[pa];
            if (U(rng) < pMut) mutate(child, rng);
            next.push_back(std::move(child));
        }

        pop = std::move(next);
        double avg = 0.0;

        for (int i = 0; i < popSize; ++i) {
            fit[i] = evaluatePenaltyOnly(cities, travel, pop[i], budget, timeLimit, alphaTime, alphaBudget).fitness;
            avg += fit[i];
            if (fit[i] > bestFit) { bestFit = fit[i]; best = pop[i]; }
        }
        avg /= popSize;

        cout << "Gen " << gen << ": Best fitness = " << bestFit << " | Avg = " << avg << "\n";
    }

    Metrics m = evaluatePenaltyOnly(cities, travel, best, budget, timeLimit, alphaTime, alphaBudget);

    cout << "\n========== Best Solution ==========\n";
    cout << "Fitness: " << m.fitness << "\n";
    cout << "Quality: " << m.quality << "\n";
    cout << "Time: " << m.timeDays << " days\n";
    cout << "Cost: " << m.costEUR << " EUR\n";
    cout << "Feasible: " << (m.feasible ? "YES" : "NO") << "\n\n";

    cout << "Itinerary (" << best.len << " cities):\n";
    for (int i = 0; i < best.len; ++i) {
        const City& c = cities[best.perm[i]];
        cout << "  " << (i + 1) << ". " << c.name << ", " << c.country
             << " (sites=" << c.sites << ", satisfaction=" << c.satisfaction << ")\n";
    }

    return 0;
}
