// g++ -O2 -std=c++17 travel_ga.cpp -o travel_ga
// ./travel_ga [cities.csv] [travel.csv]

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

using namespace std;

constexpr double HOURS_PER_DAY = 8.0;

// ===================== Data Structures =====================

struct City {
    string country, name;
    double x, y;
    int sites;
    double satisfaction, visitHours, dailyStayEUR, attractionEUR;
};

struct Genome {
    vector<int> perm;  // city permutation
    int len = 1;       // number of cities to visit
};

struct Metrics {
    double quality = 0, timeDays = 0, costEUR = 0, fitness = -1e300;
    bool feasible = false;
};

// ===================== CSV Loading =====================

vector<string> split(const string& s, char delim = ',') {
    vector<string> result;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) result.push_back(item);
    return result;
}

void loadDataset(const string& citiesPath, const string& travelPath,
                 vector<City>& cities, vector<vector<double>>& travelHours) {
    ifstream fin(citiesPath);
    if (!fin) throw runtime_error("Cannot open: " + citiesPath);
    
    string line;
    getline(fin, line); // skip header
    
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto c = split(line);
        if (c.size() < 10) throw runtime_error("Bad cities row");
        
        City city;
        city.country = c[1];
        city.name = c[2];
        city.x = stod(c[3]);
        city.y = stod(c[4]);
        city.sites = stoi(c[5]);
        city.satisfaction = stod(c[6]);
        city.visitHours = stod(c[7]);
        city.dailyStayEUR = stod(c[8]);
        city.attractionEUR = stod(c[9]);
        cities.push_back(city);
    }
    fin.close();
    
    int N = cities.size();
    travelHours.assign(N, vector<double>(N, 0.0));
    
    fin.open(travelPath);
    if (!fin) throw runtime_error("Cannot open: " + travelPath);
    
    getline(fin, line); // skip header
    while (getline(fin, line)) {
        if (line.empty()) continue;
        auto e = split(line);
        if (e.size() < 3) throw runtime_error("Bad travel row");
        
        int i = stoi(e[0]), j = stoi(e[1]);
        double h = stod(e[2]);
        travelHours[i][j] = travelHours[j][i] = h;
    }
}

// ===================== Evaluation & Repair =====================

Metrics evaluate(const vector<City>& cities, const vector<vector<double>>& travel,
                 const Genome& g, double budget, double timeLimit,
                 double alphaTime, double alphaBudget) {
    Metrics m;
    int L = min(g.len, (int)cities.size());
    
    double timeHours = 0, cost = 0, quality = 0;
    
    for (int i = 0; i < L; ++i) {
        const City& c = cities[g.perm[i]];
        
        quality += 5.0 + c.sites + c.satisfaction;
        timeHours += c.visitHours;
        cost += c.dailyStayEUR * ceil(c.visitHours / HOURS_PER_DAY);
        cost += c.attractionEUR * c.sites;
        
        if (i + 1 < L) {
            timeHours += travel[g.perm[i]][g.perm[i + 1]];
        }
    }
    
    m.quality = quality;
    m.timeDays = timeHours / HOURS_PER_DAY;
    m.costEUR = cost;
    m.feasible = (m.timeDays <= timeLimit && m.costEUR <= budget);
    
    if (m.feasible) {
        m.fitness = quality;
    } else {
        double overT = max(0.0, m.timeDays - timeLimit);
        double overB = max(0.0, m.costEUR - budget);
        m.fitness = quality - alphaTime * overT * overT - alphaBudget * (overB / 100) * (overB / 100);
    }
    
    return m;
}

// Hybrid repair: try swaps to reduce violation (doesn't guarantee feasibility)
void hybridRepair(const vector<City>& cities, const vector<vector<double>>& travel,
                  Genome& g, double budget, double timeLimit, int maxSwaps, mt19937& rng) {
    int N = cities.size();
    g.len = max(1, min(g.len, N));
    
    auto violation = [&](const Metrics& m) {
        return max(0.0, m.timeDays - timeLimit) + max(0.0, m.costEUR - budget) / 100.0;
    };
    
    Metrics cur = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
    if (cur.feasible) return;
    
    // Try swaps between in-plan and out-of-plan cities to reduce violation
    if (g.len < N) {
        uniform_int_distribution<int> inDist(0, g.len - 1);
        uniform_int_distribution<int> outDist(g.len, N - 1);
        
        for (int it = 0; it < maxSwaps; ++it) {
            int i = inDist(rng), j = outDist(rng);
            swap(g.perm[i], g.perm[j]);
            
            Metrics cand = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
            if (violation(cand) < violation(cur)) {
                cur = cand;
                if (cur.feasible) return;
            } else {
                swap(g.perm[i], g.perm[j]); // revert
            }
        }
    }
    // Note: We don't force feasibility - penalties will guide the GA
}

// ===================== GA Operators =====================

vector<int> orderCrossover(const vector<int>& p1, const vector<int>& p2, mt19937& rng) {
    int N = p1.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int l = dist(rng), r = dist(rng);
    if (l > r) swap(l, r);
    
    vector<int> child(N, -1);
    vector<bool> used(N, false);
    
    for (int i = l; i <= r; ++i) {
        child[i] = p1[i];
        used[p1[i]] = true;
    }
    
    int idx = (r + 1) % N;
    for (int k = 0; k < N; ++k) {
        int v = p2[(r + 1 + k) % N];
        if (!used[v]) {
            while (child[idx] != -1) idx = (idx + 1) % N;
            child[idx] = v;
        }
    }
    return child;
}

// ===================== GA Engine =====================

template<typename T>
struct GAResult {
    T best;
    double fitness = -1e300;
};

template<typename T>
GAResult<T> runGA(int popSize, int gens, int tournK, int elites, double pCross, double pMut,
                  function<T(mt19937&)> makeRandom, function<double(const T&, mt19937&)> fitness,
                  function<T(const T&, const T&, mt19937&)> crossover,
                  function<void(T&, mt19937&)> mutate,
                  function<void(int, const T&, double, double)> onGen, unsigned seed) {
    mt19937 rng(seed);
    vector<T> pop(popSize);
    vector<double> fit(popSize);
    
    for (int i = 0; i < popSize; ++i) {
        pop[i] = makeRandom(rng);
        fit[i] = fitness(pop[i], rng);
    }
    
    GAResult<T> best;
    best.best = pop[0];
    best.fitness = fit[0];
    for (int i = 1; i < popSize; ++i) {
        if (fit[i] > best.fitness) {
            best.fitness = fit[i];
            best.best = pop[i];
        }
    }
    
    uniform_real_distribution<double> U(0, 1);
    uniform_int_distribution<int> D(0, popSize - 1);
    
    for (int gen = 0; gen < gens; ++gen) {
        vector<int> idx(popSize);
        iota(idx.begin(), idx.end(), 0);
        partial_sort(idx.begin(), idx.begin() + elites, idx.end(),
                     [&](int a, int b) { return fit[a] > fit[b]; });
        
        vector<T> next;
        for (int e = 0; e < elites; ++e) next.push_back(pop[idx[e]]);
        
        while ((int)next.size() < popSize) {
            auto tournament = [&]() {
                int b = D(rng);
                for (int i = 1; i < tournK; ++i) {
                    int j = D(rng);
                    if (fit[j] > fit[b]) b = j;
                }
                return b;
            };
            
            int pa = tournament(), pb = tournament();
            T child = (U(rng) < pCross) ? crossover(pop[pa], pop[pb], rng) : pop[pa];
            if (U(rng) < pMut) mutate(child, rng);
            next.push_back(std::move(child));
        }
        
        pop = std::move(next);
        for (int i = 0; i < popSize; ++i) fit[i] = fitness(pop[i], rng);
        
        double avg = accumulate(fit.begin(), fit.end(), 0.0) / popSize;
        for (int i = 0; i < popSize; ++i) {
            if (fit[i] > best.fitness) {
                best.fitness = fit[i];
                best.best = pop[i];
            }
        }
        
        onGen(gen, best.best, best.fitness, avg);
    }
    
    return best;
}

// ===================== Main =====================

int main(int argc, char** argv) {
    string citiesPath = (argc > 1) ? argv[1] : "west_europe_cities.csv";
    string travelPath = (argc > 2) ? argv[2] : "west_europe_travel_hours.csv";
    
    vector<City> cities;
    vector<vector<double>> travel;
    
    try {
        loadDataset(citiesPath, travelPath, cities, travel);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    int N = cities.size();
    cout << "Loaded " << N << " cities\n\n";
    
    // Problem parameters
    const double budget = 500.0, timeLimit = 4.0;
    const double alphaTime = 200.0, alphaBudget = 120.0;
    const int hybridSwaps = 50;
    
    // GA parameters
    const int popSize = 10, gens = 50, tournK = 3, elites = 3;
    const double pCross = 0.9, pMut = 0.35;
    const unsigned seed = 12345;
    
    auto makeRandom = [&](mt19937& rng) {
        Genome g;
        g.perm.resize(N);
        iota(g.perm.begin(), g.perm.end(), 0);
        shuffle(g.perm.begin(), g.perm.end(), rng);
        g.len = uniform_int_distribution<int>(3, min(18, N))(rng);
        return g;
    };
    
    auto fitness = [&](const Genome& g, mt19937& rng) {
        Genome temp = g;
        hybridRepair(cities, travel, temp, budget, timeLimit, hybridSwaps, rng);
        Metrics m = evaluate(cities, travel, temp, budget, timeLimit, alphaTime, alphaBudget);
        return m.fitness;
    };
    
    auto crossover = [&](const Genome& a, const Genome& b, mt19937& rng) {
        Genome c;
        c.perm = orderCrossover(a.perm, b.perm, rng);
        c.len = (uniform_real_distribution<double>(0, 1)(rng) < 0.5) ? a.len : b.len;
        c.len = max(1, min(c.len, N));
        return c;
    };
    
    auto mutate = [&](Genome& g, mt19937& rng) {
        uniform_int_distribution<int> D(0, N - 1);
        swap(g.perm[D(rng)], g.perm[D(rng)]);
        
        if (uniform_real_distribution<double>(0, 1)(rng) < 0.35) {
            int delta = (uniform_real_distribution<double>(0, 1)(rng) < 0.5) ? -1 : 1;
            g.len = max(1, min(N, g.len + delta));
        }
    };
    
    cout << "Hybrid Constraint Handling GA\n";
    cout << "Budget: " << budget << " EUR | Time: " << timeLimit << " days\n\n";
    cout << "Gen  BestFit   Quality   Time(d)  Cost(EUR)  Feasible  AvgFit\n";
    cout << "---- --------- --------- -------  ---------  --------  -------\n";
    
    auto onGen = [&](int gen, const Genome& best, double bestFit, double avgFit) {
        mt19937 tmp(999 + gen);
        Genome temp = best;
        hybridRepair(cities, travel, temp, budget, timeLimit, hybridSwaps, tmp);
        Metrics m = evaluate(cities, travel, temp, budget, timeLimit, alphaTime, alphaBudget);
        
        cout << setw(4) << gen << " "
             << setw(9) << fixed << setprecision(2) << bestFit << " "
             << setw(9) << m.quality << " "
             << setw(7) << m.timeDays << " "
             << setw(9) << m.costEUR << "  "
             << setw(8) << (m.feasible ? "yes" : "no") << "  "
             << setw(7) << avgFit << "\n";
    };
    
    auto result = runGA<Genome>(popSize, gens, tournK, elites, pCross, pMut,
                                makeRandom, fitness, crossover, mutate, onGen, seed);
    
    // Final report
    mt19937 tmp(424242);
    Genome final = result.best;
    hybridRepair(cities, travel, final, budget, timeLimit, hybridSwaps, tmp);
    Metrics m = evaluate(cities, travel, final, budget, timeLimit, alphaTime, alphaBudget);
    
    cout << "\nBest Solution:\n";
    cout << "Fitness: " << m.fitness << " | Quality: " << m.quality
         << " | Time: " << m.timeDays << " days | Cost: " << m.costEUR << " EUR\n";
    cout << "Feasible: " << (m.feasible ? "yes" : "no") << "\n\n";
    cout << "Itinerary (" << final.len << " cities):\n";
    
    for (int i = 0; i < final.len; ++i) {
        const City& c = cities[final.perm[i]];
        cout << "  " << (i + 1) << ". " << c.name << ", " << c.country
             << " (sites=" << c.sites << ", satisfaction=" << c.satisfaction << ")\n";
    }
    
    return 0;
}