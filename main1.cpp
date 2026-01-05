#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>
using namespace std;
//implement Data structures
    //City struct
    struct City {
        int sites;
        string name;
        string country;
        double satisfaction;
        double visitHours;
        double dailyStayEUR;
        double attractionEUR;

    };
    //Genome struct
    struct Genome {
        vector<int> perm;
        int len;
    };
    //MEtrics struct initialized with default values
    struct Metrics {
        double quality = 0;
        double timeDays = 0;
        double costEUR = 0;
        double fitness = numeric_limits<double>::lowest();
        bool feasible = false;
    };
//implement problem specific functions
    //fitness function
    Metrics evaluate(const vector<City>& cities, const vector<vector<double>>& travel,
                 const Genome& g, double budget, double timeLimit,
                 double alphaTime, double alphaBudget) {
    Metrics m;
    int L = min(g.len, (int)cities.size());
    
    double timeHours = 0, cost = 0, quality = 0;
    
    // Calculate metrics for selected cities
    for (int i = 0; i < L; ++i) {
        const City& c = cities[g.perm[i]];
        quality += 5.0 + c.sites + c.satisfaction;
        timeHours += c.visitHours;
        cost += c.dailyStayEUR * ceil(c.visitHours / 8.0) + c.attractionEUR * c.sites;
        
        // Travel time between cities
        if (i + 1 < L) {
            timeHours += travel[g.perm[i]][g.perm[i + 1]];
        }
    }
    
    m.quality = quality;
    m.timeDays = timeHours / 8.0;  // HOURS_PER_DAY = 8
    m.costEUR = cost;
    m.feasible = (m.timeDays <= timeLimit && m.costEUR <= budget);
    
    // Fitness = quality if feasible, else apply penalties
    if (m.feasible) {
        m.fitness = quality;
    } else {
        double overTime = max(0.0, m.timeDays - timeLimit);
        double overBudget = max(0.0, m.costEUR - budget);
        m.fitness = quality - alphaTime * overTime * overTime - alphaBudget * (overBudget / 100.0) * (overBudget / 100.0);
    }
    
    return m;
}
    //constraint evaluation
        //Hybrid repair function
        void hybridRepair(const vector<City>& cities, const vector<vector<double>>& travel,
                        Genome& g, double budget, double timeLimit, int maxSwaps, mt19937& rng) {
            
            int N = cities.size();
            g.len = max(1, min(g.len, N));
            
            // Check if already feasible
            Metrics cur = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
            if (cur.feasible) return;
            
            // Try swapping cities to reduce violation
            if (g.len < N) {
                uniform_int_distribution<int> inDist(0, g.len - 1);
                uniform_int_distribution<int> outDist(g.len, N - 1);
                
                for (int it = 0; it < maxSwaps; ++it) {
                    int i = inDist(rng);
                    int j = outDist(rng);
                    
                    swap(g.perm[i], g.perm[j]);
                    
                    Metrics cand = evaluate(cities, travel, g, budget, timeLimit, 0, 0);
                    double curViolation = max(0.0, cur.timeDays - timeLimit) + max(0.0, cur.costEUR - budget);
                    double candViolation = max(0.0, cand.timeDays - timeLimit) + max(0.0, cand.costEUR - budget);
                    
                    if (candViolation < curViolation) {
                        cur = cand;
                        if (cur.feasible) return;
                    } else {
                        swap(g.perm[i], g.perm[j]);  // Revert swap
                    }
                }
            }
        }
//implement GA baseline
    //GA result struct
    struct GAResult{
        Genome best;
        double fitness = numeric_limits<double>::lowest();
    };
    //selection (tournament)
    int tournament(const vector<Metrics>& metrics, int tournK, mt19937& rng, int popSize)
    {
        uniform_int_distribution<int> dist(0, popSize - 1);
        int best = dist(rng);
        for (int i = 1; i<tournK; i++){
            int candidate = dist(rng);
            if(metrics[candidate].fitness > metrics[best].fitness){
                best = candidate;
            }
        }
        return best;
    }
    //mutation (swap cities randomly)
    void mutate(Genome& g, mt19937& rng) {
        uniform_int_distribution<int> dist(0, (int)g.perm.size() - 1);
        swap(g.perm[dist(rng)], g.perm[dist(rng)]);
        
        if (uniform_real_distribution<double>(0, 1)(rng) < 0.35) {
            int delta = (uniform_real_distribution<double>(0, 1)(rng) < 0.5) ? -1 : 1;
            g.len = max(1, min((int)g.perm.size(), g.len + delta));
        }
    }

    //crossover (ordered crossover since we are dealing with permutations as solutions)
    Genome crossover(const Genome& p1, const Genome& p2, mt19937& rng) {
        int N = p1.perm.size();
        uniform_int_distribution<int> dist(0, N - 1);
        int l = dist(rng), r = dist(rng);
        if (l > r) swap(l, r);
        
        Genome child;
        child.perm.assign(N, -1);
        vector<bool> used(N, false);
        
        // Copy segment from p1
        for (int i = l; i <= r; ++i) {
            child.perm[i] = p1.perm[i];
            used[p1.perm[i]] = true;
        }
        
        // Fill remaining from p2
        int idx = (r + 1) % N;
        for (int k = 0; k < N; ++k) {
            int v = p2.perm[(r + 1 + k) % N];
            if (!used[v]) {
                while (child.perm[idx] != -1) idx = (idx + 1) % N;
                child.perm[idx] = v;
            }
        }
        
        child.len = (uniform_real_distribution<double>(0, 1)(rng) < 0.5) ? p1.len : p2.len;
        return child;
    }

    //evaluate (fitness+constraint handling)
    double compute_fitness(const Genome& g, const vector<City>& cities, 
                       const vector<vector<double>>& travel, double budget, double timeLimit,
                       double alphaTime, double alphaBudget, int hybridSwaps, mt19937& rng) {
        Genome temp = g;
        hybridRepair(cities, travel, temp, budget, timeLimit, hybridSwaps, rng);
        Metrics m = evaluate(cities, travel, temp, budget, timeLimit, alphaTime, alphaBudget);
        return m.fitness;
    }

    //elimination (keep best discard worst)
    void eliminate(vector<Genome>& pop, vector<Metrics>& metrics, int elites) {
        vector<int> idx(pop.size());
        iota(idx.begin(), idx.end(), 0);
        partial_sort(idx.begin(), idx.begin() + elites, idx.end(),
                    [&](int a, int b) { return metrics[a].fitness > metrics[b].fitness; });
        
        vector<Genome> new_pop;
        for (int i = 0; i < elites; ++i) {
            new_pop.push_back(pop[idx[i]]);
        }
        pop = new_pop;
    }

    //main GA function
    GAResult runGA(int popSize, int gens, int tournK, int elites, double pCross, double pMut,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit, double alphaTime, double alphaBudget,
               int hybridSwaps, unsigned seed) {
    
    mt19937 rng(seed);
    uniform_real_distribution<double> U(0, 1);
    int N = cities.size();
    
    // Initialize population
    vector<Genome> pop(popSize);
    vector<Metrics> metrics(popSize);
    
    for (int i = 0; i < popSize; ++i) {
        pop[i].perm.resize(N);
        iota(pop[i].perm.begin(), pop[i].perm.end(), 0);
        shuffle(pop[i].perm.begin(), pop[i].perm.end(), rng);
        pop[i].len = uniform_int_distribution<int>(3, min(18, N))(rng);
        
        Genome temp = pop[i];
        hybridRepair(cities, travel, temp, budget, timeLimit, hybridSwaps, rng);
        metrics[i] = evaluate(cities, travel, temp, budget, timeLimit, alphaTime, alphaBudget);
    }
    
    // Track best
    GAResult result;
    result.best = pop[0];
    result.fitness = metrics[0].fitness;
    
    // Main GA loop
    for (int gen = 0; gen < gens; ++gen) {
        // Generate offspring (from full population)
        vector<Genome> offspring;
        vector<Metrics> offspring_metrics;
        
        while ((int)offspring.size() < popSize) {
            int pa = tournament(metrics, tournK, rng, popSize);
            int pb = tournament(metrics, tournK, rng, popSize);
            
            Genome child = (U(rng) < pCross) ? crossover(pop[pa], pop[pb], rng) : pop[pa];
            if (U(rng) < pMut) mutate(child, rng);
            
            Genome temp = child;
            hybridRepair(cities, travel, temp, budget, timeLimit, hybridSwaps, rng);
            Metrics m = evaluate(cities, travel, temp, budget, timeLimit, alphaTime, alphaBudget);
            offspring_metrics.push_back(m);
            offspring.push_back(temp);
        }
        
        // Merge parent + offspring (mu + lambda)
        vector<Genome> combined = pop;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        vector<Metrics> combined_metrics = metrics;
        combined_metrics.insert(combined_metrics.end(), offspring_metrics.begin(), offspring_metrics.end());
        
        // Select best popSize (elitism automatic)
        vector<int> idx(combined.size());
        iota(idx.begin(), idx.end(), 0);
        partial_sort(idx.begin(), idx.begin() + popSize, idx.end(),
                    [&](int a, int b) { return combined_metrics[a].fitness > combined_metrics[b].fitness; });
        
        pop.clear();
        metrics.clear();
        for (int i = 0; i < popSize; ++i) {
            pop.push_back(combined[idx[i]]);
            metrics.push_back(combined_metrics[idx[i]]);
        }
        
        // Update best
        for (int i = 0; i < popSize; ++i) {
            if (metrics[i].fitness > result.fitness) {
                result.fitness = metrics[i].fitness;
                result.best = pop[i];
            }
        }
        
        cout << "Gen " << gen << ": Best fitness = " << result.fitness << "\n";
    }
    
    return result;
}
    
//load dataset from CSV
    //split for CSV parsing
    vector<string> split(const string& s, char delim = ','){
        vector<string> result;
        stringstream ss(s);
        string item;
        while(getline(ss, item, delim)){
            result.push_back(item);
        }
        return result;
    }
    //readcsv()
    void load_dataset(string& citiesPath, string& travelPath, vector<City>& cities, vector<vector<double>>& travelMatrix){
        ifstream fin;
        fin.open(citiesPath, ios::in);

        string line;
        getline(fin, line);//skipping header

        while(getline(fin, line)){
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

        int N = cities.size();
        travelMatrix.assign(N, vector<double>(N, 0.0));

        fin.open(travelPath, ios::in);

        getline(fin, line);

        while(getline(fin, line)){
            vector<string> t = split(line);
            int i = stoi(t[0]);
            int j = stoi(t[1]);
            double distance = stod(t[2]);
            travelMatrix[i][j] = distance;
            travelMatrix[j][i] = distance;
        }
        fin.close();
    }

//main function
int main(int argc, char** argv) {
    //filepaths
    string citiesPath = (argc > 1) ? argv[1] : "west_europe_cities.csv";
    string travelPath = (argc > 2) ? argv[2] : "west_europe_travel_hours.csv";
    
    vector<City> cities;
    vector<vector<double>> travel;
    
    //load dataset
    try {
        load_dataset(citiesPath, travelPath, cities, travel);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    int N = cities.size();
    cout << "Loaded " << N << " cities\n\n";
    
    //problem parameters
    const double budget = 500.0;
    const double timeLimit = 4.0;
    const double alphaTime = 200.0;
    const double alphaBudget = 120.0;
    const int hybridSwaps = 50;
    
    //GA parameters
    const int popSize = 10;
    const int gens = 50;
    const int tournK = 3;
    const int elites = 3;
    const double pCross = 0.9;
    const double pMut = 0.35;
    const unsigned seed = 12345;
    
    cout << "Hybrid Constraint Handling GA\n";
    cout << "Budget: " << budget << " EUR | Time: " << timeLimit << " days\n\n";
    
    // Run GA
    GAResult result = runGA(popSize, gens, tournK, elites, pCross, pMut,
                            cities, travel, budget, timeLimit, 
                            alphaTime, alphaBudget, hybridSwaps, seed);
    
    //evaluate final best
    Genome final = result.best;
    mt19937 tmp(seed);
    hybridRepair(cities, travel, final, budget, timeLimit, hybridSwaps, tmp);
    Metrics m = evaluate(cities, travel, final, budget, timeLimit, alphaTime, alphaBudget);
    
    //report results
    cout << "\n========== Best Solution ==========\n";
    cout << "Fitness: " << m.fitness << "\n";
    cout << "Quality: " << m.quality << "\n";
    cout << "Time: " << m.timeDays << " days\n";
    cout << "Cost: " << m.costEUR << " EUR\n";
    cout << "Feasible: " << (m.feasible ? "YES" : "NO") << "\n\n";
    
    cout << "Itinerary (" << final.len << " cities):\n";
    for (int i = 0; i < final.len; ++i) {
        const City& c = cities[final.perm[i]];
        cout << "  " << (i + 1) << ". " << c.name << ", " << c.country 
             << " (sites=" << c.sites << ", satisfaction=" << c.satisfaction << ")\n";
    }
    
    return 0;
}