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
using namespace std;

// data structures
struct City {
    int sites;
    string name;
    string country;
    double satisfaction;//satisfaction of previous tourists
    
    double visitHours;//time it takes to visit the city
    double dailyStayEUR;//daily spending rate
    double attractionEUR;//fee for travelling sites
};
//permutation of cities held as int vector
struct Genome {
    vector<int> perm;
    int len;
};
//for keeping track of contraints and fitness
struct Metrics {
    double quality = 0;//overall quality of the itinerary
    double timeDays = 0;//time it takes for the travel
    double costEUR = 0;//cost for budget comparison
    double fitness = std::numeric_limits<double>::lowest();//fitness might be negative because of large penalties
    bool feasible = false;
};

//problem specific functions
//evaluate function to calculate metrics and fitness
//O(L) where L = g.len prefix length of the genome
//O(1) space since there are no data structures that grow with input size
Metrics evaluate(const vector<City>& cities, const vector<vector<double>>& travel,
                 const Genome& g, double budget, double timeLimit,
                 double alphaTime, double alphaBudget) {
    Metrics m;
    int L = g.len;

    double timeHours = 0, cost = 0, quality = 0;

    for (int i = 0; i < L; ++i) {//O(L) where L = g.len prefix length of the genome
        const City& c = cities[g.perm[i]];
        //added a constant 5 since more cities should be encouraged 
        quality += 5.0 + c.sites + c.satisfaction;
        timeHours += c.visitHours;
        cost += c.dailyStayEUR * ceil(c.visitHours / 8.0) + c.attractionEUR * c.sites;

        if (i + 1 < L) timeHours += travel[g.perm[i]][g.perm[i + 1]];
    }

    m.quality = quality;
    m.timeDays = timeHours / 8.0;
    m.costEUR = cost;
    m.feasible = (m.timeDays <= timeLimit && m.costEUR <= budget);

    if (m.feasible) {
        m.fitness = quality;
    } else {
        double overTime = max(0.0, m.timeDays - timeLimit);
        double overBudget = max(0.0, m.costEUR - budget);
        m.fitness = quality
            - alphaTime * overTime * overTime
            - alphaBudget * (overBudget / 100.0) * (overBudget / 100.0);
    }

    return m;
}
//constraints methods
//#include "constraint_penalty.h"
//#include "constraint_repair.h"
#include "constraint_hybrid.h"

//GA components
struct GAResult {
    Genome best;
    double fitness = numeric_limits<double>::lowest();
};

//O(1) time and space
static int randi(int lo, int hi) { // inclusive
    return lo + (rand() % (hi - lo + 1));
}
//O(1) time and space
static double rand01() {//O(1)
    return rand() / (double)RAND_MAX;
}

//Time: O(K) where K is the tournK parameter
//Space: O(1)
int tournament(const vector<Metrics>& metrics, int tournK) {
    int best = randi(0, metrics.size() - 1);
    for (int i = 1; i < tournK; ++i) {
        int c = randi(0, metrics.size() - 1);
        if (metrics[c].fitness > metrics[best].fitness) best = c;
    }
    return best;
}

//mutation: swap two cities, small chance to tweak length
//O(1) time and space complexity
void mutate(Genome& g) {
    int N = g.perm.size();
    int i = randi(0, N - 1), j = randi(0, N - 1);//O(1)
    swap(g.perm[i], g.perm[j]);//O(1)

    // small chance to tweak length
    //O(1)
    if (rand01() < 0.35) {
        g.len += (rand01() < 0.5) ? -1 : 1;
        g.len = max(1, min(N, g.len));
    }
}
// ordered crossover
//O( N) Time where N is the number of cities in the permutation
//Space O(N) because of child.perm and used
Genome crossover(const Genome& p1, const Genome& p2) { 
    int N = p1.perm.size();
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
//space is O(1)
//time is O(n) where n is the size of the vector
void shuffle(vector<int>& a) {
    for (int i = a.size() - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        swap(a[i], a[j]);
    }
}

//main GA function
//number of cities in permutation is assumed to be n
//Time: O(gens * popSize * n^3) in worst case if repair method is used
//Space: O(popSize * n) where n is the number of cities in the permutation
//But we hold pop + offspring + comb in the memory at the same time so O(3 * popSize * n) => O(popSize * n)
GAResult runGA(int popSize, int gens, int tournK, double pCross, double pMut,
               const vector<City>& cities, const vector<vector<double>>& travel,
               double budget, double timeLimit, double alphaTime, double alphaBudget,
               int methodParam /* maxSwaps for hybrid method, ignored otherwise */) {

    int N = cities.size();
    vector<Genome> pop(popSize);
    vector<Metrics> metrics(popSize);

    //initialize population
    //O(popSize * n) where n is the number of cities while disregarding assess function
    //In worst case O(popSize * n^3)
    for (int i = 0; i < popSize; ++i) {
        pop[i].perm.resize(N);
        iota(pop[i].perm.begin(), pop[i].perm.end(), 0);//O(n)
        shuffle(pop[i].perm);//O(n)
        pop[i].len = randi(3, min(18, N));

        Genome g = pop[i];
        //assess function handles constraint method internally
        //O(n) or O(n^2) or O(n*maxSwaps) depending on the method
        metrics[i] = assess(g, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
        pop[i] = g; // method may modify genome (repair/hybrid)
    }

    GAResult best;
    best.best = pop[0];
    best.fitness = metrics[0].fitness;

    //worst case O(gens * popSize * n^3) if repair method is used
    for (int gen = 0; gen < gens; ++gen) {//O(gens * popSize * n) disregarding assess function
        vector<Genome> offspring;
        vector<Metrics> offm;

        while (offspring.size() < popSize) {//O(popSize * n)
            int a = tournament(metrics, tournK);
            int b = tournament(metrics, tournK);

            Genome child = (rand01() < pCross) ? crossover(pop[a], pop[b]) : pop[a];
            if (rand01() < pMut) mutate(child);

            Metrics m = assess(child, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);
            offspring.push_back(child);
            offm.push_back(m);
        }

        //(mu + lambda) keep best popSize from parents+offspring => elitism
        vector<Genome> comb = pop;
        comb.insert(comb.end(), offspring.begin(), offspring.end());
        vector<Metrics> combm = metrics;
        combm.insert(combm.end(), offm.begin(), offm.end());

        vector<int> idx(comb.size());
        iota(idx.begin(), idx.end(), 0);

        //O(PlogP) where P = popSize
        sort(idx.begin(), idx.end(),
            [&](int i, int j) { return combm[i].fitness > combm[j].fitness; });

        for (int i = 0; i < popSize; ++i) {//O(popSize * N) where N is the copied vector's size
            pop[i] = comb[idx[i]];//copying genome
            metrics[i] = combm[idx[i]];
            if (metrics[i].fitness > best.fitness) {
                best.fitness = metrics[i].fitness;
                best.best = pop[i];
            }
        }


        cout << "Gen " << gen << ": Best fitness = " << best.fitness << "\n";
    }

    return best;
}

//loading dataset
//parse the csv row
//Time: O(n) where n is the size of row in the csv file
//Space: O(c) where c is the characters in the row
vector<string> split(const string& s, char delim = ',') {
    vector<string> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) out.push_back(item);
    return out;
}
//read from csv line by line
//O(N^2) where N is the number of cities since we create an adjacency matrix for travel times

void load_dataset(const string& citiesPath, const string& travelPath,
                  vector<City>& cities, vector<vector<double>>& travelMatrix) {

    ifstream fin(citiesPath.c_str());
    string line;
    getline(fin, line); // header
    
    //O(n) space complexity where n is the number of cities
    while (getline(fin, line)) {//O(n) time where n is the number of cities
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

    fin.open(travelPath.c_str());
    getline(fin, line); // header

    //O(n^2) space complexity for n by n travelMatrix
    while (getline(fin, line)) {//O(E) where E is the travel entries in the csv file
                                //can be N^2 in worst case where n is the number of cities
        vector<string> t = split(line);
        int i = stoi(t[0]);
        int j = stoi(t[1]);
        double hours = stod(t[2]);
        travelMatrix[i][j] = hours;
        travelMatrix[j][i] = hours;
    }
    fin.close();
}

//main function
//O(gens * popSize * n + n) where n is the number of cities in the permutation
//space comp: cities O(n), travel O(n^2), runGA O(P * n) where P is popSize and n is cities
int main(int argc, char** argv) {
    srand(12345); 

    string citiesPath = "west_europe_cities.csv";
    string travelPath = "west_europe_travel_hours.csv";

    vector<City> cities;
    vector<vector<double>> travel;

    load_dataset(citiesPath, travelPath, cities, travel);//O(n^2)

    cout << "Loaded " << cities.size() << " cities\n\n";

    // problem params
    const double budget = 500.0;
    const double timeLimit = 4.0;
    const double alphaTime = 200.0;
    const double alphaBudget = 120.0;

    // GA params
    const int popSize = 10;
    const int gens = 50;
    const int tournK = 5;
    const double pCross = 0.9;
    const double pMut = 0.35;

    // methodParam meaning depends on header (hybrid uses it as maxSwaps)
    const int methodParam = 50;

    GAResult result = runGA(popSize, gens, tournK, pCross, pMut,
                            cities, travel, budget, timeLimit,
                            alphaTime, alphaBudget, methodParam);//O(gens * popSize * n) n is number of cities in permutation

    Genome final = result.best;
    Metrics m = assess(final, cities, travel, budget, timeLimit, alphaTime, alphaBudget, methodParam);//O(n) n is number of cities in permutation

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
