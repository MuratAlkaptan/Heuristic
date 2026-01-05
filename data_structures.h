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