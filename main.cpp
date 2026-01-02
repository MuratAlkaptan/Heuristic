#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Vec = std::vector<double>;

static inline double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

struct RNG {
  std::mt19937_64 eng;
  explicit RNG(uint64_t seed) : eng(seed) {}
  double uni01() {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(eng);
  }
  double uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(eng);
  }
  int randint(int lo, int hi) { // inclusive
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(eng);
  }
};

struct ConstraintEval {
  Vec g; // inequalities: g_i(x) <= 0 feasible
  Vec h; // equalities: h_j(x) = 0 (converted by tolerance)
};

struct Individual {
  Vec x;
  double obj = std::numeric_limits<double>::infinity();
  ConstraintEval c;
  double violation = std::numeric_limits<double>::infinity();
  bool feasible = false;
  double fitness = std::numeric_limits<double>::infinity(); // used by penalty modes
};

struct Problem {
  virtual ~Problem() = default;
  virtual std::string name() const = 0;
  virtual size_t dim() const = 0;
  virtual const Vec& lb() const = 0;
  virtual const Vec& ub() const = 0;

  virtual double objective(const Vec& x) const = 0;
  virtual ConstraintEval constraints(const Vec& x) const = 0;
};

constexpr double EQ_EPS = 1e-4; // CEC 2006 uses epsilon = 0.0001 for equality->inequality feasibility check

static inline double total_violation(const ConstraintEval& c) {
  double v = 0.0;
  for (double gi : c.g) v += std::max(0.0, gi);
  for (double hj : c.h) v += std::max(0.0, std::fabs(hj) - EQ_EPS);
  return v;
}

// ----------------------- CEC 2006 benchmark problems -----------------------
// Implemented from: "Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session
// on Constrained Real-Parameter Optimization" (Liang et al., 2006).  :contentReference[oaicite:2]{index=2}

// g06: 2 vars, 2 inequalities
struct G06 : Problem {
  Vec _lb{13.0, 0.0};
  Vec _ub{100.0, 100.0};
  std::string name() const override { return "g06"; }
  size_t dim() const override { return 2; }
  const Vec& lb() const override { return _lb; }
  const Vec& ub() const override { return _ub; }

  double objective(const Vec& x) const override {
    // f(x) = (x1-10)^3 + (x2-20)^3
    double a = x[0] - 10.0;
    double b = x[1] - 20.0;
    return a*a*a + b*b*b;
  }

  ConstraintEval constraints(const Vec& x) const override {
    ConstraintEval c;
    // g1 = -(x1-5)^2 - (x2-5)^2 + 100 <= 0
    // g2 = (x1-6)^2 + (x2-5)^2 - 82.81 <= 0
    double g1 = -(x[0]-5.0)*(x[0]-5.0) - (x[1]-5.0)*(x[1]-5.0) + 100.0;
    double g2 =  (x[0]-6.0)*(x[0]-6.0) + (x[1]-5.0)*(x[1]-5.0) - 82.81;
    c.g = {g1, g2};
    return c;
  }
};

// g03: n=10, 1 equality
struct G03 : Problem {
  Vec _lb, _ub;
  G03() : _lb(10, 0.0), _ub(10, 1.0) {}
  std::string name() const override { return "g03"; }
  size_t dim() const override { return 10; }
  const Vec& lb() const override { return _lb; }
  const Vec& ub() const override { return _ub; }

  double objective(const Vec& x) const override {
    // f(x) = - (sqrt(n))^n * prod(xi), n=10
    const double n = 10.0;
    const double coeff = std::pow(std::sqrt(n), n);
    double prod = 1.0;
    for (double xi : x) prod *= xi;
    return -coeff * prod;
  }

  ConstraintEval constraints(const Vec& x) const override {
    ConstraintEval c;
    // h1(x) = sum(xi^2) - 1 = 0
    double s = 0.0;
    for (double xi : x) s += xi*xi;
    c.h = {s - 1.0};
    return c;
  }
};

// g01: 13 vars, 9 linear inequalities
struct G01 : Problem {
  Vec _lb, _ub;
  G01() {
    _lb.assign(13, 0.0);
    _ub.assign(13, 1.0);
    // x10..x12 in [0,100] (1-indexed in report)
    _ub[9]  = 100.0;
    _ub[10] = 100.0;
    _ub[11] = 100.0;
    // x13 in [0,1] already
  }

  std::string name() const override { return "g01"; }
  size_t dim() const override { return 13; }
  const Vec& lb() const override { return _lb; }
  const Vec& ub() const override { return _ub; }

  double objective(const Vec& x) const override {
    // f = 5*sum_{i=1..4} xi - 5*sum_{i=1..4} xi^2 - sum_{i=5..13} xi
    double s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < 4; ++i) { s1 += x[i]; s2 += x[i]*x[i]; }
    for (int i = 4; i < 13; ++i) s3 += x[i];
    return 5.0*s1 - 5.0*s2 - s3;
  }

  ConstraintEval constraints(const Vec& x) const override {
    ConstraintEval c;
    // Using 0-based indexing: x1->x[0], ..., x13->x[12]
    double x1=x[0], x2=x[1], x3=x[2], x4=x[3], x5=x[4], x6=x[5], x7=x[6], x8=x[7], x9=x[8];
    double x10=x[9], x11=x[10], x12=x[11];

    double g1 = 2*x1 + 2*x2 + x10 + x11 - 10;
    double g2 = 2*x1 + 2*x3 + x10 + x12 - 10;
    double g3 = 2*x2 + 2*x3 + x11 + x12 - 10;
    double g4 = -8*x1 + x10;
    double g5 = -8*x2 + x11;
    double g6 = -8*x3 + x12;
    double g7 = -2*x4 - x5 + x10;
    double g8 = -2*x6 - x7 + x11;
    double g9 = -2*x8 - x9 + x12;

    c.g = {g1,g2,g3,g4,g5,g6,g7,g8,g9};
    return c;
  }
};

// g07: 10 vars, 8 inequalities
struct G07 : Problem {
  Vec _lb, _ub;
  G07() : _lb(10, -10.0), _ub(10, 10.0) {}
  std::string name() const override { return "g07"; }
  size_t dim() const override { return 10; }
  const Vec& lb() const override { return _lb; }
  const Vec& ub() const override { return _ub; }

  double objective(const Vec& x) const override {
    // f(x) = x1^2 + x2^2 + x1*x2 - 14x1 - 16x2
    //      + (x3-10)^2 + 4(x4-5)^2 + (x5-3)^2 + 2(x6-1)^2
    //      + 5x7^2 + 7(x8-11)^2 + 2(x9-10)^2 + (x10-7)^2 + 45
    double x1=x[0],x2=x[1],x3=x[2],x4=x[3],x5=x[4],x6=x[5],x7=x[6],x8=x[7],x9=x[8],x10=x[9];
    double f = x1*x1 + x2*x2 + x1*x2 - 14*x1 - 16*x2
             + (x3-10)*(x3-10)
             + 4*(x4-5)*(x4-5)
             + (x5-3)*(x5-3)
             + 2*(x6-1)*(x6-1)
             + 5*x7*x7
             + 7*(x8-11)*(x8-11)
             + 2*(x9-10)*(x9-10)
             + (x10-7)*(x10-7)
             + 45.0;
    return f;
  }

  ConstraintEval constraints(const Vec& x) const override {
    ConstraintEval c;
    double x1=x[0],x2=x[1],x3=x[2],x4=x[3],x5=x[4],x6=x[5],x7=x[6],x8=x[7],x9=x[8],x10=x[9];

    double g1 = -105 + 4*x1 + 5*x2 - 3*x7 + 9*x8;
    double g2 =  10*x1 - 8*x2 - 17*x7 + 2*x8;
    double g3 = -8*x1 + 2*x2 + 5*x9 - 2*x10 - 12;
    double g4 =  3*(x1-2)*(x1-2) + 4*(x2-3)*(x2-3) + 2*x3*x3 - 7*x4 - 120;
    double g5 =  5*x1*x1 + 8*x2 + (x3-6)*(x3-6) - 2*x4 - 40;
    double g6 =  x1*x1 + 2*(x2-2)*(x2-2) - 2*x1*x2 + 14*x5 - 6*x6;
    double g7 =  0.5*(x1-8)*(x1-8) + 2*(x2-4)*(x2-4) + 3*x5*x5 - x6 - 30;
    double g8 = -3*x1 + 6*x2 + 12*(x9-8)*(x9-8) - 7*x10;

    c.g = {g1,g2,g3,g4,g5,g6,g7,g8};
    return c;
  }
};

// ----------------------- GA operators (SBX + poly mutation) -----------------------
static void sbx_crossover(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2,
                          const Vec& lb, const Vec& ub, double eta_c, RNG& rng) {
  const double EPS = 1e-14;
  const size_t n = p1.size();
  c1 = p1; c2 = p2;
  for (size_t i = 0; i < n; ++i) {
    if (rng.uni01() > 0.5) continue;
    double y1 = p1[i], y2 = p2[i];
    if (std::fabs(y1 - y2) < EPS) continue;

    if (y2 < y1) std::swap(y1, y2);
    const double yl = lb[i], yu = ub[i];

    double rand = rng.uni01();
    // Child 1
    double beta = 1.0 + (2.0*(y1 - yl)/(y2 - y1));
    double alpha = 2.0 - std::pow(beta, -(eta_c + 1.0));
    double betaq;
    if (rand <= 1.0/alpha) betaq = std::pow(rand*alpha, 1.0/(eta_c + 1.0));
    else                   betaq = std::pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c + 1.0));
    double child1 = 0.5*((y1 + y2) - betaq*(y2 - y1));

    // Child 2
    rand = rng.uni01();
    beta = 1.0 + (2.0*(yu - y2)/(y2 - y1));
    alpha = 2.0 - std::pow(beta, -(eta_c + 1.0));
    if (rand <= 1.0/alpha) betaq = std::pow(rand*alpha, 1.0/(eta_c + 1.0));
    else                   betaq = std::pow(1.0/(2.0 - rand*alpha), 1.0/(eta_c + 1.0));
    double child2 = 0.5*((y1 + y2) + betaq*(y2 - y1));

    child1 = clamp(child1, yl, yu);
    child2 = clamp(child2, yl, yu);

    // Randomly assign to c1/c2 to avoid positional bias
    if (rng.uni01() < 0.5) { c1[i] = child1; c2[i] = child2; }
    else                   { c1[i] = child2; c2[i] = child1; }
  }
}

static void polynomial_mutation(Vec& x, const Vec& lb, const Vec& ub,
                                double eta_m, double pm_gene, RNG& rng) {
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    if (rng.uni01() > pm_gene) continue;
    const double yl = lb[i], yu = ub[i];
    if (yu <= yl) continue;

    double y = x[i];
    double delta1 = (y - yl) / (yu - yl);
    double delta2 = (yu - y) / (yu - yl);
    double rnd = rng.uni01();
    double mut_pow = 1.0 / (eta_m + 1.0);
    double deltaq;

    if (rnd <= 0.5) {
      double xy = 1.0 - delta1;
      double val = 2.0*rnd + (1.0 - 2.0*rnd)*std::pow(xy, eta_m + 1.0);
      deltaq = std::pow(val, mut_pow) - 1.0;
    } else {
      double xy = 1.0 - delta2;
      double val = 2.0*(1.0 - rnd) + 2.0*(rnd - 0.5)*std::pow(xy, eta_m + 1.0);
      deltaq = 1.0 - std::pow(val, mut_pow);
    }

    y += deltaq * (yu - yl);
    x[i] = clamp(y, yl, yu);
  }
}

// ----------------------- Constraint handling modes -----------------------
enum class Method {
  StaticPenalty,
  DynamicPenalty,
  AdaptivePenalty,
  EliteProximityPenalty,
  DeathPenalty,
  DebRules
};

struct RunStats {
  double feasible_ratio = 0.0;
  double best_feasible_obj = std::numeric_limits<double>::infinity();
  double best_infeasible_obj = std::numeric_limits<double>::infinity();
  double best_overall_obj = std::numeric_limits<double>::infinity();
  double mean_infeasible_violation = 0.0;
  double obj_scale = 1.0;
};

struct EliteArchive {
  struct Item { Vec x; double obj; };
  std::vector<Item> items;
  size_t max_size = 50;

  void update_from_population(const std::vector<Individual>& pop) {
    for (const auto& ind : pop) {
      if (!ind.feasible) continue;
      items.push_back({ind.x, ind.obj});
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b){ return a.obj < b.obj; });
    if (items.size() > max_size) items.resize(max_size);
  }

  bool ready(size_t dim) const {
    return items.size() >= std::min<size_t>(10, dim + 1); // heuristic for stable variance
  }

  // Diagonal "Mahalanobis": sum ((x-mu)^2 / (var+eps))
  double diag_mahalanobis(const Vec& x, Vec& out_mu, Vec& out_var, double var_eps = 1e-12) const {
    const size_t d = x.size();
    out_mu.assign(d, 0.0);
    out_var.assign(d, 0.0);

    // mean
    for (const auto& it : items)
      for (size_t j = 0; j < d; ++j) out_mu[j] += it.x[j];
    for (size_t j = 0; j < d; ++j) out_mu[j] /= std::max<size_t>(1, items.size());

    // variance
    for (const auto& it : items)
      for (size_t j = 0; j < d; ++j) {
        double diff = it.x[j] - out_mu[j];
        out_var[j] += diff*diff;
      }
    for (size_t j = 0; j < d; ++j) out_var[j] /= std::max<size_t>(1, items.size());

    double acc = 0.0;
    for (size_t j = 0; j < d; ++j) {
      double diff = x[j] - out_mu[j];
      acc += (diff*diff) / (out_var[j] + var_eps);
    }
    return std::sqrt(acc);
  }
};

struct PenaltyController {
  // Penalty parameters (evolve over time for adaptive/dynamic)
  double lambda = 100.0;
  double lambda_min = 1e-6;
  double lambda_max = 1e12;

  double alpha = 1.0;  // weight for elite proximity
  double alpha_min = 0.0;
  double alpha_max = 1e9;

  // dynamic schedule params
  double lambda0 = 100.0;
  double dyn_k = 10.0;     // larger => stronger growth
  double dyn_p = 1.0;      // exponent

  // adaptive params
  double target_feasible = 0.4;
  double eta = 2.0;        // update aggressiveness

  void reset() {
    lambda = lambda0;
    alpha = 1.0;
  }

  void update_dynamic(int gen, int max_gen) {
    double t = (max_gen <= 0) ? 1.0 : (double)gen / (double)max_gen;
    lambda = lambda0 * std::pow(1.0 + dyn_k*t, dyn_p);
    lambda = clamp(lambda, lambda_min, lambda_max);
  }

  void update_adaptive(const RunStats& s) {
    // lambda <- lambda * exp(eta*(target - feasible_ratio))
    double factor = std::exp(eta * (target_feasible - s.feasible_ratio));
    lambda = clamp(lambda * factor, lambda_min, lambda_max);
  }
};

// Comparator for selecting "better" individuals
struct Comparator {
  Method method;
  bool operator()(const Individual& a, const Individual& b) const {
    if (method == Method::DebRules) {
      // feasibility first; then objective; else violation
      if (a.feasible != b.feasible) return a.feasible; // feasible is "better"
      if (a.feasible) return a.obj < b.obj;
      return a.violation < b.violation;
    }
    // penalty-based
    return a.fitness < b.fitness;
  }
};

// ----------------------- GA configuration & engine -----------------------
struct GAConfig {
  int pop_size = 120;
  int generations = 2000;

  double pc = 0.9;
  double eta_c = 15.0;

  // per-gene mutation probability (often 1/dim)
  double pm_gene = -1.0; // if <0 => auto = 1/dim
  double eta_m = 20.0;

  int tournament_k = 2; // binary tournament

  Method method = Method::EliteProximityPenalty;

  // Logging
  std::string out_csv = "";
};

static void evaluate_individual(Individual& ind, const Problem& prob) {
  ind.obj = prob.objective(ind.x);
  ind.c = prob.constraints(ind.x);
  ind.violation = total_violation(ind.c);
  ind.feasible = (ind.violation <= 0.0);
}

static RunStats compute_stats(const std::vector<Individual>& pop) {
  RunStats s;
  int feasible_count = 0;
  double sum_v = 0.0;
  int infeas_count = 0;

  for (const auto& ind : pop) {
    s.best_overall_obj = std::min(s.best_overall_obj, ind.obj);
    if (ind.feasible) {
      feasible_count++;
      s.best_feasible_obj = std::min(s.best_feasible_obj, ind.obj);
    } else {
      infeas_count++;
      s.best_infeasible_obj = std::min(s.best_infeasible_obj, ind.obj);
      sum_v += ind.violation;
    }
  }
  s.feasible_ratio = pop.empty() ? 0.0 : (double)feasible_count / (double)pop.size();
  s.mean_infeasible_violation = (infeas_count == 0) ? 0.0 : (sum_v / (double)infeas_count);

  // objective scaling for penalty robustness
  double scale = std::fabs(s.best_overall_obj);
  if (!std::isfinite(scale) || scale < 1.0) scale = 1.0;
  s.obj_scale = scale;
  return s;
}

static double compute_penalized_fitness(const Individual& ind,
                                       const RunStats& s,
                                       const EliteArchive& arch,
                                       PenaltyController& pc,
                                       Method m,
                                       int gen,
                                       int max_gen) {
  if (m == Method::DebRules) {
    return ind.obj; // unused in Deb comparator
  }

  if (m == Method::DeathPenalty) {
    if (ind.feasible) return ind.obj;
    return ind.obj + 1e12 * (1.0 + ind.violation);
  }

  // Update penalty schedule parameters if needed
  if (m == Method::DynamicPenalty || m == Method::EliteProximityPenalty) {
    pc.update_dynamic(gen, max_gen);
  }
  if (m == Method::AdaptivePenalty || m == Method::EliteProximityPenalty) {
    // elite mode can still adapt lambda (optional; here enabled)
    pc.update_adaptive(s);
  }

  // Base penalty: lambda * violation
  double V = ind.violation;
  double pen = pc.lambda * V * s.obj_scale;

  if (m == Method::EliteProximityPenalty && !ind.feasible) {
    // Elite proximity term
    if (arch.ready(ind.x.size())) {
      Vec mu, var;
      double D = arch.diag_mahalanobis(ind.x, mu, var);
      // alpha can also be scheduled mildly with time (optional)
      double t = (max_gen <= 0) ? 1.0 : (double)gen / (double)max_gen;
      double alpha_eff = clamp(pc.alpha * (1.0 + 0.5*t), pc.alpha_min, pc.alpha_max);
      pen += alpha_eff * D * s.obj_scale;
    }
  }

  // Static / Dynamic / Adaptive differ only by how pc.lambda is controlled.
  return ind.obj + pen;
}

static int tournament_select(const std::vector<Individual>& pop,
                             const Comparator& better,
                             RNG& rng,
                             int k = 2) {
  int best = rng.randint(0, (int)pop.size() - 1);
  for (int i = 1; i < k; ++i) {
    int cand = rng.randint(0, (int)pop.size() - 1);
    if (better(pop[cand], pop[best])) best = cand;
  }
  return best;
}

static Individual random_individual(const Problem& prob, RNG& rng) {
  Individual ind;
  ind.x.resize(prob.dim());
  for (size_t i = 0; i < prob.dim(); ++i) {
    ind.x[i] = rng.uniform(prob.lb()[i], prob.ub()[i]);
  }
  evaluate_individual(ind, prob);
  return ind;
}

static void run_ga(const Problem& prob, GAConfig cfg, uint64_t seed) {
  RNG rng(seed);

  if (cfg.pm_gene < 0.0) cfg.pm_gene = 1.0 / (double)prob.dim();

  // Initialize
  std::vector<Individual> pop;
  pop.reserve(cfg.pop_size);
  for (int i = 0; i < cfg.pop_size; ++i) pop.push_back(random_individual(prob, rng));

  EliteArchive archive;
  archive.max_size = 50;

  PenaltyController pc;
  pc.lambda0 = 100.0;
  pc.lambda = pc.lambda0;
  pc.alpha = 5.0;       // starting proximity weight; tune in experiments
  pc.target_feasible = 0.4;
  pc.eta = 2.0;
  pc.dyn_k = 20.0;
  pc.dyn_p = 2.0;

  // Timing
  auto run_start = std::chrono::high_resolution_clock::now();
  auto gen_start = run_start;

  // Prepare logging
  std::ofstream out;
  if (!cfg.out_csv.empty()) {
    out.open(cfg.out_csv);
    out << "gen,best_feasible_obj,best_overall_obj,feasible_ratio,mean_infeasible_violation,lambda,alpha,elapsed_sec,gen_time_sec\n";
  }

  // Main loop
  for (int gen = 0; gen <= cfg.generations; ++gen) {
    // Stats & archive update
    RunStats stats = compute_stats(pop);
    archive.update_from_population(pop);

    // Compute fitness values if in penalty mode
    if (cfg.method != Method::DebRules) {
      for (auto& ind : pop) {
        ind.fitness = compute_penalized_fitness(ind, stats, archive, pc, cfg.method, gen, cfg.generations);
      }
    } else {
      for (auto& ind : pop) ind.fitness = ind.obj; // not used, but keeps fields finite
    }

    // Log
    auto gen_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(gen_end - run_start).count();
    double gen_time = std::chrono::duration<double>(gen_end - gen_start).count();
    
    if (out.is_open()) {
      out << gen << ","
          << (std::isfinite(stats.best_feasible_obj) ? stats.best_feasible_obj : std::numeric_limits<double>::quiet_NaN()) << ","
          << stats.best_overall_obj << ","
          << stats.feasible_ratio << ","
          << stats.mean_infeasible_violation << ","
          << pc.lambda << ","
          << pc.alpha << ","
          << elapsed << ","
          << gen_time
          << "\n";
    }
    gen_start = gen_end;

    if (gen == cfg.generations) break;

    // Reproduction
    Comparator better{cfg.method};
    std::vector<Individual> offspring;
    offspring.reserve(cfg.pop_size);

    while ((int)offspring.size() < cfg.pop_size) {
      int i1 = tournament_select(pop, better, rng, cfg.tournament_k);
      int i2 = tournament_select(pop, better, rng, cfg.tournament_k);

      Vec c1 = pop[i1].x, c2 = pop[i2].x;
      if (rng.uni01() < cfg.pc) {
        sbx_crossover(pop[i1].x, pop[i2].x, c1, c2, prob.lb(), prob.ub(), cfg.eta_c, rng);
      }

      polynomial_mutation(c1, prob.lb(), prob.ub(), cfg.eta_m, cfg.pm_gene, rng);
      polynomial_mutation(c2, prob.lb(), prob.ub(), cfg.eta_m, cfg.pm_gene, rng);

      Individual ch1; ch1.x = std::move(c1);
      Individual ch2; ch2.x = std::move(c2);

      evaluate_individual(ch1, prob);
      evaluate_individual(ch2, prob);

      offspring.push_back(std::move(ch1));
      if ((int)offspring.size() < cfg.pop_size) offspring.push_back(std::move(ch2));
    }

    // Survivor selection: (mu + lambda) strategy
    std::vector<Individual> combined = pop;
    combined.insert(combined.end(), offspring.begin(), offspring.end());

    // Recompute stats for combined, compute fitness for combined according to current parameters
    RunStats st2 = compute_stats(combined);
    archive.update_from_population(combined);

    if (cfg.method != Method::DebRules) {
      for (auto& ind : combined) {
        ind.fitness = compute_penalized_fitness(ind, st2, archive, pc, cfg.method, gen, cfg.generations);
      }
    }

    std::sort(combined.begin(), combined.end(), better);
    combined.resize(cfg.pop_size);
    pop = std::move(combined);
  }

  // Report best (feasible-first)
  Comparator deb_like{Method::DebRules};
  std::sort(pop.begin(), pop.end(), deb_like);
  const Individual& best = pop.front();

  auto run_end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration<double>(run_end - run_start).count();

  std::cout << "Problem=" << prob.name()
            << " Method=" << (int)cfg.method
            << " Seed=" << seed << "\n";
  std::cout << "Best: feasible=" << best.feasible
            << " obj=" << std::setprecision(15) << best.obj
            << " violation=" << best.violation << "\n";
  std::cout << "x=[";
  for (size_t i = 0; i < best.x.size(); ++i) {
    std::cout << best.x[i] << (i + 1 == best.x.size() ? "" : ", ");
  }
  std::cout << "]\n";
  std::cout << "Total Runtime: " << total_time << " sec\n";
}

// ----------------------- CLI -----------------------
static void usage() {
  std::cerr <<
    "Usage:\n"
    "  ./ga --problem g01|g03|g06|g07 --method static|dynamic|adaptive|elite|death|debrules [options]\n"
    "Options:\n"
    "  --seed N        (default 1)\n"
    "  --pop N         (default 120)\n"
    "  --gen N         (default 2000)\n"
    "  --pc  X         (default 0.9)\n"
    "  --eta_c X       (default 15)\n"
    "  --eta_m X       (default 20)\n"
    "  --pm_gene X     (default auto=1/dim)\n"
    "  --out file.csv  (default none)\n";
}

static Method parse_method(const std::string& s) {
  if (s == "static")   return Method::StaticPenalty;
  if (s == "dynamic")  return Method::DynamicPenalty;
  if (s == "adaptive") return Method::AdaptivePenalty;
  if (s == "elite")    return Method::EliteProximityPenalty;
  if (s == "death")    return Method::DeathPenalty;
  if (s == "debrules") return Method::DebRules;
  throw std::runtime_error("Unknown method: " + s);
}

int main(int argc, char** argv) {
  std::string problem_name;
  std::string method_name;
  uint64_t seed = 1;

  GAConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* key)->std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + key);
      return argv[++i];
    };

    if (a == "--problem") problem_name = need("--problem");
    else if (a == "--method") method_name = need("--method");
    else if (a == "--seed") seed = (uint64_t)std::stoull(need("--seed"));
    else if (a == "--pop") cfg.pop_size = std::stoi(need("--pop"));
    else if (a == "--gen") cfg.generations = std::stoi(need("--gen"));
    else if (a == "--pc") cfg.pc = std::stod(need("--pc"));
    else if (a == "--eta_c") cfg.eta_c = std::stod(need("--eta_c"));
    else if (a == "--eta_m") cfg.eta_m = std::stod(need("--eta_m"));
    else if (a == "--pm_gene") cfg.pm_gene = std::stod(need("--pm_gene"));
    else if (a == "--out") cfg.out_csv = need("--out");
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else {
      std::cerr << "Unknown arg: " << a << "\n";
      usage();
      return 1;
    }
  }

  if (problem_name.empty() || method_name.empty()) {
    usage();
    return 1;
  }

  cfg.method = parse_method(method_name);

  std::unique_ptr<Problem> prob;
  if (problem_name == "g01") prob = std::make_unique<G01>();
  else if (problem_name == "g03") prob = std::make_unique<G03>();
  else if (problem_name == "g06") prob = std::make_unique<G06>();
  else if (problem_name == "g07") prob = std::make_unique<G07>();
  else {
    std::cerr << "Unknown problem: " << problem_name << "\n";
    usage();
    return 1;
  }

  run_ga(*prob, cfg, seed);
  return 0;
}
