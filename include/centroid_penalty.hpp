#pragma once
#include <algorithm>
#include <cmath>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <random>
#include <vector>

namespace cp {

struct params {
  double lambda = 1e4;     // Base penalty weight
  double boost = 50.0;     // Proximity penalty multiplier
  double sigma = -1.0;     // Gaussian width (auto if < 0)
  unsigned samples = 1000; // Samples per constraint for centroid
  unsigned seed = 42;      // Random seed for sampling
};

class centroid_udp {
private:
  pagmo::problem m_base;
  params m_p;
  pagmo::vector_double m_lb, m_ub;             // Space: O(n) each
  std::vector<pagmo::vector_double> m_centers; // Space: O(m × n)
public:
  /*
    Default constructor
    Time: O(1), Space: O(1)
   */
  centroid_udp() = default;

  /*
   Constructor:
   -Wraps a constrained problem with centroid proximity penalty
   -Stores the base problem
   -Finds centroids for each constraint via random sampling

   Time Complexity: O(S × m × (C + n))
   where S = samples per constraint (default 1000)
         m = number of constraints (nec + nic)
         C = cost of base problem fitness evaluation
         n = problem dimensions
   Space Complexities: O(m × n + S × n)
      - m × n for storing m centroids of dimension n
      - S × n for temporary storage of violated points (worst case)
   */
  explicit centroid_udp(pagmo::problem base, params p)
      : m_base(std::move(base)), m_p(p) {
    auto b = m_base.get_bounds();
    m_lb = b.first;
    m_ub = b.second;

    // Time: O(n), where n = dimensions
    if (m_p.sigma < 0.0) {
      double diag = 0.0;
      for (size_t i = 0; i < m_lb.size(); ++i) { // O(n) loop
        double r = m_ub[i] - m_lb[i];
        diag += r * r;
      }
      m_p.sigma = 0.15 * std::sqrt(diag); // O(1)
    }

    // Find centroids by sampling
    // Time: O(S × m × (C + n))
    find_centroids();
  }

  /*
   Function for fitness evaluation with centroid proximity penalty:
    -Evaluates original objective and constraints
    -Computes base quadratic penalty: Lambda × Sum of(violations²)
    -Adds proximity penalty for each violated constraint:
    boost × exp(-d²/2(sigma)²) × violation² where d = distance from
   individual to constraint's centroid

    Time Complexity: O(C + m × n)
      where C = base problem fitness evaluation cost
            m = number of constraints
            n = problem dimensions

    Space Complexity: O(1) - only uses local variables
   */
  pagmo::vector_double fitness(const pagmo::vector_double &x) const {
    // Evaluate base problem - Time: O(C)
    auto f = m_base.fitness(x);

    size_t nobj = m_base.get_nobj();
    size_t nec = m_base.get_nec();
    size_t nic = m_base.get_nic();

    double obj = f[0];

    // Base quadratic penalty - Time: O(m) where m = nec + nic
    // Loop through ineqauality and equality constraints and add
    // violations respectively
    double penalty = 0.0;
    for (size_t i = 0; i < nec; ++i) { // O(nec)
      double v = std::abs(f[nobj + i]);
      penalty += m_p.lambda * v * v;
    }
    for (size_t i = 0; i < nic; ++i) { // O(nic)
      double v = std::max(0.0, f[nobj + nec + i]);
      penalty += m_p.lambda * v * v;
    }

    // Proximity penalty
    // Time: O(m × n)
    double sig2 = m_p.sigma * m_p.sigma;

    // Process equality constraints with proximity
    //  Time: O(nec × n)
    for (size_t i = 0; i < nec; ++i) {
      double v = std::abs(f[nobj + i]);
      if (v > 0) {
        // Compute squared distance
        //  Time: O(n)
        double d = 0.0;
        for (size_t k = 0; k < x.size(); ++k) { // O(n) loop
          double diff = x[k] - m_centers[i][k];
          d += diff * diff;
        }
        // Gaussian weight and penalty
        //  Time: O(1)
        double w = std::exp(-d / (2.0 * sig2));
        penalty += m_p.boost * w * v * v;
      }
    }

    // Process inequality constraints
    //  Time: O(nic × n)
    for (size_t j = 0; j < nic; ++j) {
      double v = std::max(0.0, f[nobj + nec + j]);
      if (v > 0) {
        // Compute squared distance
        //  Time: O(n)
        double d = 0.0;
        for (size_t k = 0; k < x.size(); ++k) { // O(n) loop
          double diff = x[k] - m_centers[nec + j][k];
          d += diff * diff;
        }
        // Gaussian weight and penalty
        //  Time: O(1)
        double w = std::exp(-d / (2.0 * sig2));
        penalty += m_p.boost * w * v * v;
      }
    }

    return {obj + penalty};
  }

  /*
   Get problem bounds
   Time: O(1)
    Space: O(1)
   */
  std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
    return {m_lb, m_ub};
  }

  /*
   Get number of equality constraints (always 0 - converted to penalty)
   used for turning the problem to unconstrained to handle the penalty in a
   custom way
   Time: O(1)
    Space: O(1)
   */
  size_t get_nec() const { return 0; }

  /*
    Get number of inequality for the same logic as nec
    Time: O(1)
     Space: O(1)
   */
  size_t get_nic() const { return 0; }

  /*
    Get number of objectives (always 1)
    Time: O(1)
    Space: O(1)
   */
  size_t get_nobj() const { return 1; }

  /*
    Serialization as pagmo requirement
    Time: O(m × n)
    Space: O(1)
   */
  template <typename Archive> void serialize(Archive &ar, unsigned) {
    ar & m_base & m_p.lambda & m_p.boost & m_p.sigma & m_p.samples & m_p.seed;
    ar & m_lb & m_ub & m_centers;
  }

private:
  /*
    Find centroids by random sampling
    -For each constraint, samples S random points
    -Evaluates constraint at each point
    -Keeps points where constraint is violated
    -Computes centroid = average of violated points
    -Falls back to midpoint if no violations found

    Time Complexity: O(S × m × (C + n))
      where S = samples per constraint (e.g., 1000)
            m = total constraints (nec + nic)
            C = cost of base fitness evaluation
            n = problem dimensions

      Outer loop: O(m) - iterate over constraints
      Middle loop: O(S) - samples per constraint
       Inner operations: O(C + n)
        -Sampling point: O(n)
        -Fitness evaluation: O(C)
        - Averaging: O(n)

    Space Complexity: O(m × n + S × n)
      -m × n for m_centers storage
      -S × n for violated_points (worst case: all samples violate)
   */
  void find_centroids() {
    size_t nec = m_base.get_nec();
    size_t nic = m_base.get_nic();
    size_t m = nec + nic; // Total constraints

    m_centers.resize(m); // O(m) time, allocates O(m × n) space total

    // Setup random number generator
    // Time: O(n)
    std::mt19937 gen(m_p.seed);
    std::vector<std::uniform_real_distribution<double>> dists;
    for (size_t i = 0; i < m_lb.size(); ++i) { // O(n) loop
      dists.emplace_back(m_lb[i], m_ub[i]);
    }

    // For each constraint, find centroid
    // Time: O(m × S × (C + n))
    for (size_t c = 0; c < m; ++c) { // O(m) outer loop
      std::vector<pagmo::vector_double> violated_points;
      violated_points.reserve(m_p.samples); // Pre-allocate for efficiency

      // Sample random points
      // Time: O(S × (C + n))
      for (unsigned s = 0; s < m_p.samples; ++s) { // O(S) middle loop
        // Generate random point - Time: O(n)
        pagmo::vector_double x(m_lb.size());
        for (size_t i = 0; i < x.size(); ++i) { // O(n)
          x[i] = dists[i](gen);
        }

        // Evaluate constraint
        // Time: O(C)
        auto f = m_base.fitness(x);
        size_t nobj = m_base.get_nobj();

        // Check if constraint is violated
        // Time: O(1)
        bool violated = false;
        if (c < nec) {
          // Equality constraint: violated if |c(x)| > tolerance
          violated = (std::abs(f[nobj + c]) > 1e-6);
        } else {
          // Inequality constraint: violated if c(x) > 0
          violated = (f[nobj + c] > 0.0);
        }

        if (violated) {
          violated_points.push_back(x); // O(n) copy
        }
      }

      // Compute centroid of violated points
      // Time: O(V × n)
      // where V = number of violated points (≤ S)
      if (!violated_points.empty()) {
        pagmo::vector_double centroid(m_lb.size(), 0.0);

        // Sum all violated points
        // Time: O(V × n)
        for (const auto &pt : violated_points) {         // O(V) loop
          for (size_t i = 0; i < centroid.size(); ++i) { // O(n) loop
            centroid[i] += pt[i];
          }
        }

        // Average to get centroid
        // Time: O(n)
        for (size_t i = 0; i < centroid.size(); ++i) {
          centroid[i] /= violated_points.size();
        }

        m_centers[c] = centroid;
      } else {
        // Fallback to midpoint if no violations found
        // Time: O(n)
        pagmo::vector_double mid(m_lb.size());
        for (size_t i = 0; i < mid.size(); ++i) { // O(n)
          mid[i] = 0.5 * (m_lb[i] + m_ub[i]);
        }
        m_centers[c] = mid;
      }
    }
  }
};

} // namespace cp
