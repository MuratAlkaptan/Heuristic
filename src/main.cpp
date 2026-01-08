// Main file for centroid proximity penalty method

#include "centroid_penalty.hpp"
#include <iomanip>
#include <iostream>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2006.hpp>

/*
  Time: O(min(n, 10)) => O(1) for display
  Space: O(1)
 */
void print_vector(const pagmo::vector_double &v, const std::string &name) {
  std::cout << name << " = [";
  // Print at most 10 elements - Time: O(min(n, 10))
  for (size_t i = 0; i < std::min(v.size(), size_t(10)); ++i) {
    std::cout << std::fixed << std::setprecision(6) << v[i];
    if (i < std::min(v.size(), size_t(10)) - 1)
      std::cout << ", ";
  }
  if (v.size() > 10)
    std::cout << ", ...";
  std::cout << "]\n";
}

/*
  Main function: Run centroid proximity penalty GA

  Overall Time Complexity: O(S × m × (C + n) + P × G × (C + m × n))
    Phase 1 (Initialization): O(S × m × (C + n))
    Phase 2 (Evolution): O(P × G × (C + m × n))
  Overall Space Complexity: O(m × n + P × n)
    - Centers: O(m × n)
    - Population: O(P × n)
 */
int main(int argc, char **argv) {
  // Command-line argument parsing
  //  Time: O(1)
  //   Space: O(1)
  int prob = (argc > 1) ? std::atoi(argv[1]) : 1;
  if (prob < 1 || prob > 24)
    prob = 1;

  // Problem setup
  //  Time: O(1)
  //  Space: O(n)
  pagmo::problem base{pagmo::cec2006(prob)};

  // penalty parameters
  // Time: O(1)
  //  Space: O(1)
  cp::params p;
  p.lambda = 1e4; // Base penalty weight
  p.boost = 50.0; // Proximity penalty multiplier
  p.sigma = -1.0; // Auto-compute sigma (15% of diagonal)

  // 1 Initialization (Centroid Finding)
  // Time: O(S × m × (C + n))
  // Space: O(m × n + S × n)
  cp::centroid_udp udp(base, p); // Centroid finding happens here

  // Wrap in pagmo problem
  //  Time: O(1)
  //  Space: O(1)
  pagmo::problem penalized{udp};

  // 2 Evolution (Genetic Algorithm)
  //  Setup GA
  //  Time: O(1)
  //  Space: O(1)
  pagmo::algorithm ga{pagmo::sga(200)}; // 200 generations

  // Initialize population
  // Time: O(P × (C + m × n))
  //  Space: O(P × n)
  pagmo::population pop(penalized, 60, 777);

  // run evolution
  //  Time: O(P × G × (C + m × n))
  //    -P=60 individuals × G=200 generations
  //    -Each evaluation: O(C + m × n)
  pop = ga.evolve(pop);

  // Extract results
  //  Time: O(n)
  //  Space: O(n)
  auto x = pop.champion_x();       // Best solution found - O(n) space
  auto fpen = pop.champion_f()[0]; // Penalized fitness - O(1)
  auto fbase = base.fitness(x);    // Re-evaluate on original - O(C)

  double obj = fbase[0]; // Original objective value

  // Get problem metadata - Time: O(1)
  size_t nobj = base.get_nobj();
  size_t nec = base.get_nec();
  size_t nic = base.get_nic();
  auto bounds = base.get_bounds();

  // Calculate constraint violations
  //  Time: O(m) Space: O(1)
  double eq_viol = 0.0;
  double ineq_viol = 0.0;

  // Sum equality violations
  // Time: O(nec)
  for (size_t i = 0; i < nec; ++i) {
    eq_viol += std::abs(fbase[nobj + i]);
  }

  // Sum inequality violations
  //  Time: O(nic)
  for (size_t i = 0; i < nic; ++i) {
    ineq_viol += std::max(0.0, fbase[nobj + nec + i]);
  }

  double total_viol = eq_viol + ineq_viol;

  // Display results
  //  Time: O(1) for output (limited vector printing)
  std::cout << "\n========================================\n";
  std::cout << "CEC2006 Problem " << prob << ": " << base.get_name() << "\n";
  std::cout << "========================================\n";
  std::cout << "Dimensions: " << x.size() << "\n";
  std::cout << "Equality constraints: " << nec << "\n";
  std::cout << "Inequality constraints: " << nic << "\n";
  std::cout << "Bounds: [" << bounds.first[0] << ", " << bounds.second[0]
            << "]";
  if (x.size() > 1)
    std::cout << " × " << x.size();
  std::cout << "\n\n";

  std::cout << "RESULTS:\n";
  std::cout << "--------\n";
  std::cout << "Objective value: " << std::fixed << std::setprecision(8) << obj
            << "\n";
  std::cout << "Equality violation: " << std::scientific << eq_viol << "\n";
  std::cout << "Inequality violation: " << ineq_viol << "\n";
  std::cout << "Total violation: " << total_viol << "\n";
  std::cout << "Feasible: " << (total_viol < 1e-4 ? "YES ✓" : "NO ✗") << "\n";
  std::cout << "Penalized fitness: " << std::fixed << fpen << "\n\n";

  print_vector(x, "Solution"); // O(1) - prints only first 10 elements

  std::cout << "\n========================================\n\n";

  return 0;
}
