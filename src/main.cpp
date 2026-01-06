#include <pagmo/pagmo.hpp>
#include <iostream>

#include "demo_constrained_udp.hpp"
#include "penalized_udp.hpp"

int main() {
    // 1) Original constrained UDP
    demo_constrained_udp udp;

    // 2) Penalty parameters (tune here)
    PenaltyParams pp;
    pp.w_ineq = 10.0;   // raise to enforce feasibility harder
    pp.p      = 2.0;
    pp.lambda = 4.0;    // strength of center avoidance
    pp.sigma  = 0.12;   // radius in normalized space

    // 3) Wrap to make it unconstrained via penalty (now penalty has access to x)
    penalized_udp<demo_constrained_udp> wrapped(udp, pp);

    // OPTIONAL: set true known centers (raw x space). This is the strongest "avoid center" mode.
    // If you comment this out, the wrapper will learn centers online from violating x values.
    {
        std::vector<pagmo::vector_double> centers_raw;
        centers_raw.push_back({1.5, 1.0});    // center for constraint g1
        centers_raw.push_back({-2.0, -1.5});  // center for constraint g2
        wrapped.set_known_centers_raw(centers_raw);
    }

    // 4) Create problem from wrapped UDP (unconstrained)
    pagmo::problem prob{wrapped};

    // 5) Choose an algorithm (DE is fine for demo)
    pagmo::algorithm algo{pagmo::de(200u, 0.8, 0.9, 1u)};
    algo.set_verbosity(50);

    // 6) Evolve
    pagmo::population pop(prob, 80u, 123u);
    pop = algo.evolve(pop);

    // 7) Report
    const auto x = pop.champion_x();
    const auto f = pop.champion_f();

    std::cout << "Champion x = [" << x[0] << ", " << x[1] << "]\n";
    std::cout << "Champion f = " << f[0] << "\n";

    // NOTE: f is penalized objective. If you want raw objective/constraints, evaluate udp.fitness(x):
    const auto raw = udp.fitness(x);
    std::cout << "Raw obj = " << raw[0] << "\n";
    std::cout << "Raw g1  = " << raw[1] << " (<=0 feasible)\n";
    std::cout << "Raw g2  = " << raw[2] << " (<=0 feasible)\n";

    return 0;
}
