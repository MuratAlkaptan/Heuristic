#pragma once
#include <pagmo/types.hpp>
#include <utility>
#include <string>
#include <cmath>

struct demo_constrained_udp {
    // 2D
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        return {pagmo::vector_double{-5.0, -5.0}, pagmo::vector_double{5.0, 5.0}};
    }

    // 1 objective, 0 equalities, 2 inequalities
    std::size_t get_nobj() const { return 1u; }
    std::size_t get_nec()  const { return 0u; }
    std::size_t get_nic()  const { return 2u; }

    // Returns [obj | eq | ineq] following pagmo conventions
    pagmo::vector_double fitness(const pagmo::vector_double &x) const {
        const double x0 = x[0], x1 = x[1];

        // objective: sphere
        const double obj = x0 * x0 + x1 * x1;

        // forbidden circles centers and radii
        const double c1x = 1.5, c1y = 1.0, r1 = 1.2;
        const double c2x = -2.0, c2y = -1.5, r2 = 1.0;

        const double d1 = (x0 - c1x) * (x0 - c1x) + (x1 - c1y) * (x1 - c1y);
        const double d2 = (x0 - c2x) * (x0 - c2x) + (x1 - c2y) * (x1 - c2y);

        // ineq <= 0 is feasible. Here: r^2 - dist^2 <= 0 => dist^2 >= r^2 (outside circle feasible)
        const double g1 = (r1 * r1) - d1;
        const double g2 = (r2 * r2) - d2;

        return pagmo::vector_double{obj, /*eq none*/, g1, g2};
    }

    std::string get_name() const {
        return "demo_constrained_udp";
    }
};
