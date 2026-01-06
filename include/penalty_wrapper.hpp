#pragma once
#include <algorithm>
#include <string>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "custom_penalty.hpp"

struct penalty_wrapper_udp {
    pagmo::problem base;   // original constrained problem
    double rho = 1.0;      // penalty multiplier (keep this configurable)

    // Make the wrapped problem unconstrained.
    pagmo::vector_double::size_type get_nec() const { return 0u; }
    pagmo::vector_double::size_type get_nic() const { return 0u; }

    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        return base.get_bounds();
    }

    std::string get_name() const {
        return "PenaltyWrapper(" + base.get_name() + ")";
    }

    pagmo::vector_double fitness(const pagmo::vector_double &x) const {
        const auto fv = base.fitness(x);

        const auto nobj = base.get_nobj();
        const auto nec  = base.get_nec();
        const auto nic  = base.get_nic();

        // Split constraints from pagmo fitness vector: [obj | eq | ineq]
        pagmo::vector_double eq, ineq;
        eq.reserve(nec);
        ineq.reserve(nic);

        // equality constraints start at index nobj
        for (pagmo::vector_double::size_type i = 0; i < nec; ++i) {
            eq.push_back(fv[nobj + i]);
        }
        // inequality constraints start at index nobj + nec
        for (pagmo::vector_double::size_type i = 0; i < nic; ++i) {
            ineq.push_back(fv[nobj + nec + i]);
        }

        const double pen = std::max(0.0, custom_penalty(eq, ineq));
        // NOTE: pagmo is minimization-oriented. Penalize by ADDING.
        const double penalized_obj = fv[0] + rho * pen;

        return {penalized_obj};
    }
};
