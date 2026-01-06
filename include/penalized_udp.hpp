#pragma once
#include <pagmo/types.hpp>
#include <utility>
#include <stdexcept>
#include <memory>
#include <string>

#include "penalty_params.hpp"
#include "penalty_state.hpp"
#include "normalize.hpp"
#include "custom_penalty.hpp"

template <typename ConstrainedUDP>
struct penalized_udp {
    ConstrainedUDP base;
    PenaltyParams par;
    PenaltyStatePtr st;

    penalized_udp() = default;

    explicit penalized_udp(ConstrainedUDP udp, PenaltyParams params = {})
        : base(std::move(udp)), par(params), st(std::make_shared<PenaltyState>()) {}

    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        return base.get_bounds();
    }

    // Set known centers in RAW x-space. We normalize and store them.
    // centers_raw must be size nic; each vector size dim.
    void set_known_centers_raw(const std::vector<pagmo::vector_double> &centers_raw) {
        const auto bounds = get_bounds();
        const std::size_t dim = bounds.first.size();
        const std::size_t nic = base.get_nic();

        st->ensure(nic, dim);

        if (centers_raw.size() != nic) {
            throw std::runtime_error("set_known_centers_raw(): centers_raw size != nic.");
        }

        std::vector<pagmo::vector_double> centers_norm;
        centers_norm.reserve(nic);
        for (const auto &c : centers_raw) {
            if (c.size() != dim) {
                throw std::runtime_error("set_known_centers_raw(): center dim mismatch.");
            }
            centers_norm.push_back(normalize01(c, bounds));
        }
        st->set_known_centers_normalized(std::move(centers_norm));
    }

    // Unconstrained: return objectives only (penalized)
    pagmo::vector_double fitness(const pagmo::vector_double &x) const {
        const auto f = base.fitness(x);

        const std::size_t nobj = base.get_nobj();
        const std::size_t nec  = base.get_nec();
        const std::size_t nic  = base.get_nic();

        if (f.size() != nobj + nec + nic) {
            throw std::runtime_error("penalized_udp: base.fitness(x) size != nobj+nec+nic.");
        }

        const auto bounds = get_bounds();
        const auto x_norm = normalize01(x, bounds);

        // ensure state dimensions
        if (st) st->ensure(nic, x.size());

        // slice objective/constraints
        pagmo::vector_double obj(f.begin(), f.begin() + nobj);
        pagmo::vector_double eq (f.begin() + nobj, f.begin() + nobj + nec);
        pagmo::vector_double ineq(f.begin() + nobj + nec, f.end());

        const double pen = custom_penalty(x_norm, eq, ineq, st, par);

        for (auto &o : obj) o += pen;
        return obj;
    }

    std::size_t get_nobj() const { return base.get_nobj(); }
    std::size_t get_nec()  const { return 0u; } // now unconstrained
    std::size_t get_nic()  const { return 0u; } // now unconstrained

    std::string get_name() const {
        return "penalized_udp(" + base.get_name() + ")";
    }
};
