#pragma once

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cp {

using pagmo::vector_double;

// PaGMO conventions:
// - equality constraints:   c_eq(x) = 0
// - inequality constraints: c_ineq(x) <= 0  (violation is max(0, c_ineq(x)))

struct params {
    // Sampling for offline center estimation
    std::size_t center_samples = 5000;   // increase to make centers more “stable”
    unsigned center_seed = 123u;

    // Equality violation threshold (what counts as "violated" for sampling/penalty)
    double eq_tol = 1e-6;

    // Penalty strength multipliers (make these huge to “show how bad it is”)
    double lambda_eq = 1e4;
    double lambda_ineq = 1e4;

    // Extra “center attraction” multiplier:
    // penalty factor becomes (1 + boost * exp(-d^2/(2*sigma^2)))
    double boost = 50.0;

    // If <= 0, sigma is auto-picked from bounds (domain diameter).
    double sigma = -1.0;

    // Prevent division/zero issues
    double eps = 1e-12;
};

inline vector_double midpoint(const std::pair<vector_double, vector_double> &bounds) {
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    vector_double m(lb.size(), 0.0);
    for (std::size_t i = 0; i < lb.size(); ++i) {
        const double a = std::isfinite(lb[i]) ? lb[i] : -1.0;
        const double b = std::isfinite(ub[i]) ? ub[i] :  1.0;
        m[i] = 0.5 * (a + b);
    }
    return m;
}

inline double domain_diameter(const std::pair<vector_double, vector_double> &bounds) {
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < lb.size(); ++i) {
        const double a = std::isfinite(lb[i]) ? lb[i] : -1.0;
        const double b = std::isfinite(ub[i]) ? ub[i] :  1.0;
        const double r = (b - a);
        sum_sq += r * r;
    }
    return std::sqrt(sum_sq);
}

inline vector_double uniform_sample_in_bounds(const std::pair<vector_double, vector_double> &bounds,
                                              std::mt19937_64 &rng) {
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;

    std::uniform_real_distribution<double> U(0.0, 1.0);
    vector_double x(lb.size(), 0.0);

    for (std::size_t i = 0; i < lb.size(); ++i) {
        const double a = std::isfinite(lb[i]) ? lb[i] : -1.0;
        const double b = std::isfinite(ub[i]) ? ub[i] :  1.0;
        const double u = U(rng);
        x[i] = a + u * (b - a);
    }
    return x;
}

inline double sq_dist(const vector_double &a, const vector_double &b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

struct split_constraints_result {
    vector_double obj;   // size = nobj (we assume nobj=1 in this demo)
    vector_double eq;    // size = nec
    vector_double ineq;  // size = nic
};

inline split_constraints_result split_fitness(const pagmo::problem &prob,
                                              const vector_double &fit) {
    const auto nobj = prob.get_nobj();
    const auto nec  = prob.get_nec();
    const auto nic  = prob.get_nic();

    split_constraints_result out;
    out.obj.assign(fit.begin(), fit.begin() + static_cast<std::ptrdiff_t>(nobj));
    out.eq.assign(fit.begin() + static_cast<std::ptrdiff_t>(nobj),
                  fit.begin() + static_cast<std::ptrdiff_t>(nobj + nec));
    out.ineq.assign(fit.begin() + static_cast<std::ptrdiff_t>(nobj + nec),
                    fit.begin() + static_cast<std::ptrdiff_t>(nobj + nec + nic));
    return out;
}

inline double total_violation(const vector_double &eq, const vector_double &ineq, double eq_tol) {
    double v = 0.0;
    for (double c : eq) {
        const double a = std::abs(c);
        if (a > eq_tol) v += a;
    }
    for (double g : ineq) {
        if (g > 0.0) v += g;
    }
    return v;
}

// Compute one "center" per constraint index:
// indices [0..nec-1]    => equality constraints
// indices [nec..nec+nic-1] => inequality constraints
inline std::vector<vector_double> compute_constraint_centers(const pagmo::problem &base,
                                                             const params &p) {
    const auto nec = base.get_nec();
    const auto nic = base.get_nic();
    const auto nx  = base.get_nx();
    const std::size_t nc = static_cast<std::size_t>(nec + nic);

    std::vector<vector_double> sum(nc, vector_double(nx, 0.0));
    std::vector<std::size_t> count(nc, 0);

    const auto bounds = base.get_bounds();
    std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(p.center_seed));

    for (std::size_t s = 0; s < p.center_samples; ++s) {
        vector_double x = uniform_sample_in_bounds(bounds, rng);
        vector_double fit = base.fitness(x);
        auto parts = split_fitness(base, fit);

        // Equality constraints
        for (std::size_t i = 0; i < static_cast<std::size_t>(nec); ++i) {
            if (std::abs(parts.eq[i]) > p.eq_tol) {
                for (std::size_t k = 0; k < nx; ++k) sum[i][k] += x[k];
                count[i] += 1;
            }
        }
        // Inequality constraints
        for (std::size_t j = 0; j < static_cast<std::size_t>(nic); ++j) {
            if (parts.ineq[j] > 0.0) {
                const std::size_t idx = static_cast<std::size_t>(nec) + j;
                for (std::size_t k = 0; k < nx; ++k) sum[idx][k] += x[k];
                count[idx] += 1;
            }
        }
    }

    std::vector<vector_double> centers(nc);
    const auto mid = midpoint(bounds);
    for (std::size_t c = 0; c < nc; ++c) {
        if (count[c] == 0) {
            centers[c] = mid; // fallback: if never observed violated, use domain midpoint
        } else {
            centers[c] = sum[c];
            for (double &v : centers[c]) v /= static_cast<double>(count[c]);
        }
    }
    return centers;
}

// Your "center proximity" penalty:
// - scales with violation magnitude
// - multiplies by (1 + boost * exp(-d^2/(2*sigma^2))) so closer to center => larger penalty
inline double centroid_penalty(const vector_double &x,
                               const vector_double &eq,
                               const vector_double &ineq,
                               const std::vector<vector_double> &centers,
                               std::size_t nec,
                               const params &p,
                               double sigma) {
    const double s = (sigma > 0.0) ? sigma : 1.0;
    const double two_s2 = 2.0 * s * s + p.eps;

    double pen = 0.0;

    // Equality constraints [0..nec-1]
    for (std::size_t i = 0; i < eq.size(); ++i) {
        const double v = std::abs(eq[i]);
        if (v <= p.eq_tol) continue;

        const double d2 = sq_dist(x, centers[i]);
        const double proximity = std::exp(-d2 / two_s2);
        pen += p.lambda_eq * v * (1.0 + p.boost * proximity);
    }

    // Inequality constraints [nec..nec+nic-1]
    for (std::size_t j = 0; j < ineq.size(); ++j) {
        const double v = std::max(0.0, ineq[j]);
        if (v <= 0.0) continue;

        const std::size_t idx = nec + j;
        const double d2 = sq_dist(x, centers[idx]);
        const double proximity = std::exp(-d2 / two_s2);
        pen += p.lambda_ineq * v * (1.0 + p.boost * proximity);
    }

    // Must be non-negative
    return (pen >= 0.0) ? pen : 0.0;
}

inline std::string to_string_vec(const vector_double &v, std::size_t max_elems = 6) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < v.size() && i < max_elems; ++i) {
        oss << v[i];
        if (i + 1 < v.size() && i + 1 < max_elems) oss << ", ";
    }
    if (v.size() > max_elems) oss << ", ...";
    oss << "]";
    return oss.str();
}

} // namespace cp
