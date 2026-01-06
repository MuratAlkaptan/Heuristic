#pragma once
#include <pagmo/types.hpp>
#include <algorithm>
#include <utility>
#include <cstddef>

inline pagmo::vector_double normalize01(const pagmo::vector_double &x,
                                       const std::pair<pagmo::vector_double, pagmo::vector_double> &bounds)
{
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;

    const std::size_t n = x.size();
    pagmo::vector_double xn(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        const double lo = lb[i];
        const double hi = ub[i];
        const double den = hi - lo;

        if (den <= 0.0) {
            xn[i] = 0.5; // degenerate bound
        } else {
            double t = (x[i] - lo) / den;
            xn[i] = std::clamp(t, 0.0, 1.0);
        }
    }
    return xn;
}

inline double l2_dist2(const pagmo::vector_double &a, const pagmo::vector_double &b) {
    const std::size_t n = std::min(a.size(), b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
