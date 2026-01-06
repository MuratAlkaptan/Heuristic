#pragma once
#include <cstddef>

struct PenaltyParams {
    // Base weights for constraints
    double w_ineq = 1.0;
    double w_eq   = 1.0;

    // Power for violation shaping (>= 1). 2 is common.
    double p = 2.0;

    // Center avoidance factor: penalty *= (1 + lambda * proximity)
    double lambda = 3.0;

    // Proximity radius in NORMALIZED space (x in [0,1]^d).
    // Smaller => strong repulsion only when very close to the center.
    double sigma = 0.15;

    // Cap violations to avoid blow-ups when constraint functions can go huge.
    double v_cap_ineq = 1e6;
    double v_cap_eq   = 1e6;

    // Equality tolerance band (0 means strict |eq|).
    double eq_tol = 0.0;

    // Online centroid update cap (limits "effective N" in centroid update).
    std::size_t centroid_cap_n = 200;
};
