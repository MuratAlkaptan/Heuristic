#pragma once
#include <pagmo/types.hpp>

#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>
#include <stdexcept>

struct PenaltyState {
    using vec = pagmo::vector_double;

    std::size_t dim = 0;   // decision dimension
    std::size_t nic = 0;   // number of inequality constraints

    // Known centers in normalized space (optional)
    bool has_known_centers = false;
    std::vector<vec> known_centers; // size nic, each size dim

    // Learned centers (online centroid) in normalized space
    std::vector<vec> mu;             // size nic, each size dim
    std::vector<std::size_t> count;  // size nic

    mutable std::mutex mtx;

    void ensure(std::size_t nic_, std::size_t dim_) {
        std::lock_guard<std::mutex> lk(mtx);
        if (dim == 0) dim = dim_;
        if (nic == 0) nic = nic_;

        if (dim != dim_ || nic != nic_) {
            throw std::runtime_error("PenaltyState::ensure(): dimension/constraint mismatch.");
        }

        if (mu.empty()) {
            mu.assign(nic, vec(dim, 0.0));
            count.assign(nic, 0);
        }
        if (has_known_centers && known_centers.size() != nic) {
            throw std::runtime_error("PenaltyState: known_centers size != nic.");
        }
    }

    void set_known_centers_normalized(std::vector<vec> centers_norm) {
        std::lock_guard<std::mutex> lk(mtx);
        if (dim == 0 || nic == 0) {
            throw std::runtime_error("PenaltyState::set_known_centers_normalized(): call ensure() first.");
        }
        if (centers_norm.size() != nic) {
            throw std::runtime_error("PenaltyState::set_known_centers_normalized(): centers size != nic.");
        }
        for (const auto &c : centers_norm) {
            if (c.size() != dim) {
                throw std::runtime_error("PenaltyState::set_known_centers_normalized(): center dim mismatch.");
            }
        }
        known_centers = std::move(centers_norm);
        has_known_centers = true;
    }

    // Online centroid update for constraint i (normalized x)
    void update_centroid(std::size_t i, const vec &x_norm, std::size_t cap_n) {
        std::lock_guard<std::mutex> lk(mtx);
        if (i >= nic || x_norm.size() != dim) return;

        count[i] += 1;
        const std::size_t eff_n = (count[i] < cap_n) ? count[i] : cap_n;
        const double eta = 1.0 / static_cast<double>(eff_n);

        for (std::size_t k = 0; k < dim; ++k) {
            mu[i][k] += eta * (x_norm[k] - mu[i][k]);
        }
    }

    vec get_center(std::size_t i) const {
        std::lock_guard<std::mutex> lk(mtx);
        if (i >= nic) return vec(dim, 0.0);
        if (has_known_centers) return known_centers[i];
        return mu[i];
    }
};

using PenaltyStatePtr = std::shared_ptr<PenaltyState>;
