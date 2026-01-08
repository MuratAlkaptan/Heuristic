#include <pagmo/problems/cec2006.hpp>
#include <iostream>
#include <iomanip>

int main() {
    for (int i = 1; i <= 10; ++i) {
        pagmo::cec2006 prob(i);
        auto best = prob.best_known();
        std::cout << "g" << i << ": " << std::fixed << std::setprecision(6) << best[0] << "\n";
    }
    return 0;
}
