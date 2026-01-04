// ga_baseline.h
// Minimal GA engine (maximize fitness) with:
// - tournament selection
// - elitism
// - crossover + mutation hooks
// Inspired by standard GA skeleton patterns (selection/crossover/mutation/elitism).
#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

struct GAParams {
    int popSize = 200;
    int generations = 300;
    int tournamentK = 3;
    int elites = 2;
    double pCrossover = 0.9;
    double pMutation = 0.2;
    unsigned seed = 42;
};

template <typename Genome>
struct GAResult {
    Genome bestGenome{};
    double bestFitness = -1e300;
};

template <typename Genome>
GAResult<Genome> runGA(
    const GAParams& P,
    const std::function<Genome(std::mt19937&)>& makeRandom,
    const std::function<double(const Genome&, std::mt19937&)>& fitness,      // maximize
    const std::function<Genome(const Genome&, const Genome&, std::mt19937&)>& crossover,
    const std::function<void(Genome&, std::mt19937&)>& mutate,
    const std::function<void(int gen, const Genome& best, double bestFit, double avgFit)>& onGeneration
) {
    std::mt19937 rng(P.seed);

    std::vector<Genome> pop(P.popSize);
    std::vector<double> fit(P.popSize, -1e300);

    for (int i = 0; i < P.popSize; ++i) {
        pop[i] = makeRandom(rng);
        fit[i] = fitness(pop[i], rng);
    }

    auto argmaxFit = [&]() {
        int bi = 0;
        for (int i = 1; i < (int)fit.size(); ++i) if (fit[i] > fit[bi]) bi = i;
        return bi;
    };

    auto tournamentPick = [&]() {
        std::uniform_int_distribution<int> D(0, P.popSize - 1);
        int best = D(rng);
        for (int i = 1; i < P.tournamentK; ++i) {
            int j = D(rng);
            if (fit[j] > fit[best]) best = j;
        }
        return best;
    };

    GAResult<Genome> globalBest;
    {
        int bi = argmaxFit();
        globalBest.bestGenome = pop[bi];
        globalBest.bestFitness = fit[bi];
    }

    for (int gen = 0; gen < P.generations; ++gen) {
        // elitism indices
        std::vector<int> idx(P.popSize);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min(P.elites, P.popSize), idx.end(),
                          [&](int a, int b){ return fit[a] > fit[b]; });

        std::vector<Genome> nextPop;
        nextPop.reserve(P.popSize);

        // copy elites
        for (int e = 0; e < std::min(P.elites, P.popSize); ++e) {
            nextPop.push_back(pop[idx[e]]);
        }

        std::uniform_real_distribution<double> U(0.0, 1.0);

        // fill rest
        while ((int)nextPop.size() < P.popSize) {
            int pa = tournamentPick();
            int pb = tournamentPick();

            Genome child = pop[pa];

            if (U(rng) < P.pCrossover) {
                child = crossover(pop[pa], pop[pb], rng);
            }
            if (U(rng) < P.pMutation) {
                mutate(child, rng);
            }
            nextPop.push_back(std::move(child));
        }

        pop = std::move(nextPop);

        // re-evaluate
        for (int i = 0; i < P.popSize; ++i) {
            fit[i] = fitness(pop[i], rng);
        }

        // stats
        double avg = 0.0;
        for (double v : fit) avg += v;
        avg /= (double)fit.size();

        int bi = argmaxFit();
        if (fit[bi] > globalBest.bestFitness) {
            globalBest.bestFitness = fit[bi];
            globalBest.bestGenome = pop[bi];
        }

        onGeneration(gen, globalBest.bestGenome, globalBest.bestFitness, avg);
    }

    return globalBest;
}
