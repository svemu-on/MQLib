#include "heuristics/qubo/dwave_qpu.h"
#include "heuristics/qubo/qubo_solution.h"

#include <iostream>
#include <tuple>
#include <vector>

DWaveQPU::DWaveQPU(const QUBOInstance& qi,
                   double runtime_limit,
                   bool validation,
                   QUBOCallback *qc)
    : QUBOHeuristic(qi, runtime_limit, validation, qc)
{
    // Build list of QUBO terms (i, j, weight) from the instance.
    std::vector<std::tuple<int,int,double>> terms;
    int n = qi.get_size();
    const std::vector<double>& lin = qi.get_lin();

    terms.reserve(n + qi.get_edge_count());

    // Diagonal terms
    for (int i = 0; i < n; ++i) {
        double w = lin[i];
        if (w != 0.0) {
            terms.emplace_back(i, i, w);
        }
    }

    // Off-diagonal terms: stored once per pair with (min(i,j), max(i,j))
    for (auto it = qi.get_all_nonzero_begin();
         it != qi.get_all_nonzero_end(); ++it)
    {
        int i = it->first.first;
        int j = it->first.second;
        double w = it->second;
        if (w != 0.0) {
            terms.emplace_back(i, j, w);
        }
    }

    std::string error;
    DWaveResult res = run_dwave_solver(terms, "qpu", std::string(), &error);

    if (!error.empty()) {
        std::cerr << "DWaveQPU error: " << error << std::endl;
        return;
    }
    if (res.best_sample.empty()) {
        std::cerr << "DWaveQPU: no sample returned" << std::endl;
        return;
    }

    // Convert the returned sample into a QUBOSolution and report it.
    QUBOSolution sol(res.best_sample, qi, this);
    Report(sol);
}
