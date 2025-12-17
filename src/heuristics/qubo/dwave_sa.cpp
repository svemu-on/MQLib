#include "heuristics/qubo/dwave_sa.h"
#include "heuristics/qubo/qubo_solution.h"

#include <iostream>
#include <tuple>
#include <vector>

DWaveSA::DWaveSA(const QUBOInstance& qi,
                 double runtime_limit,
                 bool validation,
                 QUBOCallback *qc)
    : QUBOHeuristic(qi, runtime_limit, validation, qc)
{
    std::vector<std::tuple<int,int,double>> terms;
    int n = qi.get_size();
    const std::vector<double>& lin = qi.get_lin();

    terms.reserve(n + qi.get_edge_count());

    for (int i = 0; i < n; ++i) {
        double w = lin[i];
        if (w != 0.0) {
            terms.emplace_back(i, i, w);
        }
    }

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
    DWaveResult res = run_dwave_solver(terms, "sa", std::string(), &error);

    if (!error.empty()) {
        std::cerr << "DWaveSA error: " << error << std::endl;
        return;
    }
    if (res.best_sample.empty()) {
        std::cerr << "DWaveSA: no sample returned" << std::endl;
        return;
    }

    QUBOSolution sol(res.best_sample, qi, this);
    Report(sol);
}
