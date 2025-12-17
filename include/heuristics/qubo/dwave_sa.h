#ifndef HEURISTICS_QUBO_DWAVE_SA_H_
#define HEURISTICS_QUBO_DWAVE_SA_H_

#include "problem/qubo_heuristic.h"
#include "problem/qubo_instance.h"
#include "heuristics/qubo/dwave_bridge.h"

// D-Wave simulated annealing heuristic: same pattern as DWaveQPU but
// using the "sa" backend provided by dwave.samplers.SimulatedAnnealingSampler.
class DWaveSA : public QUBOHeuristic {
public:
    DWaveSA(const QUBOInstance& qi,
            double runtime_limit,
            bool validation,
            QUBOCallback *qc);
};

#endif  // HEURISTICS_QUBO_DWAVE_SA_H_
