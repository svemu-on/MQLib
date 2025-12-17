#ifndef HEURISTICS_QUBO_DWAVE_QPU_H_
#define HEURISTICS_QUBO_DWAVE_QPU_H_

#include "problem/qubo_heuristic.h"
#include "problem/qubo_instance.h"
#include "heuristics/qubo/dwave_bridge.h"

// D-Wave QPU heuristic: builds the QUBO from QUBOInstance and calls
// the Python helper using the "qpu" backend.  The algorithm runs in
// the constructor, consistent with other MQLib heuristics.
class DWaveQPU : public QUBOHeuristic {
public:
    DWaveQPU(const QUBOInstance& qi,
             double runtime_limit,
             bool validation,
             QUBOCallback *qc);
};

#endif  // HEURISTICS_QUBO_DWAVE_QPU_H_
