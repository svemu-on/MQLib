#ifndef HEURISTICS_QUBO_DWAVE_BRIDGE_H_
#define HEURISTICS_QUBO_DWAVE_BRIDGE_H_

#include <string>
#include <tuple>
#include <vector>

// Result of a D-Wave solve: a 0/1 assignment and its objective value
// in the MQLib maximisation convention.
struct DWaveResult {
    std::vector<int> best_sample;
    double best_weight;
};

// Call into the Python helper (mqlib_dwave.solve_qubo).
//
//  - qubo_terms: list of (i, j, weight) triples in the MQLib QUBO convention
//  - backend: "qpu" or "sa"
//  - config_json_path: optional path to a JSON config file; pass an empty
//                      string "" to use only defaults / env variables.
//  - error_message: if non-null, receives a human-readable error string.
//                   On error, best_sample will be empty.
//
// When MQLib is compiled without USE_DWAVE defined, this function
// returns an empty best_sample and sets error_message appropriately.
DWaveResult run_dwave_solver(
    const std::vector<std::tuple<int,int,double>>& qubo_terms,
    const std::string& backend,
    const std::string& config_json_path,
    std::string* error_message);

#endif  // HEURISTICS_QUBO_DWAVE_BRIDGE_H_
