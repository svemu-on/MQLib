"""
mqlib_dwave.py
----------------

This module provides a minimal wrapper around D‑Wave’s Ocean SDK for
solving QUBO instances encoded in the MQLib format.  It exposes a
single function, ``solve_qubo``, which takes a list of (i, j, weight)
terms, a backend identifier (``"qpu"`` or ``"sa"``) and an optional
path to a JSON configuration file.  The module constructs a dimod
BinaryQuadraticModel, applies default parameters with optional
overrides and returns the best sample and its objective value in the
MQLib maximisation convention.

The QUBO coefficients provided by MQLib are **maximisation**
coefficients: the objective in MQLib is

    maximise sum_i lin[i] * x_i + 2 * sum_{i<j} w_ij * x_i * x_j.

Ocean, however, solves a **minimisation** QUBO.  To convert between
the two, the signs of all coefficients are flipped (i.e., we solve
minimise -weight).  The best_energy returned by this module is
therefore the negative of the dimod energy.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Optional

import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler


def default_config() -> Dict[str, Any]:
    """Return the default configuration for both QPU and SA backends."""
    return {
        "dwave": {
            "qpu": {
                "num_reads": 100,
                "anneal_time": 250,  # microseconds
                "solver": "Advantage2_system1.8",
            },
            "sa": {
                "num_reads": 100,
                "num_sweeps": 1000,
            },
        }
    }


def load_config_json(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a JSON configuration file if available.  The search order is:

    1. If ``path`` is a non‑empty string and the file exists, load it.
    2. Otherwise, if the environment variable ``MQLIB_DWAVE_CONFIG`` is
       set and points to an existing file, load it.
    3. Otherwise, if a file named ``dwave_config.json`` exists in the
       current working directory, load it.
    4. Otherwise, return an empty dict.

    Errors while reading JSON will propagate to the caller.
    """
    candidate: Optional[str] = None
    if path:
        candidate = os.path.expanduser(path)
    elif os.environ.get("MQLIB_DWAVE_CONFIG"):
        candidate = os.path.expanduser(os.environ["MQLIB_DWAVE_CONFIG"])
    else:
        if os.path.isfile("dwave_config.json"):
            candidate = "dwave_config.json"
    if candidate and os.path.isfile(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge ``override`` into ``base``.  Values in
    ``override`` take precedence.  Only keys present in ``override``
    will overwrite those in ``base``; extra keys are added.  This is a
    shallow merge suitable for the simple nested configuration used
    here.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            merge_config(base[key], val)
        else:
            base[key] = val
    return base


def build_qubo_dict(
    terms: Iterable[Tuple[int, int, float]],
) -> Dict[Tuple[int, int], float]:
    """
    Convert a list of QUBO terms into a dictionary of coefficients
    suitable for ``dimod.BinaryQuadraticModel.from_qubo``.  The input
    terms are (i, j, weight) triples representing MQLib’s maximisation
    coefficients.  The returned coefficients are negated so that
    solving the QUBO minimises the negative of the MQLib weight.

    For each diagonal term (i == j) the coefficient -lin[i] is added.
    For each off‑diagonal term (i != j) the coefficient -2*w is added
    (the factor of two arises because MQLib stores each off‑diagonal
    weight once but the objective sums over i<j with a 2 multiplier).
    """
    qubo: Dict[Tuple[int, int], float] = {}
    for i, j, w in terms:
        i_int = int(i)
        j_int = int(j)
        coeff = float(w)
        if i_int == j_int:
            # linear term: negate to convert maximisation to minimisation
            key = (i_int, i_int)
            qubo[key] = qubo.get(key, 0.0) - coeff
        else:
            # off‑diagonal term: ensure (a,b) ordering and apply factor of 2
            a, b = (i_int, j_int) if i_int <= j_int else (j_int, i_int)
            qubo[(a, b)] = qubo.get((a, b), 0.0) - 2.0 * coeff
    return qubo


def _solve_qpu(bqm: dimod.BinaryQuadraticModel, cfg: Dict[str, Any]) -> dimod.SampleSet:
    """Submit the BQM to the quantum processing unit (QPU)."""
    solver_name = cfg.get("solver", "Advantage2_system1.8")
    num_reads = cfg.get("num_reads", 100)
    anneal_time = cfg.get("anneal_time", 250)
    # Attempt to select the requested solver
    try:
        sampler = EmbeddingComposite(DWaveSampler(solver={"name": solver_name}))
    except Exception:
        # Fall back to default solver selection
        sampler = EmbeddingComposite(DWaveSampler())
    return sampler.sample(
        bqm,
        num_reads=num_reads,
        annealing_time=anneal_time,
    )


def _solve_sa(bqm: dimod.BinaryQuadraticModel, cfg: Dict[str, Any]) -> dimod.SampleSet:
    """Solve the BQM using the classical simulated annealer."""
    sampler = SimulatedAnnealingSampler()
    num_reads = cfg.get("num_reads", 100)
    num_sweeps = cfg.get("num_sweeps", 1000)
    return sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
    )


def solve_qubo(
    terms: Iterable[Tuple[int, int, float]],
    backend: str,
    config_json_path: Optional[str] = None,
) -> Tuple[List[int], float]:
    """
    Solve the provided QUBO using either the quantum processor or the
    classical simulated annealer.  The ``terms`` iterable contains
    triples (i, j, weight) describing MQLib’s maximisation objective.

    :param terms: An iterable of (i, j, weight) entries.  Diagonal
        entries (i == j) encode linear terms.  Off‑diagonal entries
        encode pairwise interactions.
    :param backend: Either ``"qpu"`` or ``"sa"``.
    :param config_json_path: Optional path to a JSON file specifying
        solver parameters.  If empty or None, a default search path
        will be used (see :func:`load_config_json`).
    :returns: A pair ``(sample, weight)`` where ``sample`` is a list of
        0/1 assignments and ``weight`` is the maximised objective value
        in MQLib’s convention.
    :raises ValueError: If the backend is unknown or configuration
        loading fails.
    """
    # Begin with defaults and apply overrides
    cfg = default_config()
    if config_json_path is None or config_json_path == "":
        override = load_config_json(None)
    else:
        override = load_config_json(config_json_path)
    cfg = merge_config(cfg, override)
    # Build the QUBO dict and BQM.  The coefficients are negated so
    # that minimising returns the maximising solution.
    qubo_dict = build_qubo_dict(terms)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
    # Dispatch to the appropriate backend
    if backend == "qpu":
        sampleset = _solve_qpu(bqm, cfg["dwave"]["qpu"])
    elif backend == "sa":
        sampleset = _solve_sa(bqm, cfg["dwave"]["sa"])
    else:
        raise ValueError(f"Unknown backend: {backend}")
    # Select the sample with lowest energy (which corresponds to
    # highest weight because we negated the coefficients).  dimod
    # samplesets are sorted by energy so the first record is best.
    # Extract the best sample.  Samplesets are sorted by energy so
    # ``first`` returns the lowest‑energy sample.  Its ``sample``
    # attribute is a mapping {variable_index: value}.
    best = sampleset.first
    sample_dict = best.sample
    energy = float(best.energy)
    # Convert to a dense list ordered by variable index
    assignments = [int(sample_dict[i]) for i in range(len(sample_dict))]
    # Negate the energy to obtain the maximised weight
    weight = -energy
    return assignments, weight
