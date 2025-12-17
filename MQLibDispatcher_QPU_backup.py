#!/usr/bin/env python3

"""
MQLibDispatcher_QPU.py
======================

This script benchmarks the DWAVEQPU heuristic from MQLib on a
collection of problem instances (QUBO or MAX‑CUT).  Unlike the general dispatcher that
computes a per‑instance runtime based on a baseline heuristic, this
version assumes the D‑Wave quantum annealer is invoked once per
instance with fixed annealing parameters (e.g. `anneal_time=250` and
`num_reads=1000`), so no runtime caps or baseline iterations are
required.  The script processes instances in ascending order of
problem size and stops as soon as the QPU reports an embedding
failure (which typically indicates that larger instances will also
fail to embed).

Features:

* **Sorts instances by size.**  Uses `data/instance_header_info.csv`
  to determine the number of variables (`n`) **and, if available, the number
  of edges (`m`)** for each instance.  Instances are ordered first by
  increasing `n` and, for equal sizes, by increasing `m`; unknown sizes
  or edge counts are placed at the end.  The script also reads
  `data/standard.csv` to determine the problem type, passing `-fQ` or
  `-fM` accordingly when invoking MQLib.
* **Runs DWAVEQPU once per instance.**  Because the D‑Wave QPU backend
  does not support random seeding, the heuristic is invoked exactly
  once for each instance.  A `--seed` option records an arbitrary
  integer seed value in the output for compatibility with the
  analysis pipeline, but the seed has no effect on the QPU run.  The
  `-r` flag is set to a small positive number (e.g. 1.0) since the
  DWAVEQPU heuristic ignores the runtime and always performs a
  single QPU call.
* **Parses objective values and detects errors.**  The script reads
  the last non‑empty line of MQLib’s output to extract the best
  objective value.  If the output contains the word “error” (case
  insensitive) when using a D‑Wave heuristic, it is interpreted as an
  embedding failure and the script stops processing further
  instances.  Errors are logged to a separate file.
* **Outputs results in CSV format.**  Results are appended to
  `results.csv` (or another file specified via `--results_file`).  Each
  row has the columns `timestamp,graphname,heuristic,seed,limit,objective`.
  The `limit` field is set to `0.0` for all QPU runs to satisfy the
  analysis pipeline's expected column structure.  Errors are appended
  to `errors.txt` (or another file specified via `--errors_file`).

* **Supports resuming partially completed runs.**  With the
  `--skip_existing` flag, the dispatcher reads the existing results
  file and skips any instance that already has a recorded result.
  This allows you to restart a long run after a crash without
  recomputing earlier instances.

Usage example:

    python3 MQLibDispatcher_QPU.py \
        --seed 0 \
        --instances_file data/instances.txt \
        --results_file data/dwaveqpu_results.csv \
        --errors_file data/dwaveqpu_errors.txt \
        --skip_existing

This assumes that MQLib has been built with `USE_DWAVE=1` and that
the zipped benchmark instances are available locally.  D‑Wave Leap
credentials must be configured so that the QPU can be accessed.  The
heuristic list is not configurable—DWAVEQPU is the only heuristic
used in this script.  To benchmark the classical SA backend, use
MQLibDispatcher_local.py with the `DWAVESA` heuristic instead.
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except ImportError:
    print(
        "Error: pandas is required to run this script. Please install it via pip.",
        file=sys.stderr,
    )
    sys.exit(1)


def run_command(cmd):
    """Run a subprocess and return its combined stdout/stderr as a string."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out, _ = proc.communicate()
    return out


def extract_graph(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the contents of a zip file into a directory.

    Returns the path to the directory containing the extracted files.
    """
    # Ensure the extraction directory exists and is empty
    if extract_dir.exists():
        for item in extract_dir.iterdir():
            if item.is_dir():
                for sub in item.iterdir():
                    sub.unlink()
                item.rmdir()
            else:
                item.unlink()
    else:
        extract_dir.mkdir(parents=True)
    # Extract zip contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def parse_objective_from_output(output: str) -> float | None:
    """Extract the best objective from the last line of MQLib output.

    Returns the objective as a float if parsing succeeds, otherwise
    None.  MQLib prints CSV lines of the form:

        seed,heuristic,filename,best_objective,runtime,[...]

    For DWAVEQPU the runtime is irrelevant, so we focus on the
    objective in column 4 (index 3).
    """
    lines = [l for l in output.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    parts = last.split(",")
    if len(parts) < 4:
        return None
    try:
        return float(parts[3])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local dispatcher for DWAVEQPU heuristic on QUBO instances"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed value for MQLib.  The DWAVEQPU heuristic does not use random seeds, "
            "but this value is recorded for compatibility with the analysis pipeline "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip instances that already have entries in the results file. "
            "This allows restarting a crashed run without re-running completed instances."
        ),
    )
    parser.add_argument(
        "--instances_file",
        type=str,
        default="data/instances.txt",
        help="File containing list of instance zip filenames",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results.csv",
        help="Output CSV file for results (appended)",
    )
    parser.add_argument(
        "--errors_file",
        type=str,
        default="errors.txt",
        help="Output log file for errors (appended)",
    )
    args = parser.parse_args()

    # Extract single seed
    seed_value = args.seed

    # Check required files
    for req in [
        args.instances_file,
        "data/instance_header_info.csv",
        "data/standard.csv",
    ]:
        if not os.path.isfile(req):
            print(f"Error: required file {req} not found.", file=sys.stderr)
            sys.exit(1)

    # Load instance sizes and types
    header_df = pd.read_csv("data/instance_header_info.csv")
    header_df.set_index("fname", inplace=True)
    standard_df = pd.read_csv("data/standard.csv")
    problem_map = standard_df.set_index("graphname")["problem"].to_dict()

    # Read list of instances
    with open(args.instances_file, "r") as fp:
        instances = [line.strip() for line in fp if line.strip()]

    # Sort by number of nodes (n) and number of edges (m) ascending; unknown values go last
    def instance_key(inst: str):
        if inst in header_df.index:
            n_val = header_df.loc[inst, "n"]
            # Attempt to use number of edges if available; fall back to 0
            m_val = (
                header_df.loc[inst, "m"]
                if "m" in header_df.columns
                else header_df.loc[inst, "num_edges"]
                if "num_edges" in header_df.columns
                else 0
            )
            return (n_val, m_val)
        else:
            return (float("inf"), float("inf"))

    instances_sorted = sorted(instances, key=instance_key)

    # Prepare output writers
    results_path = Path(args.results_file)
    errors_path = Path(args.errors_file)
    results_file_exists = results_path.exists()

    # Determine which instances already have results (if skip_existing is set)
    existing_graphs: set[str] = set()
    if args.skip_existing and results_path.exists():
        try:
            with results_path.open("r", newline="") as f:
                reader = csv.reader(f)
                # Skip header
                header = next(reader, None)
                # Determine index of graphname
                graph_idx = 1
                if header and "graphname" in header:
                    graph_idx = header.index("graphname")
                for row in reader:
                    if not row:
                        continue
                    if len(row) > graph_idx:
                        existing_graphs.add(row[graph_idx])
        except Exception:
            # If reading fails, fall back to empty set
            existing_graphs = set()

    with (
        results_path.open("a", newline="") as results_fp,
        errors_path.open("a") as errors_fp,
    ):
        results_writer = csv.writer(results_fp)
        if not results_file_exists:
            results_writer.writerow(
                ["timestamp", "graphname", "heuristic", "seed", "limit", "objective"]
            )

        # Working directory for extracted graphs
        work_dir = Path("curgraph")
        encountered_error = False

        for graphname in instances_sorted:
            # Skip if results already present and skip_existing is requested
            if args.skip_existing and graphname in existing_graphs:
                print(f"Skipping {graphname} because it already has results")
                continue
            # Determine problem type; default to MAX‑CUT if unknown
            problem = problem_map.get(graphname, "MAX-CUT")

            if encountered_error:
                print(
                    f"Skipping {graphname} and subsequent instances due to previous embedding error"
                )
                break

            # Ensure the zip file exists
            zip_path = Path("data/zips/" + graphname)
            if not zip_path.is_file():
                errors_fp.write(f"Zip file {graphname} not found locally\n")
                errors_fp.flush()
                continue

            # Extract the graph into working directory
            extract_graph(zip_path, work_dir)

            # Only run once per instance since QPU does not support seeding
            # Select appropriate flag for problem type
            file_flag = "-fQ" if problem == "QUBO" else "-fM"
            cmd = [
                "./bin/MQLib",
                file_flag,
                str(work_dir),
                "-h",
                "DWAVEQPU",
                "-r",
                "1.0",  # runtime ignored by DWAVEQPU
                "-s",
                str(seed_value),
            ]
            print(f"Running DWAVEQPU on {graphname} (problem={problem}) ...")
            output = run_command(cmd)

            # Detect embedding error
            if "error" in output.lower():
                errors_fp.write(f"Error on {graphname}:\n{output}\n")
                errors_fp.flush()
                encountered_error = True
            else:
                objective = parse_objective_from_output(output)
                if objective is None:
                    errors_fp.write(f"Could not parse objective for {graphname}\n")
                    errors_fp.flush()
                else:
                    timestamp = datetime.datetime.now().isoformat()
                    # For QPU runs, the time limit is irrelevant; record 0.0 to satisfy the analysis pipeline
                    results_writer.writerow(
                        [timestamp, graphname, "DWAVEQPU", seed_value, 0.0, objective]
                    )
                    results_fp.flush()

            # Clean up extracted graph directory
            if work_dir.exists():
                for item in work_dir.iterdir():
                    if item.is_dir():
                        for sub in item.iterdir():
                            sub.unlink()
                        item.rmdir()
                    else:
                        item.unlink()
            if encountered_error:
                break

    print("DWAVEQPU benchmarking complete.")


if __name__ == "__main__":
    main()
