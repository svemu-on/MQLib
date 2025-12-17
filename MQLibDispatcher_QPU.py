#!/usr/bin/env python3

"""
MQLibDispatcher_QPU.py
======================

This script benchmarks the DWAVEQPU heuristic from MQLib on a collection of
problem instances (QUBO or MAX-CUT).  Unlike the general dispatcher that
computes a per-instance runtime based on a baseline heuristic, this version
simply invokes the QPU on each instance once with fixed annealing parameters
(the QPU backend ignores the MQLib runtime parameter).

Features:
- **Sorts instances by size.**
  Uses `data/instance_header_info.csv` to determine the number of variables
  (`n`) and number of edges (`m`) for each instance. Instances are sorted
  first by `n` ascending, then by `m` ascending.  If values are missing, they
  are placed later.

- **Supports QUBO and MAX-CUT.**
  Uses `data/standard.csv` to detect problem type and passes the appropriate
  `-fQ` or `-fM` flag to MQLib. Both types work with DWAVEQPU because MQLib
  converts between them automatically.

- **Runs each instance once.**
  The QPU does not support random seeding, so each instance is run exactly
  once. A single seed value (default 0) is recorded for compatibility with
  the analysis pipeline; it has no effect on the QPU’s behavior.

- **Results and errors.**
  Results are appended to a CSV file: each row has
  `timestamp,graphname,heuristic,seed,limit,objective`.
  Errors are appended to a separate file.

- **Resumable.**
  With `--skip_existing`, instances already present in the results file are
  skipped. This allows easy restarting of partially completed runs.

- **Stops on embedding error.**
  If the QPU cannot embed an instance, the script logs the error and stops,
  since larger instances will likely also fail.

Usage example:
    python3 MQLibDispatcher_QPU.py \
        --seed 0 \
        --instances_file data/instances.txt \
        --results_file data/dwaveqpu_results.csv \
        --errors_file data/dwaveqpu_errors.txt \
        --skip_existing

By default the script expects the instance archives (`<graphname>.zip`) to live
under `data/zips/`.  If your `.zip` files are elsewhere, pass `--zip_dir`.
"""

import argparse
import csv
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------


def extract_graph(zip_path: Path, work_dir: Path) -> None:
    """Extract a graph archive into work_dir."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)


def find_single_txt(work_dir: Path) -> Path:
    """Return the single .txt file inside work_dir, or None if not exactly one."""
    txts = list(work_dir.glob("*.txt"))
    return txts[0] if len(txts) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local dispatcher for running DWAVEQPU on MQLib benchmarks."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=("Seed value to record in results (DWAVEQPU ignores seeds). Default: 0."),
    )

    parser.add_argument(
        "--instances_file",
        type=str,
        default="data/instances.txt",
        help="Text file with one <graphname>.zip per line.",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="results.csv",
        help="Output CSV file (appended).",
    )

    parser.add_argument(
        "--errors_file",
        type=str,
        default="errors.txt",
        help="Output log file for errors (appended).",
    )

    parser.add_argument(
        "--zip_dir",
        type=str,
        default="data/zips",
        help=(
            "Directory containing the instance .zip files.  Entries in "
            "instances_file are joined with this directory.  Defaults to "
            "data/zips."
        ),
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "If set, graphs already present in results_file are skipped. "
            "Useful for resuming crashed runs."
        ),
    )

    args = parser.parse_args()

    seed_value = args.seed
    zip_dir = Path(args.zip_dir)

    # Load list of instances (.zip filenames)
    with open(args.instances_file, "r") as f:
        instances = [line.strip() for line in f if line.strip()]

    # Load problem type
    std_path = Path("data/standard.csv")
    if not std_path.is_file():
        print("ERROR: data/standard.csv not found.")
        sys.exit(1)

    standard_df = pd.read_csv(std_path)
    problem_map = dict(zip(standard_df["graphname"], standard_df["problem"]))

    # Load header info (sizes)
    header_path = Path("data/instance_header_info.csv")
    if not header_path.is_file():
        print("ERROR: data/instance_header_info.csv not found.")
        sys.exit(1)

    header_df = pd.read_csv(header_path)
    header_df.set_index("fname", inplace=True)

    # Sorting key: (n, m)
    def instance_key(inst):
        if inst in header_df.index:
            n = header_df.loc[inst, "n"]
            m = header_df.loc[inst, "m"]
            return (n, m)
        return (float("inf"), float("inf"))

    instances_sorted = sorted(instances, key=instance_key)

    # Prepare results file
    results_path = Path(args.results_file)
    errors_path = Path(args.errors_file)
    results_file_exists = results_path.is_file()

    # Load existing graph names if skipping
    existing_graphs = set()
    if results_file_exists and args.skip_existing:
        with open(results_path, "r") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if row and row[0] != "timestamp":
                    existing_graphs.add(row[1])

    # Begin writing output
    with (
        results_path.open("a", newline="") as results_fp,
        errors_path.open("a") as errors_fp,
    ):
        writer = csv.writer(results_fp)
        if not results_file_exists:
            writer.writerow(
                ["timestamp", "graphname", "heuristic", "seed", "limit", "objective"]
            )

        # Working directory for extracting graphs
        work_dir = Path("data/curgraph")
        work_dir.mkdir(exist_ok=True)

        encountered_error = False

        for graphname in instances_sorted:
            if args.skip_existing and graphname in existing_graphs:
                print(f"Skipping {graphname} (already done).")
                continue

            if encountered_error:
                print(f"Skipping {graphname} due to previous embedding failure.")
                continue

            problem = problem_map.get(graphname)
            if problem not in ("QUBO", "MAXCUT"):
                msg = f"Skipping {graphname}: unknown problem type {problem}"
                print(msg)
                errors_fp.write(msg + "\n")
                errors_fp.flush()
                continue

            # ZIP file location
            zip_path = zip_dir / graphname
            if not zip_path.is_file():
                msg = f"Zip file not found: {zip_path}"
                print(msg)
                errors_fp.write(msg + "\n")
                errors_fp.flush()
                continue

            # Extract
            for p in work_dir.iterdir():
                if p.is_file():
                    p.unlink()
            extract_graph(zip_path, work_dir)

            # Find extracted .txt file
            input_path = find_single_txt(work_dir)
            if input_path is None:
                msg = f"Error: {graphname} did not extract to a single .txt file"
                print(msg)
                errors_fp.write(msg + "\n")
                errors_fp.flush()
                continue

            # Problem flag
            file_flag = "-fQ" if problem == "QUBO" else "-fM"

            cmd = [
                "./bin/MQLib",
                file_flag,
                str(input_path),
                "-h",
                "DWAVEQPU",
                "-r",
                "1.0",
                "-s",
                str(seed_value),
            ]

            print(f"Running {graphname}...")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except Exception as e:
                msg = f"Error running {graphname}: {e}"
                print(msg)
                errors_fp.write(msg + "\n")
                errors_fp.flush()
                encountered_error = True
                continue

            # Process output
            outlines = (proc.stdout or "").splitlines()
            best_obj = None

            for line in outlines:
                if line.startswith("Error:"):
                    errors_fp.write(f"{graphname} :: {line}\n")
                    errors_fp.flush()
                    encountered_error = True
                    break

                if line.startswith("2,"):
                    parts = line.split(",")
                    if len(parts) >= 5:
                        try:
                            best_obj = float(parts[4])
                        except ValueError:
                            pass

            if encountered_error:
                continue

            if best_obj is None:
                msg = f"Warning: {graphname} produced no objective line."
                print(msg)
                errors_fp.write(msg + "\n")
                errors_fp.flush()
                continue

            timestamp = time.time()
            writer.writerow(
                [
                    timestamp,
                    graphname,
                    "DWAVEQPU",
                    seed_value,
                    0.0,  # QPU does not use MQLib runtime limit
                    best_obj,
                ]
            )
            results_fp.flush()

    print("All done.")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
