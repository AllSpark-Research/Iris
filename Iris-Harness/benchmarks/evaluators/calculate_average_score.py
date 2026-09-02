#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

import glob
import os
import re
import statistics
import sys


def detect_pass_at_k(results_dir: str) -> tuple:
    """Detect the pass_at_k value used in the results directory"""

    # Find all possible pass_at_k files
    pattern = os.path.join(
        results_dir, "run_*", "benchmark_results_pass_at_*_accuracy.txt"
    )
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"No accuracy files found in {results_dir}")
        print(f"Expected pattern: {pattern}")
        return None, []

    # Extract pass_at_k value from the first file
    filename = os.path.basename(all_files[0])
    match = re.search(r"pass_at_(\d+)_accuracy\.txt", filename)

    if not match:
        print(f"Cannot extract pass_at_k from filename: {filename}")
        return None, []

    k = int(match.group(1))

    # Get all files with this k value
    accuracy_files = glob.glob(
        os.path.join(
            results_dir, "run_*", f"benchmark_results_pass_at_{k}_accuracy.txt"
        )
    )

    return k, accuracy_files


def calculate_average_scores(results_dir: str) -> dict:
    """Calculate average scores from multiple runs - automatically detect pass_at_k value"""

    # Detect pass_at_k value and corresponding files
    pass_at_k, accuracy_files = detect_pass_at_k(results_dir)

    if pass_at_k is None:
        return None

    print(f"Detected pass_at_{pass_at_k} files")
    print(f"Found {len(accuracy_files)} accuracy files")

    scores = []

    # Read each accuracy file. An accuracy file starts with the score line
    # ("pass@1: 40.00%") and is followed by tool-call and termination blocks,
    # so pull the pass@k line out rather than parsing the whole file.
    score_re = re.compile(rf"^pass@{pass_at_k}:\s*([\d.]+)\s*%", re.MULTILINE)
    for i, file_path in enumerate(sorted(accuracy_files), 1):
        try:
            with open(file_path, "r") as f:
                content = f.read()
        except OSError as e:
            print(f"Error reading {file_path}: {e}")
            continue

        match = score_re.search(content) or re.fullmatch(
            r"\s*([\d.]+)\s*%?\s*", content
        )
        if match is None:
            print(f"Error reading {file_path}: no 'pass@{pass_at_k}' score found")
            continue

        score = float(match.group(1))
        scores.append(score)
        print(f"Run {i}: {score:.2f}%")

    if not scores:
        print("No valid scores found")
        return None

    # Calculate statistics
    stats = {
        "pass_at_k": pass_at_k,
        "num_runs": len(scores),
        "individual_scores": scores,
        "average_score": statistics.mean(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "min_score": min(scores),
        "max_score": max(scores),
    }

    return stats


def print_results(stats: dict):
    """Print results"""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"Pass@{stats['pass_at_k']} Results:")
    print(f"Number of runs: {stats['num_runs']}")
    print(f"Individual scores: {[f'{s:.2f}%' for s in stats['individual_scores']]}")
    print()
    print(f"Standard deviation: {stats['std_dev']:.2f}%")
    print(f"Min score: {stats['min_score']:.2f}%")
    print(f"Max score: {stats['max_score']:.2f}%")
    print(f"Average score: {stats['average_score']:.2f}%")
    print("=" * 50)


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_average_score.py <results_directory> [expected_runs]")
        print("Example: python calculate_average_score.py logs/<model-tag>/browsecomp/ave3 3")
        sys.exit(1)

    results_dir = sys.argv[1]
    expected_runs = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    print(f"Analyzing results from: {results_dir}")

    stats = calculate_average_scores(results_dir)

    if stats:
        # Averaging 2 of 3 runs and labelling it "ave3" would misreport the
        # result, so say so loudly and exit non-zero.
        if expected_runs is not None and stats["num_runs"] != expected_runs:
            print(
                f"\nERROR: found {stats['num_runs']} run(s) but expected "
                f"{expected_runs}. Some runs produced no accuracy file; the average "
                "below is over the surviving runs only and must not be reported "
                f"as ave@{expected_runs}."
            )
            print_results(stats)
            sys.exit(1)

        print_results(stats)
        # Save simple statistics results
        output_file = os.path.join(
            results_dir, f"average_scores_pass_at_{stats['pass_at_k']}.txt"
        )
        with open(output_file, "w") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Pass@{stats['pass_at_k']} Results:\n")
            f.write(f"Number of runs: {stats['num_runs']}\n")
            f.write(
                f"Individual scores: {[f'{s:.2f}%' for s in stats['individual_scores']]}\n"
            )
            f.write(f"Standard deviation: {stats['std_dev']:.2f}%\n")
            f.write(f"Min score: {stats['min_score']:.2f}%\n")
            f.write(f"Max score: {stats['max_score']:.2f}%\n")
            f.write(f"Average score: {stats['average_score']:.2f}%\n")
            f.write("=" * 50 + "\n")

        print(f"\nResults saved to: {output_file}")
    else:
        print("Failed to calculate statistics")
        sys.exit(1)


if __name__ == "__main__":
    main()
