#!/usr/bin/env python3
"""Print dyad and round counts by treatment for completed Ultimatum GameTrials.

Uses the same inclusion rules as ``game_trial_analysis.py``:
completed, non-failed dyads from GameTrial.csv, with round counts from
UltimatumSession.csv for matching group_ids.

Reads batch exports from the path in ``data_with_treatment_path.txt`` or
``data_path.txt``.
"""

from __future__ import annotations

from extract_module_times import get_data_root, iter_treatment_batches
from game_trial_analysis import (
    _completed_dyads,
    load_game_trials,
    load_round_dataframe,
)


def print_dyad_counts(data_root) -> None:
    batches = [batch_dir.name for _, batch_dir in iter_treatment_batches(data_root)]
    game_trials = load_game_trials(data_root)
    completed_dyads = _completed_dyads(game_trials)
    rounds = load_round_dataframe(data_root, game_trials)
    all_dyads = game_trials.drop_duplicates(subset=["treatment", "batch_id", "group_id"])

    print(f"Data root: {data_root}")
    print(f"Batches: {batches}\n")

    print("=== Completed, non-failed dyads (analysis sample) ===")
    for treatment in sorted(completed_dyads["treatment"].unique(), key=str):
        count = int((completed_dyads["treatment"] == treatment).sum())
        label = treatment or "(no treatment label)"
        print(f"  {label}: {count}")
    print(f"  Total: {len(completed_dyads)}")

    print("\n=== All GameTrial dyads (any status) ===")
    for treatment in sorted(all_dyads["treatment"].unique(), key=str):
        count = int((all_dyads["treatment"] == treatment).sum())
        label = treatment or "(no treatment label)"
        print(f"  {label}: {count}")
    print(f"  Total: {len(all_dyads)}")

    print("\n=== Round-level rows (completed dyads only) ===")
    for treatment in sorted(rounds["treatment"].unique(), key=str):
        count = int((rounds["treatment"] == treatment).sum())
        label = treatment or "(no treatment label)"
        print(f"  {label}: {count}")
    print(f"  Total: {len(rounds)}")

    excluded = len(all_dyads) - len(completed_dyads)
    print(f"\n=== Excluded dyads (incomplete or failed) ===")
    print(f"  Total: {excluded}")


def main() -> None:
    print_dyad_counts(get_data_root())


if __name__ == "__main__":
    main()
