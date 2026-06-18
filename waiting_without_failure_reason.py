#!/usr/bin/env python3
"""List participants who reached waiting but have no failed_reason in Participant.csv.

Compares ``last_stage_reached`` from ``extract_module_times.py`` against
``failed_reason`` in Participant.csv and prints waiting-stage participants
without a non-empty failure reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from extract_module_times import (
    _unique_id_to_worker_id,
    build_participant_total_times_dataframe,
    find_participant_csv,
    get_data_root,
    iter_treatment_batches,
)
from lobby_trial_analysis import build_participant_summary, load_lobby_trials

TASK_TYPES = ("personality", "guessing")


def load_participants(data_root: Path) -> pd.DataFrame:
    batches = iter_treatment_batches(data_root)
    if not batches:
        raise FileNotFoundError(f"No batch-n directories found in {data_root}")

    frames: list[pd.DataFrame] = []
    for treatment, batch_dir in batches:
        csv_path = find_participant_csv(batch_dir)
        df = pd.read_csv(
            csv_path,
            usecols=["id", "unique_id", "failed", "failed_reason"],
        )
        df["batch_id"] = batch_dir.name
        df["treatment"] = treatment
        df["participant_id"] = df["id"]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["worker_id"] = combined["unique_id"].map(_unique_id_to_worker_id)
    combined["failed_reason"] = (
        combined["failed_reason"].fillna("").astype(str).str.strip()
    )
    return combined


def load_waiting_participants(data_root: Path) -> pd.DataFrame:
    totals = build_participant_total_times_dataframe(data_root)
    waiting = totals.loc[totals["last_stage_reached"] == "waiting"].copy()
    return waiting[
        ["treatment", "batch_id", "participant_id", "worker_id", "last_stage_reached"]
    ]


def add_task_type_answer_counts(waiting: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    lobby_trials = load_lobby_trials(data_root)
    answer_counts = build_participant_summary(lobby_trials)[
        [
            "treatment",
            "batch_id",
            "participant_id",
            "personality_answered_count",
            "guessing_completed_count",
        ]
    ].rename(
        columns={
            "personality_answered_count": "personality",
            "guessing_completed_count": "guessing",
        }
    )

    result = waiting.merge(
        answer_counts, on=["treatment", "batch_id", "participant_id"], how="left"
    )
    for task_type in TASK_TYPES:
        result[task_type] = result[task_type].fillna(0).astype(int)
    return result


def find_waiting_without_failure_reason(data_root: Path) -> pd.DataFrame:
    participants = load_participants(data_root)
    waiting = load_waiting_participants(data_root)

    merged = waiting.merge(
        participants[["treatment", "batch_id", "participant_id", "failed", "failed_reason"]],
        on=["treatment", "batch_id", "participant_id"],
        how="left",
    )
    missing = merged.loc[merged["failed_reason"] == ""].copy()
    return add_task_type_answer_counts(missing, data_root)


def print_waiting_without_failure_reason(missing: pd.DataFrame) -> None:
    if missing.empty:
        print("All waiting-stage participants have a failed_reason.")
        return

    columns = [
        "treatment",
        "participant_id",
        "worker_id",
        "failed",
        "last_stage_reached",
        *TASK_TYPES,
    ]
    for treatment, group in missing.groupby("treatment", sort=True):
        if treatment:
            print(f"Treatment: {treatment}")
            print("-" * (10 + len(treatment)))
        summary = (
            group[columns]
            .drop_duplicates()
            .sort_values(["participant_id", "worker_id"])
            .reset_index(drop=True)
        )
        if summary.empty:
            continue
        print(summary.to_string(index=False))
        n = len(summary)
        print(f"\n{n} participant{'s' if n != 1 else ''} reached waiting without a failed_reason")
        print()


def main() -> None:
    data_root = get_data_root()
    missing = find_waiting_without_failure_reason(data_root)
    print_waiting_without_failure_reason(missing)


if __name__ == "__main__":
    main()
