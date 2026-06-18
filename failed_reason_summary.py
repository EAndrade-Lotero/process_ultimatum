#!/usr/bin/env python3
"""Summarize failed_reason values in Participant.csv across batch exports.

Reads batch directories from ``data_path.txt`` and prints each failed_reason
with the worker_id (Prolific ID prefix from ``unique_id``), ``last_stage_reached``,
per-task answer counts for participants who stopped in the waiting stage, grouped
by both ``failed_reason`` and any ``failed_reason_*`` key found in ``vars``. Also
includes a special category for participants whose ``elt_id`` reached
``successful_end`` while ``vars`` contains a ``failed_reason_*`` key.
"""

from __future__ import annotations

import json
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
VARS_FAILED_REASON_PREFIX = "failed_reason_"
MISSING_VARS_FAILED_REASON = "(none)"
SUCCESSFUL_END_VARS_FAILED_REASON = "successfulEnd with vars failed_reason"
SUCCESSFUL_END_LABELS = {"successful_end", "successfulend"}


def parse_elt_id(elt_id: str | float | None) -> list[object] | str | None:
    if elt_id is None or (isinstance(elt_id, float) and pd.isna(elt_id)):
        return None

    if isinstance(elt_id, str):
        try:
            parsed = json.loads(elt_id)
        except json.JSONDecodeError:
            return elt_id
        return parsed
    return elt_id


def reached_successful_end(elt_id: str | float | None) -> bool:
    parsed = parse_elt_id(elt_id)
    if isinstance(parsed, list) and parsed:
        label = str(parsed[0]).lower().replace("-", "_")
        return label in SUCCESSFUL_END_LABELS
    label = str(parsed or "").lower().replace("-", "_")
    return label in SUCCESSFUL_END_LABELS


def extract_vars_failed_reason(vars_json: str | float | None) -> str:
    if vars_json is None or (isinstance(vars_json, float) and pd.isna(vars_json)):
        return ""

    data = json.loads(vars_json)
    matches = sorted(key for key in data if key.startswith(VARS_FAILED_REASON_PREFIX))
    if not matches:
        return ""

    key = matches[0]
    value = data[key]
    if value is not None and str(value).strip():
        return str(value).strip()
    return key.removeprefix(VARS_FAILED_REASON_PREFIX)


def load_participants(data_root: Path) -> pd.DataFrame:
    batches = iter_treatment_batches(data_root)
    if not batches:
        raise FileNotFoundError(f"No batch-n directories found in {data_root}")

    frames: list[pd.DataFrame] = []
    for treatment, batch_dir in batches:
        csv_path = find_participant_csv(batch_dir)
        df = pd.read_csv(
            csv_path,
            usecols=["id", "unique_id", "failed", "failed_reason", "vars", "elt_id"],
        )
        df["batch_id"] = batch_dir.name
        df["treatment"] = treatment
        df["participant_id"] = df["id"]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["worker_id"] = combined["unique_id"].map(_unique_id_to_worker_id)
    combined["failed_reason"] = combined["failed_reason"].fillna("").astype(str).str.strip()
    combined["vars_failed_reason"] = combined["vars"].map(extract_vars_failed_reason)
    return combined


def load_failed_participants(data_root: Path) -> pd.DataFrame:
    combined = load_participants(data_root)
    failed = combined.loc[
        combined["failed"].astype(bool) & (combined["failed_reason"] != "")
    ].copy()
    return failed.drop(columns=["vars", "elt_id"])


def load_successful_end_with_vars_failed_reason(data_root: Path) -> pd.DataFrame:
    combined = load_participants(data_root)
    has_failed_reason = combined["failed"].astype(bool) & (combined["failed_reason"] != "")
    special = combined.loc[
        combined["elt_id"].map(reached_successful_end)
        & (combined["vars_failed_reason"] != "")
        & ~has_failed_reason
    ].copy()
    special["failed_reason"] = SUCCESSFUL_END_VARS_FAILED_REASON
    return special.drop(columns=["vars", "elt_id"])


def load_all_summarized_participants(data_root: Path) -> pd.DataFrame:
    failed = load_failed_participants(data_root)
    special = load_successful_end_with_vars_failed_reason(data_root)
    return pd.concat([failed, special], ignore_index=True)


def add_last_stage_reached(failed: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    stages = build_participant_total_times_dataframe(data_root)[
        ["treatment", "batch_id", "participant_id", "last_stage_reached"]
    ]
    return failed.merge(stages, on=["treatment", "batch_id", "participant_id"], how="left")


def add_task_type_answer_counts(failed: pd.DataFrame, data_root: Path) -> pd.DataFrame:
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

    result = failed.merge(
        answer_counts, on=["treatment", "batch_id", "participant_id"], how="left"
    )
    waiting = result["last_stage_reached"] == "waiting"
    for task_type in TASK_TYPES:
        counts = pd.Series(pd.NA, index=result.index, dtype="Int64")
        counts.loc[waiting] = result.loc[waiting, task_type].fillna(0).astype(int)
        result[task_type] = counts
    return result


def print_failed_reason_summary(failed: pd.DataFrame) -> None:
    if failed.empty:
        print("No failed participants with a failed_reason found.")
        return

    columns = ["treatment", "participant_id", "worker_id", "last_stage_reached", *TASK_TYPES]
    reason_order = [
        *[
            reason
            for reason in failed["failed_reason"].unique()
            if reason != SUCCESSFUL_END_VARS_FAILED_REASON
        ],
        SUCCESSFUL_END_VARS_FAILED_REASON,
    ]
    for treatment, treatment_group in failed.groupby("treatment", sort=True):
        if treatment:
            print(f"Treatment: {treatment}")
            print("-" * (10 + len(treatment)))
        for reason in reason_order:
            reason_group = treatment_group.loc[treatment_group["failed_reason"] == reason]
            if reason_group.empty:
                continue
            print(reason)
            for vars_reason, group in reason_group.groupby("vars_failed_reason", sort=False):
                label = vars_reason or MISSING_VARS_FAILED_REASON
                print(f"  {label}")
                summary = (
                    group[columns]
                    .drop_duplicates()
                    .sort_values(["participant_id", "worker_id"])
                    .reset_index(drop=True)
                )
                indented = summary.to_string(index=False).replace("\n", "\n  ")
                print(f"  {indented}")
                n = len(summary)
                print(f"  {n} participant{'s' if n != 1 else ''}")
                print()
            print()

    total = failed.drop_duplicates(subset=["treatment", "batch_id", "participant_id"])
    n_total = len(total)
    print(f"Total: {n_total} participant{'s' if n_total != 1 else ''} with failures")


def main() -> None:
    data_root = get_data_root()
    failed = load_all_summarized_participants(data_root)
    failed = add_last_stage_reached(failed, data_root)
    failed = add_task_type_answer_counts(failed, data_root)
    print_failed_reason_summary(failed)


if __name__ == "__main__":
    main()
