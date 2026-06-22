#!/usr/bin/env python3
"""Build summary tables for Ultimatum GameTrial outcomes.

Computes each participant's accumulated score relative to the maximum
achievable score (10 coins per round) and the overall Gini index across
completed participants.

Reads batch exports from the path in ``data_path.txt`` and writes tables to
``analysis/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extract_module_times import find_csv_in_batch, get_data_root, iter_treatment_batches

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"
COINS_PER_ROUND = 10


def load_game_trials(data_root: Path) -> pd.DataFrame:
    batches = iter_treatment_batches(data_root)
    if not batches:
        raise FileNotFoundError(f"No batch-n directories found in {data_root}")

    frames: list[pd.DataFrame] = []
    for treatment, batch_dir in batches:
        csv_path = find_csv_in_batch(batch_dir, "GameTrial.csv")
        df = pd.read_csv(csv_path)
        df["batch_id"] = batch_dir.name
        df["treatment"] = treatment
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def completed_participants(game_trials: pd.DataFrame) -> pd.DataFrame:
    return game_trials.loc[
        game_trials["complete"].astype(bool) & ~game_trials["failed"].astype(bool)
    ].copy()


def rounds_played(row: pd.Series) -> int:
    for column in ("counted_rounds", "round_index", "rounds_required"):
        value = row.get(column)
        if pd.notna(value):
            return int(value)
    raise ValueError(f"Could not determine rounds played for participant {row['participant_id']}")


def gini_coefficient(values: list[float] | np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0:
        return float("nan")
    total = arr.sum()
    if total == 0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2 * np.sum(indices * arr) - (n + 1) * total) / (n * total))


def build_participant_table(game_trials: pd.DataFrame) -> pd.DataFrame:
    participants = completed_participants(game_trials)
    participants["rounds_played"] = participants.apply(rounds_played, axis=1)
    participants["total_possible_score"] = (
        participants["rounds_played"] * COINS_PER_ROUND
    )
    participants["accumulated_score"] = participants["total_score"].astype(float)
    participants["score_ratio"] = (
        participants["accumulated_score"] / participants["total_possible_score"]
    )

    return participants[
        [
            "treatment",
            "batch_id",
            "participant_id",
            "group_id",
            "rounds_played",
            "accumulated_score",
            "total_possible_score",
            "score_ratio",
        ]
    ].sort_values(["treatment", "batch_id", "participant_id"])


def build_summary_table(participant_table: pd.DataFrame) -> pd.DataFrame:
    if "treatment" in participant_table.columns and participant_table["treatment"].nunique() > 1:
        frames = []
        for treatment, group in participant_table.groupby("treatment", sort=True):
            frames.append(
                pd.DataFrame(
                    {
                        "treatment": treatment,
                        "metric": [
                            "Participant count",
                            "Accumulated score / total possible score",
                            "Overall Gini index",
                        ],
                        "value": [
                            len(group),
                            group["score_ratio"].mean(),
                            gini_coefficient(group["accumulated_score"].tolist()),
                        ],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(
        {
            "metric": [
                "Participant count",
                "Accumulated score / total possible score",
                "Overall Gini index",
            ],
            "value": [
                len(participant_table),
                participant_table["score_ratio"].mean(),
                gini_coefficient(participant_table["accumulated_score"].tolist()),
            ],
        }
    )


def _format_summary_value(row: pd.Series) -> str:
    if row["metric"] == "Participant count":
        return str(int(row["value"]))
    return f"{row['value']:.4f}"


def format_summary_table(summary: pd.DataFrame) -> str:
    if "treatment" in summary.columns and summary["treatment"].nunique() > 1:
        lines: list[str] = []
        for treatment, group in summary.groupby("treatment", sort=True):
            lines.append(f"\n{treatment}")
            lines.append(f"{'Metric':<45} {'Value':>12}")
            lines.append("-" * 58)
            for _, row in group.iterrows():
                lines.append(f"{row['metric']:<45} {_format_summary_value(row):>12}")
        return "\n".join(lines).lstrip()

    lines = [
        f"{'Metric':<45} {'Value':>12}",
        "-" * 58,
    ]
    for _, row in summary.iterrows():
        lines.append(f"{row['metric']:<45} {_format_summary_value(row):>12}")
    return "\n".join(lines)


def main() -> None:
    data_root = get_data_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_trials = load_game_trials(data_root)
    participant_table = build_participant_table(game_trials)
    summary_table = build_summary_table(participant_table)

    participant_path = OUTPUT_DIR / "participant_score_summary.csv"
    summary_path = OUTPUT_DIR / "overall_score_summary.csv"

    participant_table.to_csv(participant_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    print(format_summary_table(summary_table))
    print()
    print(f"Participant table: {participant_path}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()
