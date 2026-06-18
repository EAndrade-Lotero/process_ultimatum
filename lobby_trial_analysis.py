#!/usr/bin/env python3
"""Summarize LobbyTrial data per participant by task type.

Loads LobbyTrial.csv from batch directories listed in ``data_path.txt`` and
writes summary tables plus a guessing-attempts histogram under ``analysis/``.

For ``task_type == "personality"``, exports one row per answered item with the
question text, facet, and participant choice. For ``task_type == "guessing"``,
plots the distribution of attempts (``n_guesses``) required to find the target.
Participant-level means are averaged across participants in
``lobby_trial_averages.csv``. Round formation is the share of participants
with at least one lobby trial of that task type who reached the game phase
(at least one GameTrial row).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from extract_module_times import (
    add_worker_id_to_dataframe,
    find_game_trial_csv,
    find_waiting_trial_csv,
    get_data_root,
    iter_treatment_batches,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"
FIGURE_SIZE = (5.5, 4.0)
SAVE_DPI = 300
PALETTE = {
    "personality": "#1d3557",
    "guessing": "#457b9d",
    "primary": "#1d3557",
    "accent": "#e76f51",
}
TREATMENT_PALETTE = {
    "random": "#1d3557",
    "constant": "#e76f51",
}


def configure_plot_style() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.15,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.dpi": SAVE_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        },
    )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _despine(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.grid(False)


def load_lobby_trials(data_root: Path) -> pd.DataFrame:
    batches = iter_treatment_batches(data_root)
    if not batches:
        raise FileNotFoundError(f"No batch-n directories found in {data_root}")

    frames: list[pd.DataFrame] = []
    for treatment, batch_dir in batches:
        csv_path = find_waiting_trial_csv(batch_dir)
        df = pd.read_csv(csv_path)
        df["batch_id"] = batch_dir.name
        df["treatment"] = treatment
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return add_worker_id_to_dataframe(combined, data_root)


def load_game_phase_participants(data_root: Path) -> pd.DataFrame:
    """Participants with at least one GameTrial row (reached the game phase)."""
    batches = iter_treatment_batches(data_root)
    if not batches:
        raise FileNotFoundError(f"No batch-n directories found in {data_root}")

    frames: list[pd.DataFrame] = []
    for treatment, batch_dir in batches:
        csv_path = find_game_trial_csv(batch_dir)
        df = pd.read_csv(csv_path, usecols=["participant_id"])
        df = df.drop_duplicates()
        df["batch_id"] = batch_dir.name
        df["treatment"] = treatment
        df["reached_game_phase"] = True
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _round_formation_pct(
    lobby_trials: pd.DataFrame,
    game_phase_participants: pd.DataFrame,
    task_type: str,
) -> float:
    task_participants = (
        lobby_trials.loc[
            lobby_trials["task_type"] == task_type,
            ["treatment", "batch_id", "participant_id"],
        ]
        .drop_duplicates()
    )
    if task_participants.empty:
        return float("nan")

    merged = task_participants.merge(
        game_phase_participants,
        on=["treatment", "batch_id", "participant_id"],
        how="left",
    )
    merged["reached_game_phase"] = merged["reached_game_phase"].fillna(False)
    return 100 * merged["reached_game_phase"].mean()


def build_participant_summary(lobby_trials: pd.DataFrame) -> pd.DataFrame:
    task_counts = (
        lobby_trials.groupby(
            ["treatment", "batch_id", "participant_id", "task_type"], as_index=False
        )
        .size()
        .rename(columns={"size": "trial_count"})
    )
    pivot = (
        task_counts.pivot_table(
            index=["treatment", "batch_id", "participant_id"],
            columns="task_type",
            values="trial_count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    pivot.columns.name = None

    for task_type in ("personality", "guessing"):
        if task_type not in pivot.columns:
            pivot[task_type] = 0

    pivot = pivot.rename(
        columns={
            "personality": "personality_trial_count",
            "guessing": "guessing_trial_count",
        }
    )
    pivot["lobby_trial_count"] = (
        pivot["personality_trial_count"] + pivot["guessing_trial_count"]
    )

    personality_answered = (
        lobby_trials[
            (lobby_trials["task_type"] == "personality")
            & lobby_trials["choice"].notna()
        ]
        .groupby(["treatment", "batch_id", "participant_id"], as_index=False)
        .size()
        .rename(columns={"size": "personality_answered_count"})
    )
    guessing_completed = (
        lobby_trials[
            (lobby_trials["task_type"] == "guessing")
            & lobby_trials["complete"].fillna(False)
        ]
        .groupby(["treatment", "batch_id", "participant_id"], as_index=False)
        .size()
        .rename(columns={"size": "guessing_completed_count"})
    )

    summary = pivot.merge(
        personality_answered, how="left", on=["treatment", "batch_id", "participant_id"]
    )
    summary = summary.merge(
        guessing_completed, how="left", on=["treatment", "batch_id", "participant_id"]
    )
    summary[["personality_answered_count", "guessing_completed_count"]] = (
        summary[["personality_answered_count", "guessing_completed_count"]].fillna(0).astype(int)
    )

    worker_ids = (
        lobby_trials[["treatment", "batch_id", "participant_id", "worker_id"]]
        .drop_duplicates()
    )
    summary = summary.merge(
        worker_ids, on=["treatment", "batch_id", "participant_id"], how="left"
    )

    column_order = [
        "treatment",
        "batch_id",
        "participant_id",
        "worker_id",
        "lobby_trial_count",
        "personality_trial_count",
        "personality_answered_count",
        "guessing_trial_count",
        "guessing_completed_count",
    ]
    return summary[column_order].sort_values(
        ["treatment", "batch_id", "participant_id"]
    ).reset_index(drop=True)


def build_overall_averages(
    lobby_trials: pd.DataFrame,
    game_phase_participants: pd.DataFrame,
    treatment: str | None = None,
) -> pd.DataFrame:
    """Average trial metrics within each participant, then average across participants."""
    subset = lobby_trials
    game_subset = game_phase_participants
    if treatment is not None:
        subset = subset.loc[subset["treatment"] == treatment]
        game_subset = game_subset.loc[game_subset["treatment"] == treatment]

    all_participants = subset[["batch_id", "participant_id"]].drop_duplicates()

    personality = subset[
        (subset["task_type"] == "personality")
        & subset["choice"].notna()
    ]
    personality_answer_counts = (
        personality.groupby(["batch_id", "participant_id"], as_index=False)
        .size()
        .rename(columns={"size": "answer_count"})
    )
    personality_by_participant = (
        personality.groupby(["batch_id", "participant_id"], as_index=False)
        .agg(avg_time_taken=("time_taken", "mean"))
    )

    guessing = subset[
        (subset["task_type"] == "guessing")
        & subset["complete"].fillna(False)
        & subset["n_guesses"].notna()
    ]
    guessing_by_participant = (
        guessing.groupby(["batch_id", "participant_id"], as_index=False)
        .agg(
            avg_attempts=("n_guesses", "mean"),
            avg_time_taken=("time_taken", "mean"),
        )
    )

    personality_counts = all_participants.merge(
        personality_answer_counts, on=["batch_id", "participant_id"], how="left"
    )
    personality_counts["answer_count"] = personality_counts["answer_count"].fillna(0)

    averages = {
        "treatment": treatment or "",
        "n_participants": len(all_participants),
        "n_participants_with_personality_answers": int(
            (personality_counts["answer_count"] > 0).sum()
        ),
        "avg_personality_answers": personality_counts["answer_count"].mean(),
        "avg_personality_time_taken": personality_by_participant["avg_time_taken"].mean(),
        "pct_personality_round_formation": _round_formation_pct(
            subset, game_subset, "personality"
        ),
        "n_participants_with_guessing_completed": len(guessing_by_participant),
        "avg_guessing_attempts": guessing_by_participant["avg_attempts"].mean(),
        "avg_guessing_time_taken": guessing_by_participant["avg_time_taken"].mean(),
        "pct_guessing_round_formation": _round_formation_pct(
            subset, game_subset, "guessing"
        ),
    }
    return pd.DataFrame([averages])


def build_all_overall_averages(
    lobby_trials: pd.DataFrame,
    game_phase_participants: pd.DataFrame,
) -> pd.DataFrame:
    if "treatment" in lobby_trials.columns and lobby_trials["treatment"].nunique() > 1:
        frames = [
            build_overall_averages(lobby_trials, game_phase_participants, treatment)
            for treatment in sorted(lobby_trials["treatment"].unique())
        ]
        return pd.concat(frames, ignore_index=True)

    overall = build_overall_averages(lobby_trials, game_phase_participants)
    return overall.drop(columns=["treatment"])


def _format_average_value(value: float, kind: str) -> str:
    if pd.isna(value):
        return "—"
    if kind == "count":
        return f"{int(value)}"
    if kind == "seconds":
        return f"{value:.2f} s"
    if kind == "percent":
        return f"{value:.1f}%"
    return f"{value:.2f}"


def print_pretty_averages(overall_averages: pd.DataFrame) -> None:
    metric_rows = [
        ("n_participants", "Total participants", "count"),
        (
            "n_participants_with_personality_answers",
            "With personality answers",
            "count",
        ),
        (
            "n_participants_with_guessing_completed",
            "With completed guessing rounds",
            "count",
        ),
        ("avg_personality_answers", "Avg answers per participant", "float"),
        ("avg_personality_time_taken", "Avg time per item", "seconds"),
        ("pct_personality_round_formation", "Round formation", "percent"),
        ("avg_guessing_attempts", "Avg attempts to target", "float"),
        ("avg_guessing_time_taken", "Avg time per round", "seconds"),
        ("pct_guessing_round_formation", "Round formation", "percent"),
    ]
    sections = [
        ("Participants", metric_rows[:3]),
        ("Personality", metric_rows[3:6]),
        ("Guessing", metric_rows[6:]),
    ]

    label_width = max(len(label) for _, rows in sections for _, label, _ in rows)
    value_width = 10

    if "treatment" in overall_averages.columns and overall_averages["treatment"].nunique() > 1:
        for _, row in overall_averages.iterrows():
            treatment = row["treatment"] or "all"
            lines = [
                "",
                f"Averages over participants ({treatment})",
                "─" * (label_width + value_width + 4),
            ]
            for section_title, rows in sections:
                lines.append(section_title)
                for metric_key, label, kind in rows:
                    formatted = _format_average_value(row.get(metric_key), kind)
                    lines.append(f"  {label:<{label_width}}  {formatted:>{value_width}}")
                lines.append("")
            print("\n".join(lines).rstrip())
        return

    values = overall_averages.iloc[0].to_dict()
    lines = ["", "Averages over participants", "─" * (label_width + value_width + 4)]
    for section_title, rows in sections:
        lines.append(section_title)
        for metric_key, label, kind in rows:
            formatted = _format_average_value(values.get(metric_key), kind)
            lines.append(f"  {label:<{label_width}}  {formatted:>{value_width}}")
        lines.append("")
    print("\n".join(lines).rstrip())


def build_personality_answers(lobby_trials: pd.DataFrame) -> pd.DataFrame:
    personality = lobby_trials[lobby_trials["task_type"] == "personality"].copy()
    answered = personality[personality["choice"].notna()].copy()

    answers = answered.assign(
        question=answered["item"],
        answer=answered["choice"],
    )
    columns = [
        "treatment",
        "batch_id",
        "participant_id",
        "worker_id",
        "lobby_index",
        "question",
        "facet",
        "answer",
        "time_taken",
        "complete",
        "failed",
    ]
    return (
        answers[columns]
        .sort_values(["treatment", "batch_id", "participant_id", "lobby_index"])
        .reset_index(drop=True)
    )


def build_guessing_attempts(lobby_trials: pd.DataFrame) -> pd.DataFrame:
    guessing = lobby_trials[lobby_trials["task_type"] == "guessing"].copy()
    completed = guessing[guessing["complete"].fillna(False) & guessing["n_guesses"].notna()].copy()
    completed["attempts_to_target"] = completed["n_guesses"].astype(int)

    columns = [
        "treatment",
        "batch_id",
        "participant_id",
        "worker_id",
        "lobby_index",
        "target",
        "attempts_to_target",
        "time_taken",
        "failed",
    ]
    return (
        completed[columns]
        .sort_values(["treatment", "batch_id", "participant_id", "lobby_index"])
        .reset_index(drop=True)
    )


def plot_guessing_attempts_histogram(
    guessing_attempts: pd.DataFrame,
    output_path: Path,
) -> None:
    if guessing_attempts.empty:
        print("No completed guessing trials; skipping histogram.")
        return

    fig, ax = plt.subplots(
        figsize=(6.5, 4.0)
        if "treatment" in guessing_attempts.columns
        and guessing_attempts["treatment"].nunique() > 1
        else FIGURE_SIZE
    )
    if "treatment" in guessing_attempts.columns and guessing_attempts["treatment"].nunique() > 1:
        sns.histplot(
            data=guessing_attempts,
            x="attempts_to_target",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            discrete=True,
            multiple="dodge",
            shrink=0.8,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.histplot(
            data=guessing_attempts,
            x="attempts_to_target",
            color=PALETTE["guessing"],
            discrete=True,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
    ax.set_xlabel("Attempts to find target")
    ax.set_ylabel("Number of guessing rounds")
    ax.set_title("Guessing attempts prior to finding the target")
    _despine(ax)
    _save_figure(fig, output_path)


def main() -> None:
    configure_plot_style()
    data_root = get_data_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lobby_trials = load_lobby_trials(data_root)
    game_phase_participants = load_game_phase_participants(data_root)
    summary = build_participant_summary(lobby_trials)
    overall_averages = build_all_overall_averages(lobby_trials, game_phase_participants)
    personality_answers = build_personality_answers(lobby_trials)
    guessing_attempts = build_guessing_attempts(lobby_trials)

    summary.to_csv(OUTPUT_DIR / "lobby_trial_summary.csv", index=False)
    overall_averages.to_csv(OUTPUT_DIR / "lobby_trial_averages.csv", index=False)
    personality_answers.to_csv(OUTPUT_DIR / "lobby_personality_answers.csv", index=False)
    guessing_attempts.to_csv(OUTPUT_DIR / "lobby_guessing_attempts.csv", index=False)
    plot_guessing_attempts_histogram(
        guessing_attempts,
        OUTPUT_DIR / "guessing_attempts_histogram.png",
    )

    print(f"Loaded {len(lobby_trials)} LobbyTrial rows from {data_root}")
    print(f"Participants in summary: {len(summary)}")
    print(f"Personality answers exported: {len(personality_answers)}")
    print(f"Completed guessing rounds: {len(guessing_attempts)}")
    print(f"Wrote outputs to {OUTPUT_DIR}")
    print_pretty_averages(overall_averages)


if __name__ == "__main__":
    main()
