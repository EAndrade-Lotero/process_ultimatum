#!/usr/bin/env python3
"""Analyze Ultimatum GameTrial outcomes and produce summary plots.

Loads GameTrial.csv from batch directories listed in ``data_path.txt`` and
writes figures under ``analysis/``.

Round-level offer, decision, and payoff data are taken from UltimatumSession.csv
(one row per dyad), because GameTrial rows only store each participant's final
trial state. Sessions are filtered to dyads that appear in completed GameTrials.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from extract_module_times import find_csv_in_batch, get_data_root, iter_treatment_batches

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"
TOTAL_COINS = 10
FIGURE_SIZE = (5.5, 4.0)
SAVE_DPI = 300
PALETTE = {
    "accept": "#2d6a4f",
    "reject": "#9d0208",
    "primary": "#1d3557",
    "secondary": "#457b9d",
    "accent": "#e76f51",
    "neutral": "#6c757d",
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


def _completed_dyads(game_trials: pd.DataFrame) -> pd.DataFrame:
    completed = game_trials.loc[
        game_trials["complete"].astype(bool) & ~game_trials["failed"].astype(bool)
    ]
    return completed[["treatment", "batch_id", "group_id"]].drop_duplicates()


def _round_metrics(payoffs: dict[str, int | float]) -> tuple[float, float]:
    scores = [float(value) for value in payoffs.values()]
    if not scores:
        return float("nan"), float("nan")

    min_score = min(scores)
    max_score = max(scores)
    efficiency = sum(scores)
    if max_score == 0:
        fairness = float("nan")
    else:
        fairness = min_score / max_score
    return fairness, efficiency


def load_round_dataframe(
    data_root: Path, game_trials: pd.DataFrame
) -> pd.DataFrame:
    dyads = _completed_dyads(game_trials)
    if dyads.empty:
        raise ValueError("No completed GameTrial dyads found.")

    rows: list[dict[str, object]] = []
    for treatment, batch_dir in iter_treatment_batches(data_root):
        session_path = batch_dir / "UltimatumSession.csv"
        if not session_path.is_file():
            continue

        batch_dyads = dyads.loc[
            (dyads["treatment"] == treatment) & (dyads["batch_id"] == batch_dir.name),
            "group_id",
        ]
        if batch_dyads.empty:
            continue

        sessions = pd.read_csv(session_path)
        for _, session_row in sessions.iterrows():
            group_id = session_row["group_id"]
            if group_id not in set(batch_dyads):
                continue

            state = json.loads(session_row["state_json"])
            for round_row in state.get("history", []):
                if round_row.get("skipped"):
                    continue

                payoffs = {
                    str(participant_id): value
                    for participant_id, value in round_row.get("payoffs", {}).items()
                }
                fairness, efficiency = _round_metrics(payoffs)
                decision = str(round_row.get("decision", "")).lower()
                rows.append(
                    {
                        "treatment": treatment,
                        "batch_id": batch_dir.name,
                        "group_id": group_id,
                        "round_index": int(round_row["round_index"]),
                        "offer": int(round_row["offer"]),
                        "decision": decision,
                        "rejected": decision == "reject",
                        "fairness": fairness,
                        "efficiency": efficiency,
                    }
                )

    if not rows:
        raise ValueError(
            "No round history found. Expected UltimatumSession.csv alongside GameTrial.csv."
        )

    return pd.DataFrame(rows)


def plot_acceptance_rates(rounds: pd.DataFrame, output_path: Path) -> None:
    labels = ["accept", "reject"]
    group_cols = (
        ["treatment"]
        if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1
        else []
    )
    if group_cols:
        plot_data = (
            rounds.groupby(group_cols + ["decision"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        totals = (
            rounds.groupby(group_cols, as_index=False)
            .size()
            .rename(columns={"size": "total"})
        )
        plot_data = plot_data.merge(totals, on=group_cols)
        plot_data["Percentage"] = 100 * plot_data["count"] / plot_data["total"]
        plot_data["Decision"] = plot_data["decision"]
        plot_data = plot_data[plot_data["Decision"].isin(labels)]

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        sns.barplot(
            data=plot_data,
            x="Decision",
            y="Percentage",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            width=0.7,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        decision_counts = rounds["decision"].value_counts()
        plot_data = pd.DataFrame(
            {
                "Decision": labels,
                "Percentage": [
                    100 * decision_counts.get(label, 0) / len(rounds) for label in labels
                ],
                "Count": [decision_counts.get(label, 0) for label in labels],
            }
        )

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        sns.barplot(
            data=plot_data,
            x="Decision",
            y="Percentage",
            hue="Decision",
            palette=[PALETTE["accept"], PALETTE["reject"]],
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
            legend=False,
        )
        for patch, (_, row) in zip(ax.patches, plot_data.iterrows()):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                row["Percentage"] + 2,
                f"{row['Percentage']:.1f}%\n(n = {int(row['Count'])})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Percentage of offers (%)")
    ax.set_xlabel("")
    ax.set_title("Offer decisions")
    ax.set_ylim(0, 100)
    _despine(ax)
    _save_figure(fig, output_path)


def plot_offer_histogram(rounds: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0) if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1 else FIGURE_SIZE)
    if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1:
        sns.histplot(
            data=rounds,
            x="offer",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            bins=range(0, TOTAL_COINS + 2),
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
            data=rounds,
            x="offer",
            bins=range(0, TOTAL_COINS + 2),
            discrete=True,
            color=PALETTE["primary"],
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
    ax.set_xlabel("Coins offered to responder")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of offers")
    ax.set_xticks(range(0, TOTAL_COINS + 1))
    _despine(ax)
    _save_figure(fig, output_path)


def plot_accumulated_score_histogram(
    game_trials: pd.DataFrame, output_path: Path
) -> None:
    completed = game_trials.loc[
        game_trials["complete"].astype(bool) & ~game_trials["failed"].astype(bool)
    ]
    scores = completed[["total_score", "treatment"]].dropna(subset=["total_score"])

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if "treatment" in scores.columns and scores["treatment"].nunique() > 1 else FIGURE_SIZE)
    if "treatment" in scores.columns and scores["treatment"].nunique() > 1:
        sns.histplot(
            data=scores,
            x="total_score",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            bins="auto",
            multiple="dodge",
            shrink=0.8,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.histplot(
            data=scores,
            x="total_score",
            bins="auto",
            color=PALETTE["secondary"],
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
    ax.set_xlabel("Accumulated score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of accumulated scores")
    _despine(ax)
    _save_figure(fig, output_path)


def plot_fairness_by_round(rounds: pd.DataFrame, output_path: Path) -> None:
    group_cols = ["treatment", "round_index"] if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1 else ["round_index"]
    summary = (
        rounds.groupby(group_cols, as_index=False)
        .agg(mean=("fairness", "mean"), std=("fairness", "std"))
        .sort_values(group_cols)
    )
    summary["std"] = summary["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if "treatment" in summary.columns else FIGURE_SIZE)
    if "treatment" in summary.columns:
        sns.lineplot(
            data=summary,
            x="round_index",
            y="mean",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            marker="o",
            markersize=5,
            linewidth=1.5,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.lineplot(
            data=summary,
            x="round_index",
            y="mean",
            color=PALETTE["primary"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            ax=ax,
        )
        ax.errorbar(
            summary["round_index"],
            summary["mean"],
            yerr=summary["std"],
            fmt="none",
            ecolor=PALETTE["primary"],
            elinewidth=1,
            capsize=3,
            capthick=1,
            alpha=0.8,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Fairness (min payoff / max payoff)")
    ax.set_title("Fairness by round")
    ax.set_ylim(0, 1.05)
    _despine(ax)
    _save_figure(fig, output_path)


def plot_efficiency_by_round(rounds: pd.DataFrame, output_path: Path) -> None:
    group_cols = ["treatment", "round_index"] if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1 else ["round_index"]
    summary = (
        rounds.groupby(group_cols, as_index=False)
        .agg(mean=("efficiency", "mean"), std=("efficiency", "std"))
        .sort_values(group_cols)
    )
    summary["std"] = summary["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if "treatment" in summary.columns else FIGURE_SIZE)
    if "treatment" in summary.columns:
        sns.lineplot(
            data=summary,
            x="round_index",
            y="mean",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            marker="o",
            markersize=5,
            linewidth=1.5,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.lineplot(
            data=summary,
            x="round_index",
            y="mean",
            color=PALETTE["accent"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            ax=ax,
        )
        ax.errorbar(
            summary["round_index"],
            summary["mean"],
            yerr=summary["std"],
            fmt="none",
            ecolor=PALETTE["accent"],
            elinewidth=1,
            capsize=3,
            capthick=1,
            alpha=0.8,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Efficiency (sum of dyad payoffs)")
    ax.set_title("Efficiency by round")
    ax.set_ylim(-0.05, TOTAL_COINS + 0.05)
    _despine(ax)
    _save_figure(fig, output_path)


def plot_rejection_by_offer_and_round(rounds: pd.DataFrame, output_path: Path) -> None:
    offer_values = list(range(0, TOTAL_COINS + 1))
    group_cols = ["treatment", "offer"] if "treatment" in rounds.columns and rounds["treatment"].nunique() > 1 else ["offer"]

    total_counts = (
        rounds.groupby(group_cols)
        .size()
        .reset_index(name="n_offers")
    )
    reject_counts = (
        rounds.loc[rounds["rejected"]]
        .groupby(group_cols)
        .size()
        .reset_index(name="n_rejected")
    )
    plot_data = total_counts.merge(reject_counts, on=group_cols, how="left")
    plot_data["n_rejected"] = plot_data["n_rejected"].fillna(0)
    plot_data["rejection_rate"] = (
        plot_data["n_rejected"] / plot_data["n_offers"].replace(0, pd.NA)
    ).fillna(0)
    plot_data = plot_data.loc[plot_data["n_offers"] > 0]

    fig, ax = plt.subplots(figsize=(7.0, 4.0) if "treatment" in plot_data.columns else FIGURE_SIZE)
    if "treatment" in plot_data.columns:
        sns.barplot(
            data=plot_data,
            x="offer",
            y="rejection_rate",
            hue="treatment",
            palette=TREATMENT_PALETTE,
            width=0.75,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.barplot(
            data=plot_data,
            x="offer",
            y="rejection_rate",
            color=PALETTE["reject"],
            width=0.75,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for patch, (_, row) in zip(ax.patches, plot_data.iterrows()):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                row["rejection_rate"] + 0.03,
                f"{row['rejection_rate']:.0%}\n(n = {int(row['n_offers'])})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Coins offered to responder")
    ax.set_ylabel("Proportion rejected")
    ax.set_title("Rejection rate by offer")
    ax.set_xticks(range(0, TOTAL_COINS + 1))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    _despine(ax)
    _save_figure(fig, output_path)


def main() -> None:
    configure_plot_style()
    data_root = get_data_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_trials = load_game_trials(data_root)
    rounds = load_round_dataframe(data_root, game_trials)

    game_trials.to_csv(OUTPUT_DIR / "game_trials.csv", index=False)
    rounds.to_csv(OUTPUT_DIR / "game_trial_rounds.csv", index=False)

    plot_acceptance_rates(rounds, OUTPUT_DIR / "acceptance_rates.png")
    plot_offer_histogram(rounds, OUTPUT_DIR / "offer_histogram.png")
    plot_accumulated_score_histogram(
        game_trials, OUTPUT_DIR / "accumulated_score_histogram.png"
    )
    plot_fairness_by_round(rounds, OUTPUT_DIR / "fairness_by_round.png")
    plot_efficiency_by_round(rounds, OUTPUT_DIR / "efficiency_by_round.png")
    plot_rejection_by_offer_and_round(
        rounds, OUTPUT_DIR / "rejection_by_offer_round.png"
    )

    print(f"Loaded {len(game_trials)} GameTrial rows from {data_root}")
    print(f"Expanded to {len(rounds)} round-level rows across {rounds['group_id'].nunique()} dyads")
    print(f"Wrote plots and CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
