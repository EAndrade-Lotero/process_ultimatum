#!/usr/bin/env python3
"""Offer and acceptance figures for the constant treatment.

Figure 1: acceptance rate by offer bin and experience (rounds 1-5 vs 6-10).
Figure 2: acceptance rate by offer bin and change relative to the previous offer.
Figure 3: average current offer by lagged offer bin and lagged outcome.

Uses completed dyads from GameTrial.csv and round history from UltimatumSession.csv,
following the same inclusion rules as ``game_trial_analysis.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from game_trial_analysis import (
    OUTPUT_DIR,
    TOTAL_COINS,
    configure_plot_style,
    get_data_root,
    load_game_trials,
    load_round_dataframe,
)

FIGURE1_BIN_LABELS = [
    "0 ≤ Offer < 10",
    "10 ≤ Offer < 20",
    "20 ≤ Offer < 30",
    "30 ≤ Offer < 40",
    "40 ≤ Offer < 50",
    "Offer ≥ 50",
]
FIGURE2_BIN_LABELS = [
    "20 ≤ Offer < 30",
    "30 ≤ Offer < 40",
    "40 ≤ Offer < 50",
]
EXPERIENCE_ORDER = ["Rounds 1 - 5", "Rounds 6 - 10"]
EXPERIENCE_PALETTE = {
    "Rounds 1 - 5": "#4472C4",
    "Rounds 6 - 10": "#C00000",
}
PREVIOUS_OFFER_ORDER = [
    "Lower than Previous",
    "Same as Previous",
    "Higher than Previous",
]
PREVIOUS_OFFER_PALETTE = {
    "Lower than Previous": "#4472C4",
    "Same as Previous": "#C00000",
    "Higher than Previous": "#548235",
}
LAGGED_OUTCOME_ORDER = [
    "Lagged Offer Rejected",
    "Lagged Offer Accepted",
]
LAGGED_OUTCOME_PALETTE = {
    "Lagged Offer Rejected": "#4472C4",
    "Lagged Offer Accepted": "#C00000",
}
MAX_ROUND = 10
EARLY_ROUND_CUTOFF = 5
DYAD_KEYS = ["batch_id", "group_id"]


def _percent_offered(offer: int) -> float:
    return offer / TOTAL_COINS * 100


def _figure1_offer_bin_label(percent_offered: float) -> str:
    if percent_offered < 10:
        return FIGURE1_BIN_LABELS[0]
    if percent_offered < 20:
        return FIGURE1_BIN_LABELS[1]
    if percent_offered < 30:
        return FIGURE1_BIN_LABELS[2]
    if percent_offered < 40:
        return FIGURE1_BIN_LABELS[3]
    if percent_offered < 50:
        return FIGURE1_BIN_LABELS[4]
    return FIGURE1_BIN_LABELS[5]


def _figure2_offer_bin_label(percent_offered: float) -> str | None:
    if 20 <= percent_offered < 30:
        return FIGURE2_BIN_LABELS[0]
    if 30 <= percent_offered < 40:
        return FIGURE2_BIN_LABELS[1]
    if 40 <= percent_offered < 50:
        return FIGURE2_BIN_LABELS[2]
    return None


def _experience_label(round_index: int) -> str:
    if round_index <= EARLY_ROUND_CUTOFF:
        return EXPERIENCE_ORDER[0]
    return EXPERIENCE_ORDER[1]


def _previous_offer_category(current_offer: int, previous_offer: int) -> str:
    if current_offer < previous_offer:
        return PREVIOUS_OFFER_ORDER[0]
    if current_offer == previous_offer:
        return PREVIOUS_OFFER_ORDER[1]
    return PREVIOUS_OFFER_ORDER[2]


def _constant_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    data = rounds.loc[
        (rounds["treatment"] == "constant")
        & rounds["round_index"].between(1, MAX_ROUND)
    ].copy()
    if data.empty:
        raise ValueError("No constant-treatment rows found for rounds 1-10.")
    return data


def build_acceptance_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate acceptance rates for Figure 1."""
    data = _constant_rounds(rounds)
    data["percent_offered"] = data["offer"].map(_percent_offered)
    data["offer_bin"] = data["percent_offered"].map(_figure1_offer_bin_label)
    data["experience"] = data["round_index"].map(_experience_label)
    data["accepted"] = data["decision"].eq("accept")

    summary = (
        data.groupby(["offer_bin", "experience"], observed=True)
        .agg(
            n=("accepted", "size"),
            acceptance_rate=("accepted", "mean"),
        )
        .reset_index()
    )
    summary["offer_bin"] = pd.Categorical(
        summary["offer_bin"], categories=FIGURE1_BIN_LABELS, ordered=True
    )
    summary["experience"] = pd.Categorical(
        summary["experience"], categories=EXPERIENCE_ORDER, ordered=True
    )
    return summary.sort_values(["offer_bin", "experience"]).reset_index(drop=True)


def build_previous_offer_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate acceptance rates for Figure 2."""
    data = rounds.loc[rounds["treatment"] == "constant"].copy()
    if data.empty:
        raise ValueError("No constant-treatment rows found.")
    data = data.sort_values([*DYAD_KEYS, "round_index"])
    data["previous_offer"] = data.groupby(DYAD_KEYS)["offer"].shift(1)
    data = data.loc[data["previous_offer"].notna()].copy()
    data["previous_offer"] = data["previous_offer"].astype(int)
    data["percent_offered"] = data["offer"].map(_percent_offered)
    data["offer_bin"] = data["percent_offered"].map(_figure2_offer_bin_label)
    data = data.loc[data["offer_bin"].notna()].copy()
    data["previous_offer_category"] = [
        _previous_offer_category(current, previous)
        for current, previous in zip(data["offer"], data["previous_offer"])
    ]
    data["accepted"] = data["decision"].eq("accept")

    summary = (
        data.groupby(["offer_bin", "previous_offer_category"], observed=True)
        .agg(
            n=("accepted", "size"),
            acceptance_rate=("accepted", "mean"),
        )
        .reset_index()
    )
    summary["offer_bin"] = pd.Categorical(
        summary["offer_bin"], categories=FIGURE2_BIN_LABELS, ordered=True
    )
    summary["previous_offer_category"] = pd.Categorical(
        summary["previous_offer_category"],
        categories=PREVIOUS_OFFER_ORDER,
        ordered=True,
    )
    return summary.sort_values(
        ["offer_bin", "previous_offer_category"]
    ).reset_index(drop=True)


def build_lagged_outcome_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate average current offers for Figure 3."""
    data = rounds.loc[rounds["treatment"] == "constant"].copy()
    if data.empty:
        raise ValueError("No constant-treatment rows found.")
    data = data.sort_values([*DYAD_KEYS, "round_index"])
    data["lagged_offer"] = data.groupby(DYAD_KEYS)["offer"].shift(1)
    data["lagged_decision"] = data.groupby(DYAD_KEYS)["decision"].shift(1)
    data = data.loc[data["lagged_offer"].notna() & data["lagged_decision"].notna()].copy()
    data["lagged_offer"] = data["lagged_offer"].astype(int)
    data["lagged_percent_offered"] = data["lagged_offer"].map(_percent_offered)
    data["current_percent_offered"] = data["offer"].map(_percent_offered)
    data["lagged_offer_bin"] = data["lagged_percent_offered"].map(_figure1_offer_bin_label)
    data["lagged_outcome"] = np.where(
        data["lagged_decision"].eq("accept"),
        LAGGED_OUTCOME_ORDER[1],
        LAGGED_OUTCOME_ORDER[0],
    )

    summary = (
        data.groupby(["lagged_offer_bin", "lagged_outcome"], observed=True)
        .agg(
            n=("current_percent_offered", "size"),
            avg_current_offer=("current_percent_offered", "mean"),
        )
        .reset_index()
    )
    summary["lagged_offer_bin"] = pd.Categorical(
        summary["lagged_offer_bin"], categories=FIGURE1_BIN_LABELS, ordered=True
    )
    summary["lagged_outcome"] = pd.Categorical(
        summary["lagged_outcome"], categories=LAGGED_OUTCOME_ORDER, ordered=True
    )
    return summary.sort_values(["lagged_offer_bin", "lagged_outcome"]).reset_index(
        drop=True
    )


def _plot_grouped_bars(
    summary: pd.DataFrame,
    *,
    x_bin_labels: list[str],
    x_bin_col: str,
    group_order: list[str],
    palette: dict[str, str],
    group_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: np.ndarray | None = None,
    count_offset: float = 0.02,
) -> None:
    x = np.arange(len(x_bin_labels))
    n_groups = len(group_order)
    width = 0.8 / n_groups
    offset_start = -(n_groups - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for idx, group in enumerate(group_order):
        subset = summary.loc[summary[group_col] == group].set_index(x_bin_col)
        values = [
            subset.loc[label, value_col] if label in subset.index else 0.0
            for label in x_bin_labels
        ]
        counts = [
            int(subset.loc[label, "n"]) if label in subset.index else 0
            for label in x_bin_labels
        ]
        bars = ax.bar(
            x + offset_start + idx * width,
            values,
            width,
            label=group,
            color=palette[group],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, count in zip(bars, counts):
            if count == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + count_offset,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_bin_labels, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(group_order),
        frameon=False,
    )

    fig.suptitle(title, y=0.98, fontsize=12)
    fig.text(
        0.5,
        0.915,
        "Note: The numbers above the bars give the number of observations for that bar.",
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.subplots_adjust(top=0.82, bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_acceptance_by_offer_and_experience(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Replicate the grouped bar chart from Figure 1."""
    _plot_grouped_bars(
        summary,
        x_bin_labels=FIGURE1_BIN_LABELS,
        x_bin_col="offer_bin",
        group_order=EXPERIENCE_ORDER,
        palette=EXPERIENCE_PALETTE,
        group_col="experience",
        value_col="acceptance_rate",
        output_path=output_path,
        title="Figure 1: Acceptance Rate as a Function of Experience",
        xlabel="% Offered",
        ylabel="Acceptance Rate",
        ylim=(0.0, 1.0),
        yticks=np.arange(0, 1.01, 0.1),
    )


def plot_acceptance_by_previous_offer_category(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Replicate the grouped bar chart from Figure 2."""
    _plot_grouped_bars(
        summary,
        x_bin_labels=FIGURE2_BIN_LABELS,
        x_bin_col="offer_bin",
        group_order=PREVIOUS_OFFER_ORDER,
        palette=PREVIOUS_OFFER_PALETTE,
        group_col="previous_offer_category",
        value_col="acceptance_rate",
        output_path=output_path,
        title="Figure 2: Acceptance Rate as a Function of Previous Offer Category",
        xlabel="% Offered",
        ylabel="Acceptance Rate",
        ylim=(0.2, 1.0),
        yticks=np.arange(0.2, 1.01, 0.1),
    )


def plot_current_offer_by_lagged_outcome(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Replicate the grouped bar chart from Figure 3."""
    _plot_grouped_bars(
        summary,
        x_bin_labels=FIGURE1_BIN_LABELS,
        x_bin_col="lagged_offer_bin",
        group_order=LAGGED_OUTCOME_ORDER,
        palette=LAGGED_OUTCOME_PALETTE,
        group_col="lagged_outcome",
        value_col="avg_current_offer",
        output_path=output_path,
        title="Figure 3: Current Offers as a Function of Lagged Outcome",
        xlabel="Lagged Percent Offered",
        ylabel="Average Current Offer",
        ylim=(0.0, 60.0),
        yticks=np.arange(0, 61, 10),
        count_offset=1.0,
    )


def _print_summary_table(summary: pd.DataFrame, value_cols: list[str]) -> None:
    formatted = summary.copy()
    for col in value_cols:
        formatted[col] = formatted[col].map(lambda v: f"{v:.3f}")
    print(formatted.to_string(index=False))


def main() -> None:
    configure_plot_style()
    data_root = get_data_root()
    game_trials = load_game_trials(data_root)
    rounds = load_round_dataframe(data_root, game_trials)

    figure1_summary = build_acceptance_summary(rounds)
    figure1_path = OUTPUT_DIR / "acceptance_by_offer_experience.png"
    plot_acceptance_by_offer_and_experience(figure1_summary, figure1_path)

    figure2_summary = build_previous_offer_summary(rounds)
    figure2_path = OUTPUT_DIR / "acceptance_by_previous_offer_category.png"
    plot_acceptance_by_previous_offer_category(figure2_summary, figure2_path)

    figure3_summary = build_lagged_outcome_summary(rounds)
    figure3_path = OUTPUT_DIR / "current_offer_by_lagged_outcome.png"
    plot_current_offer_by_lagged_outcome(figure3_summary, figure3_path)

    print(f"Data root: {data_root}")
    constant_rows = _constant_rounds(rounds)
    print(f"Constant-treatment rows (rounds 1-{MAX_ROUND}): {len(constant_rows)}")

    print("\nFigure 1 summary:")
    _print_summary_table(figure1_summary, ["acceptance_rate"])
    print(f"\nWrote {figure1_path}")

    print("\nFigure 2 summary:")
    _print_summary_table(figure2_summary, ["acceptance_rate"])
    print(f"\nWrote {figure2_path}")

    print("\nFigure 3 summary:")
    _print_summary_table(figure3_summary, ["avg_current_offer"])
    print(f"\nWrote {figure3_path}")


if __name__ == "__main__":
    main()
