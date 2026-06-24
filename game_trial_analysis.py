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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, fisher_exact

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
    ax.grid(True, linestyle="--", alpha=0.5)


def _acceptance_rate_data(rounds: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    filtered = rounds.loc[rounds["decision"].isin(labels)]
    return filtered.assign(
        Percentage=(filtered["decision"] == "accept").astype(float) * 100,
    )


def _accept_reject_proportion_pvalue(accept_count: int, reject_count: int) -> float:
    total = accept_count + reject_count
    if total == 0:
        return float("nan")
    return float(
        binomtest(int(accept_count), int(total), p=0.5, alternative="two-sided").pvalue
    )


def _treatment_outcome_pvalue(
    rounds: pd.DataFrame,
    decision: str,
    treatments: list[str],
) -> float:
    if len(treatments) != 2:
        return float("nan")

    table: list[list[int]] = []
    for treatment in treatments:
        subset = rounds.loc[rounds["treatment"] == treatment]
        n_outcome = int((subset["decision"] == decision).sum())
        n_other = int((subset["decision"] != decision).sum())
        if n_outcome + n_other == 0:
            return float("nan")
        table.append([n_outcome, n_other])

    return float(fisher_exact(table)[1])


def _treatment_efficiency_pvalue(
    dyads: pd.DataFrame,
    treatments: list[str],
) -> float:
    if len(treatments) != 2:
        return float("nan")

    table: list[list[int]] = []
    for treatment in treatments:
        subset = dyads.loc[dyads["treatment"] == treatment].dropna(
            subset=["efficiency", "rounds_played"]
        )
        perfect = subset["efficiency"] == subset["rounds_played"] * TOTAL_COINS
        n_full = int(perfect.sum())
        n_partial = int((~perfect).sum())
        if n_full + n_partial == 0:
            return float("nan")
        table.append([n_full, n_partial])

    return float(fisher_exact(table)[1])


def _treatment_fairness_pvalue(
    dyads: pd.DataFrame,
    treatments: list[str],
) -> float:
    if len(treatments) != 2:
        return float("nan")

    table: list[list[int]] = []
    for treatment in treatments:
        subset = dyads.loc[dyads["treatment"] == treatment, "fairness"].dropna()
        n_perfect = int((subset == 1.0).sum())
        n_imperfect = int((subset != 1.0).sum())
        if n_perfect + n_imperfect == 0:
            return float("nan")
        table.append([n_perfect, n_imperfect])

    return float(fisher_exact(table)[1])


def _format_pvalue(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _bar_group_top(ax: plt.Axes, bars: list) -> float:
    x_min = min(bar.get_x() for bar in bars) - 0.1
    x_max = max(bar.get_x() + bar.get_width() for bar in bars) + 0.1
    y_top = max(bar.get_y() + bar.get_height() for bar in bars)
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            if pd.notna(x) and pd.notna(y) and x_min <= x <= x_max:
                y_top = max(y_top, y)
    return y_top


def _annotate_accept_reject_proportion_tests(
    ax: plt.Axes,
    rounds: pd.DataFrame,
    labels: list[str],
    *,
    treatments: list[str] | None = None,
    label_y: float = 0.0,
) -> float:
    filtered = rounds.loc[rounds["decision"].isin(labels)]

    if treatments:
        bars = list(ax.patches)
        x_center = sum(bar.get_x() + bar.get_width() / 2 for bar in bars) / len(bars)
        y_top = _bar_group_top(ax, bars)
        p_value = _treatment_outcome_pvalue(filtered, "accept", treatments)
        p_str = _format_pvalue(p_value)
        if p_str:
            label_y = max(y_top + 3, label_y)
            ax.text(
                x_center,
                label_y,
                p_str,
                ha="center",
                va="bottom",
                fontsize=9,
            )
            return label_y + 5
        return label_y
    else:
        bars = [container[0] for container in ax.containers]
        x_center = sum(bar.get_x() + bar.get_width() / 2 for bar in bars) / len(bars)
        y_top = _bar_group_top(ax, bars)
        n_accept = int((filtered["decision"] == "accept").sum())
        n_reject = int((filtered["decision"] == "reject").sum())
        p_value = _accept_reject_proportion_pvalue(n_accept, n_reject)
        label_y = y_top + 3
        ax.text(
            x_center,
            label_y,
            _format_pvalue(p_value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return label_y + 5


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


def normalize_round_index(rounds: pd.DataFrame) -> pd.DataFrame:
    """Map engine round indices to 1..N counted rounds within each dyad."""
    dyad_cols = ["treatment", "batch_id", "group_id"]
    normalized = rounds.copy()
    normalized["engine_round_index"] = normalized["round_index"]
    normalized["round_index"] = (
        normalized.groupby(dyad_cols, sort=False)["engine_round_index"]
        .rank(method="first")
        .astype(int)
    )
    return normalized


def _accumulated_payoffs_from_history(history: list) -> dict[str, float]:
    totals: dict[str, float] = {}
    for round_row in history:
        if round_row.get("skipped"):
            continue
        for participant_id, value in round_row.get("payoffs", {}).items():
            totals[str(participant_id)] = totals.get(str(participant_id), 0.0) + float(
                value
            )
    return totals


def _rounds_played_from_history(history: list) -> int:
    return sum(1 for round_row in history if not round_row.get("skipped"))


def load_dyad_metrics_dataframe(
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
            history = state.get("history", [])
            payoffs = _accumulated_payoffs_from_history(history)
            fairness, efficiency = _round_metrics(payoffs)
            rows.append(
                {
                    "treatment": treatment,
                    "batch_id": batch_dir.name,
                    "group_id": group_id,
                    "rounds_played": _rounds_played_from_history(history),
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
    has_treatment = (
        "treatment" in rounds.columns and rounds["treatment"].nunique() > 1
    )
    plot_data = _acceptance_rate_data(rounds, labels)
    filtered = rounds.loc[rounds["decision"].isin(labels)]
    errorbar_kwargs = {
        "errorbar": ("ci", 95),
        "capsize": 0.1,
        "err_kws": {"linewidth": 1},
    }
    treatment_order = [
        treatment
        for treatment in TREATMENT_PALETTE
        if treatment in plot_data["treatment"].unique()
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        sns.barplot(
            data=plot_data,
            x="treatment",
            y="Percentage",
            hue="treatment",
            hue_order=treatment_order,
            order=treatment_order,
            palette=TREATMENT_PALETTE,
            legend=False,
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
            **errorbar_kwargs,
        )
        label_y = 0.0
        for bar, treatment in zip(ax.patches, treatment_order):
            subset = filtered.loc[filtered["treatment"] == treatment]
            n_accept = int((subset["decision"] == "accept").sum())
            n_total = len(subset)
            percentage = 100 * n_accept / n_total if n_total else 0
            y_top = _bar_group_top(ax, [bar])
            text_y = y_top + 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{percentage:.1f}%\n(n = {n_accept})",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            label_y = max(label_y, text_y + 10)
        y_top = _annotate_accept_reject_proportion_tests(
            ax, rounds, labels, treatments=treatment_order, label_y=label_y
        )
    else:
        plot_data = plot_data.assign(category="")
        n_accept = int((filtered["decision"] == "accept").sum())
        n_reject = int((filtered["decision"] == "reject").sum())
        n_total = len(filtered)
        sns.barplot(
            data=plot_data,
            x="category",
            y="Percentage",
            color=PALETTE["accept"],
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
            **errorbar_kwargs,
        )
        bar = ax.containers[0][0]
        y_top = _bar_group_top(ax, [bar])
        x = bar.get_x() + bar.get_width() / 2
        percentage = 100 * n_accept / n_total if n_total else 0
        p_value = _accept_reject_proportion_pvalue(n_accept, n_reject)
        annotation_lines = [
            f"{percentage:.1f}%",
            f"(n = {n_accept})",
        ]
        p_str = _format_pvalue(p_value)
        if p_str:
            annotation_lines.append(p_str)
        ax.text(
            x,
            y_top + 2,
            "\n".join(annotation_lines),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        y_top = y_top + 2 + 5 * len(annotation_lines)

    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xlabel("")
    ax.set_title("Offer acceptance")
    if not has_treatment:
        ax.set_xticks([])
    ax.set_ylim(0, max(110, y_top))
    _despine(ax)
    _save_figure(fig, output_path)


def plot_offer_histogram(rounds: pd.DataFrame, output_path: Path) -> None:
    has_treatment = (
        "treatment" in rounds.columns and rounds["treatment"].nunique() > 1
    )
    bin_edges = np.arange(0, TOTAL_COINS + 2)
    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        treatment_order = [
            treatment
            for treatment in TREATMENT_PALETTE
            if treatment in rounds["treatment"].unique()
        ]
        sns.histplot(
            data=rounds,
            x="offer",
            hue="treatment",
            hue_order=treatment_order,
            palette=TREATMENT_PALETTE,
            bins=bin_edges,
            discrete=True,
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for container, treatment in _match_histplot_containers_to_treatments(
            ax, rounds, treatment_order, bin_edges, "offer"
        ):
            group_values = rounds.loc[
                rounds["treatment"] == treatment, "offer"
            ].to_numpy()
            _add_histogram_proportion_errorbars(
                ax,
                container,
                group_values,
                bin_edges,
                TREATMENT_PALETTE[treatment],
            )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Treatment")
    else:
        sns.histplot(
            data=rounds,
            x="offer",
            bins=bin_edges,
            discrete=True,
            stat="probability",
            color=PALETTE["primary"],
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        container = ax.containers[0] if ax.containers else ax.patches
        _add_histogram_proportion_errorbars(
            ax,
            container,
            rounds["offer"].to_numpy(),
            bin_edges,
            PALETTE["primary"],
        )
    ax.set_xlabel("Coins offered to responder")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Distribution of offers")
    ax.set_xticks(range(0, TOTAL_COINS + 1))
    _despine(ax)
    _save_figure(fig, output_path)


def _match_histplot_containers_to_treatments(
    ax: plt.Axes,
    scores: pd.DataFrame,
    treatment_order: list[str],
    bin_edges: np.ndarray,
    value_col: str,
) -> list[tuple[object, str]]:
    """Pair histplot bar containers with treatments by matching bar heights."""
    expected: dict[str, np.ndarray] = {}
    for treatment in treatment_order:
        values = scores.loc[scores["treatment"] == treatment, value_col].to_numpy()
        counts, _ = np.histogram(values, bins=bin_edges)
        total = counts.sum()
        expected[treatment] = counts / total if total else np.zeros(len(counts))

    matched: list[tuple[object, str]] = []
    available = list(enumerate(ax.containers))
    for treatment in treatment_order:
        props = expected[treatment]
        best_index: int | None = None
        best_container = None
        best_err = float("inf")
        for index, container in available:
            heights = np.array([bar.get_height() for bar in container])
            if len(heights) != len(props):
                continue
            err = float(np.abs(heights - props).sum())
            if err < best_err:
                best_err = err
                best_index = index
                best_container = container
        if best_container is None or best_err > 1e-6:
            raise ValueError(
                f"Could not match histplot container to treatment {treatment!r}"
            )
        available = [(index, container) for index, container in available if index != best_index]
        matched.append((best_container, treatment))
    return matched


def _bootstrap_histogram_proportion_ci(
    values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    confidence: float = 95,
    n_bootstrap: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n_bins = len(bin_edges) - 1
    if values.size == 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    rng = np.random.default_rng()
    boot_props = np.zeros((n_bootstrap, n_bins))
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        counts, _ = np.histogram(sample, bins=bin_edges)
        total = counts.sum()
        boot_props[index] = counts / total if total else 0

    alpha = (100 - confidence) / 2
    low = np.percentile(boot_props, alpha, axis=0)
    high = np.percentile(boot_props, 100 - alpha, axis=0)
    return low, high


def _add_histogram_proportion_errorbars(
    ax: plt.Axes,
    container,
    values: np.ndarray,
    bin_edges: np.ndarray,
    color: str,
    *,
    confidence: float = 95,
    n_bootstrap: int = 1000,
) -> None:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return

    low, high = _bootstrap_histogram_proportion_ci(
        values,
        bin_edges,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )
    for bar, lo, hi in zip(container, low, high):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        lo = min(lo, y)
        hi = max(hi, y)
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=2,
            capthick=1,
            alpha=0.8,
        )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0

    rng = np.random.default_rng()
    boot_means = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[index] = sample.mean()

    alpha = (100 - confidence) / 2
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def _offer_from_bar_center(bar, xticks: np.ndarray) -> int:
    center = bar.get_x() + bar.get_width() / 2
    return int(xticks[int(np.argmin(np.abs(xticks - center)))])


def _match_barplot_containers_to_treatments(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    treatment_order: list[str],
    rate_col: str,
) -> list[tuple[object, str]]:
    """Pair barplot containers with treatments by matching bar heights."""
    expected: dict[str, dict[int, float]] = {}
    for treatment in treatment_order:
        subset = plot_data.loc[plot_data["treatment"] == treatment]
        expected[treatment] = dict(zip(subset["offer"], subset[rate_col]))

    xticks = ax.get_xticks()
    matched: list[tuple[object, str]] = []
    available = list(enumerate(ax.containers))
    for treatment in treatment_order:
        rates = expected[treatment]
        best_index: int | None = None
        best_container = None
        best_err = float("inf")
        for index, container in available:
            err = 0.0
            count = 0
            for bar in container:
                offer = _offer_from_bar_center(bar, xticks)
                if offer not in rates:
                    continue
                err += abs(bar.get_height() - rates[offer])
                count += 1
            if count == 0:
                continue
            err /= count
            if err < best_err:
                best_err = err
                best_index = index
                best_container = container
        if best_container is None or best_err > 1e-6:
            raise ValueError(
                f"Could not match barplot container to treatment {treatment!r}"
            )
        available = [(index, container) for index, container in available if index != best_index]
        matched.append((best_container, treatment))
    return matched


def _add_barplot_proportion_errorbars(
    ax: plt.Axes,
    container,
    rounds: pd.DataFrame,
    color: str,
    *,
    treatment: str | None = None,
    confidence: float = 95,
    n_bootstrap: int = 1000,
) -> None:
    xticks = ax.get_xticks()
    for bar in container:
        offer = _offer_from_bar_center(bar, xticks)
        if treatment is None:
            mask = rounds["offer"] == offer
        else:
            mask = (rounds["treatment"] == treatment) & (rounds["offer"] == offer)
        values = rounds.loc[mask, "rejected"].astype(float).to_numpy()
        if values.size == 0:
            continue

        lo, hi = _bootstrap_mean_ci(
            values, confidence=confidence, n_bootstrap=n_bootstrap
        )
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        lo = min(lo, y)
        hi = max(hi, y)
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=2,
            capthick=1,
            alpha=0.8,
        )


def plot_accumulated_score_histogram(
    game_trials: pd.DataFrame, output_path: Path
) -> None:
    completed = game_trials.loc[
        game_trials["complete"].astype(bool) & ~game_trials["failed"].astype(bool)
    ]
    scores = completed[["total_score", "treatment"]].dropna(subset=["total_score"])
    has_treatment = (
        "treatment" in scores.columns and scores["treatment"].nunique() > 1
    )
    bin_edges = np.histogram_bin_edges(scores["total_score"], bins="auto")

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        treatment_order = [
            treatment
            for treatment in TREATMENT_PALETTE
            if treatment in scores["treatment"].unique()
        ]
        sns.histplot(
            data=scores,
            x="total_score",
            hue="treatment",
            hue_order=treatment_order,
            palette=TREATMENT_PALETTE,
            bins=bin_edges,
            stat="probability",
            common_norm=False,
            multiple="dodge",
            shrink=0.8,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for container, treatment in _match_histplot_containers_to_treatments(
            ax, scores, treatment_order, bin_edges, "total_score"
        ):
            group_values = scores.loc[
                scores["treatment"] == treatment, "total_score"
            ].to_numpy()
            _add_histogram_proportion_errorbars(
                ax,
                container,
                group_values,
                bin_edges,
                TREATMENT_PALETTE[treatment],
            )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Treatment")
    else:
        sns.histplot(
            data=scores,
            x="total_score",
            bins=bin_edges,
            stat="probability",
            color=PALETTE["secondary"],
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        container = ax.containers[0] if ax.containers else ax.patches
        _add_histogram_proportion_errorbars(
            ax,
            container,
            scores["total_score"].to_numpy(),
            bin_edges,
            PALETTE["secondary"],
        )
    ax.set_xlabel("Accumulated score")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Distribution of accumulated scores")
    _despine(ax)
    _save_figure(fig, output_path)


def plot_offer_by_round(rounds: pd.DataFrame, output_path: Path) -> None:
    has_treatment = (
        "treatment" in rounds.columns and rounds["treatment"].nunique() > 1
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        treatment_order = [
            treatment
            for treatment in TREATMENT_PALETTE
            if treatment in rounds["treatment"].unique()
        ]
        sns.lineplot(
            data=rounds,
            x="round_index",
            y="offer",
            hue="treatment",
            hue_order=treatment_order,
            palette=TREATMENT_PALETTE,
            marker="o",
            markersize=5,
            linewidth=1.5,
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={"capsize": 3, "linewidth": 1},
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.lineplot(
            data=rounds,
            x="round_index",
            y="offer",
            color=PALETTE["primary"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={"capsize": 3, "linewidth": 1},
            ax=ax,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Coins offered to responder")
    ax.set_title("Mean offer by round")
    ax.set_ylim(-0.05, TOTAL_COINS + 0.05)
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


def plot_fairness_by_treatment(dyads: pd.DataFrame, output_path: Path) -> None:
    has_treatment = (
        "treatment" in dyads.columns and dyads["treatment"].nunique() > 1
    )
    treatment_order = [
        treatment
        for treatment in TREATMENT_PALETTE
        if treatment in dyads["treatment"].unique()
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        sns.barplot(
            data=dyads,
            x="treatment",
            y="fairness",
            hue="treatment",
            hue_order=treatment_order,
            order=treatment_order,
            palette=TREATMENT_PALETTE,
            legend=False,
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for bar, treatment in zip(ax.patches, treatment_order):
            values = dyads.loc[
                dyads["treatment"] == treatment, "fairness"
            ].dropna().to_numpy()
            lo, hi = _bootstrap_mean_ci(values)
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            lo = min(lo, y)
            hi = max(hi, y)
            ax.errorbar(
                x,
                y,
                yerr=[[y - lo], [hi - y]],
                fmt="none",
                ecolor=TREATMENT_PALETTE[treatment],
                elinewidth=1,
                capsize=2,
                capthick=1,
                alpha=0.8,
            )
        bars = list(ax.patches)
        x_center = sum(bar.get_x() + bar.get_width() / 2 for bar in bars) / len(bars)
        y_top = _bar_group_top(ax, bars)
        p_value = _treatment_fairness_pvalue(dyads, treatment_order)
        ax.text(
            x_center,
            y_top + 0.03,
            _format_pvalue(p_value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        y_limit = max(1.05, y_top + 0.12)
    else:
        sns.barplot(
            data=dyads,
            x="treatment",
            y="fairness",
            color=PALETTE["primary"],
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        bar = ax.patches[0]
        values = dyads["fairness"].dropna().to_numpy()
        lo, hi = _bootstrap_mean_ci(values)
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        lo = min(lo, y)
        hi = max(hi, y)
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt="none",
            ecolor=PALETTE["primary"],
            elinewidth=1,
            capsize=2,
            capthick=1,
            alpha=0.8,
        )
        y_limit = 1.05

    ax.set_xlabel("")
    ax.set_ylabel("Fairness (min accumulated score / max accumulated score)")
    ax.set_title("Accumulated fairness by treatment")
    ax.set_ylim(0, y_limit)
    _despine(ax)
    _save_figure(fig, output_path)


def plot_efficiency_by_treatment(dyads: pd.DataFrame, output_path: Path) -> None:
    has_treatment = (
        "treatment" in dyads.columns and dyads["treatment"].nunique() > 1
    )
    treatment_order = [
        treatment
        for treatment in TREATMENT_PALETTE
        if treatment in dyads["treatment"].unique()
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        sns.barplot(
            data=dyads,
            x="treatment",
            y="efficiency",
            hue="treatment",
            hue_order=treatment_order,
            order=treatment_order,
            palette=TREATMENT_PALETTE,
            legend=False,
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for bar, treatment in zip(ax.patches, treatment_order):
            values = dyads.loc[
                dyads["treatment"] == treatment, "efficiency"
            ].dropna().to_numpy()
            lo, hi = _bootstrap_mean_ci(values)
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            lo = min(lo, y)
            hi = max(hi, y)
            ax.errorbar(
                x,
                y,
                yerr=[[y - lo], [hi - y]],
                fmt="none",
                ecolor=TREATMENT_PALETTE[treatment],
                elinewidth=1,
                capsize=2,
                capthick=1,
                alpha=0.8,
            )
        bars = list(ax.patches)
        x_center = sum(bar.get_x() + bar.get_width() / 2 for bar in bars) / len(bars)
        y_top = _bar_group_top(ax, bars)
        p_value = _treatment_efficiency_pvalue(dyads, treatment_order)
        label_offset = max(5.0, y_top * 0.04)
        ax.text(
            x_center,
            y_top + label_offset,
            _format_pvalue(p_value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        y_limit = y_top + max(15.0, y_top * 0.12)
    else:
        sns.barplot(
            data=dyads,
            x="treatment",
            y="efficiency",
            color=PALETTE["accent"],
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        bar = ax.patches[0]
        values = dyads["efficiency"].dropna().to_numpy()
        lo, hi = _bootstrap_mean_ci(values)
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        lo = min(lo, y)
        hi = max(hi, y)
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt="none",
            ecolor=PALETTE["accent"],
            elinewidth=1,
            capsize=2,
            capthick=1,
            alpha=0.8,
        )
        y_top = _bar_group_top(ax, [bar])
        y_limit = y_top + max(15.0, y_top * 0.12)

    ax.set_xlabel("")
    ax.set_ylabel("Efficiency (sum of accumulated dyad payoffs)")
    ax.set_title("Accumulated efficiency by treatment")
    ax.set_ylim(-0.05, y_limit)
    _despine(ax)
    _save_figure(fig, output_path)


def plot_rejection_by_round(rounds: pd.DataFrame, output_path: Path) -> None:
    plot_data = rounds.loc[rounds["decision"].isin(["accept", "reject"])].copy()
    has_treatment = (
        "treatment" in plot_data.columns and plot_data["treatment"].nunique() > 1
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0) if has_treatment else FIGURE_SIZE)
    if has_treatment:
        treatment_order = [
            treatment
            for treatment in TREATMENT_PALETTE
            if treatment in plot_data["treatment"].unique()
        ]
        sns.lineplot(
            data=plot_data,
            x="round_index",
            y="rejected",
            hue="treatment",
            hue_order=treatment_order,
            palette=TREATMENT_PALETTE,
            marker="o",
            markersize=5,
            linewidth=1.5,
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={"capsize": 3, "linewidth": 1},
            ax=ax,
        )
        ax.legend(title="Treatment")
    else:
        sns.lineplot(
            data=plot_data,
            x="round_index",
            y="rejected",
            color=PALETTE["reject"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={"capsize": 3, "linewidth": 1},
            ax=ax,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion rejected")
    ax.set_title("Rejection rate by round")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    _despine(ax)
    _save_figure(fig, output_path)


def plot_rejection_by_offer_and_round(rounds: pd.DataFrame, output_path: Path) -> None:
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
        treatment_order = [
            treatment
            for treatment in TREATMENT_PALETTE
            if treatment in plot_data["treatment"].unique()
        ]
        sns.barplot(
            data=plot_data,
            x="offer",
            y="rejection_rate",
            hue="treatment",
            hue_order=treatment_order,
            palette=TREATMENT_PALETTE,
            width=0.75,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        for container, treatment in _match_barplot_containers_to_treatments(
            ax, plot_data, treatment_order, "rejection_rate"
        ):
            _add_barplot_proportion_errorbars(
                ax,
                container,
                rounds,
                TREATMENT_PALETTE[treatment],
                treatment=treatment,
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
        container = ax.containers[0] if ax.containers else ax.patches
        _add_barplot_proportion_errorbars(
            ax, container, rounds, PALETTE["reject"]
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
    rounds = normalize_round_index(load_round_dataframe(data_root, game_trials))
    dyad_metrics = load_dyad_metrics_dataframe(data_root, game_trials)

    game_trials.to_csv(OUTPUT_DIR / "game_trials.csv", index=False)
    rounds.to_csv(OUTPUT_DIR / "game_trial_rounds.csv", index=False)

    plot_acceptance_rates(rounds, OUTPUT_DIR / "acceptance_rates.png")
    plot_offer_histogram(rounds, OUTPUT_DIR / "offer_histogram.png")
    plot_offer_by_round(rounds, OUTPUT_DIR / "offer_by_round.png")
    plot_accumulated_score_histogram(
        game_trials, OUTPUT_DIR / "accumulated_score_histogram.png"
    )
    plot_fairness_by_round(rounds, OUTPUT_DIR / "fairness_by_round.png")
    plot_fairness_by_treatment(
        dyad_metrics, OUTPUT_DIR / "fairness_by_treatment.png"
    )
    plot_efficiency_by_round(rounds, OUTPUT_DIR / "efficiency_by_round.png")
    plot_efficiency_by_treatment(
        dyad_metrics, OUTPUT_DIR / "efficiency_by_treatment.png"
    )
    plot_rejection_by_round(rounds, OUTPUT_DIR / "rejection_by_round.png")
    plot_rejection_by_offer_and_round(
        rounds, OUTPUT_DIR / "rejection_by_offer_round.png"
    )

    print(f"Loaded {len(game_trials)} GameTrial rows from {data_root}")
    print(f"Expanded to {len(rounds)} round-level rows across {rounds['group_id'].nunique()} dyads")
    print(f"Wrote plots and CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
