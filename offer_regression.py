#!/usr/bin/env python3
"""Linear regression of mean offer on round and treatment.

Uses round-level observations from completed Ultimatum dyads. The dependent
variable is the coins offered to the responder. Predictors are round index,
treatment, and their interaction.

Writes ``analysis/offer_regression_summary.txt`` and
``analysis/offer_by_round_treatment.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from extract_module_times import get_data_root
from game_trial_analysis import (
    load_game_trials,
    load_round_dataframe,
    normalize_round_index,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"
SUMMARY_PATH = OUTPUT_DIR / "offer_regression_summary.txt"
AGGREGATE_PATH = OUTPUT_DIR / "offer_by_round_treatment.csv"
REFERENCE_TREATMENT = "random"


def load_offer_rounds() -> pd.DataFrame:
    data_root = get_data_root()
    game_trials = load_game_trials(data_root)
    return normalize_round_index(load_round_dataframe(data_root, game_trials))


def aggregate_offer_by_round_treatment(rounds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rounds.groupby(["treatment", "round_index"], as_index=False)
        .agg(
            n_rounds=("offer", "size"),
            mean_offer=("offer", "mean"),
            std_offer=("offer", "std"),
        )
        .sort_values(["treatment", "round_index"])
    )
    return summary


def fit_offer_regression(rounds: pd.DataFrame):
    model_data = rounds.copy()
    model_data["treatment"] = pd.Categorical(
        model_data["treatment"],
        categories=[REFERENCE_TREATMENT, "constant"],
    )
    formula = f"offer ~ round_index * C(treatment, Treatment('{REFERENCE_TREATMENT}'))"
    return smf.ols(formula, data=model_data).fit()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rounds = load_offer_rounds()
    aggregate = aggregate_offer_by_round_treatment(rounds)
    aggregate.to_csv(AGGREGATE_PATH, index=False)

    result = fit_offer_regression(rounds)
    summary_text = result.summary().as_text()

    SUMMARY_PATH.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    print(f"\nWrote aggregate data to {AGGREGATE_PATH}")
    print(f"Wrote regression summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
