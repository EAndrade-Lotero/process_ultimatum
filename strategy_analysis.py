#!/usr/bin/env python3
"""Compile Strategy open answers into a PDF and summarize salient themes.

Reads Response.csv from batch exports (see ``data_with_treatment_path.txt`` or
``data_path.txt``) and writes:

- ``analysis/strategy_responses.pdf`` — one entry per participant
- ``analysis/strategy_responses.csv`` — tabular export of verbal answers
- ``analysis/strategy_summary.txt`` — qualitative summary of common strategies
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from extract_module_times import find_response_csv, get_data_root, iter_treatment_batches

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis"
PDF_PATH = OUTPUT_DIR / "strategy_responses.pdf"
CSV_PATH = OUTPUT_DIR / "strategy_responses.csv"
SUMMARY_PATH = OUTPUT_DIR / "strategy_summary.txt"

THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    "equal_split": re.compile(
        r"\b(50.?50|fifty.?fifty|equal|even(?:ly)?|half|split.{0,20}middle|same amount)\b",
        re.IGNORECASE,
    ),
    "fairness_norm": re.compile(
        r"\b(fair|equit|moral|generous|trust)\b",
        re.IGNORECASE,
    ),
    "rejection_threshold": re.compile(
        r"\b(reject|below|under|less than|minimum|at least|lowball|low offer|punish)\b",
        re.IGNORECASE,
    ),
    "probe_minimum": re.compile(
        r"\b(?:test|see (?:what|how)|find|lower bound|sweet spot|probe|experiment|try)\b",
        re.IGNORECASE,
    ),
    "maximize_self": re.compile(
        r"\b(maxim|as (?:much|low) as|keep more|selfish|gain|profit|extra coin|more coin)\b",
        re.IGNORECASE,
    ),
    "accept_any": re.compile(
        r"\b(accept(?:ed)? all|anything|better than nothing|take what|any offer)\b",
        re.IGNORECASE,
    ),
    "partner_adaptation": re.compile(
        r"\b(partner|other (?:person|player)|they|respond|retaliat|advantage)\b",
        re.IGNORECASE,
    ),
}

THEME_LABELS = {
    "equal_split": "Equal / 50-50 split",
    "fairness_norm": "Fairness and trust",
    "rejection_threshold": "Rejection threshold (minimum acceptable offer)",
    "probe_minimum": "Probing for minimum acceptable offer",
    "maximize_self": "Maximizing own payoff",
    "accept_any": "Accept most or all offers",
    "partner_adaptation": "Adapting to partner behavior",
}


def load_strategy_responses(data_root: Path | None = None) -> pd.DataFrame:
    root = data_root or get_data_root()
    rows: list[dict[str, object]] = []

    for treatment, batch_dir in iter_treatment_batches(root):
        csv_path = find_response_csv(batch_dir)
        df = pd.read_csv(csv_path)
        strategy_rows = df[df["question"] == "Strategy"].copy()

        for _, row in strategy_rows.iterrows():
            answer = json.loads(row["answer"]) if pd.notna(row["answer"]) else {}
            rows.append(
                {
                    "treatment": treatment,
                    "batch_id": batch_dir.name,
                    "participant_id": int(row["participant_id"]),
                    "verbal_strategy": (answer.get("verbal_strategy") or "").strip(),
                    "own_benefit": answer.get("own_benefit"),
                    "other_benefit": answer.get("other_benefit"),
                    "importance_of_fairness": answer.get("importance_of_fairness"),
                    "assessment_of_fairness": answer.get("assessment_of_fairness"),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No Strategy responses found under {root}")

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["treatment", "batch_id", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    return result


def _wrap_paragraph(text: str, width: int = 88) -> str:
    if not text:
        return "(no response)"
    return "\n".join(textwrap.fill(line, width=width) if line else "" for line in text.splitlines())


def write_strategy_pdf(responses: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for _, row in responses.iterrows():
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")

            treatment = row["treatment"] or "(no treatment label)"
            header = (
                f"Treatment: {treatment}\n"
                f"Batch: {row['batch_id']}\n"
                f"Participant ID: {row['participant_id']}\n"
            )
            body = _wrap_paragraph(str(row["verbal_strategy"]))

            ax.text(
                0.05,
                0.95,
                "Strategy open answer",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
            )
            ax.text(
                0.05,
                0.88,
                header,
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                family="monospace",
            )
            ax.text(
                0.05,
                0.72,
                body,
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                wrap=True,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _count_themes(responses: pd.DataFrame) -> tuple[pd.Series, int]:
    nonempty = responses[responses["verbal_strategy"] != ""].copy()
    counts: dict[str, int] = {}

    for theme, pattern in THEME_PATTERNS.items():
        counts[theme] = int(
            nonempty["verbal_strategy"].str.contains(pattern, na=False).sum()
        )

    return pd.Series(counts).sort_values(ascending=False), len(nonempty)


def build_summary_text(responses: pd.DataFrame) -> str:
    theme_counts, n_nonempty = _count_themes(responses)
    n_empty = int((responses["verbal_strategy"] == "").sum())
    n_total = len(responses)

    by_treatment = (
        responses.groupby("treatment", dropna=False)
        .agg(
            responses=("participant_id", "count"),
            nonempty=("verbal_strategy", lambda s: int((s != "").sum())),
        )
        .reset_index()
    )

    lines = [
        "Strategy open-answer summary",
        "=" * 32,
        "",
        f"Total Strategy responses: {n_total}",
        f"Non-empty verbal answers: {n_nonempty}",
        f"Empty verbal answers: {n_empty}",
        "",
        "Counts by treatment:",
    ]
    for _, row in by_treatment.iterrows():
        label = row["treatment"] or "(no treatment label)"
        lines.append(
            f"  {label}: {int(row['responses'])} responses "
            f"({int(row['nonempty'])} with text)"
        )

    lines.extend(
        [
            "",
            "Most salient strategy themes",
            "-" * 28,
            "",
            "Responses often combine several themes. Counts below count participants "
            "whose answer mentions each theme at least once.",
            "",
        ]
    )

    for theme, count in theme_counts.items():
        pct = 100 * count / n_nonempty if n_nonempty else 0
        lines.append(f"  {THEME_LABELS[theme]}: {count} ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "Qualitative synthesis",
            "-" * 20,
            "",
            "1. Equal splitting was the dominant stated strategy. Many participants "
            "described offering or accepting a 50/50 division and sticking with it "
            "once accepted, treating equal splits as both fair and efficient.",
            "",
            "2. Fairness norms were widely invoked. Participants framed decisions in "
            "terms of equity, generosity, trust-building, and moral obligation, not "
            "only payoff maximization.",
            "",
            "3. Rejection thresholds were common among responders. A frequent rule "
            "was to reject offers below about 4-5 coins, using rejection to signal "
            "unacceptable lowball proposals or punish unfair partners.",
            "",
            "4. Proposers often probed for the minimum acceptable offer. Several "
            "participants tested progressively lower offers, searched for a 'sweet "
            "spot,' or alternated between low and higher offers to learn partner "
            "tolerance.",
            "",
            "5. Some participants prioritized any positive payoff. A smaller group "
            "reported accepting all or nearly all offers on the grounds that some "
            "coins are better than none.",
            "",
            "6. Adaptive play was salient. Many answers described watching the "
            "partner's behavior and adjusting offers or acceptance rules over "
            "rounds, including retaliation after unfair proposals.",
            "",
            "7. Explicit payoff maximization was less common in open text than "
            "fairness language, but still appeared when participants described "
            "keeping extra coins, offering as little as possible, or pushing for "
            "7+ coins when in the responder role.",
            "",
            "Notable edge cases:",
            "- A few participants reported no strategy or gave very short answers.",
            "- Some answers described alternating patterns between two or three offer "
            "levels after finding an acceptable range.",
            "- Occasional end-game deviations appeared, such as taking one extra "
            "coin in the final round after maintaining fairness earlier.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    responses = load_strategy_responses()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    responses.to_csv(CSV_PATH, index=False)
    write_strategy_pdf(responses, PDF_PATH)
    SUMMARY_PATH.write_text(build_summary_text(responses), encoding="utf-8")

    print(f"Wrote {len(responses)} strategy responses to:")
    print(f"  {PDF_PATH}")
    print(f"  {CSV_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
