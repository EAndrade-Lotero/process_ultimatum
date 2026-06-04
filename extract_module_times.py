#!/usr/bin/env python3
"""Extract participant_id and time_taken from ModuleState.csv in batch-n directories."""

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

TUTORIAL_TIME = 90
HOURLY_PAYMENT = 25
SECONDS_PER_HOUR = 3600
BONUS_PER_SECOND = HOURLY_PAYMENT / SECONDS_PER_HOUR
STAGE_ORDER = ("conscent", "waiting", "game")
STAGE_RANK = {stage: rank for rank, stage in enumerate(STAGE_ORDER)}
REWARD = 6.25
MIN_GAME_TRIAL_COUNT = 10


def get_data_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    data_path_file = script_dir / "data_path.txt"
    return Path(data_path_file.read_text().strip())


def find_batch_dirs(data_root: Path) -> list[Path]:
    pattern = re.compile(r"^batch-\d+$")
    return sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and pattern.match(path.name)
    )


def find_csv_in_batch(batch_dir: Path, filename: str) -> Path:
    direct = batch_dir / filename
    if direct.is_file():
        return direct

    matches = sorted(batch_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No {filename} found in {batch_dir}")

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple {filename} files found in {batch_dir}: {matches}"
        )

    return matches[0]


def find_module_state_csv(batch_dir: Path) -> Path:
    return find_csv_in_batch(batch_dir, "ModuleState.csv")


def find_waiting_trial_csv(batch_dir: Path) -> Path:
    return find_csv_in_batch(batch_dir, "WaitingTrial.csv")


def find_game_trial_csv(batch_dir: Path) -> Path:
    return find_csv_in_batch(batch_dir, "GameTrial.csv")


def find_participant_csv(batch_dir: Path) -> Path:
    return find_csv_in_batch(batch_dir, "Participant.csv")


def load_batch_module_state(batch_dir: Path) -> pd.DataFrame:
    csv_path = find_module_state_csv(batch_dir)
    df = pd.read_csv(csv_path)

    time_started = pd.to_datetime(df["time_started"], errors="coerce")
    time_finished = pd.to_datetime(df["time_finished"], errors="coerce")

    df = pd.DataFrame(
        {
            "participant_id": df["participant_id"],
            "time_taken": time_finished - time_started,
        }
    )
    df["batch_id"] = batch_dir.name
    df["stage"] = "conscent"

    return df


def load_batch_waiting_trial(batch_dir: Path) -> pd.DataFrame:
    csv_path = find_waiting_trial_csv(batch_dir)
    df = pd.read_csv(csv_path)

    grouped = (
        df.groupby("participant_id", as_index=False)["time_taken"]
        .sum()
    )
    grouped["batch_id"] = batch_dir.name
    grouped["stage"] = "waiting"

    return grouped


def load_batch_game_trial(batch_dir: Path) -> pd.DataFrame:
    csv_path = find_game_trial_csv(batch_dir)
    df = pd.read_csv(csv_path)

    grouped = (
        df.groupby("participant_id", as_index=False)["time_taken"]
        .sum()
    )
    grouped["batch_id"] = batch_dir.name
    grouped["stage"] = "game"

    return grouped


def load_batch_game_trial_row_counts(batch_dir: Path) -> pd.DataFrame:
    csv_path = find_game_trial_csv(batch_dir)
    df = pd.read_csv(csv_path, usecols=["participant_id"])

    return (
        df.groupby("participant_id", as_index=False)
        .size()
        .rename(columns={"size": "game_trial_count"})
    )


def build_waiting_trial_times_dataframe(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = data_root or get_data_root()
    batch_dirs = find_batch_dirs(root)

    if not batch_dirs:
        raise FileNotFoundError(f"No batch-n directories found in {root}")

    frames = [load_batch_waiting_trial(batch_dir) for batch_dir in batch_dirs]
    return pd.concat(frames, ignore_index=True)


def build_game_trial_times_dataframe(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = data_root or get_data_root()
    batch_dirs = find_batch_dirs(root)

    if not batch_dirs:
        raise FileNotFoundError(f"No batch-n directories found in {root}")

    frames = [load_batch_game_trial(batch_dir) for batch_dir in batch_dirs]
    return pd.concat(frames, ignore_index=True)


def build_module_times_dataframe(data_root: Optional[Path] = None) -> pd.DataFrame:
    root = data_root or get_data_root()
    batch_dirs = find_batch_dirs(root)

    if not batch_dirs:
        raise FileNotFoundError(f"No batch-n directories found in {root}")

    frames = [load_batch_module_state(batch_dir) for batch_dir in batch_dirs]
    return pd.concat(frames, ignore_index=True)


def _unique_id_to_worker_id(unique_id: str) -> str:
    return str(unique_id).split(":", 1)[0]


def load_batch_participant_worker_map(batch_dir: Path) -> Dict[int, str]:
    csv_path = find_participant_csv(batch_dir)
    df = pd.read_csv(csv_path, usecols=["id", "unique_id"])

    return {
        row["id"]: _unique_id_to_worker_id(row["unique_id"])
        for _, row in df.iterrows()
    }


def build_participant_worker_id_maps(
    data_root: Optional[Path] = None,
) -> Dict[str, Dict[int, str]]:
    root = data_root or get_data_root()
    batch_dirs = find_batch_dirs(root)

    if not batch_dirs:
        raise FileNotFoundError(f"No batch-n directories found in {root}")

    return {
        batch_dir.name: load_batch_participant_worker_map(batch_dir)
        for batch_dir in batch_dirs
    }


def add_worker_id_to_dataframe(
    df: pd.DataFrame,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    worker_maps = build_participant_worker_id_maps(data_root)
    result = df.copy()
    result["worker_id"] = result.apply(
        lambda row: worker_maps.get(row["batch_id"], {}).get(row["participant_id"]),
        axis=1,
    )
    return result


def _time_taken_to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return series


def _last_stage_reached_by_participant(df: pd.DataFrame) -> pd.DataFrame:
    stage_ranks = df[["participant_id", "stage"]].copy()
    stage_ranks["stage_rank"] = stage_ranks["stage"].map(STAGE_RANK)
    rank_to_stage = {rank: stage for stage, rank in STAGE_RANK.items()}
    return (
        stage_ranks.groupby("participant_id", as_index=False)["stage_rank"]
        .max()
        .assign(last_stage_reached=lambda x: x["stage_rank"].map(rank_to_stage))
        [["participant_id", "last_stage_reached"]]
    )


def build_participant_total_times_dataframe(
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    root = data_root or get_data_root()
    batch_dirs = find_batch_dirs(root)

    if not batch_dirs:
        raise FileNotFoundError(f"No batch-n directories found in {root}")

    batch_totals = []
    for batch_dir in batch_dirs:
        stage_frames = []
        for loader in (
            load_batch_module_state,
            load_batch_waiting_trial,
            load_batch_game_trial,
        ):
            df = loader(batch_dir)
            normalized = df[["participant_id", "time_taken", "stage"]].copy()
            normalized["time_taken"] = _time_taken_to_seconds(normalized["time_taken"])
            stage_frames.append(normalized)

        combined = pd.concat(stage_frames, ignore_index=True)
        totals = (
            combined.groupby("participant_id", as_index=False)["time_taken"]
            .sum()
        )
        totals = totals.merge(
            _last_stage_reached_by_participant(combined),
            on="participant_id",
        )
        totals = totals.merge(
            load_batch_game_trial_row_counts(batch_dir),
            on="participant_id",
            how="left",
        )
        totals["game_trial_count"] = totals["game_trial_count"].fillna(0).astype(int)
        totals["batch_id"] = batch_dir.name
        batch_totals.append(totals)

    result = pd.concat(batch_totals, ignore_index=True)
    result["time_taken"] = result["time_taken"] + TUTORIAL_TIME
    result = add_worker_id_to_dataframe(result, data_root)
    return result.sort_values(["batch_id", "participant_id"]).reset_index(drop=True)


def main() -> None:
    total_df = build_participant_total_times_dataframe()
    total_df["bonus"] = (total_df["time_taken"] * BONUS_PER_SECOND).round(2)
    print("Total time per participant:")
    print(total_df)

    wrote_back_worker_ids = [
        "69c0ae59e7e1aabc3d03cc78",
        "69cc5b991e26f316db82dd76",
        "5cab7cc86d3a6e001519d9e9",
        "678adaaa5d10296603b4b2d6",
        "629e9a5d81c583f5ebe90f05",
        "65e8cf8f4c7424fa062e54a3",
        "694332aee9accc7da4b9503e",
    ]

    returned_df = total_df[total_df["worker_id"].isin(wrote_back_worker_ids)]
    print("Bonuses for wrote back:")
    print(returned_df)

    wrote_back_df = total_df[total_df["worker_id"].isin(wrote_back_worker_ids)]

    low_game_trial_count = wrote_back_df[
        wrote_back_df["game_trial_count"] <= MIN_GAME_TRIAL_COUNT
    ]
    print("Wrote back with game_trial_count <= MIN_GAME_TRIAL_COUNT (worker_id, bonus):")
    for _, row in low_game_trial_count.iterrows():
        print(f"{row['worker_id']},{row['bonus']}")

    high_game_trial_count_worker_ids = wrote_back_df.loc[
        wrote_back_df["game_trial_count"] > MIN_GAME_TRIAL_COUNT, "worker_id"
    ].tolist()

    print("Wrote back with game_trial_count > MIN_GAME_TRIAL_COUNT (worker_id):")
    for worker_id in high_game_trial_count_worker_ids:
        print(worker_id)



if __name__ == "__main__":
    main()
