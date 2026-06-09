#!/usr/bin/env python3
"""Participant survival analysis for the Ultimatum PsyNet export.

Run with no arguments to read data from ``data_path.txt``, print the survival
table, and write ``analysis/participant_survival.csv`` plus
``analysis/participant_survival_sankey.svg``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

def get_data_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    data_path_file = script_dir / "data_path.txt"
    return Path(data_path_file.read_text().strip())


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = get_data_root()
TABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "participant": ("participant", "Participant"),
    "sync_group": ("sync_group", "SimpleSyncGroup"),
    "participant_link_sync_group": (
        "participant_link_sync_group",
        "ParticipantLinkSyncGroup",
    ),
    "participant_link_barrier": (
        "participant_link_barrier",
        "ParticipantLinkBarrier",
    ),
    "info": ("info", "GameTrial"),
    "response": ("response", "Response"),
    "module_state": ("module_state", "ModuleState"),
    "waiting_trial": ("waiting_trial", "WaitingTrial"),
}
TIMESTAMP_KEYS = (
    "__barrier:chain_grouper__loop_start_time__loop_start_time",
    "__barrier:init_participant__loop_start_time__loop_start_time",
    "__barrier:prepare_trial__loop_start_time__loop_start_time",
    "__grid_trial_maker__trial_loop__loop_start_time__loop_start_time",
    "__barrier:instructions_barrier__loop_start_time__loop_start_time",
    "__barrier:adoption_barrier__loop_start_time__loop_start_time",
    "__barrier:finished_trial__loop_start_time__loop_start_time",
    "__personality__trial_loop__loop_start_time__loop_start_time",
    "__Waiting for trial__loop_start_time__loop_start_time",
)


def get_data_root() -> Path:
    data_path_file = SCRIPT_DIR / "data_path.txt"
    if data_path_file.is_file():
        return Path(data_path_file.read_text().strip())
    return DEFAULT_DATA_DIR


def find_batch_dirs(data_root: Path) -> list[Path]:
    pattern = re.compile(r"^batch-\d+$")
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and pattern.match(path.name)
    )


def resolve_table_path(data_dir: Path, name: str) -> Path | None:
    for stem in TABLE_ALIASES.get(name, (name,)):
        path = data_dir / f"{stem}.csv"
        if path.is_file():
            return path
    return None


def read_table(data_dir: Path, name: str, *, required: bool = True) -> list[dict[str, str]]:
    path = resolve_table_path(data_dir, name)
    if path is None:
        if required:
            raise FileNotFoundError(f"No CSV found for table {name!r} in {data_dir}")
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return value in {"t", "True", "true", "1"}


def int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def short_type(value: str | None) -> str:
    if not value:
        return ""
    return value.rsplit(".", 1)[-1]


def summarize_definition(definition: str | None) -> str:
    data = parse_json(definition, {})
    if not isinstance(data, dict):
        return ""
    if data.get("task"):
        label = str(data["task"])
        if data.get("order") is not None:
            label += f" order={data['order']}"
        if data.get("item"):
            label += f" item={data['item']!r}"
        return label
    if data.get("generation") is not None:
        options = data.get("options") or []
        return f"generation={data['generation']} options={len(options)}"
    return json.dumps(data, sort_keys=True)[:120]


def _remap_id(raw_id: int, id_offset: int) -> int:
    return id_offset + raw_id


def load_single_batch_export(
    data_dir: Path,
    *,
    participant_id_offset: int = 0,
    sync_group_id_offset: int = 0,
    info_id_offset: int = 0,
) -> dict[str, Any]:
    participants = {
        _remap_id(int(row["id"]), participant_id_offset): row
        for row in read_table(data_dir, "participant")
    }
    sync_groups = {
        _remap_id(int(row["id"]), sync_group_id_offset): row
        for row in read_table(data_dir, "sync_group", required=False)
    }

    group_members: dict[int, list[int]] = defaultdict(list)
    participant_groups: dict[int, list[int]] = defaultdict(list)
    for row in read_table(data_dir, "participant_link_sync_group", required=False):
        participant_id = _remap_id(int(row["participant_id"]), participant_id_offset)
        group_id = _remap_id(int(row["sync_group_id"]), sync_group_id_offset)
        group_members[group_id].append(participant_id)
        participant_groups[participant_id].append(group_id)

    barrier_links_by_participant: dict[int, list[dict[str, str]]] = defaultdict(list)
    barrier_links_by_group: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_table(data_dir, "participant_link_barrier", required=False):
        participant_id = _remap_id(int(row["participant_id"]), participant_id_offset)
        remapped_row = dict(row)
        remapped_row["participant_id"] = str(participant_id)
        barrier_links_by_participant[participant_id].append(remapped_row)
        for group_id in participant_groups.get(participant_id, []):
            barrier_links_by_group[group_id].append(remapped_row)

    info_by_participant: dict[int, list[dict[str, str]]] = defaultdict(list)
    info_by_id: dict[int, dict[str, str]] = {}
    for row in read_table(data_dir, "info", required=False):
        info_id = _remap_id(int(row["id"]), info_id_offset)
        remapped_row = dict(row)
        remapped_row["id"] = str(info_id)
        info_by_id[info_id] = remapped_row
        participant_id = int_or_none(row.get("participant_id"))
        if participant_id is not None:
            participant_id = _remap_id(participant_id, participant_id_offset)
            remapped_row["participant_id"] = str(participant_id)
            info_by_participant[participant_id].append(remapped_row)

    response_by_participant: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_table(data_dir, "response", required=False):
        participant_id = int_or_none(row.get("participant_id"))
        if participant_id is not None:
            participant_id = _remap_id(participant_id, participant_id_offset)
            response_by_participant[participant_id].append(row)

    for values in barrier_links_by_participant.values():
        values.sort(key=lambda row: row["arrival_time"])
    for values in barrier_links_by_group.values():
        values.sort(key=lambda row: row["arrival_time"])
    for values in info_by_participant.values():
        values.sort(key=lambda row: row["creation_time"])
    for values in response_by_participant.values():
        values.sort(key=lambda row: row["creation_time"])
    for values in group_members.values():
        values.sort()

    consent_finished: set[int] = set()
    for row in read_table(data_dir, "module_state", required=False):
        if truthy(row.get("finished")):
            consent_finished.add(
                _remap_id(int(row["participant_id"]), participant_id_offset)
            )

    waiting_pages_participants: set[int] = set()
    for row in read_table(data_dir, "waiting_trial", required=False):
        waiting_pages_participants.add(
            _remap_id(int(row["participant_id"]), participant_id_offset)
        )

    game_trial_counts: Counter[int] = Counter()
    for row in read_table(data_dir, "info", required=False):
        participant_id = int_or_none(row.get("participant_id"))
        if participant_id is not None:
            game_trial_counts[
                _remap_id(participant_id, participant_id_offset)
            ] += 1

    return {
        "participants": participants,
        "sync_groups": sync_groups,
        "group_members": dict(group_members),
        "participant_groups": dict(participant_groups),
        "barrier_links_by_participant": barrier_links_by_participant,
        "barrier_links_by_group": barrier_links_by_group,
        "info_by_participant": info_by_participant,
        "info_by_id": info_by_id,
        "response_by_participant": response_by_participant,
        "consent_finished": consent_finished,
        "waiting_pages_participants": waiting_pages_participants,
        "game_trial_counts": game_trial_counts,
    }


def _merge_exports(exports: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "participants": {},
        "sync_groups": {},
        "group_members": {},
        "participant_groups": {},
        "barrier_links_by_participant": defaultdict(list),
        "barrier_links_by_group": defaultdict(list),
        "info_by_participant": defaultdict(list),
        "info_by_id": {},
        "response_by_participant": defaultdict(list),
        "consent_finished": set(),
        "waiting_pages_participants": set(),
        "game_trial_counts": Counter(),
    }
    for export in exports:
        merged["participants"].update(export["participants"])
        merged["sync_groups"].update(export["sync_groups"])
        merged["group_members"].update(export["group_members"])
        merged["participant_groups"].update(export["participant_groups"])
        merged["info_by_id"].update(export["info_by_id"])
        merged["consent_finished"].update(export["consent_finished"])
        merged["waiting_pages_participants"].update(export["waiting_pages_participants"])
        merged["game_trial_counts"].update(export["game_trial_counts"])
        for participant_id, rows in export["barrier_links_by_participant"].items():
            merged["barrier_links_by_participant"][participant_id].extend(rows)
        for group_id, rows in export["barrier_links_by_group"].items():
            merged["barrier_links_by_group"][group_id].extend(rows)
        for participant_id, rows in export["info_by_participant"].items():
            merged["info_by_participant"][participant_id].extend(rows)
        for participant_id, rows in export["response_by_participant"].items():
            merged["response_by_participant"][participant_id].extend(rows)

    for values in merged["barrier_links_by_participant"].values():
        values.sort(key=lambda row: row["arrival_time"])
    for values in merged["barrier_links_by_group"].values():
        values.sort(key=lambda row: row["arrival_time"])
    for values in merged["info_by_participant"].values():
        values.sort(key=lambda row: row["creation_time"])
    for values in merged["response_by_participant"].values():
        values.sort(key=lambda row: row["creation_time"])

    merged["barrier_links_by_participant"] = dict(merged["barrier_links_by_participant"])
    merged["barrier_links_by_group"] = dict(merged["barrier_links_by_group"])
    merged["info_by_participant"] = dict(merged["info_by_participant"])
    merged["response_by_participant"] = dict(merged["response_by_participant"])
    return merged


def load_export(data_dir: Path) -> dict[str, Any]:
    batch_dirs = find_batch_dirs(data_dir)
    if batch_dirs:
        exports = []
        participant_offset = 0
        sync_group_offset = 0
        info_offset = 0
        for batch_dir in batch_dirs:
            batch_export = load_single_batch_export(
                batch_dir,
                participant_id_offset=participant_offset,
                sync_group_id_offset=sync_group_offset,
                info_id_offset=info_offset,
            )
            exports.append(batch_export)
            participant_offset += max(batch_export["participants"]) if batch_export["participants"] else 0
            sync_group_offset += max(batch_export["sync_groups"]) if batch_export["sync_groups"] else 0
            info_offset += max(batch_export["info_by_id"]) if batch_export["info_by_id"] else 0
        return _merge_exports(exports)

    return load_single_batch_export(data_dir)


def failed_barrier_name(reason: str) -> str | None:
    match = re.search(r"barrier:([A-Za-z0-9_]+)", reason or "")
    if match:
        return match.group(1)
    return None


def current_trial(participant: dict[str, str], info_by_id: dict[int, dict[str, str]]) -> dict[str, str] | None:
    trial_id = int_or_none(participant.get("current_trial_id"))
    if trial_id is None:
        return None
    return info_by_id.get(trial_id)


def last_trial(info_rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not info_rows:
        return None
    return info_rows[-1]


def find_barrier_link(
    participant_id: int,
    barrier_id: str,
    barrier_links_by_participant: dict[int, list[dict[str, str]]],
) -> dict[str, str] | None:
    matches = [
        row
        for row in barrier_links_by_participant.get(participant_id, [])
        if row["barrier_id"] == barrier_id
    ]
    if matches:
        return matches[-1]
    return None


def infer_missing_group_barrier(
    participant_id: int,
    group_id: int,
    export: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Infer the group barrier a participant failed to reach.

    Some PsyNet failures only say "participant timeout at barrier" or
    "manual_failure". In these cases the failed participant's last barrier is
    often the previous barrier, while groupmates subsequently arrive at the
    barrier that could not complete.
    """

    participant_links = export["barrier_links_by_participant"].get(participant_id, [])
    group_links = export["barrier_links_by_group"].get(group_id, [])
    if not participant_links:
        return (
            "chain_grouper",
            f"P{participant_id} has no barrier-link rows inside group {group_id}",
        )
    if not group_links:
        return None, None

    participant_barriers = {row["barrier_id"] for row in participant_links}
    last_link = participant_links[-1]
    cutoff = last_link["departure_time"] or last_link["arrival_time"]
    candidates = [
        row
        for row in group_links
        if int(row["participant_id"]) != participant_id
        and row["arrival_time"] >= cutoff
        and row["barrier_id"] not in participant_barriers
    ]
    if not candidates:
        participant_counts = Counter(row["barrier_id"] for row in participant_links)
        groupmate_counts: dict[int, Counter[str]] = defaultdict(Counter)
        for row in group_links:
            groupmate_counts[int(row["participant_id"])][row["barrier_id"]] += 1
        for barrier_id in (
            "init_participant",
            "prepare_trial",
            "instructions_barrier",
            "adoption_barrier",
            "finished_trial",
            "chain_grouper",
        ):
            max_groupmate_count = max(
                (
                    counts[barrier_id]
                    for other_participant_id, counts in groupmate_counts.items()
                    if other_participant_id != participant_id
                ),
                default=0,
            )
            if max_groupmate_count > participant_counts[barrier_id]:
                return (
                    barrier_id,
                    (
                        f"inferred because groupmates reached {barrier_id} "
                        f"{max_groupmate_count} time(s), while P{participant_id} "
                        f"reached it {participant_counts[barrier_id]} time(s)"
                    ),
                )
        return (
            last_link["barrier_id"],
            (
                f"fallback to P{participant_id}'s last observed barrier "
                f"{last_link['barrier_id']} at {cutoff}"
            ),
        )

    barrier_id = candidates[0]["barrier_id"]
    participant = export["participants"][participant_id]
    return (
        barrier_id,
        (
            f"inferred from groupmates reaching {barrier_id} at "
            f"{candidates[0]['arrival_time']} after P{participant_id}'s last "
            f"barrier {last_link['barrier_id']} at {cutoff}; "
            f"P{participant_id} failed at {participant.get('time_of_death')}"
        ),
    )


def failure_barrier_attribution(
    participant_id: int,
    participant: dict[str, str],
    export: dict[str, Any],
    group_id: int | None = None,
) -> tuple[str | None, str | None]:
    explicit = failed_barrier_name(participant.get("failed_reason", ""))
    if explicit:
        return explicit, "from failed_reason"
    if group_id is not None:
        inferred, evidence = infer_missing_group_barrier(participant_id, group_id, export)
        if inferred:
            return inferred, evidence
    return None, None


def infer_failure_step(
    participant_id: int,
    participant: dict[str, str],
    export: dict[str, Any],
    group_id: int | None = None,
) -> tuple[str, list[str]]:
    reason = participant.get("failed_reason", "")
    vars_ = parse_json(participant.get("vars"), {})
    info_rows = export["info_by_participant"].get(participant_id, [])
    trial = current_trial(participant, export["info_by_id"]) or last_trial(info_rows)
    barrier_id = failed_barrier_name(reason)
    evidence: list[str] = []

    if barrier_id:
        link = find_barrier_link(
            participant_id,
            barrier_id,
            export["barrier_links_by_participant"],
        )
        if link:
            evidence.append(
                "barrier "
                f"{barrier_id}: arrived {link['arrival_time']}, "
                f"departed {link['departure_time'] or '<never>'}, "
                f"released={link['released']}"
            )
        start_key = f"__barrier:{barrier_id}__loop_start_time__loop_start_time"
        if isinstance(vars_, dict) and vars_.get(start_key):
            evidence.append(f"participant loop for {barrier_id} began {vars_[start_key]}")
        return f"barrier:{barrier_id}", evidence

    if "participant timeout at barrier" in reason:
        if group_id is not None:
            barrier, barrier_evidence = failure_barrier_attribution(
                participant_id, participant, export, group_id
            )
            if barrier:
                evidence.append(barrier_evidence or "")
                if trial:
                    evidence.append(describe_trial("current/last trial", trial))
                return f"barrier:{barrier}", evidence
        if trial:
            evidence.append(describe_trial("current/last trial", trial))
            return "trial before next group barrier", evidence
        return "participant timeout at barrier", evidence

    if "personality_timeout" in reason:
        if trial:
            evidence.append(describe_trial("current/last trial", trial))
        return "personality trial timeout", evidence

    if reason == "manual_failure":
        if group_id is not None:
            barrier, barrier_evidence = failure_barrier_attribution(
                participant_id, participant, export, group_id
            )
            if barrier:
                evidence.append(barrier_evidence or "")
                if trial:
                    evidence.append(describe_trial("current/last trial", trial))
                return f"barrier:{barrier}", evidence
        if trial:
            evidence.append(describe_trial("current/last trial", trial))
        return "manual failure", evidence

    if trial:
        evidence.append(describe_trial("current/last trial", trial))
        return "current trial", evidence

    return reason or "unknown", evidence


def describe_trial(prefix: str, row: dict[str, str]) -> str:
    complete = truthy(row.get("complete"))
    failed = truthy(row.get("failed"))
    return (
        f"{prefix}: info_id={row['id']} {short_type(row.get('type'))} "
        f"at {row['creation_time']}, complete={complete}, failed={failed}, "
        f"node={row.get('node_id') or '<none>'}, {summarize_definition(row.get('definition'))}"
    )


def participant_line(
    participant_id: int,
    participant: dict[str, str],
    export: dict[str, Any],
    group_id: int | None = None,
) -> list[str]:
    failed = truthy(participant.get("failed"))
    complete = truthy(participant.get("complete"))
    lines = [
        (
            f"P{participant_id}: failed={failed}, complete={complete}, "
            f"status={participant.get('status')}, progress={participant.get('progress')}, "
            f"page_count={participant.get('page_count')}, death={participant.get('time_of_death') or '<none>'}"
        )
    ]
    if failed:
        step, evidence = infer_failure_step(participant_id, participant, export, group_id)
        lines.append(f"  reason: {participant.get('failed_reason')}")
        lines.append(f"  inferred failure step: {step}")
        if group_id is not None:
            barrier, barrier_evidence = failure_barrier_attribution(
                participant_id, participant, export, group_id
            )
            if barrier:
                lines.append(f"  attributed failure barrier: {barrier}")
                if barrier_evidence:
                    lines.append(f"  barrier attribution: {barrier_evidence}")
        for item in evidence:
            lines.append(f"  evidence: {item}")
    else:
        trial = current_trial(participant, export["info_by_id"])
        if trial and not complete:
            lines.append("  participant did not fail, but was left incomplete")
            lines.append(f"  {describe_trial('current trial', trial)}")
    return lines


def print_recent_trials(
    participant_id: int,
    export: dict[str, Any],
    count: int,
) -> None:
    rows = export["info_by_participant"].get(participant_id, [])[-count:]
    if not rows:
        print("    no trial/info rows")
        return
    for row in rows:
        answer = (row.get("answer") or "").replace("\n", " ")
        if len(answer) > 70:
            answer = answer[:67] + "..."
        print(f"    {describe_trial('trial', row)}")
        if answer:
            print(f"      answer={answer}")


def print_recent_barriers(
    group_id: int,
    export: dict[str, Any],
    count: int,
) -> None:
    rows = export["barrier_links_by_group"].get(group_id, [])[-count:]
    if not rows:
        print("    no barrier-link rows")
        return
    for row in rows:
        print(
            "    "
            f"{row['arrival_time']} {row['barrier_id']} P{row['participant_id']} "
            f"departed={row['departure_time'] or '<never>'} released={row['released']}"
        )


def group_failure_barrier_distribution(
    group_id: int,
    responsible: list[int],
    export: dict[str, Any],
) -> tuple[Counter[str], dict[int, tuple[str | None, str | None]]]:
    participants = export["participants"]
    attribution = {
        participant_id: failure_barrier_attribution(
            participant_id,
            participants[participant_id],
            export,
            group_id,
        )
        for participant_id in responsible
    }
    distribution = Counter(
        barrier or "<unknown>"
        for barrier, _evidence in attribution.values()
    )
    return distribution, attribution


def failed_group_ids(export: dict[str, Any]) -> list[int]:
    """Return formed groups that are closed and empty."""

    return sorted(
        group_id
        for group_id, group in export["sync_groups"].items()
        if not truthy(group.get("active"))
        and int_or_none(group.get("n_active_participants")) == 0
    )


def responsible_participants(group_id: int, export: dict[str, Any]) -> list[int]:
    participants = export["participants"]
    responsible = [
        participant_id
        for participant_id in export["group_members"][group_id]
        if truthy(participants[participant_id].get("failed"))
        or not truthy(participants[participant_id].get("complete"))
    ]
    if responsible:
        return responsible
    return list(export["group_members"][group_id])


def print_group_report(export: dict[str, Any], data_dir: Path, recent: int) -> None:
    participants = export["participants"]
    sync_groups = export["sync_groups"]
    group_members = export["group_members"]

    detected_failed_group_ids = failed_group_ids(export)
    by_barrier_groups: dict[str, set[int]] = defaultdict(set)
    by_barrier_participants: Counter[str] = Counter()
    for group_id in detected_failed_group_ids:
        responsible = responsible_participants(group_id, export)
        barrier_distribution, _barrier_attribution = group_failure_barrier_distribution(
            group_id, responsible, export
        )
        for barrier, count in barrier_distribution.items():
            by_barrier_groups[barrier].add(group_id)
            by_barrier_participants[barrier] += count

    print("Failure analysis")
    print("================")
    print(f"Data directory: {data_dir}")
    print(f"Participants: {len(participants)}")
    print(f"Formed sync groups: {len(sync_groups)}")
    print("Failed group detection: sync_group.active=f and n_active_participants=0")
    print(f"Failed formed sync groups: {len(detected_failed_group_ids)}")
    print()
    print("barrier,failed_groups,implicated_participants")
    for barrier in sorted(
        by_barrier_participants,
        key=lambda item: (-len(by_barrier_groups[item]), item),
    ):
        print(
            f"{barrier},"
            f"{len(by_barrier_groups[barrier])},"
            f"{by_barrier_participants[barrier]}"
        )


def participant_barrier_release_counts(
    participant_id: int,
    export: dict[str, Any],
) -> Counter[str]:
    return Counter(
        row["barrier_id"]
        for row in export["barrier_links_by_participant"].get(participant_id, [])
        if truthy(row.get("released"))
    )


def participant_started_final_personality(
    participant_id: int,
    export: dict[str, Any],
) -> bool:
    return any(
        short_type(row.get("type")) == "PersonalityTrial"
        and row.get("trial_maker_id") == "personality"
        for row in export["info_by_participant"].get(participant_id, [])
    )


def participant_completed_required_personality(
    participant_id: int,
    export: dict[str, Any],
) -> bool:
    completed_items = set()
    for row in export["info_by_participant"].get(participant_id, []):
        if short_type(row.get("type")) != "PersonalityTrial" or not truthy(row.get("complete")):
            continue
        definition = parse_json(row.get("definition"), {})
        task = definition.get("task")
        if task in {"dat", "big_five_short"}:
            completed_items.add((task, definition.get("order")))
    return len(completed_items) >= 11


def participant_response_questions(participant_id: int, export: dict[str, Any]) -> set[str]:
    return {
        row.get("question", "")
        for row in export["response_by_participant"].get(participant_id, [])
    }


def participant_completed_game_trials(participant_id: int, export: dict[str, Any]) -> int:
    return sum(
        1
        for row in export["info_by_participant"].get(participant_id, [])
        if short_type(row.get("type")) in {"GameTrial", "GridCreateTrial"}
        and truthy(row.get("complete"))
    )


def participant_completed_grid_trials(participant_id: int, export: dict[str, Any]) -> int:
    return participant_completed_game_trials(participant_id, export)


def participant_reached_successful_end(participant_id: int, export: dict[str, Any]) -> bool:
    return "SuccessfulEndLogic" in participant_response_questions(participant_id, export)


def participant_reached_conscent(participant_id: int, export: dict[str, Any]) -> bool:
    if participant_id in export.get("consent_finished", set()):
        return True
    return "consent_choice" in participant_response_questions(participant_id, export)


def participant_reached_waiting_pages(participant_id: int, export: dict[str, Any]) -> bool:
    return participant_id in export.get("waiting_pages_participants", set())


def participant_game_trial_count(participant_id: int, export: dict[str, Any]) -> int:
    return export.get("game_trial_counts", Counter()).get(participant_id, 0)


def max_observed_game_trial_count(export: dict[str, Any]) -> int:
    counts = export.get("game_trial_counts", Counter())
    if not counts:
        return 0
    return max(counts.values())


def survival_steps(export: dict[str, Any]) -> list[tuple[str, Callable[[int], bool]]]:
    steps: list[tuple[str, Callable[[int], bool]]] = [
        ("participant_created", lambda _pid: True),
        ("conscent", lambda pid: participant_reached_conscent(pid, export)),
        ("waitingPages", lambda pid: participant_reached_waiting_pages(pid, export)),
    ]

    for trial_count in range(1, max_observed_game_trial_count(export) + 1):
        steps.append(
            (
                f"game_game_trial_count_{trial_count:02d}",
                lambda pid, count=trial_count: participant_game_trial_count(pid, export)
                >= count,
            )
        )

    return steps


def survival_statistics(export: dict[str, Any]) -> list[dict[str, Any]]:
    participants = export["participants"]
    total = len(participants)
    survivors = set(participants)
    rows = []
    previous_count = total
    for step, predicate in survival_steps(export):
        survivors = {participant_id for participant_id in survivors if predicate(participant_id)}
        count = len(survivors)
        dropped = previous_count - count
        survival_rate = count / total if total else 0
        conditional_survival = count / previous_count if previous_count else 0
        rows.append(
            {
                "step": step,
                "participants_survived": count,
                "dropped_since_previous": dropped,
                "survival_rate": survival_rate,
                "conditional_survival": conditional_survival,
            }
        )
        previous_count = count
    return rows


def print_survival_report(export: dict[str, Any], data_dir: Path) -> None:
    participants = export["participants"]
    total = len(participants)

    print("Participant Survival")
    print("====================")
    print(f"Data directory: {data_dir}")
    print(f"Participants: {total}")
    print()
    print(
        "step,participants_survived,dropped_since_previous,"
        "survival_rate,conditional_survival"
    )

    for row in survival_statistics(export):
        print(
            f"{row['step']},{row['participants_survived']},"
            f"{row['dropped_since_previous']},"
            f"{row['survival_rate']:.4f},{row['conditional_survival']:.4f}"
        )


def compress_survival_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) <= 2:
        return rows

    keep_indices = {0, len(rows) - 1}
    game_trial_milestone = re.compile(r"^game_game_trial_count_(\d+)$")
    for idx, row in enumerate(rows):
        if row["step"] in {"conscent", "waitingPages"}:
            keep_indices.add(idx)
        if row["dropped_since_previous"] > 0:
            keep_indices.add(idx)
            continue
        match = game_trial_milestone.match(row["step"])
        if match and int(match.group(1)) % 5 == 0:
            keep_indices.add(idx)

    compressed: list[dict[str, Any]] = []
    for idx in sorted(keep_indices):
        row = dict(rows[idx])
        if idx > 0 and (idx - 1) not in keep_indices:
            previous_kept = max(i for i in keep_indices if i < idx)
            row["dropped_since_previous"] = (
                rows[previous_kept]["participants_survived"] - row["participants_survived"]
            )
            previous_count = rows[previous_kept]["participants_survived"]
            row["conditional_survival"] = (
                row["participants_survived"] / previous_count if previous_count else 0
            )
        compressed.append(row)
    return compressed


def short_step_label(step: str) -> str:
    if step == "participant_created":
        return "created"
    if step == "conscent":
        return "conscent"
    if step == "waitingPages":
        return "waitingPages"
    match = re.fullmatch(r"game_game_trial_count_(\d+)", step)
    if match:
        return f"game (trial {int(match.group(1))})"
    return step.replace("_", " ")


def sankey_path(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> str:
    dy = (y2 - y1) * 0.5
    return f"M {x1:.1f} {y1:.1f} C {x1:.1f} {y1 + dy:.1f}, {x2:.1f} {y2 - dy:.1f}, {x2:.1f} {y2:.1f}"


def write_survival_sankey(
    export: dict[str, Any],
    output_path: Path,
    *,
    compress: bool = True,
) -> None:
    rows = survival_statistics(export)
    if not rows:
        raise ValueError("No survival rows to plot.")
    if compress:
        rows = compress_survival_rows(rows)

    total = rows[0]["participants_survived"]
    left = 260
    top = 95
    y_gap = max(28, min(76, int(2600 / max(len(rows) - 1, 1))))
    survivor_width = 360
    drop_left = 690
    node_height = 12
    width = 980
    height = top * 2 + y_gap * (len(rows) - 1) + 105
    scale = survivor_width / total if total else 1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #222; }",
        ".small { font-size: 11px; }",
        ".label { font-size: 12px; font-weight: 600; }",
        ".title { font-size: 22px; font-weight: 700; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="40" y="38">Participant Survival by Experiment Stage</text>',
        (
            f'<text class="small" x="40" y="60">N={total} across all batches. '
            "Stages: conscent, waitingPages, game (by game_trial_count). "
            "Blue flows are survivors; orange branches are drop-offs."
            + (
                " Compressed to stages with attrition or every 5th game trial."
                if compress
                else ""
            )
            + "</text>"
        ),
    ]

    # Draw flows first so nodes and labels sit on top.
    for idx in range(1, len(rows)):
        previous = rows[idx - 1]
        current = rows[idx]
        y1 = top + (idx - 1) * y_gap + node_height
        y2 = top + idx * y_gap
        previous_count = previous["participants_survived"]
        current_count = current["participants_survived"]
        dropped = current["dropped_since_previous"]

        if current_count:
            stroke_width = max(current_count * scale, 1.0)
            x1 = left + stroke_width / 2
            x2 = left + stroke_width / 2
            lines.append(
                f'<path d="{sankey_path(x1, y1, x2, y2)}" '
                f'stroke="#4C78A8" stroke-width="{stroke_width:.2f}" '
                'fill="none" opacity="0.62" stroke-linecap="butt"/>'
            )

        if dropped:
            stroke_width = max(dropped * scale, 1.0)
            survivor_stroke_width = max(current_count * scale, 0)
            x1 = left + survivor_stroke_width + stroke_width / 2
            x2 = drop_left + stroke_width / 2
            lines.append(
                f'<path d="{sankey_path(x1, y1, x2, y2)}" '
                f'stroke="#F58518" stroke-width="{stroke_width:.2f}" '
                'fill="none" opacity="0.68" stroke-linecap="butt"/>'
            )

    for idx, row in enumerate(rows):
        y = top + idx * y_gap
        count = row["participants_survived"]
        dropped = row["dropped_since_previous"]
        survivor_node_width = max(count * scale, 1.0)
        lines.append(
            f'<rect x="{left}" y="{y}" width="{survivor_node_width:.2f}" height="{node_height}" '
            'fill="#1F77B4" opacity="0.9"/>'
        )
        if dropped:
            drop_width = max(dropped * scale, 1.0)
            previous_count = rows[idx - 1]["participants_survived"] if idx else total
            dropped_percent = dropped / previous_count if previous_count else 0
            lines.append(
                f'<rect x="{drop_left}" y="{y}" width="{drop_width:.2f}" height="{node_height}" '
                'fill="#E4572E" opacity="0.9"/>'
            )
            lines.append(
                f'<text class="small" x="{drop_left + drop_width + 6:.1f}" y="{y + node_height:.1f}">'
                f'-{dropped} ({dropped_percent:.1%})</text>'
            )

        escaped_label = html.escape(short_step_label(row["step"]))
        survived_percent = count / total if total else 0
        lines.append(
            f'<text class="label" x="{left - 14}" y="{y + node_height:.1f}" '
            f'text-anchor="end">{escaped_label}</text>'
        )
        lines.append(
            f'<text class="small" x="{left + survivor_node_width + 6:.1f}" y="{y + node_height:.1f}">'
            f'{count} ({survived_percent:.1%})</text>'
        )

    lines.extend(
        [
            f'<rect x="40" y="{height - 45}" width="16" height="10" fill="#1F77B4" opacity="0.9"/>',
            f'<text class="small" x="62" y="{height - 36}">survived to milestone</text>',
            f'<rect x="220" y="{height - 45}" width="16" height="10" fill="#E4572E" opacity="0.9"/>',
            f'<text class="small" x="242" y="{height - 36}">dropped before milestone</text>',
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_survival_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "participants_survived",
        "dropped_since_previous",
        "survival_rate",
        "conditional_survival",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain participant and sync-group failures in a PsyNet CSV export."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing PsyNet CSV exports, or a parent directory with "
            "batch-* subdirectories. Defaults to data_path.txt when present."
        ),
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=5,
        help="Number of recent barrier/trial rows to print for each participant/group.",
    )
    parser.add_argument(
        "--report",
        choices=("survival", "barriers", "sankey", "all"),
        default="sankey",
        help=(
            "Which report to produce (default: sankey prints survival stats and "
            "writes analysis/participant_survival.csv plus the Sankey SVG)."
        ),
    )
    parser.add_argument(
        "--sankey-output",
        type=Path,
        default=Path("analysis/participant_survival_sankey.svg"),
        help="Path where the Sankey SVG should be written.",
    )
    parser.add_argument(
        "--survival-csv-output",
        type=Path,
        default=Path("analysis/participant_survival.csv"),
        help="Path where the full survival table should be written.",
    )
    parser.add_argument(
        "--full-sankey",
        action="store_true",
        help="Include every milestone in the Sankey instead of compressing quiet stages.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or get_data_root()
    export = load_export(data_dir)
    survival_rows = survival_statistics(export)
    if args.report in ("survival", "sankey", "all"):
        print_survival_report(export, data_dir)
    if args.report == "all":
        print()
    if args.report in ("barriers", "all"):
        print_group_report(export, data_dir, args.recent)
    if args.report in ("sankey", "all"):
        write_survival_csv(survival_rows, args.survival_csv_output)
        write_survival_sankey(
            export,
            args.sankey_output,
            compress=not args.full_sankey,
        )
        print(f"Wrote survival table to {args.survival_csv_output}")
        print(f"Wrote Sankey plot to {args.sankey_output}")


if __name__ == "__main__":
    main()