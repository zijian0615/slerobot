"""Persist and increment dataset repo_id index between UI sessions."""

from __future__ import annotations

import json
import re
from pathlib import Path

MODES = ("attack", "detect", "mitigation")

STATE_PATH = Path(__file__).resolve().parent / "session_state.json"


def _extract_eval_index(repo_id: str) -> int | None:
    match = re.search(r"eval(\d+)", repo_id, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def format_repo_id(template: str, index: int) -> str:
    if "{index}" in template:
        return template.format(index=index)
    return template


def _template_for_mode(cfg: dict, mode: str) -> str:
    key = f"repo_id_{mode}_template"
    if key in cfg:
        return cfg[key]
    legacy = {
        "detect": cfg.get("repo_id_defense_template", cfg.get("repo_id_defense")),
        "mitigation": cfg.get("repo_id_mitigation_template"),
    }
    fallback = legacy.get(mode) or cfg.get(f"repo_id_{mode}", f"zijian2022/eval{{index}}_xa4_125_{mode}")
    return fallback


def load_state(cfg: dict) -> dict:
    if STATE_PATH.is_file():
        with STATE_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    start = cfg.get("repo_index_start")
    if start is None:
        for key in ("repo_id_attack", "repo_id_detect", "repo_id_defense", "repo_id_mitigation"):
            if key in cfg:
                parsed = _extract_eval_index(cfg[key])
                if parsed is not None:
                    start = parsed
                    break
    if start is None:
        start = 1

    return {"next_repo_index": int(start)}


def save_state(state: dict) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def peek_next_repo_ids(cfg: dict) -> tuple[int, dict[str, str]]:
    state = load_state(cfg)
    index = int(state["next_repo_index"])
    return index, {mode: format_repo_id(_template_for_mode(cfg, mode), index) for mode in MODES}


def allocate_repo_index(cfg: dict) -> tuple[int, dict[str, str]]:
    """Increment shared index; return (index, {attack, detect, mitigation} repo ids for this run)."""
    state = load_state(cfg)
    index = int(state["next_repo_index"])
    state["next_repo_index"] = index + 1
    save_state(state)
    repos = {mode: format_repo_id(_template_for_mode(cfg, mode), index) for mode in MODES}
    return index, repos
