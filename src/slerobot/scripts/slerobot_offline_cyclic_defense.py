#!/usr/bin/env python
"""Offline cyclic SusScore defense experiment (C = n_action_steps).

Compares three action trajectories on dataset replay:
  - baseline: no masking, normal select_action queue
  - viz_only: same actions as baseline (detection logged only)
  - cyclic: detect at each cycle step0; on trigger mask current cycle + next step0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.policies.act.modeling_act import ACTPolicy, create_act_attention_helper
from slerobot.policies.utils import prepare_observation_for_inference
from slerobot.utils.constants import OBS_IMAGES
from slerobot.utils.susscore import build_intervention_mask, compute_sus_map, mask_rgb_by_intervention

logger = logging.getLogger(__name__)

TAU_QUANTILE = 0.99
DEFAULT_CALIB_EPISODES = "0,8,16"
CALIB_SUBSAMPLE = 4
DEFAULT_CALIB_MAX_FRAMES = 40


@dataclass
class StepLog:
    frame_in_episode: int
    cycle_index: int
    cycle_step: int
    queue_refilled: bool
    score: float | None
    detected: bool
    intervening: bool
    bridged_step0: bool


@dataclass
class RunResult:
    mode: str
    actions: np.ndarray
    detections: int
    intervention_steps: int
    logs: list[StepLog] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline cyclic SusScore defense experiment.")
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="zijian2022/xa3_clean123_bad2")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None)
    parser.add_argument("--policy.path", dest="policy_path", default="zijian2022/xa3_clean123_bad2_work")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/susscore_cyclic_defense"))
    parser.add_argument("--episode", type=int, default=20)
    parser.add_argument("--clean_episodes", type=str, default=DEFAULT_CALIB_EPISODES)
    parser.add_argument("--calib_repo_id", type=str, default=None)
    parser.add_argument("--calib_root", type=Path, default=None)
    parser.add_argument("--calib_max_frames", type=int, default=DEFAULT_CALIB_MAX_FRAMES)
    parser.add_argument("--camera", type=str, default="side")
    parser.add_argument("--cam_method", type=str, default="grad_cam_pp", choices=("eigen_cam", "grad_cam_pp"))
    parser.add_argument("--task_mask_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def parse_episode_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def resolve_camera_key(image_features: dict, camera: str) -> str:
    for key in image_features:
        if key == camera or key.split(".")[-1] == camera:
            return key
    available = [k.split(".")[-1] for k in image_features]
    raise ValueError(f"Camera {camera!r} not found. Available: {available}")


def tensor_image_to_rgb_uint8(image: Tensor) -> np.ndarray:
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] in (3, 4):
        arr = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        arr = image.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def resize_heatmap(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def load_task_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    return (mask > 0).astype(np.float32)


class EpisodeRunner:
    def __init__(self, policy: ACTPolicy, cam_key: str, device: torch.device, cycle_size: int):
        self.policy = policy
        self.cam_key = cam_key
        self.device = device
        self.cycle_size = cycle_size
        self.helper = policy._attention_helper
        if self.helper is None:
            raise RuntimeError("Policy attention helper is not initialized.")

    def reset_policy(self) -> None:
        self.policy.reset()
        self.helper.clear_warning_masks()

    def flush_action_queue(self) -> None:
        self.policy._action_queue.clear()

    def queue_empty(self) -> bool:
        return len(self.policy._action_queue) == 0

    def _build_batch(self, item: dict, rgb_override: np.ndarray | None = None) -> dict[str, Tensor]:
        observation: dict[str, np.ndarray] = {}
        for key, value in item.items():
            if not key.startswith("observation."):
                continue
            if key in self.policy.config.image_features:
                if rgb_override is not None and key == self.cam_key:
                    observation[key] = rgb_override
                else:
                    observation[key] = tensor_image_to_rgb_uint8(value)
            else:
                tensor_value = value if isinstance(value, Tensor) else torch.as_tensor(value)
                observation[key] = tensor_value.detach().cpu().numpy()
        batch = prepare_observation_for_inference(
            observation,
            self.device,
            task=item.get("task", ""),
        )
        batch[OBS_IMAGES] = [batch[key] for key in self.policy.config.image_features]
        return batch

    def run_cam(self, item: dict, rgb_override: np.ndarray | None = None) -> np.ndarray:
        batch = self._build_batch(item, rgb_override=rgb_override)
        self.helper.predict_action_chunk_with_attention(batch)
        heatmap = self.helper.last_attention_maps.get(self.cam_key)
        if heatmap is None:
            raise RuntimeError(f"Missing CAM for {self.cam_key}")
        rgb = rgb_override if rgb_override is not None else tensor_image_to_rgb_uint8(item[self.cam_key])
        return resize_heatmap(heatmap, rgb.shape[0], rgb.shape[1])

    def select_action(self, item: dict, rgb_override: np.ndarray | None = None) -> np.ndarray:
        batch = self._build_batch(item, rgb_override=rgb_override)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return action.detach().cpu().numpy().reshape(-1)


def build_episode_indices(dataset: sLerobotDataset) -> dict[int, list[int]]:
    episode_indices: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        episode_index = dataset.hf_dataset[idx]["episode_index"].item()
        episode_indices.setdefault(episode_index, []).append(idx)
    return episode_indices


def calibrate_tau(
    runner: EpisodeRunner,
    dataset: sLerobotDataset,
    episode_indices: dict[int, list[int]],
    clean_episodes: list[int],
    task_mask: np.ndarray,
    calib_max_frames: int,
    cycle_size: int,
) -> float:
    scores: list[float] = []
    for episode in clean_episodes:
        runner.reset_policy()
        indices = episode_indices.get(episode, [])[:calib_max_frames]
        for frame_idx, dataset_index in enumerate(indices):
            item = dataset[dataset_index]
            if not runner.queue_empty():
                runner.select_action(item)
                continue
            if frame_idx % CALIB_SUBSAMPLE != 0:
                runner.select_action(item)
                continue
            heatmap = runner.run_cam(item)
            _, score = compute_sus_map(heatmap, task_mask)
            scores.append(score)
            runner.select_action(item)
    if not scores:
        raise RuntimeError("Failed to calibrate tau.")
    tau = float(np.quantile(scores, TAU_QUANTILE))
    logger.info("Calibrated tau=%.6f from %d cycle-start samples.", tau, len(scores))
    return tau


def run_episode(
    mode: str,
    runner: EpisodeRunner,
    dataset: sLerobotDataset,
    dataset_indices: list[int],
    task_mask: np.ndarray,
    tau: float,
) -> RunResult:
    runner.reset_policy()
    actions: list[np.ndarray] = []
    logs: list[StepLog] = []

    pending_mask: np.ndarray | None = None
    intervene_active = False
    bridge_next_step0 = False

    detections = 0
    intervention_steps = 0
    cycle_index = 0
    cycle_step = 0

    for frame_in_episode, dataset_index in enumerate(dataset_indices):
        item = dataset[dataset_index]
        rgb_raw = tensor_image_to_rgb_uint8(item[runner.cam_key])
        queue_refilled = runner.queue_empty()

        if queue_refilled:
            cycle_step = 0
        else:
            cycle_step += 1

        score: float | None = None
        detected = False
        bridged_step0 = False

        if queue_refilled:
            rgb_detect = rgb_raw
            if mode == "cyclic" and bridge_next_step0 and pending_mask is not None:
                rgb_detect = mask_rgb_by_intervention(rgb_raw, pending_mask)
                bridged_step0 = True
                bridge_next_step0 = False

            heatmap = runner.run_cam(item, rgb_override=rgb_detect)
            sus_map, score = compute_sus_map(heatmap, task_mask)
            detected = score > tau

            if mode == "cyclic" and detected:
                pending_mask = build_intervention_mask(sus_map, tau)
                intervene_active = True
                bridge_next_step0 = True
                runner.flush_action_queue()
                detections += 1
            elif mode == "cyclic" and intervene_active and not detected:
                intervene_active = False
                pending_mask = None
            elif mode in ("baseline", "viz_only") and detected:
                detections += 1

        use_mask = mode == "cyclic" and intervene_active and pending_mask is not None
        rgb_policy = mask_rgb_by_intervention(rgb_raw, pending_mask) if use_mask else rgb_raw
        if use_mask:
            intervention_steps += 1

        action = runner.select_action(item, rgb_override=rgb_policy if use_mask else None)
        actions.append(action)

        logs.append(
            StepLog(
                frame_in_episode=frame_in_episode,
                cycle_index=cycle_index,
                cycle_step=cycle_step,
                queue_refilled=queue_refilled,
                score=score,
                detected=detected,
                intervening=use_mask,
                bridged_step0=bridged_step0,
            )
        )

        if queue_refilled:
            cycle_index += 1

    return RunResult(
        mode=mode,
        actions=np.stack(actions, axis=0),
        detections=detections,
        intervention_steps=intervention_steps,
        logs=logs,
    )


def summarize_results(
    baseline: RunResult,
    viz_only: RunResult,
    cyclic: RunResult,
    cycle_size: int,
    tau: float,
) -> dict:
    viz_action_diff = float(np.linalg.norm(viz_only.actions - baseline.actions))
    cyclic_action_diff = float(np.linalg.norm(cyclic.actions - baseline.actions))
    per_step_l2 = np.linalg.norm(cyclic.actions - baseline.actions, axis=1)
    return {
        "cycle_size_n_action_steps": cycle_size,
        "tau": tau,
        "num_frames": int(baseline.actions.shape[0]),
        "baseline_detections_at_cycle_start": baseline.detections,
        "viz_only_detections_at_cycle_start": viz_only.detections,
        "cyclic_detections_at_cycle_start": cyclic.detections,
        "cyclic_intervention_steps": cyclic.intervention_steps,
        "viz_vs_baseline_action_l2": viz_action_diff,
        "cyclic_vs_baseline_action_l2": cyclic_action_diff,
        "cyclic_vs_baseline_mean_step_l2": float(per_step_l2.mean()),
        "cyclic_vs_baseline_max_step_l2": float(per_step_l2.max()),
        "changed_action_steps": int(np.sum(per_step_l2 > 1e-5)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_episodes = parse_episode_range(args.clean_episodes)
    calib_episodes = sorted(set(clean_episodes))
    calib_repo = args.calib_repo_id or args.repo_id

    dataset = sLerobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        download_videos=args.dataset_root is None,
    )
    episode_indices = build_episode_indices(dataset)
    indices = episode_indices[args.episode]

    if calib_repo == args.repo_id and args.calib_root == args.dataset_root:
        calib_dataset = dataset
        calib_episode_indices = episode_indices
        # If scoring set has no calib episodes, fall back to scoring episode itself.
        if not set(calib_episodes) & set(episode_indices):
            calib_episodes = [args.episode]
    else:
        calib_dataset = sLerobotDataset(
            calib_repo,
            root=args.calib_root,
            episodes=calib_episodes,
            download_videos=args.calib_root is None,
        )
        calib_episode_indices = build_episode_indices(calib_dataset)

    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.config.enable_attention_visualization = True
    policy.config.attention_cam_method = args.cam_method
    policy.config.attention_camera = args.camera
    policy.config.grad_cam_edge_mean_threshold = 1.1
    policy._attention_helper = create_act_attention_helper(policy)
    policy.eval()

    device = torch.device(args.device or policy.config.device)
    policy.to(device)

    cam_key = resolve_camera_key(policy.config.image_features, args.camera)
    cycle_size = policy.config.n_action_steps
    logger.info("C = n_action_steps = %d, camera=%s", cycle_size, cam_key)

    task_mask = load_task_mask(args.task_mask_path)
    logger.info("Task mask coverage %.2f%%", 100.0 * task_mask.mean())

    runner = EpisodeRunner(policy, cam_key, device, cycle_size)
    tau = calibrate_tau(
        runner,
        calib_dataset,
        calib_episode_indices,
        calib_episodes,
        task_mask,
        args.calib_max_frames,
        cycle_size,
    )

    baseline = run_episode("baseline", runner, dataset, indices, task_mask, tau)
    viz_only = run_episode("viz_only", runner, dataset, indices, task_mask, tau)
    cyclic = run_episode("cyclic", runner, dataset, indices, task_mask, tau)

    prefix = f"ep{args.episode}"
    np.save(args.output_dir / f"{prefix}_baseline_actions.npy", baseline.actions)
    np.save(args.output_dir / f"{prefix}_viz_only_actions.npy", viz_only.actions)
    np.save(args.output_dir / f"{prefix}_cyclic_actions.npy", cyclic.actions)

    summary = summarize_results(baseline, viz_only, cyclic, cycle_size, tau)
    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved actions and summary under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
