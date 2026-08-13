#!/usr/bin/env python
"""Offline SusScore defense demo: short detect / detect+remove heatmap videos.

Generates one clean (no trigger) and one triggered episode clip each.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.policies.act.modeling_act import ACTEigenCAMHelper, ACTPolicy, create_act_attention_helper
from slerobot.policies.utils import prepare_observation_for_inference
from slerobot.utils.constants import OBS_IMAGES
from slerobot.utils.io_utils import write_video
from slerobot.utils.susscore import (
    INTERVENTION_MIN_AREA,
    build_intervention_mask,
    compute_sus_map,
    mask_rgb_by_intervention,
)

logger = logging.getLogger(__name__)

M_TASK_THRESHOLD = 0.3
TAU_QUANTILE = 0.99
DEFAULT_CLIP_SECONDS = 15.0
CALIB_SUBSAMPLE = 4
DEFAULT_CALIB_EPISODES = "0,8,16"
DEFAULT_CALIB_MAX_FRAMES = 40


@dataclass
class FrameScore:
    dataset_index: int
    frame_in_episode: int
    heatmap: np.ndarray
    sus_map: np.ndarray
    score: float
    detected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline SusScore defense demo videos.")
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="zijian2022/xa3_clean123_bad2")
    parser.add_argument(
        "--dataset.root",
        dest="dataset_root",
        type=Path,
        default=None,
        help="Optional local dataset root. If set, loads from disk instead of Hub.",
    )
    parser.add_argument("--policy.path", dest="policy_path", default="zijian2022/xa3_clean123_bad2_work")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/susscore_demo"))
    parser.add_argument("--clean_episode", type=int, default=4)
    parser.add_argument("--trigger_episode", type=int, default=20)
    parser.add_argument(
        "--run_label",
        type=str,
        default=None,
        help="If set, only score --trigger_episode once and name outputs with this label.",
    )
    parser.add_argument(
        "--calib_repo_id",
        type=str,
        default=None,
        help="Repo used to calibrate tau (defaults to --dataset.repo_id).",
    )
    parser.add_argument(
        "--calib_root",
        type=Path,
        default=None,
        help="Optional local root for tau calibration dataset.",
    )
    parser.add_argument("--clip_seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--clean_episodes", type=str, default=DEFAULT_CALIB_EPISODES)
    parser.add_argument("--calib_max_frames", type=int, default=DEFAULT_CALIB_MAX_FRAMES)
    parser.add_argument("--camera", type=str, default="side", help="Camera short name, e.g. side or front.")
    parser.add_argument(
        "--cam_method",
        type=str,
        default="grad_cam_pp",
        choices=("eigen_cam", "grad_cam_pp"),
        help="CAM method: eigen_cam (fast) or grad_cam_pp (gradient-based).",
    )
    parser.add_argument(
        "--task_mask_path",
        type=Path,
        default=None,
        help="Optional union task mask (.npy). If omitted, falls back to CAM-median mask.",
    )
    parser.add_argument(
        "--full_episode",
        action="store_true",
        help="Export the full scored episode instead of a short clip.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_camera_key(image_features: dict, camera: str) -> str:
    for key in image_features:
        if key == camera or key.split(".")[-1] == camera:
            return key
    available = [k.split(".")[-1] for k in image_features]
    raise ValueError(f"Camera {camera!r} not found. Available: {available}")


def parse_episode_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


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
    return cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_vis_map(values: np.ndarray) -> np.ndarray:
    peak = float(values.max())
    if peak <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / peak).astype(np.float32)


def draw_suspect_contours(
    image: np.ndarray,
    sus_map: np.ndarray,
    tau: float,
    min_area: int = INTERVENTION_MIN_AREA,
) -> np.ndarray:
    """Outline intervention blobs (high-anomaly connected components)."""
    output = image.copy()
    intervention = build_intervention_mask(sus_map, tau, min_area=min_area)
    mask = intervention.astype(np.uint8)
    if mask.sum() == 0:
        return output
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(output, [contour], -1, (255, 64, 64), 2)
    return output


def render_score_curve_inset(
    scores: list[float],
    tau: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a live s(t) curve with a horizontal tau line as an RGB inset."""
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 36, 10, 18, 22
    plot_x0, plot_y0 = margin_l, margin_t
    plot_x1, plot_y1 = width - margin_r, height - margin_b
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    cv2.rectangle(panel, (plot_x0, plot_y0), (plot_x1, plot_y1), (235, 235, 235), -1)

    y_max = max(tau * 1.25, max(scores) * 1.15 if scores else tau * 1.25, 1e-3)
    y_min = 0.0

    def to_xy(idx: int, value: float, n: int) -> tuple[int, int]:
        if n <= 1:
            x = plot_x0
        else:
            x = plot_x0 + int(round(idx / (n - 1) * (plot_w - 1)))
        norm = (value - y_min) / (y_max - y_min)
        y = plot_y1 - int(round(np.clip(norm, 0.0, 1.0) * (plot_h - 1)))
        return x, y

    # Colors are RGB (video frames are RGB, not OpenCV BGR).
    tau_color = (40, 40, 40)
    red = (220, 40, 40)
    green = (20, 150, 40)

    tau_y = to_xy(0, tau, 2)[1]
    cv2.line(panel, (plot_x0, tau_y), (plot_x1, tau_y), tau_color, 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"tau={tau:.3f}",
        (plot_x0 + 2, max(plot_y0 + 12, tau_y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        tau_color,
        1,
        cv2.LINE_AA,
    )

    if scores:
        n = len(scores)
        for i in range(1, n):
            p0 = to_xy(i - 1, scores[i - 1], n)
            p1 = to_xy(i, scores[i], n)
            mid = 0.5 * (scores[i - 1] + scores[i])
            color = red if mid > tau else green
            cv2.line(panel, p0, p1, color, 2, cv2.LINE_AA)
        last = to_xy(n - 1, scores[-1], n)
        last_color = red if scores[-1] > tau else green
        cv2.circle(panel, last, 4, last_color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"s={scores[-1]:.3f}",
            (plot_x0 + 2, plot_y0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            last_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(panel, "s(t)", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def overlay_score_curve(
    image: np.ndarray,
    scores: list[float],
    tau: float,
    *,
    inset_w_ratio: float = 0.34,
    inset_h_ratio: float = 0.28,
    margin: int = 12,
) -> np.ndarray:
    output = image.copy()
    h, w = output.shape[:2]
    inset_w = max(160, int(round(w * inset_w_ratio)))
    inset_h = max(110, int(round(h * inset_h_ratio)))
    inset = render_score_curve_inset(scores, tau, inset_w, inset_h)
    x0 = w - inset_w - margin
    y0 = h - inset_h - margin
    # Keep the inset nearly opaque so red/green are not washed into the heatmap.
    roi = output[y0 : y0 + inset_h, x0 : x0 + inset_w]
    blended = cv2.addWeighted(roi, 0.08, inset, 0.92, 0)
    output[y0 : y0 + inset_h, x0 : x0 + inset_w] = blended
    return output


def draw_banner(image: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (_text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    banner_h = text_h + baseline + 16
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], banner_h), color, thickness=-1)
    output = cv2.addWeighted(output, 0.35, overlay, 0.65, 0)
    cv2.putText(
        output,
        label,
        (12, text_h + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    vis_map = normalize_vis_map(heatmap) if normalize else np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    return ACTEigenCAMHelper.overlay_attention_on_image(rgb, vis_map, use_rgb=True)


class SusScoreRunner:
    def __init__(self, policy: ACTPolicy, cam_key: str, device: torch.device):
        self.policy = policy
        self.cam_key = cam_key
        self.device = device
        self.helper = policy._attention_helper
        if self.helper is None:
            raise RuntimeError("Attention helper is not initialized on the policy.")

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
        return prepare_observation_for_inference(
            observation,
            self.device,
            task=item.get("task", ""),
        )

    def run_cam(self, item: dict, rgb_override: np.ndarray | None = None) -> np.ndarray:
        batch = self._build_batch(item, rgb_override=rgb_override)
        batch[OBS_IMAGES] = [batch[key] for key in self.policy.config.image_features]
        self.helper.predict_action_chunk_with_attention(batch)
        heatmap = self.helper.last_attention_maps.get(self.cam_key)
        if heatmap is None:
            raise RuntimeError(f"Missing CAM heatmap for camera key '{self.cam_key}'.")
        rgb = rgb_override if rgb_override is not None else tensor_image_to_rgb_uint8(item[self.cam_key])
        return resize_heatmap(heatmap, rgb.shape[0], rgb.shape[1])


def build_episode_indices(dataset: sLerobotDataset) -> dict[int, list[int]]:
    episode_indices: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        episode_index = dataset.hf_dataset[idx]["episode_index"].item()
        episode_indices.setdefault(episode_index, []).append(idx)
    return episode_indices


def load_task_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"Task mask must be 2D, got shape {mask.shape}")
    return (mask > 0).astype(np.float32)


def build_task_mask(
    runner: SusScoreRunner,
    dataset: sLerobotDataset,
    episode_indices: dict[int, list[int]],
    clean_episodes: list[int],
    calib_max_frames: int,
) -> np.ndarray:
    heatmaps: list[np.ndarray] = []
    for episode in clean_episodes:
        indices = episode_indices.get(episode, [])[:calib_max_frames]
        for frame_idx, dataset_index in enumerate(indices):
            if frame_idx % CALIB_SUBSAMPLE != 0:
                continue
            item = dataset[dataset_index]
            heatmaps.append(runner.run_cam(item))
    if not heatmaps:
        raise RuntimeError("Failed to collect clean heatmaps for task-mask estimation.")
    median_map = np.median(np.stack(heatmaps, axis=0), axis=0)
    return (median_map > M_TASK_THRESHOLD).astype(np.float32)


def calibrate_tau(
    runner: SusScoreRunner,
    dataset: sLerobotDataset,
    episode_indices: dict[int, list[int]],
    clean_episodes: list[int],
    task_mask: np.ndarray,
    calib_max_frames: int,
) -> float:
    scores: list[float] = []
    for episode in clean_episodes:
        for frame_idx, dataset_index in enumerate(episode_indices.get(episode, [])[:calib_max_frames]):
            if frame_idx % CALIB_SUBSAMPLE != 0:
                continue
            item = dataset[dataset_index]
            heatmap = runner.run_cam(item)
            _, score = compute_sus_map(heatmap, task_mask)
            scores.append(score)
    if not scores:
        raise RuntimeError("Failed to collect clean SusScore values for threshold calibration.")
    tau = float(np.quantile(scores, TAU_QUANTILE))
    logger.info("Calibrated tau=%.6f from %d clean frames.", tau, len(scores))
    return tau


def score_episode(
    runner: SusScoreRunner,
    dataset: sLerobotDataset,
    dataset_indices: list[int],
    task_mask: np.ndarray,
    tau: float,
) -> list[FrameScore]:
    scored_frames: list[FrameScore] = []
    for frame_in_episode, dataset_index in enumerate(dataset_indices):
        item = dataset[dataset_index]
        heatmap = runner.run_cam(item)
        sus_map, score = compute_sus_map(heatmap, task_mask)
        scored_frames.append(
            FrameScore(
                dataset_index=dataset_index,
                frame_in_episode=frame_in_episode,
                heatmap=heatmap,
                sus_map=sus_map,
                score=score,
                detected=score > tau,
            )
        )
    return scored_frames


def choose_clip_window(scored_frames: list[FrameScore], clip_frames: int, prefer_detection: bool) -> list[FrameScore]:
    if len(scored_frames) <= clip_frames:
        return scored_frames
    if prefer_detection:
        detected_frames = [frame for frame in scored_frames if frame.detected]
        if detected_frames:
            anchor = max(detected_frames, key=lambda frame: frame.score)
            anchor_idx = scored_frames.index(anchor)
            start = max(0, anchor_idx - clip_frames // 3)
            end = min(len(scored_frames), start + clip_frames)
            start = max(0, end - clip_frames)
            return scored_frames[start:end]
    start = max(0, (len(scored_frames) - clip_frames) // 2)
    return scored_frames[start : start + clip_frames]


def render_susscore_frame(
    frame: FrameScore,
    item: dict,
    cam_key: str,
    tau: float,
    score_history: list[float],
) -> np.ndarray:
    """SusScore view: full CAM heatmap + anomaly blob contours + live s(t) curve."""
    rgb = tensor_image_to_rgb_uint8(item[cam_key])
    # Show CAM everywhere (including task-mask region); SusScore only drives contours/score.
    vis = overlay_heatmap(rgb, frame.heatmap)
    vis = draw_suspect_contours(vis, frame.sus_map, tau)
    vis = overlay_score_curve(vis, score_history, tau)
    if frame.detected:
        label = "DETECT"
        color = (180, 40, 40)
    else:
        label = "CLEAN"
        color = (40, 100, 40)
    return draw_banner(vis, label, color)


def render_detect_frame(frame: FrameScore, item: dict, runner: SusScoreRunner, cam_key: str, tau: float) -> np.ndarray:
    """Detection view: raw CAM on the original frame; SusScore only marks suspect regions."""
    rgb = tensor_image_to_rgb_uint8(item[cam_key])
    vis = overlay_heatmap(rgb, frame.heatmap)
    if frame.detected:
        vis = draw_suspect_contours(vis, frame.sus_map, tau)
        label = f"DETECT  s={frame.score:.3f}  tau={tau:.3f}"
        color = (180, 40, 40)
    else:
        label = f"CLEAN  s={frame.score:.3f}  tau={tau:.3f}"
        color = (40, 100, 40)
    return draw_banner(vis, label, color)


def render_remove_frame(frame: FrameScore, item: dict, runner: SusScoreRunner, cam_key: str, tau: float) -> np.ndarray:
    """Mitigation view: masked input and CAM recomputed after removal."""
    rgb = tensor_image_to_rgb_uint8(item[cam_key])
    if frame.detected:
        intervention = build_intervention_mask(frame.sus_map, tau)
        masked_rgb = mask_rgb_by_intervention(rgb, intervention)
        heatmap = runner.run_cam(item, rgb_override=masked_rgb)
        vis = overlay_heatmap(masked_rgb, heatmap)
        label = f"REMOVE  s={frame.score:.3f}  tau={tau:.3f}"
        color = (180, 40, 40)
    else:
        vis = overlay_heatmap(rgb, frame.heatmap)
        label = f"CLEAN  s={frame.score:.3f}  tau={tau:.3f}"
        color = (40, 100, 40)
    return draw_banner(vis, label, color)


def render_clip(
    frames: list[FrameScore],
    dataset: sLerobotDataset,
    runner: SusScoreRunner,
    cam_key: str,
    tau: float,
    mode: str,
) -> list[np.ndarray]:
    rendered: list[np.ndarray] = []
    score_history: list[float] = []
    for frame in frames:
        item = dataset[frame.dataset_index]
        if mode == "detect":
            rendered.append(render_detect_frame(frame, item, runner, cam_key, tau))
        elif mode == "remove":
            rendered.append(render_remove_frame(frame, item, runner, cam_key, tau))
        elif mode == "susscore":
            score_history.append(frame.score)
            rendered.append(render_susscore_frame(frame, item, cam_key, tau, score_history))
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    return rendered


def save_clip(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output_path), frames, fps=fps)
    logger.info("Wrote %s (%d frames)", output_path, len(frames))


def generate_demo_videos(
    repo_id: str,
    policy_path: str,
    output_dir: Path,
    clean_episode: int,
    trigger_episode: int,
    clean_episodes: list[int],
    clip_seconds: float,
    device: str | None,
    calib_max_frames: int,
    camera: str,
    cam_method: str,
    task_mask_path: Path | None,
    full_episode: bool,
    dataset_root: Path | None = None,
    calib_repo_id: str | None = None,
    calib_root: Path | None = None,
    run_label: str | None = None,
) -> dict[str, Path]:
    calib_repo = calib_repo_id or repo_id
    score_episodes = [trigger_episode] if run_label else sorted({clean_episode, trigger_episode})

    score_dataset = sLerobotDataset(
        repo_id,
        root=dataset_root,
        episodes=score_episodes,
        download_videos=dataset_root is None,
    )
    score_episode_indices = build_episode_indices(score_dataset)

    if calib_repo == repo_id and calib_root == dataset_root:
        calib_dataset = score_dataset
        calib_episode_indices = score_episode_indices
        calib_episode_list = sorted(set(clean_episodes) & set(score_episodes)) or sorted(score_episodes)
    else:
        calib_episode_list = list(clean_episodes)
        calib_dataset = sLerobotDataset(
            calib_repo,
            root=calib_root,
            episodes=calib_episode_list,
            download_videos=calib_root is None,
        )
        calib_episode_indices = build_episode_indices(calib_dataset)

    policy = ACTPolicy.from_pretrained(policy_path)
    policy.config.enable_attention_visualization = True
    policy.config.attention_cam_method = cam_method
    policy.config.attention_camera = camera
    policy.config.grad_cam_edge_mean_threshold = 1.1
    policy._attention_helper = create_act_attention_helper(policy)
    policy.eval()

    if device is None:
        device = str(policy.config.device)
    torch_device = torch.device(device)
    policy.to(torch_device)

    cam_key = resolve_camera_key(policy.config.image_features, camera)
    logger.info("Using camera %s with CAM method %s.", cam_key, cam_method)
    runner = SusScoreRunner(policy, cam_key, torch_device)

    if task_mask_path is not None:
        task_mask = load_task_mask(task_mask_path)
        logger.info("Loaded SAM2 union task mask from %s (coverage %.2f%%)", task_mask_path, 100.0 * task_mask.mean())
    else:
        task_mask = build_task_mask(
            runner, calib_dataset, calib_episode_indices, calib_episode_list, calib_max_frames
        )
        logger.warning("Using CAM-median task mask fallback (not SAM2 union).")
    tau = calibrate_tau(
        runner, calib_dataset, calib_episode_indices, calib_episode_list, task_mask, calib_max_frames
    )
    clip_frames = max(1, int(round(score_dataset.fps * clip_seconds)))
    suffix = "_full" if full_episode else ""

    if run_label:
        targets = [(run_label, trigger_episode, True)]
    else:
        targets = (
            ("clean", clean_episode, False),
            ("trigger", trigger_episode, True),
        )

    outputs: dict[str, Path] = {}
    for label, episode, prefer_detection in targets:
        scored = score_episode(runner, score_dataset, score_episode_indices[episode], task_mask, tau)
        if not scored:
            raise RuntimeError(f"No scored frames for episode {episode}.")
        clip = scored if full_episode else choose_clip_window(scored, clip_frames, prefer_detection=prefer_detection)

        detect_frames = render_clip(clip, score_dataset, runner, cam_key, tau, mode="detect")
        remove_frames = render_clip(clip, score_dataset, runner, cam_key, tau, mode="remove")
        susscore_frames = render_clip(clip, score_dataset, runner, cam_key, tau, mode="susscore")

        detect_path = output_dir / f"detect_{label}{suffix}.mp4"
        remove_path = output_dir / f"detect_remove_{label}{suffix}.mp4"
        susscore_path = output_dir / f"susscore_{label}{suffix}.mp4"
        save_clip(detect_frames, detect_path, score_dataset.fps)
        save_clip(remove_frames, remove_path, score_dataset.fps)
        save_clip(susscore_frames, susscore_path, score_dataset.fps)
        outputs[f"detect_{label}{suffix}"] = detect_path
        outputs[f"detect_remove_{label}{suffix}"] = remove_path
        outputs[f"susscore_{label}{suffix}"] = susscore_path

    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    clean_episodes = parse_episode_range(args.clean_episodes)
    outputs = generate_demo_videos(
        repo_id=args.repo_id,
        policy_path=args.policy_path,
        output_dir=args.output_dir,
        clean_episode=args.clean_episode,
        trigger_episode=args.trigger_episode,
        clean_episodes=clean_episodes,
        clip_seconds=args.clip_seconds,
        device=args.device,
        calib_max_frames=args.calib_max_frames,
        camera=args.camera,
        cam_method=args.cam_method,
        task_mask_path=args.task_mask_path,
        full_episode=args.full_episode,
        dataset_root=args.dataset_root,
        calib_repo_id=args.calib_repo_id,
        calib_root=args.calib_root,
        run_label=args.run_label,
    )
    print("Generated videos:")
    for name, path in outputs.items():
        print(f"  {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
