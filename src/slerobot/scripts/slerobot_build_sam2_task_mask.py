#!/usr/bin/env python
"""Build union task mask M_task with SAM2 (interactive box on first frame)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from slerobot.datasets.slerobot_datasets import sLerobotDataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAM2 union task mask for SusScore.")
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="zijian2022/xa3_clean123_bad2")
    parser.add_argument("--camera", type=str, default="side")
    parser.add_argument("--episodes", type=str, default="0,4,8,16", help="Episodes to segment and union.")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/sam2_task_mask/M_task_side_union.npy"))
    parser.add_argument("--boxes_path", type=Path, default=Path("outputs/sam2_task_mask/boxes_side.json"))
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=Path,
        default=Path("outputs/sam2_checkpoints/sam2.1_hiera_tiny.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--reuse_boxes",
        action="store_true",
        help="Skip interactive ROI and reuse --boxes_path if it exists.",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=None,
        help="Use pre-exported JPEG frames (00000.jpg, ...) and skip dataset loading.",
    )
    parser.add_argument(
        "--segmentation_video_path",
        type=Path,
        default=None,
        help="Optional path to save per-frame segmentation overlay video.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for segmentation video output.")
    return parser.parse_args()


def parse_episodes(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def resolve_camera_key(meta_features: dict, camera: str) -> str:
    image_keys = [k for k in meta_features if "images" in k]
    for key in image_keys:
        if key == camera or key.split(".")[-1] == camera:
            return key
    available = [k.split(".")[-1] for k in image_keys]
    raise ValueError(f"Camera {camera!r} not found. Available: {available}")


def tensor_image_to_bgr_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] in (3, 4):
        arr = image.permute(1, 2, 0).numpy()
    else:
        arr = image.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def build_episode_indices(dataset: sLerobotDataset) -> dict[int, list[int]]:
    episode_indices: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        episode_index = dataset.hf_dataset[idx]["episode_index"].item()
        episode_indices.setdefault(episode_index, []).append(idx)
    return episode_indices


def export_episode_frames(
    dataset: sLerobotDataset,
    dataset_indices: list[int],
    cam_key: str,
    frames_dir: Path,
) -> tuple[int, int]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    height = width = None
    for frame_idx, dataset_index in enumerate(dataset_indices):
        item = dataset[dataset_index]
        bgr = tensor_image_to_bgr_uint8(item[cam_key])
        if height is None:
            height, width = bgr.shape[:2]
        cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), bgr)
    if height is None or width is None:
        raise RuntimeError("Episode has no frames.")
    return height, width


def select_boxes_interactive(first_frame_bgr: np.ndarray) -> list[list[float]]:
    boxes: list[list[float]] = []
    obj_names = ["arm/object 1", "arm/object 2 (optional)", "arm/object 3 (optional)"]
    print(
        "\n=== SAM2 ROI ===\n"
        "Drag a rectangle on the FIRST frame, then press ENTER/SPACE to confirm.\n"
        "Press C or ESC with empty selection to finish adding objects.\n"
    )
    preview = first_frame_bgr.copy()
    for name in obj_names:
        print(f"Select ROI for: {name}")
        x, y, w, h = cv2.selectROI(f"SAM2: {name}", preview, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(f"SAM2: {name}")
        if w <= 0 or h <= 0:
            break
        box = [float(x), float(y), float(x + w), float(y + h)]
        boxes.append(box)
        cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        print(f"  saved box: {box}")
    cv2.destroyAllWindows()
    if not boxes:
        raise RuntimeError("No ROI selected.")
    return boxes


def mask_to_bool_hw(mask_tensor, height: int, width: int) -> np.ndarray:
    mask = (mask_tensor.detach().cpu().numpy() > 0.0).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def render_segmentation_frame(bgr: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    overlay = bgr.copy()
    colored = np.zeros_like(overlay)
    colored[frame_mask.astype(bool)] = (0, 180, 255)
    return cv2.addWeighted(overlay, 0.55, colored, 0.45, 0)


def propagate_episode_union(
    predictor,
    frames_dir: Path,
    boxes: list[list[float]],
    height: int,
    width: int,
    device: str,
    segmentation_video_path: Path | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    from sam2.build_sam import build_sam2_video_predictor  # noqa: F401

    union = np.zeros((height, width), dtype=np.uint8)
    video_writer = None
    with torch.inference_mode():
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True)
        for obj_id, box in enumerate(boxes, start=1):
            box_arr = np.array(box, dtype=np.float32)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box_arr)
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            frame_mask = np.zeros((height, width), dtype=bool)
            for mask in masks:
                frame_mask |= mask_to_bool_hw(mask, height, width)
            union |= frame_mask.astype(np.uint8)
            if segmentation_video_path is not None:
                bgr = cv2.imread(str(frames_dir / f"{frame_idx:05d}.jpg"))
                if bgr is None:
                    raise RuntimeError(f"Missing frame image: {frames_dir / f'{frame_idx:05d}.jpg'}")
                vis = render_segmentation_frame(bgr, frame_mask)
                if video_writer is None:
                    segmentation_video_path.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = imageio.get_writer(
                        str(segmentation_video_path),
                        fps=fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        macro_block_size=1,
                    )
                video_writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            if frame_idx % 50 == 0:
                logger.info("  propagated frame %d/%d", frame_idx + 1, state["num_frames"])
    if video_writer is not None:
        video_writer.close()
        logger.info("Wrote segmentation video to %s", segmentation_video_path)
    return union.astype(np.float32)


def save_preview(first_frame_bgr: np.ndarray, task_mask: np.ndarray, path: Path) -> None:
    overlay = first_frame_bgr.copy()
    colored = np.zeros_like(overlay)
    colored[task_mask.astype(bool)] = (0, 180, 255)
    preview = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def load_first_frame_bgr(frames_dir: Path) -> np.ndarray:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No JPEG frames found in {frames_dir}")
    return cv2.imread(str(frame_paths[0]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    episodes = parse_episodes(args.episodes)

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    if args.frames_dir is not None:
        frames_dir = args.frames_dir
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
        first_bgr = load_first_frame_bgr(frames_dir)
        height, width = first_bgr.shape[:2]
        if args.reuse_boxes and args.boxes_path.exists():
            boxes = json.loads(args.boxes_path.read_text())
            logger.info("Reusing boxes from %s", args.boxes_path)
        else:
            boxes = select_boxes_interactive(first_bgr)
            args.boxes_path.parent.mkdir(parents=True, exist_ok=True)
            args.boxes_path.write_text(json.dumps(boxes, indent=2))
            logger.info("Saved boxes to %s", args.boxes_path)

        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            args.sam2_config,
            str(args.sam2_checkpoint),
            device=device,
        )
        logger.info("SAM2 propagate pre-exported frames (%d jpgs)", len(list(frames_dir.glob("*.jpg"))))
        union = propagate_episode_union(
            predictor,
            frames_dir,
            boxes,
            height,
            width,
            device,
            segmentation_video_path=args.segmentation_video_path,
            fps=args.fps,
        )
    else:
        dataset = sLerobotDataset(args.repo_id, episodes=episodes, download_videos=True)
        cam_key = resolve_camera_key(dataset.meta.features, args.camera)
        episode_indices = build_episode_indices(dataset)

        first_episode = episodes[0]
        first_idx = episode_indices[first_episode][0]
        first_bgr = tensor_image_to_bgr_uint8(dataset[first_idx][cam_key])

        if args.reuse_boxes and args.boxes_path.exists():
            boxes = json.loads(args.boxes_path.read_text())
            logger.info("Reusing boxes from %s", args.boxes_path)
        else:
            boxes = select_boxes_interactive(first_bgr)
            args.boxes_path.parent.mkdir(parents=True, exist_ok=True)
            args.boxes_path.write_text(json.dumps(boxes, indent=2))
            logger.info("Saved boxes to %s", args.boxes_path)

        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            args.sam2_config,
            str(args.sam2_checkpoint),
            device=device,
        )

        height, width = first_bgr.shape[:2]
        union = np.zeros((height, width), dtype=np.float32)

        for episode in episodes:
            indices = episode_indices[episode]
            logger.info("SAM2 propagate episode %d (%d frames)", episode, len(indices))
            with tempfile.TemporaryDirectory(prefix=f"sam2_ep{episode}_") as tmp:
                frames_dir = Path(tmp)
                export_episode_frames(dataset, indices, cam_key, frames_dir)
                ep_union = propagate_episode_union(
                    predictor,
                    frames_dir,
                    boxes,
                    height,
                    width,
                    device,
                    segmentation_video_path=args.segmentation_video_path if episode == episodes[-1] else None,
                    fps=args.fps,
                )
                union = np.clip(union + ep_union, 0, 1)
                logger.info("Episode %d union coverage: %.2f%%", episode, 100.0 * union.mean())

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, union)
    save_preview(first_bgr, union, args.output_path.with_suffix(".preview.jpg"))
    logger.info("Wrote task mask to %s (coverage %.2f%%)", args.output_path, 100.0 * union.mean())
    print(f"\nTask mask saved: {args.output_path.resolve()}")
    print(f"Preview: {args.output_path.with_suffix('.preview.jpg').resolve()}")


if __name__ == "__main__":
    main()
