"""
OpenCV 相机测试脚本
===================
为每个相机连接、读取一帧并保存为 PNG。

用法：
    # 自动扫描本机所有可用相机，各拍一张
    python src/slerobot/test/test_cameras.py

    # 指定设备 index
    python src/slerobot/test/test_cameras.py --indices 0,1

    # 与 xarm_record 相同的 cameras 配置格式
    python src/slerobot/test/test_cameras.py \
        --cameras '{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 20}, "side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 20}}'

    # 指定输出目录
    python src/slerobot/test/test_cameras.py --indices 0,1 --output-dir ./camera_snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from slerobot.cameras.opencv.camera_opencv import OpenCVCamera
from slerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from slerobot.cameras.utils import make_cameras_from_configs


def _parse_indices(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("indices 不能为空，例如: 0,1")
    return [int(p) for p in parts]


def _parse_cameras_json(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        raise ValueError("cameras 必须是非空 JSON 对象")
    return data


def _build_cameras_from_indices(
    indices: list[int],
    width: int | None,
    height: int | None,
    fps: int | None,
) -> dict[str, OpenCVCamera]:
    configs = {
        f"cam_{idx}": OpenCVCameraConfig(
            index_or_path=idx,
            width=width,
            height=height,
            fps=fps,
        )
        for idx in indices
    }
    return make_cameras_from_configs(configs)


def _scan_and_build_cameras(
    width: int | None,
    height: int | None,
    fps: int | None,
) -> dict[str, OpenCVCamera]:
    found = OpenCVCamera.find_cameras()
    if not found:
        raise RuntimeError("未检测到可用 OpenCV 相机。请检查 USB 连接或手动指定 --indices。")

    print(f"扫描到 {len(found)} 个相机：")
    for info in found:
        profile = info.get("default_stream_profile", {})
        print(
            f"  - id={info['id']}  backend={info.get('backend_api')}  "
            f"{profile.get('width')}x{profile.get('height')} @ {profile.get('fps')} fps"
        )

    configs = {}
    for info in found:
        cam_id = info["id"]
        name = f"cam_{cam_id}".replace("/", "_")
        configs[name] = OpenCVCameraConfig(
            index_or_path=cam_id,
            width=width,
            height=height,
            fps=fps,
        )
    return make_cameras_from_configs(configs)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 and frame.shape[2] == 3 else frame
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"保存失败: {path}")


def capture_one_photo_per_camera(
    cameras: dict[str, OpenCVCamera],
    output_dir: Path,
    warmup_frames: int,
) -> list[Path]:
    saved: list[Path] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, camera in cameras.items():
        print(f"\n[{name}] 连接中 …")
        camera.connect()

        # 丢弃前几帧，等待曝光/白平衡稳定
        frame = None
        for i in range(max(warmup_frames, 1)):
            frame = camera.read()
            if i < warmup_frames - 1:
                time.sleep(0.05)

        if frame is None:
            raise RuntimeError(f"[{name}] 读取帧失败")

        h, w = frame.shape[:2]
        out_path = output_dir / f"{name}_{timestamp}.png"
        _save_frame(out_path, frame)
        saved.append(out_path)
        print(f"[{name}] 已保存 {w}x{h} -> {out_path}")

        camera.disconnect()

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="为每个 OpenCV 相机拍摄一张照片")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="相机设备 index，逗号分隔，例如 0,1",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='相机配置 JSON，格式与 slerobot_xarm_record 的 --robot.cameras 相同',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="请求宽度（scan/indices 模式，默认 640）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="请求高度（scan/indices 模式，默认 480）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="请求帧率（scan/indices 模式，默认 20）",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="保存前预热的帧数（默认 5）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camera_snapshots"),
        help="照片输出目录（默认 ./camera_snapshots）",
    )
    args = parser.parse_args()

    if args.cameras and args.indices:
        print("错误: --cameras 与 --indices 不能同时使用", file=sys.stderr)
        sys.exit(1)

    try:
        if args.cameras:
            cameras = make_cameras_from_configs(_parse_cameras_json(args.cameras))
        elif args.indices:
            cameras = _build_cameras_from_indices(
                _parse_indices(args.indices),
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
        else:
            cameras = _scan_and_build_cameras(
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)

    if not cameras:
        print("错误: 没有可测试的相机", file=sys.stderr)
        sys.exit(1)

    print(f"\n将测试 {len(cameras)} 个相机: {list(cameras.keys())}")
    print(f"输出目录: {args.output_dir.resolve()}")

    try:
        saved = capture_one_photo_per_camera(
            cameras,
            output_dir=args.output_dir,
            warmup_frames=args.warmup_frames,
        )
    except Exception as exc:
        print(f"\n测试失败: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n完成，共保存 {} 张照片:".format(len(saved)))
    for path in saved:
        print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
