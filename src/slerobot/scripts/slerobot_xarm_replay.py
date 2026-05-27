"""
xArm 数据集回放脚本

用法示例：
    python -m slerobot.scripts.slerobot_xarm_replay \
        --dataset.repo_id=my_user/my_dataset \
        --dataset.episode=0 \
        --robot.robot_ip=192.168.1.127 \
        --robot.robot_mode=1 \
        --robot.robot_speed=300

参数说明：
    --dataset.repo_id      HuggingFace 数据集 ID 或本地路径（必填）
    --dataset.episode      要回放的 episode 索引，-1 表示全部（默认 0）
    --dataset.root         本地数据集根目录（可选）
    --dataset.fps          回放帧率，None 表示使用数据集自身 fps（默认 None）
    --dataset.use_dataset_timestamps  是否按录制时间戳控制帧间隔（默认 True）
    --dataset.start_delay_s           回放开始前等待时间（默认 2.0 s）
    --robot.*              XArmConfig 配置参数（与 record 脚本相同）
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from slerobot.configs import parser
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.robots.xarm import XArmRobot
from slerobot.robots.xarm.config_xarm import XArmConfig
from slerobot.utils.control_utils import init_keyboard_listener, is_headless
from slerobot.utils.utils import init_logging, log_say

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
#  配置                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class DatasetReplayConfig:
    repo_id: str
    episode: int = 0          # -1 = 回放全部 episode
    root: str | None = None
    fps: int | None = None    # None = 使用数据集自身 fps
    use_dataset_timestamps: bool = True
    download_videos: bool = False
    start_delay_s: float = 2.0


@dataclass
class XArmReplayConfig:
    dataset: DatasetReplayConfig
    robot: XArmConfig = field(default_factory=XArmConfig)
    play_sounds: bool = True


# ──────────────────────────────────────────────────────────────────────────── #
#  工具函数                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _to_python_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _precise_sleep(duration_s: float) -> None:
    """高精度睡眠（避免 time.sleep 误差积累）。"""
    if duration_s <= 0:
        return
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        remaining = deadline - time.perf_counter()
        if remaining > 0.002:
            time.sleep(remaining * 0.5)


def _build_xarm_action(row: dict, action_names: list[str]) -> dict[str, float]:
    """从数据集行构造 XArmRobot.send_action() 所需的动作字典 {j0…j7}。"""
    raw = _to_python_value(row.get("action"))
    if raw is None:
        return {}
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return {
        name: float(raw[idx])
        for idx, name in enumerate(action_names)
        if idx < len(raw)
    }


# ──────────────────────────────────────────────────────────────────────────── #
#  核心回放逻辑                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def replay_episode(
    robot: XArmRobot,
    dataset: sLerobotDataset,
    episode_index: int,
    cfg: XArmReplayConfig,
    events: dict,
) -> int:
    """回放单个 episode，返回实际回放的帧数。"""
    if "action" not in dataset.features:
        raise ValueError("数据集不含 `action` 特征，无法回放。")

    action_names: list[str] = dataset.features["action"].get("names") or []
    if not action_names:
        raise ValueError("数据集 `action` 特征未定义动作名称（names）。")

    num_frames = len(dataset.hf_dataset)
    if num_frames == 0:
        raise ValueError(f"Episode {episode_index} 无帧可回放。")

    replay_fps   = cfg.dataset.fps or dataset.fps
    target_dt    = 1.0 / replay_fps
    prev_ts: float | None = None
    next_t: float | None  = None
    started_t    = time.perf_counter()
    last_log_t   = started_t

    logger.info(
        "[REPLAY] 开始回放 episode=%s  共 %s 帧  目标帧率=%s fps",
        episode_index, num_frames, replay_fps,
    )

    for frame_idx in range(num_frames):
        if events.get("stop_recording") or events.get("exit_early"):
            logger.info("[REPLAY] 用户中断回放（frame=%s）", frame_idx)
            break

        row = dataset.hf_dataset[frame_idx]
        cur_ts = _to_python_value(row.get("timestamp")) if "timestamp" in row else None

        # ── 帧间定时 ──
        if frame_idx == 0:
            next_t = time.perf_counter()
        else:
            if cfg.dataset.use_dataset_timestamps and cur_ts is not None and prev_ts is not None:
                delta = max(0.0, float(cur_ts) - float(prev_ts))
            else:
                delta = target_dt
            next_t += delta
            _precise_sleep(next_t - time.perf_counter())

        # ── 构造并发送动作 ──
        action_dict = _build_xarm_action(row, action_names)
        if not action_dict:
            logger.warning("[REPLAY] frame=%s action 为空，跳过", frame_idx)
            prev_ts = float(cur_ts) if cur_ts is not None else prev_ts
            continue

        try:
            robot.send_action(action_dict)
        except Exception as exc:
            logger.error("[REPLAY] frame=%s send_action 失败：%s", frame_idx, exc)
            break

        prev_ts = float(cur_ts) if cur_ts is not None else prev_ts

        # ── 每秒打印进度 ──
        now = time.perf_counter()
        if now - last_log_t >= 1.0:
            elapsed  = now - started_t
            act_fps  = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[REPLAY] episode=%s  frame=%d/%d  %.1f%%  actual_fps=%.2f",
                episode_index,
                frame_idx + 1,
                num_frames,
                (frame_idx + 1) / num_frames * 100,
                act_fps,
            )
            last_log_t = now

    replayed = frame_idx + 1
    logger.info("[REPLAY] episode=%s 完成，共回放 %d 帧", episode_index, replayed)
    return replayed


# ──────────────────────────────────────────────────────────────────────────── #
#  主入口                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

@parser.wrap()
def replay(cfg: XArmReplayConfig) -> None:
    init_logging()

    # ── 加载数据集 ──
    total_episodes_hint = None  # 先不限制，加载元数据后再判断
    if cfg.dataset.episode >= 0:
        episodes_to_load = [cfg.dataset.episode]
    else:
        episodes_to_load = None  # 全部

    dataset = sLerobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episodes_to_load,
        download_videos=cfg.dataset.download_videos,
    )

    total = dataset.meta.total_episodes
    if cfg.dataset.episode >= 0:
        if cfg.dataset.episode >= total:
            raise IndexError(
                f"Episode {cfg.dataset.episode} 超出范围，数据集共 {total} 个 episode（0–{total-1}）。"
            )
        episodes_list = [cfg.dataset.episode]
    else:
        episodes_list = list(range(total))

    logger.info(
        "[REPLAY] 数据集：%s  共 %d episode，将回放：%s",
        cfg.dataset.repo_id, total, episodes_list,
    )

    # ── 连接机器人 ──
    robot = XArmRobot(cfg.robot)
    listener = None
    events: dict = {}

    try:
        try:
            robot.connect()
        except Exception as e:
            logger.error(
                "[REPLAY] 无法连接到 xArm（%s）：%s\n"
                "  请检查：\n"
                "  1. 机器人是否已上电\n"
                "  2. 网络是否连通（ping %s）\n"
                "  3. IP 地址是否正确（当前：%s）",
                cfg.robot.robot_ip, e,
                cfg.robot.robot_ip, cfg.robot.robot_ip,
            )
            raise
        listener, events = init_keyboard_listener()

        log_say(f"Start replay from {cfg.dataset.repo_id}", cfg.play_sounds)

        if cfg.dataset.start_delay_s > 0:
            logger.info("[REPLAY] %.1f 秒后开始回放…（按 Esc/Space 中止）", cfg.dataset.start_delay_s)
            _precise_sleep(cfg.dataset.start_delay_s)

        # ── 逐 episode 回放 ──
        for ep_idx in episodes_list:
            if events.get("stop_recording") or events.get("exit_early"):
                break

            if len(episodes_list) > 1:
                # 多 episode 时重新加载该 episode 的数据
                ep_dataset = sLerobotDataset(
                    repo_id=cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    episodes=[ep_idx],
                    download_videos=cfg.dataset.download_videos,
                )
            else:
                ep_dataset = dataset

            log_say(f"Replay episode {ep_idx}", cfg.play_sounds)
            replayed = replay_episode(
                robot=robot,
                dataset=ep_dataset,
                episode_index=ep_idx,
                cfg=cfg,
                events=events,
            )

            logger.info(
                "[REPLAY] episode=%d 完成，回放 %d 帧",
                ep_idx, replayed,
            )

            # 多 episode 之间短暂停顿
            if len(episodes_list) > 1 and ep_idx != episodes_list[-1]:
                if not events.get("stop_recording") and not events.get("exit_early"):
                    logger.info("[REPLAY] 3 秒后回放下一个 episode…")
                    _precise_sleep(3.0)

        log_say("Replay finished", cfg.play_sounds)
        logger.info("[REPLAY] 全部回放完成")

    finally:
        if robot.is_connected:
            robot.disconnect()
        if listener is not None and not is_headless():
            listener.stop()


if __name__ == "__main__":
    replay()
