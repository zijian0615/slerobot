"""
xArm Robot 单元测试脚本
=======================
独立测试 XArmRobot 的 connect / get_observation / send_action。

用法：
    python src/slerobot/test/test_xarm_robot.py --ip 192.168.1.204 --mode 6
    python src/slerobot/test/test_xarm_robot.py --ip 192.168.1.204 --mode 7
    python src/slerobot/test/test_xarm_robot.py --ip 192.168.1.204 --mode 6 --no-move
"""

import argparse
import math
import time

from slerobot.robots.xarm import XArmConfig, XArmRobot


def print_obs(obs: dict, label: str = "OBS") -> None:
    joint_keys = sorted([k for k in obs if k.startswith("j")])
    joints_str = "  ".join(f"{k}={obs[k]:+.4f}" for k in joint_keys)
    ts = obs.get("timestamp", 0.0)
    print(f"[{label}] t={ts:.3f}  {joints_str}")


def test_get_observation(robot: XArmRobot, n: int = 5) -> None:
    print("\n" + "=" * 60)
    print(f"TEST: get_observation  (×{n})")
    print("=" * 60)
    for i in range(n):
        obs = robot.get_observation()
        print_obs(obs, label=f"obs[{i}]")
        time.sleep(0.1)


def test_send_action_hold(robot: XArmRobot, duration_s: float = 3.0) -> None:
    """保持当前位置不动（发送当前观测作为目标），验证 send_action 不报错。"""
    print("\n" + "=" * 60)
    print(f"TEST: send_action HOLD  ({duration_s}s)")
    print("=" * 60)

    obs = robot.get_observation()
    # 用当前观测直接作为动作（保持不动）
    action = {k: v for k, v in obs.items() if k.startswith("j")}

    fps = 20
    dt = 1.0 / fps
    t_start = time.perf_counter()
    sent = 0

    while time.perf_counter() - t_start < duration_s:
        t0 = time.perf_counter()
        robot.send_action(action)
        sent += 1
        elapsed = time.perf_counter() - t0
        time.sleep(max(dt - elapsed, 0.0))

    print(f"  sent={sent}  actual_fps={sent/duration_s:.1f}")
    print("  ✓ send_action HOLD done")


def test_send_action_move_joint(robot: XArmRobot) -> None:
    """mode=6：小幅关节运动测试（J1 ±5°）。"""
    print("\n" + "=" * 60)
    print("TEST: send_action MOVE (mode=6, J1 ±5°)")
    print("=" * 60)

    obs = robot.get_observation()
    base_j0 = obs.get("j0", 0.0)
    delta = math.radians(5)

    targets = [
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j0": base_j0 + delta},
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j0": base_j0 - delta},
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j0": base_j0},
    ]

    for idx, action in enumerate(targets):
        print(f"  → target[{idx}] j0={math.degrees(action['j0']):.1f}°")
        for _ in range(20):
            robot.send_action(action)
            time.sleep(0.05)
        time.sleep(0.5)
        obs_now = robot.get_observation()
        print_obs(obs_now, label=f"after[{idx}]")

    print("  ✓ send_action MOVE done")


def test_send_action_move_cartesian(robot: XArmRobot) -> None:
    """mode=7：小幅笛卡尔运动（Z 轴 ±20mm）。"""
    print("\n" + "=" * 60)
    print("TEST: send_action MOVE (mode=7, Z ±20mm)")
    print("=" * 60)

    obs = robot.get_observation()
    base_z = obs.get("j2", 0.0)  # j2 = z in mm

    targets = [
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j2": base_z + 20},
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j2": base_z - 20},
        {**{k: obs[k] for k in obs if k.startswith("j")}, "j2": base_z},
    ]

    for idx, action in enumerate(targets):
        print(f"  → target[{idx}] z={action['j2']:.1f}mm")
        for _ in range(20):
            robot.send_action(action)
            time.sleep(0.05)
        time.sleep(0.5)
        obs_now = robot.get_observation()
        print_obs(obs_now, label=f"after[{idx}]")

    print("  ✓ send_action MOVE done")


def main() -> None:
    parser = argparse.ArgumentParser(description="xArm Robot 功能测试")
    parser.add_argument("--ip", default="192.168.1.204", help="xArm IP 地址")
    parser.add_argument("--dof", type=int, default=6, help="自由度 (5/6/7)")
    parser.add_argument("--mode", type=int, default=6, choices=[6, 7],
                        help="运动模式: 6=关节伺服, 7=笛卡尔在线")
    parser.add_argument("--gripper", type=int, default=0,
                        help="夹爪类型: 0=无, 1=xArmGripper, 2=G2, 11=Robotiq")
    parser.add_argument("--no-move", action="store_true",
                        help="只测试 get_observation，不发送运动指令")
    parser.add_argument("--no-start-move", action="store_true",
                        help="连接时不运动到 start_joints")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"xArm Test  ip={args.ip}  dof={args.dof}  mode={args.mode}")
    print(f"{'='*60}\n")

    cfg = XArmConfig(
        robot_ip=args.ip,
        robot_dof=args.dof,
        robot_mode=args.mode,
        gripper_type=args.gripper,
        move_to_start_on_connect=not args.no_start_move,
    )

    robot = XArmRobot(cfg)

    try:
        print("[1/4] Connecting …")
        robot.connect()
        print(f"      error_code after connect = {robot.real_arm.error_code}")
        print(f"      observation_features = {robot.observation_features}")
        print(f"      action_features      = {robot.action_features}")

        print("\n[2/4] get_observation test …")
        test_get_observation(robot, n=5)

        if args.no_move:
            print("\n[--no-move] Skipping send_action tests.")
        else:
            print("\n[3/4] send_action HOLD test …")
            test_send_action_hold(robot, duration_s=3.0)

            print("\n[4/4] send_action MOVE test …")
            if args.mode == 6:
                test_send_action_move_joint(robot)
            else:
                test_send_action_move_cartesian(robot)

        print("\n✅ All tests passed.")

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}")
        raise
    finally:
        print("\nDisconnecting …")
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
