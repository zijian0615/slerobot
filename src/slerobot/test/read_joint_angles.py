"""
读取 xArm 关节角度（用于标定 start_joints）
==========================================
持续打印当前 6/7 个关节角，同时给出度数、弧度，以及一行
可直接粘贴到 config 的 `start_joints=(...)`（弧度）。

典型用法——手动示教标定新的 home 姿态：
    1) 接好机械臂，运行（teach 模式会松开抱闸，可用手推动机械臂）：
         python src/slerobot/test/read_joint_angles.py --ip 192.168.1.204 --teach
    2) 用手把机械臂摆到你想要的 home 姿态；
    3) 屏幕上实时打印当前关节角，摆好后记下最新一行的
         start_joints=(...)
       Ctrl+C 退出时也会再打印一次最终值；
    4) 把该元组填进 config_xarm.py 的 start_joints 或用
       --robot.start_joints 传入。

只读模式（不松抱闸，仅观察当前角度）：
    python src/slerobot/test/read_joint_angles.py --ip 192.168.1.204

注意：
  - --teach 会调用 set_mode(2)（手动/示教模式），运行期间机械臂可被
    徒手推动；退出时自动切回 set_mode(0)。请在安全、无碰撞的环境下使用，
    托住机械臂以免松闸瞬间因重力下坠。
  - 本脚本直接用 XArmAPI，不经过 XArmRobot，因此不会自动运动到 start_joints。
"""

import argparse
import math
import time


def _fmt_joints(joints, dof):
    vals = list(joints)[:dof]
    deg = "  ".join(f"J{i + 1}={math.degrees(v):+8.3f}°" for i, v in enumerate(vals))
    rad = "  ".join(f"{v:+.5f}" for v in vals)
    tup = ", ".join(f"{v:.5f}" for v in vals)
    return deg, rad, tup


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 xArm 关节角，用于标定 start_joints")
    parser.add_argument("--ip", default="192.168.1.204", help="xArm IP 地址")
    parser.add_argument("--dof", type=int, default=6, help="自由度 (5/6/7)")
    parser.add_argument("--hz", type=float, default=5.0, help="打印频率 (Hz)")
    parser.add_argument(
        "--teach",
        action="store_true",
        help="启用手动/示教模式（松开抱闸，可徒手推动机械臂）",
    )
    args = parser.parse_args()

    from xarm.wrapper import XArmAPI  # type: ignore

    print(f"\n连接 xArm @ {args.ip}  dof={args.dof}  teach={args.teach}")
    arm = XArmAPI(args.ip)
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.clean_warn()

    if args.teach:
        # 手动模式：松开抱闸，可徒手拖动机械臂
        arm.set_mode(2)
        arm.set_state(0)
        time.sleep(0.5)
        print("⚠️  已进入手动模式：机械臂抱闸已松开，请托住机械臂，徒手摆到目标姿态。")
    else:
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.2)
        print("只读模式：未松开抱闸。如需徒手摆动请加 --teach。")

    period = 1.0 / max(args.hz, 0.1)
    last_tuple = None

    print("\n实时关节角（Ctrl+C 退出）：\n")
    try:
        while True:
            code, joints = arm.get_servo_angle(is_radian=True)
            if code != 0:
                print(f"[warn] get_servo_angle code={code}")
                time.sleep(period)
                continue
            deg, rad, tup = _fmt_joints(joints, args.dof)
            last_tuple = tup
            print(f"{deg}\n    弧度: [{rad}]\n    start_joints=({tup})\n", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        if args.teach:
            # 切回位置模式，重新抱闸
            arm.set_mode(0)
            arm.set_state(0)
            print("已切回位置模式（重新抱闸）。")
        if last_tuple is not None:
            print("\n最终标定值，复制到 config 的 start_joints：")
            print(f"    start_joints = ({last_tuple})")
        try:
            arm.disconnect()
        except Exception:
            pass
        print("断开连接。完成。")


if __name__ == "__main__":
    main()
