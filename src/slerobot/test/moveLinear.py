import json
import os
import socket
import time

HOST = os.getenv("ROBOT_HOST", "127.0.0.1")
PORT_CONNECT = int(os.getenv("ROBOT_PORT", "16001"))
GROUP = 1

def frc_connect(host, port):
    """建立 RMI 基础连接并获取动态端口"""
    msg = b'{"Communication": "FRC_Connect"}\r\n'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(msg)
        resp = s.recv(1024)
    print("FRC_Connect:", repr(resp))
    data = json.loads(resp.decode())
    if data.get("ErrorID", -1) != 0:
        raise RuntimeError(f"FRC_Connect failed: {data}")
    return data.get("PortNumber")

def rmi_abort(sock):
    """发送 FRC_Abort 清空之前的 RMI_MOVE 程序"""
    abort_msg = json.dumps({"Command": "FRC_Abort"}) + "\r\n"
    sock.sendall(abort_msg.encode())
    resp = sock.recv(1024)
    print("FRC_Abort response:", resp.decode())
    data = json.loads(resp.decode())
    if data.get("ErrorID", -1) != 0:
        print("⚠️ Abort returned ErrorID:", data.get("ErrorID"))

def rmi_initialize(host, port, group=1):
    """初始化 RMI_MOVE，失败自动检查并 Abort"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.settimeout(5.0)

    init_msg = json.dumps({
        "Command": "FRC_Initialize",
        "GroupMask": group
    }) + "\r\n"

    s.sendall(init_msg.encode())
    resp = s.recv(1024)
    print("FRC_Initialize:", repr(resp))
    data = json.loads(resp.decode())

    # if data.get("ErrorID") == 2556943:  # Invalid Controller State
    #     print("⚠️ Controller busy, sending Abort and retrying...")
    #     rmi_abort(s)
    #     time.sleep(0.5)
    #     s.sendall(init_msg.encode())
    #     resp = s.recv(1024)
    #     data = json.loads(resp.decode())
    #     print("Retry FRC_Initialize:", data)

    # if data.get("ErrorID") not in (0, 7015):
    #     raise RuntimeError(f"FRC_Initialize failed: {data}")
    return s

def send_linear_motion(
        sock,
        sequence_id,
        x, y, z,
        w, p, r,
        utool=1,
        uframe=1,
        speed_type="mmSec",
        speed=150,
        term_type="FINE",
        term_value=0):

    packet = {
        "Instruction": "FRC_LinearMotion",
        "SequenceID": int(sequence_id),
        "Configuration": {
            "UToolNumber": int(utool),
            "UFrameNumber": int(uframe),
            "Front": 1,
            "Up": 1,
            "Left": 0,
            "Flip": 0,
            "Turn4": 0,
            "Turn5": 0,
            "Turn6": 0
        },
        "Position": {
            "X": float(x),
            "Y": float(y),
            "Z": float(z),
            "W": float(w),
            "P": float(p),
            "R": float(r),
            "Ext1": 0.0,
            "Ext2": 0.0,
            "Ext3": 0.0
        },
        "SpeedType": str(speed_type),
        "Speed": int(speed),
        "TermType": str(term_type),
        "TermValue": int(term_value)
    }

    msg = (json.dumps(packet) + "\r\n").encode()
    print("\nSending Linear Motion:")
    print(json.dumps(packet, indent=2))

    sock.sendall(msg)
    resp = sock.recv(4096)
    print("Response:", resp.decode())
    return json.loads(resp.decode())

if __name__ == "__main__":
    dynamic_port = frc_connect(HOST, PORT_CONNECT)
    print(f"Dynamic port: {dynamic_port}")

    rmi_sock = rmi_initialize(HOST, dynamic_port, GROUP)

    try:
        resp = send_linear_motion(
            rmi_sock,
            sequence_id=1,
            x=400.176,
            y=-33.764,
            z=-97.136,
            w=160.773,
            p=19.458,
            r=142.266,
            utool=1,
            uframe=0,
            speed_type="mmSec",
            speed=100,
            term_type="FINE",
            term_value=0
        )

        if resp:
            err = resp.get("ErrorID", -1)
            if err == 0:
                print("✅ Linear motion executed successfully")
            else:
                print(f"❌ ErrorID: {err} (hex: {hex(err)})")
                print("请在示教器 ALARM 页面查看详细报警信息")

    except Exception as e:
        print(f"💥 发生异常: {e}")
        # 只有当 rmi_sock 成功建立后才能调用 abort
        if rmi_sock:
            try:
                rmi_abort(rmi_sock)
            except:
                pass
    finally:
        rmi_sock.close()