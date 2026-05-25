import socket
import json
import time

# --- 配置参数 ---
HOST = "172.30.109.22"
PORT_CONNECT = 16001

def rmi_force_cleanup():
    """
    独立清理函数：连接、中止所有任务、关闭 Session
    """
    sock = None
    try:
        # 1. 握手获取动态端口
        print(f"正在连接控制器 {HOST}...")
        connect_msg = b'{"Communication": "FRC_Connect"}\r\n'
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((HOST, PORT_CONNECT))
            s.sendall(connect_msg)
            resp = s.recv(1024)
            data = json.loads(resp.decode())
            dynamic_port = data.get("PortNumber")
        
        if not dynamic_port:
            print("❌ 未能获取动态端口，请检查机器人网络或 RMI 选项是否开启。")
            return

        print(f"✅ 已获取动态端口: {dynamic_port}")

        # 2. 建立指令 Socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((HOST, dynamic_port))

        # 3. 发送 FRC_Abort
        # 这是最核心的一步，用于清空机器人内部的 RMI 运动队列
        print("正在发送 FRC_Abort 指令...")
        abort_cmd = json.dumps({"Command": "FRC_Abort"}) + "\r\n"
        sock.sendall(abort_cmd.encode())
        
        resp_abort = sock.recv(1024)
        print(f"🤖 机器人响应: {resp_abort.decode().strip()}")

        # 4. 等待一小会儿确保控制器处理完成
        time.sleep(0.5)

        # 5. 发送 FRC_Terminate (可选)
        # 如果你想完全结束本次 RMI 会话，可以使用 Terminate
        print("正在终止当前 RMI 会话...")
        term_cmd = json.dumps({"Command": "FRC_Terminate"}) + "\r\n"
        sock.sendall(term_cmd.encode())
        
        print("\n✨ 清理完成！机器人现在应该可以接收新的 Initialize 指令了。")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        if sock:
            sock.close()
            print("🔌 连接已安全关闭。")

if __name__ == "__main__":
    rmi_force_cleanup()