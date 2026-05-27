import json
import queue
import threading
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import paho.mqtt.client as mqtt

# ───────────────── MQTT 配置 ─────────────────
MQTT_BROKER = "10.22.55.72"   # Mac 本机 IP
MQTT_PORT = 1883
MQTT_TOPIC = "quest/data"

# ───────────────── 缓冲区 ─────────────────
BUFFER_SIZE = 300

timestamps = deque(maxlen=BUFFER_SIZE)

quest_data = {
    'px': deque(maxlen=BUFFER_SIZE), 'py': deque(maxlen=BUFFER_SIZE), 'pz': deque(maxlen=BUFFER_SIZE),
    'qx': deque(maxlen=BUFFER_SIZE), 'qy': deque(maxlen=BUFFER_SIZE),
    'qz': deque(maxlen=BUFFER_SIZE), 'qw': deque(maxlen=BUFFER_SIZE)
}

fanuc_data = {
    'x': deque(maxlen=BUFFER_SIZE), 'y': deque(maxlen=BUFFER_SIZE), 'z': deque(maxlen=BUFFER_SIZE),
    'w': deque(maxlen=BUFFER_SIZE), 'p': deque(maxlen=BUFFER_SIZE), 'r': deque(maxlen=BUFFER_SIZE)
}

data_queue = queue.Queue()

print(f"[MQTT Receiver] 连接到 {MQTT_BROKER}:{MQTT_PORT}")
print("-" * 70)

# ───────────────── MQTT 回调 ─────────────────
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        timestamp = datetime.now()
        data_queue.put({'timestamp': timestamp, 'payload': payload})
    except Exception as e:
        print(f"[MQTT Error] {e}")

def mqtt_listener():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# ───────────────── 数据处理 ─────────────────
# ───────────────── 频率统计 ─────────────────
freq_counter = 0
freq_start_time = datetime.now()
current_freq = 0.0
def process_incoming_data():
    global freq_counter, freq_start_time

    while not data_queue.empty():
        try:
            item = data_queue.get_nowait()
            timestamp = item['timestamp']
            p = item['payload']

            timestamps.append(timestamp)

            for key in quest_data:
                quest_data[key].append(p.get(key, 0.0))

            for key in fanuc_data:
                fanuc_data[key].append(p.get(key, 0.0))

            freq_counter += 1  # ← 核心计数

            ts = timestamp.strftime("%H:%M:%S.%f")[:-3]

            print(
                f"[{ts}]  "
                f"Quest pos=({p.get('px',0):+.3f}, {p.get('py',0):+.3f}, {p.get('pz',0):+.3f})  |  "
                f"Fanuc X={p.get('x',0):+.1f} Y={p.get('y',0):+.1f} Z={p.get('z',0):+.1f}  "
                f"W={p.get('w',0):+.1f} P={p.get('p',0):+.1f} R={p.get('r',0):+.1f}  "
                f"trigger={p.get('triggerButton',0)} grip={p.get('gripButton',0)}"
            )

        except queue.Empty:
            break

    # ─── 每 1 秒输出频率 ───
    now = datetime.now()
    elapsed = (now - freq_start_time).total_seconds()

    if elapsed >= 1.0:
        freq = freq_counter / elapsed
        print(f"[Freq] {freq:.2f} Hz")

        freq_counter = 0
        freq_start_time = now

# ───────────────── 绘图 ─────────────────
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Real-time MQTT Data', fontsize=14)

def update_plot(frame):
    global freq_counter, freq_start_time, current_freq

    process_incoming_data()

    now = datetime.now()
    elapsed = (now - freq_start_time).total_seconds()

    if elapsed >= 1.0:
        current_freq = freq_counter / elapsed
        freq_counter = 0
        freq_start_time = now

    if not timestamps:
        return

    t0 = timestamps[0]
    time_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax1.clear()
    ax2.clear()

    # ─── Quest ───
    ax1.set_title('Quest Hand Tracker')
    ax1.grid(True, alpha=0.3)

    if len(time_seconds) > 1:
        ax1.plot(time_seconds, list(quest_data['px']), label='px')
        ax1.plot(time_seconds, list(quest_data['py']), label='py')
        ax1.plot(time_seconds, list(quest_data['pz']), label='pz')

        ax1.plot(time_seconds, list(quest_data['qx']), '--', label='qx')
        ax1.plot(time_seconds, list(quest_data['qy']), '--', label='qy')
        ax1.plot(time_seconds, list(quest_data['qz']), '--', label='qz')
        ax1.plot(time_seconds, list(quest_data['qw']), '--', label='qw')

    ax1.legend()

    # ─── Fanuc ───
    ax2.set_title('Fanuc TCP')
    ax2.grid(True, alpha=0.3)

    if len(time_seconds) > 1:
        ax2.plot(time_seconds, list(fanuc_data['x']), label='X')
        ax2.plot(time_seconds, list(fanuc_data['y']), label='Y')
        ax2.plot(time_seconds, list(fanuc_data['z']), label='Z')

        ax2.plot(time_seconds, list(fanuc_data['w']), '--', label='W')
        ax2.plot(time_seconds, list(fanuc_data['p']), '--', label='P')
        ax2.plot(time_seconds, list(fanuc_data['r']), '--', label='R')

    ax2.legend()

    # ─── 频率显示（核心） ───
    ax1.text(
        0.02, 0.95,
        f"Freq: {current_freq:.2f} Hz",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
    )

    plt.tight_layout()

# ───────────────── 主函数 ─────────────────
def main():
    t = threading.Thread(target=mqtt_listener, daemon=True)
    t.start()

    print("等待 MQTT 数据中...")

    ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n[MQTT Receiver] 已停止。")

if __name__ == "__main__":
    main()