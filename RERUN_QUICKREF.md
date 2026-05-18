# Rerun 实时显示 - 快速参考卡

## 一分钟快速开始

### 1️⃣ 安装
```bash
pip install rerun-sdk
```

### 2️⃣ 运行采集

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="my_data" \
    --dataset.single_task="pick" \
    --policy.path="./models/act_policy" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

### 3️⃣ 看画面
- ✅ Rerun 自动打开
- 📊 实时显示注意力热图
- 🎬 实时流式显示所有摄像头

---

## 命令解释

| 参数 | 含义 | 必要性 |
|------|------|--------|
| `--enable_attention_visualization=true` | 启用注意力捕获 | ✅ 必需 |
| `--realtime_attention_display=true` | 启用Rerun实时显示 | ✅ 必需 |
| `--policy.path="..."` | ACT模型路径 | ✅ 必需 |

---

## 看到的内容

### Rerun 窗口布局

```
Timeline 时间轴 ←→ Frame 0, 1, 2, ...

摄像头0:
  ├─ 原始图像
  ├─ 注意力热图 (蓝→红)
  └─ 热图叠加

摄像头1:
  ├─ 原始图像
  ├─ 注意力热图
  └─ 热图叠加

...
```

### 颜色含义

| 颜色 | 含义 |
|------|------|
| 🔵 蓝色 | 低注意力（0） |
| 🟢 绿色 | 中等注意力 |
| 🟡 黄色 | 高注意力 |
| 🔴 红色 | 最高注意力 (255) |

---

## 常见场景

### 场景 1: 查看单个摄像头

点击 `Rerun` 侧边栏中的 `camera_0/` 展开查看该摄像头的所有数据。

### 场景 2: 比较多个摄像头

点击并拖拽时间轴，观察所有摄像头的注意力变化。

### 场景 3: 导出视频帧

```bash
# 使用脚本导出
python -c "
import glob
import cv2
from pathlib import Path

imgs = sorted(glob.glob('datasets/my_data/attention_visualizations/attention_maps/frame_*/camera_0_attention.png'))
for img_path in imgs:
    print(img_path)
"
```

---

## 故障排查

### ❌ Rerun 没有打开？

```bash
# 1. 检查安装
pip show rerun-sdk

# 2. 测试导入
python -c "import rerun; print('OK')"

# 3. 检查参数
# 需要两个标志都设置为 true
```

### ❌ 窗口显示为空？

```bash
# 1. 查看日志错误
# 2. 确认策略是 ACT
# 3. 确认注意力捕获启用
```

### ❌ 性能太慢？

```bash
# 方案 1: 降低采样率（修改代码）
if frame_idx % 10 == 0:  # 每10帧显示一次
    # Rerun display

# 方案 2: 逐集采集（不同时启用）
--enable_attention_visualization=true
# 然后后处理显示
```

---

## 进阶用法

### 远程显示

在远程机器上启动 Rerun：
```bash
rerun serve
# 访问 http://remote_ip:9090
```

修改脚本连接（需要修改代码）：
```python
rr.init(..., spawn=False)
rr.connect("remote_ip:9876")
```

### 保存录制

修改初始化代码：
```python
rr.init(..., recording="/path/to/recording.rrd")
```

稍后重放：
```bash
rerun /path/to/recording.rrd
```

---

## 关键文件位置

- 📄 主集成文件: `src/slerobot/scripts/slerobot_record.py`
- 📄 详细文档: `RERUN_REALTIME_VISUALIZATION.md`
- 📄 集成总结: `RERUN_INTEGRATION_SUMMARY.md`
- 📄 注意力Mapper: `src/slerobot/secure/act_attention_mapper.py`

---

## 完整例子

### 示例 A: 简单调试

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="debug" \
    --dataset.single_task="test" \
    --dataset.episode_time_s=30 \
    --policy.path="./model" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

### 示例 B: 完整采集

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="production" \
    --dataset.single_task="assembly" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --policy.path="./model_v2" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

---

## 快速链接

- 📹 查看原始图像: Rerun → `camera_X/image`
- 📊 查看热图: Rerun → `camera_X/attention_heatmap`
- 🎨 查看叠加: Rerun → `camera_X/attention_overlay`
- ⏱️ 播放动画: Rerun 时间轴

---

## 记住这个

```
运行 ← 自动启动 Rerun ← 实时显示 ← 完成后自动关闭
   ↓
输入观察 → 策略推理 → 捕获注意力 → Rerun 显示 + 保存文件
```

---

**现在开始使用 Rerun 实时监控您的采集！🚀**

