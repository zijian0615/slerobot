# Rerun 问题排查清单

## 🚀 运行前检查清单

- [ ] 已安装 Rerun: `pip show rerun-sdk` ✓
- [ ] 已安装 PyTorch: `python -c "import torch; print(torch.__version__)"`
- [ ] 已安装 OpenCV: `python -c "import cv2; print(cv2.__version__)"`
- [ ] ACT 模型路径正确: `ls ~/model/act/pytorch_model.bin`
- [ ] 机器人已连接并就绪
- [ ] 没有其他进程占用端口 9090: `lsof -i :9090`

## 🔍 运行时检查清单

### 脚本启动
- [ ] 看到欢迎信息和配置摘要
- [ ] 看到: `[INFO] Rerun initialized for real-time visualization on http://localhost:9090`
- [ ] 看到浏览器窗口自动打开
- [ ] 浏览器显示 Rerun UI (不是白屏或错误页)

### 数据流检查
- [ ] 看到: `[DEBUG] Logged image for camera_0: shape=...`
- [ ] 看到: `[DEBUG] Frame 0 logged to Rerun successfully`
- [ ] 看到图像逐帧更新 (0, 1, 2, 3...)
- [ ] 没有看到: `[WARNING] Failed to log attention to Rerun`

### 显示验证
- [ ] 左侧数据树显示: `camera_0`, `camera_1`, 等
- [ ] 中间面板显示第一帧画面
- [ ] 下方时间轴显示进度条
- [ ] 可以拖拽时间轴查看不同帧
- [ ] 图像看起来正确 (不是黑屏或随机像素)

## ⚠️ 常见失败症状和修复

### 症状 1: "Rerun opens but no images" (最常见)

```
检查项 | 状态 | 修复
------|------|------
Rerun 窗口 | 打开 | ✓
左侧面板 | 空 | → 运行 test_rerun_simple.py
中间画面 | 空 | → 检查 camera names
时间轴 | 无变化 | → 检查 rr.set_time_sequence() 调用
```

**快速修复**:
```bash
# 方案 A: 验证独立 Rerun 功能
python test_rerun_simple.py

# 方案 B: 启用详细日志
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
" && python -m slerobot.scripts.slerobot_record ...

# 方案 C: 检查数据格式
python diagnose_rerun.py
```

### 症状 2: "Rerun window doesn't open"

```
检查项 | 诊断命令
-------|-------------------
Rerun 安装 | pip show rerun-sdk
Rerun 导入 | python -c "import rerun"
端口占用 | lsof -i :9090
网络 | curl http://localhost:9090
```

**快速修复**:
```bash
# 重新安装
pip uninstall rerun-sdk -y && pip install rerun-sdk

# 清空缓存
python -c "import os; os.system('rm -rf ~/.cache/rerun')"

# 更换端口 (在代码中改 9090)
```

### 症状 3: "Connection refused"

```
原因 | 修复
-----|------
服务器未启动 | rr.init(spawn=True) 会自动启动
火墙阻止 | 允许 localhost:9090
旧进程占用 | kill $(lsof -ti :9090) 或重启
```

### 症状 4: "Invalid dtype/shape"

```
问题 | 检查 | 修复
------|------|------
浮点数 | logger 输出: dtype=float64 | 转换为 uint8
图像倒置 | 颜色反向 (蓝=红) | 转换 BGR→RGB
维度错误 | logger 输出: shape=(C,H,W) | 转置为 (H,W,C)
范围超限 | max=1000 | clip 到 0-255 或 0-1
```

## 🧪 診斷工具

### 工具 1: `diagnose_rerun.py`
运行完整的系统诊断:
```bash
python diagnose_rerun.py
```

输出示例:
```
[✓] Rerun 包已安装 (version 0.14.1)
[✓] 可以导入 Rerun
[✓] 可以连接到 Rerun (http://localhost:9090)
[✓] 可以记录基本图像数据
[!] 配置检查: 注意力映射已启用

总结: 系统健康 ✓
```

### 工具 2: `test_rerun_simple.py`  
运行简单的 10 帧测试:
```bash
python test_rerun_simple.py
```

预期输出:
```
[INFO] Rerun 已初始化
[DEBUG] Frame 0: 已记录图像 (shape=480x640x3)
[DEBUG] Frame 1: 已记录图像 (shape=480x640x3)
...
[DEBUG] Frame 9: 已记录图像 (shape=480x640x3)
[INFO] 测试完成 ✓ 请检查浏览器窗口
```

### 工具 3: 手动数据验证

```python
import rerun as rr
import numpy as np

# 测试 uint8
rr.init('uint8_test', spawn=True)
img_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
rr.log('test', rr.Image(img_uint8))
print("uint8 test: 应该看到彩色噪声图像")

# 测试 float32  
rr.init('float32_test', spawn=True)
img_float = np.random.rand(100, 100, 3).astype(np.float32)
rr.log('test', rr.Image(img_float))
print("float32 test: 应该看到彩色噪声图像")

# 测试形状转换
img_chw = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)
img_hwc = np.transpose(img_chw, (1, 2, 0))
rr.init('shape_test', spawn=True)
rr.log('test', rr.Image(img_hwc))
print("形状转换: 应该看到图像")
```

## 📋 运行命令参考

### 完整的采集命令
```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="test_attention" \
    --dataset.single_task="test" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=30 \
    --dataset.fps=30 \
    --policy.path="./models/act" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true \
    --resume_trajectory_only=false \
    --resume_ckpt_path=null
```

### 仅启用文件保存（不用 Rerun）
```bash
python -m slerobot.scripts.slerobot_record \
    --enable_attention_visualization=true \
    --realtime_attention_display=false \
    ...
```

### 仅启用 Rerun（不保存文件）
```bash
python -m slerobot.scripts.slerobot_record \
    --enable_attention_visualization=false \
    --realtime_attention_display=true \
    ...
```

### 调试模式（详细日志）
```bash
PYTHONPATH=/Users/zhangzijian/Desktop/fanuc/scheme/slerobot/src:$PYTHONPATH \
python -m slerobot.scripts.slerobot_record \
    --enable_attention_visualization=true \
    --realtime_attention_display=true \
    -c {"logging": {"level": "DEBUG"}} \
    ...
```

## 🔧 常用修复命令

```bash
# 1. 检查環境
python -m pip check

# 2. 升级 Rerun
pip install --upgrade rerun-sdk

# 3. 清除缓存
rm -rf ~/.cache/rerun
rm -rf /tmp/rerun*

# 4. 查看 Rerun 日志
tail -f ~/.local/share/rerun/*.log

# 5. 杀死僵尸进程
pkill -f rerun
pkill -f slerobot_record

# 6. 重置环境
python -m venv venv_clean
source venv_clean/bin/activate
pip install rerun-sdk torch opencv-python
```

## 📞 信息收集清单

当需要报告问题时，收集:

- [ ] 完整命令行:
  ```bash
  # 粘贴你运行的完整命令
  ```

- [ ] 错误日志 (最后 50 行):
  ```bash
  # 粘贴日志输出
  ```

- [ ] Python 环境:
  ```bash
  python --version
  pip show rerun-sdk
  pip show torch
  pip show opencv-python
  ```

- [ ] 系统信息:
  ```bash
  uname -a
  df -h  # 磁盘空间
  ```

- [ ] 诊断结果:
  ```bash
  python diagnose_rerun.py  # 完整输出
  ```

---

## 💡 Pro Tips

1. **慢速查看**: 用低 FPS 测试，更容易看到每帧
   ```bash
   --dataset.fps=5  # 而不是 30
   ```

2. **短时间测试**: 快速验证设置
   ```bash
   --dataset.episode_time_s=10  # 而不是 600
   ```

3. **单摄像头**: 简化调试
   ```bash
   --dataset.camera_ids="camera_0"  # 只用第一个
   ```

4. **单帧暂停**: 在 Rerun 中查看细节
   - 按空白键暂停
   - 拖拽时间轴查看特定帧
   - 放大/缩小图像

5. **布局导出**: 保存 Rerun 配置
   - 设置完美布局后，选择 "Save Layout"
   - 下次会自动应用

---

更新于: 2024-01-22
最后修改: 添加完整的症状和诊断表
