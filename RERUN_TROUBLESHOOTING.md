# Rerun 显示为空 - 故障排查指南

## ⚡ 快速修复 (按顺序尝试)

### 1️⃣ 确认两个标志都设置

```bash
✅ 正确:
python -m slerobot.scripts.slerobot_record \
    --enable_attention_visualization=true \
    --realtime_attention_display=true \
    --policy.path="./model" \
    ...

❌ 错误:
python -m slerobot.scripts.slerobot_record \
    --enable_attention_visualization=true \
    --policy.path="./model" \
    ...
```

### 2️⃣ 查看浏览器窗口

```
预期看到:
├── 左侧面板: camera_0, camera_1, ... 等数据树
├── 中间面板: 图像显示区域
└── 下侧: 时间轴进度条
```

如果为空，检查浏览器控制台 (F12 → Console)，寻找红色错误。

### 3️⃣ 检查日志输出

```bash
# 运行脚本，查看是否有以下日志:

✅ 看到这个说明初始化成功:
[INFO] Rerun initialized for real-time visualization on http://localhost:9090

❌ 看到这个说明初始化失败:
[WARNING] Failed to initialize Rerun: ...
```

### 4️⃣ 运行诊断脚本

```bash
cd /Users/zhangzijian/Desktop/fanuc/scheme/slerobot

# 运行完整诊断
python diagnose_rerun.py

# 运行简单测试
python test_rerun_simple.py
```

---

## 🔍 详细故障排查

### 问题 1: Rerun 窗口打开但没有数据

**症状**: 浏览器窗口打开，但左侧面板为空，中间显示默认界面

**检查步骤**:

1. **检查数据是否被发送**
   ```bash
   # 查看日志中是否有:
   [DEBUG] Logged image for camera_0: shape=...
   [DEBUG] Frame 0 logged to Rerun successfully
   ```

2. **检查 ACT 策略是否正确加载**
   ```bash
   # 查看日志中是否有:
   [INFO] Wrapping ACT policy with attention visualization
   ```

3. **检查观察是否包含图像**
   ```bash
   # 如果日志显示:
   [DEBUG] Failed to log original image for camera_0: ...
   # 说明观察中没有 camera_0 数据
   ```

4. **手动运行测试**
   ```bash
   python test_rerun_simple.py
   
   # 如果测试正常但采集不行，问题在数据格式
   ```

### 问题 2: 看到"Failed to log attention to Rerun"错误

**症状**: 日志中出现 `[WARNING] Failed to log attention to Rerun: ...` 错误

**常见原因和解决**:

| 错误信息 | 原因 | 解决方案 |
|---------|------|--------|
| `invalid shape` | 图像维度不对 | 确保图像是 HWC (H,W,C) 或 HW (H,W) |
| `invalid dtype` | 数据类型不支持 | 转换为 uint8 (0-255) 或 float32 (0-1) |
| `no connection` | Rerun 未初始化 | 检查初始化代码是否执行 |
| `NoneType` | 数据为 None | 检查 attention_maps 是否正确生成 |

**查看详细错误**:
```bash
# 修改日志级别查看具体错误
# 在 slerobot_record.py 中将以下行改为 logging.DEBUG:
logging.debug(...) → print(...)  # 强制输出
```

### 问题 3: Rerun 窗口不打开

**症状**: 没有看到 Rerun 浏览器窗口

**检查步骤**:

1. **检查 Rerun 是否安装**
   ```bash
   pip show rerun-sdk
   
   # 如果未安装:
   pip install rerun-sdk
   ```

2. **检查是否有报错阻止初始化**
   ```bash
   # 查看日志中是否有:
   [WARNING] Failed to initialize Rerun: ...
   
   # 查看详细错误:
   [DEBUG] Traceback...
   ```

3. **手动启动 Rerun**
   ```bash
   # 终端 1: 启动 Rerun 服务器
   rerun serve --web-viewer-listen 0.0.0.0:9090
   
   # 终端 2: 在另一个终端查看日志
   tail -f /tmp/rerun.log  # 如果存在
   ```

4. **检查端口占用**
   ```bash
   # 查看 9090 端口是否被占用
   lsof -i :9090  # macOS/Linux
   netstat -ano | findstr :9090  # Windows
   
   # 如果被占用，要么关闭占用进程，要么在代码中改变端口
   ```

### 问题 4: 数据显示但很凌乱/不清晰

**症状**: 看到数据但图像模糊、颜色怪异或布局混乱

**调试步骤**:

1. **检查图像格式**
   ```bash
   # 如果看到颜色反转，可能是 BGR 而不是 RGB
   # 对比原始图像和热图是否数据类型一致
   ```

2. **检查数据范围**
   ```bash
   # 查看日志中的数据提示:
   shape=..., dtype=uint8  ✓
   shape=..., dtype=float64 ⚠️  # 可能有精度问题
   ```

3. **重新排列窗口**
   ```
   Rerun 布局调整:
   1. 点击 "+"  按钮添加新面板
   2. 选择 camera_0/original_image
   3. 拖拽调整位置和大小
   ```

---

## 🧪 验证步骤

### 步骤 1: 测试 Rerun 独立工作

```bash
python -c "
import rerun as rr
import numpy as np

rr.init('test', spawn=True)
rr.set_time_sequence('frame', 0)

# 发送测试图像
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
rr.log('test/image', rr.Image(img))

print('✓ 数据已发送到 Rerun')
print('✓ 检查浏览器窗口是否显示')
input('按 Enter 退出...')
"
```

**预期结果**: 浏览器窗口显示随机图像

### 步骤 2: 测试 ACT 注意力映射

```bash
python -c "
from slerobot.secure.act_attention_mapper import ACTPolicyWithAttention
from slerobot.policies.factory import get_policy_class

# 加载模型
policy_cls = get_policy_class('act')
policy = policy_cls.from_pretrained('./path/to/model')

# 包装
wrapped = ACTPolicyWithAttention(policy)
print('✓ ACT 策略已包装')
print(f'✓ 注意力映射器已初始化')
"
```

**预期结果**: 无错误，策略已正确包装

### 步骤 3: 端到端测试

```bash
# 使用完整命令进行测试
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="test_rerun" \
    --dataset.single_task="test" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=10 \
    --dataset.fps=5 \
    --policy.path="./models/act" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

**预期结果**: 
- ✓ Rerun 窗口打开
- ✓ 数据在 10 秒内显示
- ✓ 时间轴从 0 到 50 帧（5fps × 10s）

---

## 🛠️ 高级调试

### 启用详细日志

修改 `slerobot_record.py` 中的日志级别:

```python
# 第 696 行附近
import logging
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG
    format='[%(levelname)s] %(message)s'
)
```

### 检查数据流

在代码中添加临时打印:

```python
# 在 record_loop 中, 注意力显示代码前添加:
print(f"[DEBUG] attention_maps type: {type(attention_maps)}")
print(f"[DEBUG] attention_maps length: {len(attention_maps)}")
for i, m in enumerate(attention_maps):
    if m is not None:
        print(f"       camera_{i}: shape={m.shape}, dtype={m.dtype}, min={m.min()}, max={m.max()}")
    else:
        print(f"       camera_{i}: None")
```

### 保存调试数据

```bash
# 将 Rerun 数据保存到文件用于离线分析
rr.init(..., recording="/tmp/debug.rrd")

# 稍后重放:
rerun /tmp/debug.rrd
```

---

## 📞 联系支持

如果以上都不能解决，请提供:

1. ✅ 完整的命令行参数
2. ✅ 完整的日志输出（包括 DEBUG 级别）
3. ✅ 截图显示 Rerun 窗口
4. ✅ `diagnose_rerun.py` 的输出
5. ✅ Python 版本: `python --version`
6. ✅ Rerun 版本: `pip show rerun-sdk`

---

## ⚡ 快速参考

| 问题 | 快速修复 |
|------|---------|
| 窗口打开但无数据 | 运行 `test_rerun_simple.py` 验证 |
| Rerun 不打开 | 安装 Rerun: `pip install rerun-sdk` |
| 数据加载但清晰度低 | 检查图像 dtype 和范围 |
| 某个摄像头数据缺失 | 检查观察中是否有该摄像头键 |
| 性能太低 | 降采样或删除某些摄像头 |
| 端口冲突 | 关闭其他 Rerun 进程或改端口 |

---

## 🎓 背景知识

### Rerun 工作原理

```
脚本 → 数据序列化 → UDP 或 TCP → Rerun 服务器 → 浏览器显示
```

### 数据格式要求

```
图像必须是:
- NumPy 数组
- uint8 (0-255) 或 float32 (0-1)
- Shape: HWC (彩色) 或 HW (灰度)
```

### 时间同步

```
rr.set_time_sequence("frame", 0)  # Frame 0
rr.log("camera/image", data)       # 记录到 Frame 0
rr.set_time_sequence("frame", 1)   # Frame 1
rr.log("camera/image", data)       # 记录到 Frame 1
```

---

希望这个指南能帮你解决 Rerun 显示问题！🚀
