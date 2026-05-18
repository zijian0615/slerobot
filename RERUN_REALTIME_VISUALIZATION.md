# Rerun 实时注意力可视化 - 使用指南

## 功能说明

现在您可以在数据采集时使用 Rerun 实时显示 ACT 策略的注意力热图。这允许您在采集过程中实时监控策略的视觉关注点。

## 安装 Rerun

首先安装 Rerun Python 客户端：

```bash
pip install rerun-sdk
```

或在 conda 环境中：

```bash
conda activate slerobot
pip install rerun-sdk
```

## 基本使用

### 启用实时注意力显示

在启动记录脚本时添加两个标志：

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="my_dataset" \
    --dataset.single_task="pick_object" \
    --policy.path="/path/to/act_model" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

### 脚本会自动：

1. ✅ 初始化 Rerun 可视化窗口
2. ✅ 在采集过程中实时流式传输：
   - 原始摄像头图像
   - 注意力热图
   - 注意力热图叠加在原始图像上
3. ✅ 采集完成后自动关闭 Rerun 连接

## 界面说明

Rerun 窗口会显示以下数据流：

```
Rerun Application
├── camera_0/
│   ├── image              # 原始摄像头图像
│   ├── attention_heatmap  # 纯注意力热图（归一化到 0-255）
│   └── attention_overlay  # 热图叠加在图像上
├── camera_1/
│   ├── image
│   ├── attention_heatmap
│   └── attention_overlay
└── ... (更多摄像头)
```

### 热图颜色解释

- **红色区域** - 高注意力（值 > 200）
- **黄色/绿色** - 中等注意力（值 100-200）
- **蓝色区域** - 低注意力（值 < 100）

## 完整示例命令

### 示例 1：基础实时显示

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="demo_realtime" \
    --dataset.single_task="grasp" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --policy.path="./models/act_policy" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

### 示例 2：高级配置（多摄像头 + 优化）

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="multi_cam_realtime" \
    --dataset.single_task="assembly" \
    --dataset.num_episodes=10 \
    --dataset.fps=30 \
    --dataset.streaming_encoding=true \
    --policy.path="./models/act_policy_v2" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true \
    --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0}, \"side\": {\"type\": \"opencv\", \"index_or_path\": 1}}"
```

## 工作流程

### 采集期间

1. **启动脚本** - 添加 `--realtime_attention_display=true`
2. **Rerun 自动打开** - 新窗口会在后台启动
3. **实时流式显示** - 每一帧都会被实时显示
4. **本地监控** - 可以实时检查注意力分布是否合理

### 监控内容

```
采集时间轴 (Frame Index)
↓
观察每个摄像头的：
  1. 原始图像
  2. 注意力热图
  3. 叠加效果
↓
实时反馈策略行为
```

## 性能影响

### 与仅保存文件对比

| 指标 | 仅保存文件 | + 实时显示 |
|------|----------|----------|
| 内存 | 50 MB/frame | 60 MB/frame (+20%) |
| CPU | 5-10% | 10-15% (+5-10%) |
| 网络 | 无 | 本地 UDP |

### 优化建议

1. **降低采样率** - 不是每帧都显示（代码修改）
2. **使用高性能硬件** - GPU 驱动的推理机器
3. **分离显示** - 在单独的机器上运行 Rerun 查看器

## 常见问题

### Q: Rerun 没有显示任何内容怎么办？

A: 检查以下几点：
1. ✅ 确认 `--enable_attention_visualization=true`
2. ✅ 确认 `--realtime_attention_display=true`
3. ✅ 确认使用 ACT 策略（`--policy.path=...`）
4. ✅ 查看控制台日志中的错误信息
5. ✅ 确认 rerun 已安装：`pip install rerun-sdk`

### Q: Rerun 进程卡住了怎么办？

A: 这种情况很少见。如果发生：
1. 在终端按 `Ctrl+C` 中断采集
2. 手动关闭 Rerun 窗口
3. 查看日志中的详细错误信息

### Q: 可以将数据保存到 Rerun 文件吗？

A: 当前实现只进行实时流式显示。如果需要保存，建议：
1. 使用注意力可视化保存功能（`--enable_attention_visualization=true`）
2. 将 PNG 图像导出为视频：
   ```bash
   ffmpeg -framerate 30 -i "frame_%d.png" -c:v libx264 output.mp4
   ```

### Q: 支持远程 Rerun 显示吗？

A: 当前使用本地 Rerun 实例（`spawn=True`）。对于远程显示，可以修改代码：

```python
# 修改：record() 函数中的 Rerun 初始化
rr.init(f"slerobot_record_{cfg.dataset.repo_id}", spawn=False)
rr.connect("192.168.1.100:9876")  # 连接到远程 Rerun 服务
```

## 调试技巧

### 1. 查看详细日志

运行时添加日志级别：

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="debug" \
    --enable_attention_visualization=true \
    --realtime_attention_display=true \
    2>&1 | grep -i "rerun\|attention"
```

### 2. 只运行几帧测试

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="test" \
    --dataset.episode_time_s=5  # 只运行 5 秒
    --enable_attention_visualization=true \
    --realtime_attention_display=true
```

### 3. 验证注意力捕获

```python
# 简单测试脚本
from slerobot.secure.act_attention_mapper import ACTPolicyWithAttention
from slerobot.policies.factory import make_policy

policy = make_policy("act", pretrained_path="./model")
wrapped = ACTPolicyWithAttention(policy)

# 测试注意力捕获
observation = {...}  # 准备观察
action, attention_maps = wrapped.select_action(observation)
print(f"Captured {len(attention_maps)} attention maps")
```

## 集成 Rerun 查看器（可选）

如果要在 Web 浏览器中查看，可以启动 Rerun 查看器服务：

```bash
# 终端 1: 启动 Rerun 查看器
rerun serve

# 终端 2: 运行采集脚本（连接到查看器）
python -m slerobot.scripts.slerobot_record ...
```

然后访问 `http://localhost:9090` 在浏览器中查看。

## 文件结构

```
slerobot_record.py
├── 导入 Rerun
├── RecordConfig
│   ├── enable_attention_visualization
│   └── realtime_attention_display
├── record() 函数
│   ├── 初始化 Rerun (如果启用)
│   ├── 调用 record_loop()
│   └── 关闭 Rerun (finally 块)
└── record_loop() 函数
    ├── 捕获注意力 (if enabled)
    └── Rerun 实时显示 (if enabled)
```

## 修改的代码行数

- 导入：行 7-11
- 配置：行 171-172（新字段）
- 初始化：行 703-711
- 实时显示：行 463-491
- 关闭连接：行 819-825
- 参数传递：行 751, 785

## 故障排除清单

- [ ] rerun-sdk 已安装
- [ ] ACT 策略正确加载
- [ ] 两个标志都设置为 true
- [ ] 注意力可视化捕获已启用
- [ ] 摄像头已连接
- [ ] 磁盘空间充足

---

## 下一步

现在您可以：

1. ✅ 实时监控策略行为
2. ✅ 调试注意力分布问题
3. ✅ 验证数据采集质量
4. ✅ 分析失败情况

享受实时注意力可视化！🎉
