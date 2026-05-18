# ACT 策略注意力可视化集成

## 概述

`ACTPolicyWithAttention` mapper 已集成到 `slerobot_record.py` 中，允许在数据采集期间实时捕获和保存 ACT 策略的 Transformer 注意力可视化。

## 功能说明

### 核心组件

1. **导入** - `ACTPolicyWithAttention` 类已导入
2. **配置选项** - `RecordConfig` 新增 `enable_attention_visualization` 布尔字段
3. **策略包装** - 当启用时，ACT 策略自动包装为 `ACTPolicyWithAttention`
4. **可视化捕获** - 在每个推理步骤中捕获注意力热图并保存为图像

## 使用方法

### 启用注意力可视化

在记录数据时添加 `--enable_attention_visualization=true` 标志：

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="my_dataset" \
    --dataset.single_task="pick_up_object" \
    --policy.path="/path/to/act_model" \
    --enable_attention_visualization=true
```

### 输出目录结构

注意力可视化图像将保存到以下位置：

```
<dataset_root>/<repo_id>/attention_visualizations/
├── attention_maps/
│   ├── frame_0/
│   │   ├── camera_0_attention.png
│   │   ├── camera_1_attention.png
│   │   └── ...
│   ├── frame_1/
│   │   ├── camera_0_attention.png
│   │   └── ...
│   └── ...
```

## 技术细节

### 集成点

1. **政策初始化** (`record()` 函数)
   - 当 `enable_attention_visualization=True` 且 `policy.type="act"` 时
   - 原始 ACT 策略被 `ACTPolicyWithAttention` 包装

2. **记录循环** (`record_loop()` 函数)
   - 新参数：`attention_visualization_enabled` 和 `attention_output_dir`
   - 在策略推理后调用 `policy.select_action()` 捕获注意力权重
   - 生成热图可视化并保存为 PNG 图像

### 代码流程

```python
if attention_visualization_enabled and isinstance(policy, ACTPolicyWithAttention):
    # 调用包装的策略获取注意力
    action_from_wrapped, attention_maps = policy.select_action(observation_frame)
    
    # 生成可视化
    vis_images = policy.visualize_attention(
        attention_maps=attention_maps,
        observation=observation_frame,
        use_rgb=True,
        overlay_alpha=0.5
    )
    
    # 保存到磁盘
    for i, vis in enumerate(vis_images):
        cv2.imwrite(f"{output_dir}/camera_{i}_attention.png", vis)
```

## 环境要求

### 必需依赖

- `torch` - PyTorch（用于前向钩子）
- `cv2` - OpenCV（用于图像处理和保存）
- `numpy` - 数值计算库

这些通常已在项目环境中安装。

## 性能考虑

### 开销

- **内存**: 每帧约 10-50MB（取决于注意力张量大小）
- **计算**: ~5-10% CPU/GPU 额外开销（热图生成）
- **I/O**: 磁盘写入速度可能是瓶颈

### 优化建议

1. **使用 SSD 存储** - 加快图像写入速度
2. **适当采样** - 并非每帧都保存（可修改代码）
3. **降低分辨率** - 在 `visualize_attention()` 中调整图像大小

## 修改建议

### 按帧间隔采样

在 `record_loop()` 中添加采样逻辑：

```python
frame_count = 0
ATTENTION_SAMPLE_INTERVAL = 5  # 每 5 帧保存一次

if frame_count % ATTENTION_SAMPLE_INTERVAL == 0:
    # 保存注意力可视化
    ...

frame_count += 1
```

### 自定义热图参数

在 `policy.visualize_attention()` 调用中修改：

```python
vis_images = policy.visualize_attention(
    attention_maps=attention_maps,
    observation=observation_frame,
    use_rgb=False,           # 改为 BGR 格式
    overlay_alpha=0.7        # 调整透明度 0-1
)
```

## 故障排查

### 问题：注意力可视化保存失败

**解决方案**：
1. 检查输出目录权限
2. 确保磁盘有足够空间
3. 查看日志中的详细错误信息

### 问题：性能下降显著

**解决方案**：
1. 禁用流编码：`--dataset.streaming_encoding=false`
2. 增加采样间隔（修改代码）
3. 在高速 SSD 上运行

### 问题：注意力热图全黑或没有信息

**解决方案**：
1. 检查 ACT 模型是否正确加载
2. 验证输入观察是否正确处理
3. 尝试调整 `overlay_alpha` 参数

## 关键修改文件

- `src/slerobot/scripts/slerobot_record.py` - 主集成文件
- `src/slerobot/secure/act_attention_mapper.py` - mapper 实现（无需修改）

## 后续增强计划

1. **实时显示** - 在推理过程中实时显示注意力可视化
2. **聚合分析** - 跨多个episode分析注意力模式
3. **交互式可视化** - 创建 Web 界面查看注意力轨迹
4. **注意力统计** - 计算注意力集中度、变化率等指标
