# 注意力可视化快速开始指南

## 三步启用

### 1. 启动数据采集（带注意力可视化）

```bash
conda activate slerobot

python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="my_dataset_with_attention" \
    --dataset.single_task="grasp_object" \
    --dataset.num_episodes=10 \
    --policy.path="/path/to/act_checkpoint" \
    --enable_attention_visualization=true
```

### 2. 监控输出

采集期间，注意力可视化将自动保存到：
```
./datasets/my_dataset_with_attention/attention_visualizations/attention_maps/
```

### 3. 查看结果

在采集完成后，使用任何图像查看器打开 PNG 文件：

```bash
# Mac
open ./datasets/my_dataset_with_attention/attention_visualizations/attention_maps/frame_0/

# Linux
xdg-open ./datasets/my_dataset_with_attention/attention_visualizations/attention_maps/frame_0/

# 或使用 Python
import cv2
img = cv2.imread("./frame_0/camera_0_attention.png")
cv2.imshow("Attention Map", img)
cv2.waitKey(0)
```

## 完整示例命令

### 示例 1：基础使用（单个摄像头）

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="dataset_v1" \
    --dataset.single_task="pick" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --policy.path="./models/act_policy" \
    --enable_attention_visualization=true
```

### 示例 2：高级配置（多摄像头 + 优化）

```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="dataset_multi_cam" \
    --dataset.single_task="assembly_task" \
    --dataset.num_episodes=20 \
    --dataset.fps=30 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --policy.path="./models/act_policy_v2" \
    --enable_attention_visualization=true \
    --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0}, \"side\": {\"type\": \"opencv\", \"index_or_path\": 1}}"
```

## 配置参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `--enable_attention_visualization` | bool | 启用注意力可视化捕获（默认：False） |
| `--dataset.repo_id` | str | 数据集的标识符名称 |
| `--dataset.single_task` | str | 任务描述（用于标记） |
| `--dataset.num_episodes` | int | 要记录的 episode 数量 |
| `--policy.path` | str | 预训练 ACT 模型的路径 |

## 注意力热图解释

### 热图颜色含义

- **红色区域** - 高注意力集中，策略重点关注该区域
- **黄色/绿色区域** - 中等注意力
- **蓝色区域** - 低注意力，策略忽略该区域

### 示例场景

#### 拾取任务
```
预期：关注目标物体和抓取点
热图应显示：红色热点在物体周围
```

#### 组装任务
```
预期：关注对齐位置和插入点
热图应显示：两个红色热点（源和目标）
```

## 数据存储

### 目录结构

```
datasets/
├── my_dataset_with_attention/
│   ├── episodes/              # 录制的 episode 数据
│   ├── attention_visualizations/
│   │   └── attention_maps/
│   │       ├── frame_0/
│   │       │   ├── camera_0_attention.png
│   │       │   └── camera_1_attention.png
│   │       ├── frame_1/
│   │       │   ├── camera_0_attention.png
│   │       │   └── camera_1_attention.png
│   │       └── ...
│   └── metadata.json          # 数据集元数据
```

### 文件大小估计

- 每张注意力热图：~50-200 KB（取决于图像分辨率）
- 单个 episode（30 秒 @ 30 FPS）：~100-200 MB
- 完整数据集（10 episodes）：~1-2 GB

## 常见任务

### 检查特定 episode 的注意力

```python
from pathlib import Path
import cv2

episode_dir = Path("./datasets/my_dataset/attention_visualizations/attention_maps")

for frame_dir in sorted(episode_dir.glob("frame_*"))[:10]:  # 前 10 帧
    for img_file in sorted(frame_dir.glob("*.png")):
        img = cv2.imread(str(img_file))
        cv2.imshow(f"{img_file.name}", img)
        cv2.waitKey(100)  # 显示 100ms
```

### 批量导出为视频

```python
import cv2
from pathlib import Path

def create_attention_video(attention_dir, output_file="attention.mp4"):
    images = sorted(Path(attention_dir).glob("frame_*/camera_0_attention.png"))
    
    if not images:
        print(f"No images found in {attention_dir}")
        return
    
    # 读取第一帧来获取尺寸
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_file}")

# 使用
create_attention_video("./datasets/my_dataset/attention_visualizations/attention_maps")
```

## 性能最优化

### 推荐配置

```bash
# 快速采集（实时优先）
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="fast_collect" \
    --dataset.single_task="task" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=false \
    --policy.path="./model" \
    --enable_attention_visualization=true
```

### 高质量采集（准确性优先）

```bash
# 高分辨率、流编码
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="quality_collect" \
    --dataset.single_task="task" \
    --dataset.fps=30 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --policy.path="./model" \
    --enable_attention_visualization=true
```

## 故障排查

### Q: 注意力可视化目录为空怎么办？

A: 检查以下几点：
1. 确认 `--enable_attention_visualization=true`
2. 确认使用的是 ACT 策略（`policy.type="act"`）
3. 查看控制台日志寻找错误信息
4. 验证输出目录权限

### Q: 采集速度变慢了怎么办？

A: 
1. 禁用流编码：`--dataset.streaming_encoding=false`
2. 降低 FPS：`--dataset.fps=20`
3. 使用快速存储设备（SSD）
4. 在后台运行：`nohup python ... > output.log &`

### Q: 如何暂停并恢复采集？

A: 使用 `--resume=true` 继续上次的采集：
```bash
python -m slerobot.scripts.slerobot_record \
    --dataset.repo_id="my_dataset_with_attention" \
    --resume=true \
    --enable_attention_visualization=true
```

## 后续分析

采集完成后，可以进行以下分析：

1. **质量评估** - 检查注意力是否合理
2. **失败诊断** - 分析失败情况下的注意力分布
3. **策略理解** - 理解 AI 如何做决策
4. **数据清理** - 删除低质量的 episode

---

更多详细信息见 [ATTENTION_VISUALIZATION_INTEGRATION.md](./ATTENTION_VISUALIZATION_INTEGRATION.md)
