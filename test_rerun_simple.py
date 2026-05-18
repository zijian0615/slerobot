#!/usr/bin/env python
"""
Rerun 简单测试 - 快速验证可视化功能

用法:
    python test_rerun_simple.py
"""

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_basic_display():
    """基础显示测试"""
    try:
        import rerun as rr
    except ImportError:
        logger.error("Rerun 未安装: pip install rerun-sdk")
        return False
    
    logger.info("=" * 60)
    logger.info("Rerun 基础显示测试")
    logger.info("=" * 60)
    
    try:
        logger.info("\n1️⃣  初始化 Rerun...")
        rr.init("test_simple_display", spawn=True)
        logger.info("   ✓ 初始化成功")
        
        logger.info("\n2️⃣  创建测试数据...")
        
        # 原始图像 (RGB)
        original_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # 注意力热图 (灰度，0-1 范围)
        attention_heatmap = np.random.rand(240, 320).astype(np.float32)  # 降采样
        attention_heatmap_uint8 = (attention_heatmap * 255).astype(np.uint8)
        
        # 注意力叠加 (RGB)
        overlay = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        logger.info(f"   ✓ 原始图像: {original_image.shape} {original_image.dtype}")
        logger.info(f"   ✓ 热图: {attention_heatmap_uint8.shape} {attention_heatmap_uint8.dtype}")
        logger.info(f"   ✓ 叠加: {overlay.shape} {overlay.dtype}")
        
        logger.info("\n3️⃣  发送数据到 Rerun...")
        
        # 设置时间
        for frame_idx in range(10):
            logger.info(f"   发送 Frame {frame_idx}...")
            
            rr.set_time_sequence("frame", frame_idx)
            
            # 发送图像
            rr.log("camera_0/original_image", rr.Image(original_image))
            rr.log("camera_0/attention_heatmap", rr.Image(attention_heatmap_uint8))
            rr.log("camera_0/attention_overlay", rr.Image(overlay))
            
            # 添加一些标量数据用于对比
            rr.log("attention_stats/mean", rr.Scalar(attention_heatmap.mean()))
            rr.log("attention_stats/max", rr.Scalar(attention_heatmap.max()))
        
        logger.info("   ✓ 所有数据已发送")
        
        logger.info("\n4️⃣  验证窗口...")
        logger.info("   查看 Rerun 窗口中:")
        logger.info("   - 左侧应显示 camera_0 和 attention_stats")
        logger.info("   - 中间应显示图像")
        logger.info("   - 下方显示时间线")
        logger.info("   - 可以拖动时间线查看不同帧")
        
        logger.info("\n✅ 测试完成!")
        logger.info("   如果看到了图像数据，说明 Rerun 工作正常")
        logger.info("   如果看不到数据，检查:")
        logger.info("   - Rerun 窗口是否打开? (http://localhost:9090)")
        logger.info("   - 浏览器控制台是否有错误?")
        
        input("\n按 Enter 关闭 Rerun 连接...")
        rr.disconnect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    import sys
    success = test_basic_display()
    sys.exit(0 if success else 1)
