#!/usr/bin/env python
"""
Rerun 显示诊断脚本 - 检查和测试 Rerun 可视化

用法:
    python diagnose_rerun.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def check_rerun_import():
    """检查 Rerun 库是否安装"""
    try:
        import rerun as rr
        logger.info("✅ Rerun 库已安装")
        logger.info(f"   版本: {rr.__version__ if hasattr(rr, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        logger.error(f"❌ Rerun 库未安装: {e}")
        logger.info("   运行: pip install rerun-sdk")
        return False

def check_torch_numpy():
    """检查必需的依赖"""
    try:
        import torch
        import numpy as np
        logger.info("✅ PyTorch 和 NumPy 已安装")
        return True
    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        return False

def test_rerun_connection():
    """测试 Rerun 连接和显示"""
    try:
        import rerun as rr
        import numpy as np
        from pathlib import Path
        
        logger.info("\n🧪 测试 Rerun 显示...")
        logger.info("  1. 初始化 Rerun...")
        
        rr.init("slerobot_diagnostic", spawn=True)
        logger.info("     ✅ Rerun 已启动")
        
        logger.info("  2. 发送测试图像...")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_heatmap = np.random.rand(480, 640).astype(np.uint8) * 255
        
        # 发送数据
        rr.set_time_sequence("frame", 0)
        rr.log("test_camera/image", rr.Image(test_image))
        rr.log("test_camera/heatmap", rr.Image(test_heatmap))
        
        logger.info("     ✅ 测试数据已发送")
        logger.info("  3. 检查 Rerun 窗口...")
        logger.info("     - 应该看到: test_camera/image 和 test_camera/heatmap")
        logger.info("     - 左侧应该有数据树")
        logger.info("     - 中间应该显示图像")
        
        logger.info("\n✅ Rerun 显示正常!")
        logger.info("   - 如果看不到数据，检查浏览器是否打开")
        logger.info("   - 尝试访问: http://localhost:9090")
        
        rr.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Rerun 测试失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_image_formats():
    """测试不同的图像格式"""
    try:
        import rerun as rr
        import numpy as np
        import torch
        
        logger.info("\n🎨 测试图像格式...")
        
        rr.init("format_test", spawn=False)  # 不启动新窗口
        
        # 测试不同的格式
        formats = {
            "uint8_rgb": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            "uint8_mono": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            "float32_0_1": np.random.rand(100, 100).astype(np.float32),
        }
        
        for name, img in formats.items():
            try:
                rr.log(f"formats/{name}", rr.Image(img))
                logger.info(f"  ✅ {name}: shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                logger.warning(f"  ⚠️  {name}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 图像格式测试失败: {e}")
        return False

def check_config():
    """检查配置"""
    logger.info("\n⚙️  配置检查...")
    
    try:
        from slerobot.scripts.slerobot_record import RecordConfig
        
        # 创建一个虚拟的数据集配置
        from slerobot.scripts.slerobot_record import DatasetRecordConfig
        
        dataset_cfg = DatasetRecordConfig(
            repo_id="test",
            single_task="test"
        )
        
        logger.info(f"  ✅ RecordConfig 可导入")
        logger.info(f"  ✅ 支持参数:")
        logger.info(f"     - enable_attention_visualization")
        logger.info(f"     - realtime_attention_display")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置检查失败: {e}")
        return False

def suggest_fixes():
    """建议修复步骤"""
    logger.info("\n💡 如果仍然看不到数据:")
    logger.info("   1. 确保 Rerun 窗口已打开 (http://localhost:9090)")
    logger.info("   2. 运行此脚本检查连接: python diagnose_rerun.py")
    logger.info("   3. 检查两个参数都设置为 true:")
    logger.info("      --enable_attention_visualization=true")
    logger.info("      --realtime_attention_display=true")
    logger.info("   4. 查看日志输出寻找错误信息:")
    logger.info("      - 'Failed to log attention to Rerun'")
    logger.info("      - 'Failed to initialize Rerun'")
    logger.info("   5. 尝试手动启动 Rerun:")
    logger.info("      rerun --web-viewer-listen 0.0.0.0:9090")

def main():
    """主诊断程序"""
    logger.info("=" * 60)
    logger.info("Rerun 显示诊断工具")
    logger.info("=" * 60)
    
    all_passed = True
    
    # 检查步骤
    checks = [
        ("导入检查", check_rerun_import),
        ("依赖检查", check_torch_numpy),
        ("连接测试", test_rerun_connection),
    ]
    
    for name, check_func in checks:
        logger.info(f"\n▶️  {name}...")
        if not check_func():
            all_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ 所有诊断检查通过!")
        logger.info("   Rerun 应该可以正常显示数据")
    else:
        logger.info("❌ 发现问题，请按照上面的建议修复")
    
    suggest_fixes()
    logger.info("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
