"""
快速测试脚本 - 验证路径解析是否正确
"""

from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_path_resolution():
    """测试路径解析逻辑"""
    print("="*80)
    print("测试路径解析")
    print("="*80)

    # 模拟从 student_model 目录运行
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent

    print(f"\n脚本目录: {script_dir}")
    print(f"父目录: {parent_dir}")

    # 测试相对路径
    config_relative = Path("models_yaml/anytext2_sd15.yaml")
    ckpt_relative = Path("models/anytext_v2.0.ckpt")

    # 解析为绝对路径
    config_abs = (parent_dir / config_relative).resolve() if not config_relative.is_absolute() else config_relative
    ckpt_abs = (parent_dir / ckpt_relative).resolve() if not ckpt_relative.is_absolute() else ckpt_relative

    print(f"\n配置文件 (相对): {config_relative}")
    print(f"配置文件 (绝对): {config_abs}")
    print(f"配置文件存在: {'✓' if config_abs.exists() else '✗'}")

    print(f"\n检查点 (相对): {ckpt_relative}")
    print(f"检查点 (绝对): {ckpt_abs}")
    print(f"检查点存在: {'✓' if ckpt_abs.exists() else '✗'}")

    print("\n" + "="*80)
    print("路径解析测试完成！")
    print("="*80)

    return config_abs.exists(), ckpt_abs.exists()

if __name__ == "__main__":
    config_ok, ckpt_ok = test_path_resolution()

    if config_ok and ckpt_ok:
        print("\n✓ 所有路径解析正确！可以运行 inspect_modules.py")
    else:
        print("\n✗ 路径解析失败，请检查文件是否存在")
