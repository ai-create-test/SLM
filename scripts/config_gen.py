#!/usr/bin/env python
"""
config_gen.py - 配置生成器

交互式创建和编辑配置文件。

使用示例:
    # 交互式创建
    python scripts/config_gen.py --output configs/my_experiment.yaml
    
    # 从现有配置派生
    python scripts/config_gen.py --inherit configs/base.yaml --output configs/custom.yaml
    
    # 快速设置参数
    python scripts/config_gen.py \
        --inherit configs/base.yaml \
        --set model.d_latent=256 \
        --set training.batch_size=64 \
        --output configs/small_batch.yaml
    
    # 查看配置预设
    python scripts/config_gen.py --list-presets
    
    # 验证配置
    python scripts/config_gen.py --validate configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def cmd_create(args):
    """创建新配置"""
    from app.interfaces.config_loader import ExtendedConfig, ConfigLoader
    
    if args.inherit:
        config = ConfigLoader.load(args.inherit)
        print(f"Inherited from: {args.inherit}")
    else:
        config = ExtendedConfig()
    
    # 应用 --set 参数
    if args.set:
        for item in args.set:
            if "=" not in item:
                print(f"Invalid --set format: {item} (expected key=value)")
                continue
            
            key, value = item.split("=", 1)
            parts = key.split(".")
            
            # 解析值类型
            value = parse_value(value)
            
            # 设置值
            try:
                set_nested_attr(config, parts, value)
                print(f"  Set {key} = {value}")
            except Exception as e:
                print(f"  Failed to set {key}: {e}")
    
    # 保存
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    config.save(str(output))
    print(f"\nConfiguration saved to: {output}")


def cmd_interactive(args):
    """交互式创建配置"""
    from app.interfaces.config_loader import ExtendedConfig
    
    print("\n=== NeuralFlow Config Generator ===\n")
    
    config = ExtendedConfig()
    
    # 基本信息
    name = input(f"Experiment name [{config.name}]: ").strip()
    if name:
        config.name = name
    
    # 模型配置
    print("\n--- Model Configuration ---")
    d_latent = input(f"d_latent [{config.model.d_latent}]: ").strip()
    if d_latent:
        config.model.d_latent = int(d_latent)
    
    d_model = input(f"d_model [{config.model.d_model}]: ").strip()
    if d_model:
        config.model.d_model = int(d_model)
    
    num_layers = input(f"num_layers [{config.model.num_layers}]: ").strip()
    if num_layers:
        config.model.num_layers = int(num_layers)
    
    brain_type = input(f"brain_type (mamba/gru) [{config.model.brain_type}]: ").strip()
    if brain_type:
        config.model.brain_type = brain_type
    
    # 训练配置
    print("\n--- Training Configuration ---")
    batch_size = input(f"batch_size [{config.training.batch_size}]: ").strip()
    if batch_size:
        config.training.batch_size = int(batch_size)
    
    learning_rate = input(f"learning_rate [{config.training.learning_rate}]: ").strip()
    if learning_rate:
        config.training.learning_rate = float(learning_rate)
    
    max_epochs = input(f"max_epochs [{config.training.max_epochs}]: ").strip()
    if max_epochs:
        config.training.max_epochs = int(max_epochs)
    
    # 云配置
    print("\n--- Cloud Configuration ---")
    use_cloud = input("Configure cloud training? [y/N]: ").strip().lower()
    
    if use_cloud == "y":
        provider = input(f"provider (runpod/modal/lambda) [{config.cloud.provider}]: ").strip()
        if provider:
            config.cloud.provider = provider
        
        gpu_type = input(f"gpu_type [{config.cloud.gpu_type}]: ").strip()
        if gpu_type:
            config.cloud.gpu_type = gpu_type
        
        max_hours = input(f"max_hours [{config.cloud.max_hours}]: ").strip()
        if max_hours:
            config.cloud.max_hours = float(max_hours)
    
    # 保存
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    config.save(str(output))
    print(f"\nConfiguration saved to: {output}")


def cmd_validate(args):
    """验证配置文件"""
    from app.interfaces.config_loader import ConfigLoader
    
    try:
        config = ConfigLoader.load(args.config)
        
        print(f"\n✓ Configuration valid: {args.config}\n")
        print(f"Name: {config.name}")
        print(f"Version: {config.version}")
        print(f"\nModel:")
        print(f"  d_latent: {config.model.d_latent}")
        print(f"  d_model: {config.model.d_model}")
        print(f"  num_layers: {config.model.num_layers}")
        print(f"  brain_type: {config.model.brain_type}")
        print(f"\nTraining:")
        print(f"  batch_size: {config.training.batch_size}")
        print(f"  learning_rate: {config.training.learning_rate}")
        print(f"  max_epochs: {config.training.max_epochs}")
        print(f"\nCloud:")
        print(f"  provider: {config.cloud.provider}")
        print(f"  gpu_type: {config.cloud.gpu_type}")
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return 1
    
    return 0


def cmd_list_presets(args):
    """列出可用预设"""
    presets_dir = Path("configs/presets")
    
    print("\n=== Available Presets ===\n")
    
    # 内置预设
    print("Built-in:")
    for name in ["small", "base", "large"]:
        print(f"  - {name}")
    
    # 用户预设
    if presets_dir.exists():
        print("\nUser presets:")
        for f in presets_dir.glob("*.yaml"):
            print(f"  - {f.stem}")
    
    print("\nUsage: --preset <name> or --inherit configs/presets/<name>.yaml")


def cmd_diff(args):
    """比较两个配置"""
    from app.interfaces.config_loader import ConfigLoader
    
    config1 = ConfigLoader.load(args.config1)
    config2 = ConfigLoader.load(args.config2)
    
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    print(f"\n=== Config Diff ===")
    print(f"Base: {args.config1}")
    print(f"Compare: {args.config2}\n")
    
    diff_count = compare_dicts(dict1, dict2, "")
    
    if diff_count == 0:
        print("No differences found")


def compare_dicts(d1: dict, d2: dict, prefix: str) -> int:
    """比较字典差异"""
    count = 0
    all_keys = set(d1.keys()) | set(d2.keys())
    
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        
        v1 = d1.get(key)
        v2 = d2.get(key)
        
        if isinstance(v1, dict) and isinstance(v2, dict):
            count += compare_dicts(v1, v2, path)
        elif v1 != v2:
            print(f"  {path}:")
            print(f"    - {v1}")
            print(f"    + {v2}")
            count += 1
    
    return count


def set_nested_attr(obj, parts, value):
    """设置嵌套属性"""
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def parse_value(value: str):
    """解析字符串值"""
    import json
    
    # 布尔
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    
    # 整数
    try:
        return int(value)
    except ValueError:
        pass
    
    # 浮点
    try:
        return float(value)
    except ValueError:
        pass
    
    # JSON
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except Exception:
            pass
    
    return value


def main():
    parser = argparse.ArgumentParser(
        description="NeuralFlow Config Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Create
    create_parser = subparsers.add_parser("create", help="Create config")
    create_parser.add_argument("--inherit", "-i", help="Inherit from config")
    create_parser.add_argument("--set", "-s", action="append", help="Set values (key=value)")
    create_parser.add_argument("--output", "-o", required=True, help="Output path")
    create_parser.set_defaults(func=cmd_create)
    
    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--output", "-o", required=True)
    interactive_parser.set_defaults(func=cmd_interactive)
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validate config")
    validate_parser.add_argument("config", help="Config file path")
    validate_parser.set_defaults(func=cmd_validate)
    
    # List presets
    list_parser = subparsers.add_parser("list-presets", help="List presets")
    list_parser.set_defaults(func=cmd_list_presets)
    
    # Diff
    diff_parser = subparsers.add_parser("diff", help="Compare configs")
    diff_parser.add_argument("config1")
    diff_parser.add_argument("config2")
    diff_parser.set_defaults(func=cmd_diff)
    
    # 快捷方式
    parser.add_argument("--output", "-o", help="Output path (quick create)")
    parser.add_argument("--inherit", "-i", help="Inherit from")
    parser.add_argument("--set", "-s", action="append", help="Set values")
    parser.add_argument("--validate", "-v", help="Validate config")
    parser.add_argument("--list-presets", action="store_true")
    
    args = parser.parse_args()
    
    # 处理快捷方式
    if args.list_presets:
        cmd_list_presets(args)
        return
    
    if args.validate:
        args.config = args.validate
        return cmd_validate(args)
    
    if args.output and not args.command:
        return cmd_create(args)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
