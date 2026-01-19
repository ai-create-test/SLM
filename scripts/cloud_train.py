#!/usr/bin/env python
"""
cloud_train.py - 云训练 CLI

使用示例:
    # 提交训练任务
    python scripts/cloud_train.py submit --provider runpod --config configs/base.yaml
    
    # 查看任务状态
    python scripts/cloud_train.py status --job-id xxx
    
    # 查看日志
    python scripts/cloud_train.py logs --job-id xxx
    
    # 下载结果
    python scripts/cloud_train.py download --job-id xxx --output ./outputs
    
    # 取消任务
    python scripts/cloud_train.py cancel --job-id xxx
    
    # 列出历史任务
    python scripts/cloud_train.py list
    
    # 估算成本
    python scripts/cloud_train.py estimate --config configs/base.yaml --hours 8
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def cmd_submit(args):
    """提交训练任务"""
    from cloud import JobManager, TrainingJobConfig
    
    # 构建任务配置
    config = TrainingJobConfig(
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        config_path=args.config,
        data_path=args.data,
        stages=args.stages,
        max_hours=args.max_hours,
        name=args.name,
    )
    
    # 估算成本
    manager = JobManager(provider=args.provider)
    cost = manager.estimate_cost(config)
    
    print(f"\n{'='*50}")
    print("Cloud Training Job")
    print(f"{'='*50}")
    print(f"Provider: {args.provider}")
    print(f"GPU: {args.gpu} x {args.gpu_count}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Stages: {args.stages}")
    print(f"Max Hours: {args.max_hours}")
    print(f"Estimated Cost: ${cost:.2f}")
    print(f"{'='*50}")
    
    if not args.yes:
        confirm = input("\nSubmit job? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled")
            return
    
    job_id = manager.submit(config)
    print(f"\nJob submitted: {job_id}")
    
    if args.wait:
        print("\nWaiting for completion...")
        
        def progress_cb(info):
            print(f"\r  Status: {info.status.value} | Progress: {info.progress:.1%}", end="")
        
        info = manager.wait(job_id, callback=progress_cb)
        print()
        
        if info.status.is_success:
            print("Job completed successfully!")
            if args.output:
                manager.download(job_id, args.output)
        else:
            print(f"Job failed: {info.error_message}")


def cmd_status(args):
    """查看任务状态"""
    from cloud import JobManager
    
    manager = JobManager(provider=args.provider)
    info = manager.status(args.job_id)
    
    print(f"\nJob: {info.job_id}")
    print(f"Status: {info.status.value}")
    print(f"Provider: {info.provider}")
    
    if info.gpu_type:
        print(f"GPU: {info.gpu_type} x {info.gpu_count}")
    if info.progress > 0:
        print(f"Progress: {info.progress:.1%}")
    if info.current_stage:
        print(f"Stage: {info.current_stage}")
    if info.current_loss:
        print(f"Loss: {info.current_loss:.4f}")
    if info.error_message:
        print(f"Error: {info.error_message}")


def cmd_logs(args):
    """查看任务日志"""
    from cloud import JobManager
    
    manager = JobManager(provider=args.provider)
    logs = manager.logs(args.job_id, tail=args.tail)
    
    print(f"\n=== Logs for {args.job_id} ===\n")
    for line in logs:
        print(line)


def cmd_download(args):
    """下载任务结果"""
    from cloud import JobManager
    
    manager = JobManager(provider=args.provider)
    manager.download(args.job_id, args.output)


def cmd_cancel(args):
    """取消任务"""
    from cloud import JobManager
    
    manager = JobManager(provider=args.provider)
    manager.cancel(args.job_id)


def cmd_list(args):
    """列出历史任务"""
    from cloud import JobManager
    
    manager = JobManager(provider=args.provider)
    jobs = manager.list_jobs()
    
    if not jobs:
        print("No jobs found")
        return
    
    print(f"\n{'Job ID':<25} {'Provider':<10} {'Created':<20}")
    print("-" * 60)
    
    for job in jobs[-10:]:  # 最近 10 个
        print(f"{job['job_id']:<25} {job['provider']:<10} {job['created_at'][:19]}")


def cmd_estimate(args):
    """估算成本"""
    from cloud import JobManager, TrainingJobConfig
    from cloud.providers import get_provider
    
    providers = ["runpod", "modal", "lambda"]
    
    config = TrainingJobConfig(
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        max_hours=args.hours,
    )
    
    print(f"\nCost Estimate for {args.gpu} x {args.gpu_count} ({args.hours} hours)")
    print("-" * 40)
    
    for provider_name in providers:
        try:
            provider = get_provider(provider_name, api_key="dry-run")
            cost = provider.estimate_cost(config)
            print(f"  {provider_name:<15} ${cost:.2f}")
        except Exception:
            print(f"  {provider_name:<15} N/A")


def main():
    parser = argparse.ArgumentParser(
        description="NeuralFlow Cloud Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Submit
    submit_parser = subparsers.add_parser("submit", help="Submit training job")
    submit_parser.add_argument("--provider", "-p", default="runpod",
                               choices=["runpod", "modal", "lambda"])
    submit_parser.add_argument("--config", "-c", default="configs/base.yaml")
    submit_parser.add_argument("--data", "-d", default="data/train.jsonl")
    submit_parser.add_argument("--stages", nargs="+", default=["vqvae", "dynamics"])
    submit_parser.add_argument("--gpu", default="RTX4090")
    submit_parser.add_argument("--gpu-count", type=int, default=1)
    submit_parser.add_argument("--max-hours", type=float, default=8.0)
    submit_parser.add_argument("--name", default=None)
    submit_parser.add_argument("--wait", "-w", action="store_true")
    submit_parser.add_argument("--output", "-o", default="./outputs")
    submit_parser.add_argument("--yes", "-y", action="store_true")
    submit_parser.set_defaults(func=cmd_submit)
    
    # Status
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("--job-id", "-j", required=True)
    status_parser.add_argument("--provider", "-p", default="runpod")
    status_parser.set_defaults(func=cmd_status)
    
    # Logs
    logs_parser = subparsers.add_parser("logs", help="Get job logs")
    logs_parser.add_argument("--job-id", "-j", required=True)
    logs_parser.add_argument("--provider", "-p", default="runpod")
    logs_parser.add_argument("--tail", "-n", type=int, default=100)
    logs_parser.set_defaults(func=cmd_logs)
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download job outputs")
    download_parser.add_argument("--job-id", "-j", required=True)
    download_parser.add_argument("--provider", "-p", default="runpod")
    download_parser.add_argument("--output", "-o", default="./outputs")
    download_parser.set_defaults(func=cmd_download)
    
    # Cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel job")
    cancel_parser.add_argument("--job-id", "-j", required=True)
    cancel_parser.add_argument("--provider", "-p", default="runpod")
    cancel_parser.set_defaults(func=cmd_cancel)
    
    # List
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--provider", "-p", default="runpod")
    list_parser.set_defaults(func=cmd_list)
    
    # Estimate
    estimate_parser = subparsers.add_parser("estimate", help="Estimate cost")
    estimate_parser.add_argument("--gpu", default="RTX4090")
    estimate_parser.add_argument("--gpu-count", type=int, default=1)
    estimate_parser.add_argument("--hours", type=float, default=8.0)
    estimate_parser.set_defaults(func=cmd_estimate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
