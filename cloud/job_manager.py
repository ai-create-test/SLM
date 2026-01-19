"""
Job Manager - 训练任务管理器

统一管理云训练任务的生命周期。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import time
from datetime import datetime

from .providers.base import CloudProvider, JobStatus, JobInfo


@dataclass
class TrainingJobConfig:
    """
    训练任务配置
    
    定义云训练任务的所有参数。
    """
    # =========================================================================
    # 计算资源
    # =========================================================================
    gpu_type: str = "RTX4090"
    gpu_count: int = 1
    cpu_count: int = 4
    memory_gb: int = 32
    disk_gb: int = 50
    
    # =========================================================================
    # 训练配置
    # =========================================================================
    config_path: str = "configs/base.yaml"
    data_path: str = "data/train.jsonl"
    eval_data_path: Optional[str] = None
    stages: List[str] = field(default_factory=lambda: ["vqvae", "dynamics"])
    
    # =========================================================================
    # 任务控制
    # =========================================================================
    max_hours: float = 8.0
    auto_terminate: bool = True
    priority: int = 0                   # 优先级 (高优先执行)
    
    # =========================================================================
    # 环境配置
    # =========================================================================
    docker_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
    pip_requirements: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # =========================================================================
    # 同步配置
    # =========================================================================
    sync_checkpoints: bool = True
    sync_interval_minutes: int = 30
    output_path: str = "./outputs"
    
    # =========================================================================
    # 元信息
    # =========================================================================
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "config_path": self.config_path,
            "data_path": self.data_path,
            "eval_data_path": self.eval_data_path,
            "stages": self.stages,
            "max_hours": self.max_hours,
            "auto_terminate": self.auto_terminate,
            "docker_image": self.docker_image,
            "pip_requirements": self.pip_requirements,
            "env_vars": self.env_vars,
            "sync_checkpoints": self.sync_checkpoints,
            "sync_interval_minutes": self.sync_interval_minutes,
            "output_path": self.output_path,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJobConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class JobManager:
    """
    训练任务管理器
    
    统一管理不同云平台的训练任务。
    
    使用示例:
        manager = JobManager(provider="runpod")
        
        # 提交任务
        config = TrainingJobConfig(
            gpu_type="RTX4090",
            config_path="configs/base.yaml",
            data_path="data/train.jsonl",
        )
        job_id = manager.submit(config)
        
        # 查看状态
        info = manager.status(job_id)
        
        # 等待完成
        manager.wait(job_id, callback=lambda i: print(f"{i.progress:.1%}"))
        
        # 下载结果
        manager.download(job_id, "./outputs")
    """
    
    # 任务历史保存路径
    HISTORY_FILE = ".cloud_jobs.json"
    
    def __init__(
        self,
        provider: str = "runpod",
        api_key: Optional[str] = None,
        **provider_kwargs,
    ):
        """
        Args:
            provider: 云平台名称
            api_key: API 密钥
            **provider_kwargs: 传递给 Provider 的额外参数
        """
        from .providers import get_provider
        
        self.provider_name = provider
        self.provider: CloudProvider = get_provider(
            provider,
            api_key=api_key,
            **provider_kwargs,
        )
        
        self._jobs: Dict[str, JobInfo] = {}
        self._load_history()
    
    def submit(self, config: TrainingJobConfig) -> str:
        """
        提交训练任务
        
        Args:
            config: 任务配置
            
        Returns:
            任务 ID
        """
        print(f"Submitting job to {self.provider_name}...")
        print(f"  GPU: {config.gpu_type} x {config.gpu_count}")
        print(f"  Config: {config.config_path}")
        print(f"  Data: {config.data_path}")
        print(f"  Stages: {config.stages}")
        
        job_id = self.provider.submit_job(config)
        
        print(f"  Job ID: {job_id}")
        
        # 保存历史
        self._save_job(job_id, config)
        
        return job_id
    
    def status(self, job_id: str) -> JobInfo:
        """
        获取任务状态
        
        Args:
            job_id: 任务 ID
            
        Returns:
            任务信息
        """
        return self.provider.get_job_info(job_id)
    
    def logs(self, job_id: str, tail: int = 100) -> List[str]:
        """
        获取任务日志
        
        Args:
            job_id: 任务 ID
            tail: 返回最后 N 行
            
        Returns:
            日志行列表
        """
        return self.provider.get_logs(job_id, tail=tail)
    
    def download(self, job_id: str, local_path: str) -> None:
        """
        下载任务产出
        
        Args:
            job_id: 任务 ID
            local_path: 本地保存路径
        """
        print(f"Downloading outputs for job {job_id}...")
        self.provider.download_outputs(job_id, local_path)
        print(f"  Saved to: {local_path}")
    
    def cancel(self, job_id: str) -> None:
        """
        取消任务
        
        Args:
            job_id: 任务 ID
        """
        print(f"Cancelling job {job_id}...")
        self.provider.terminate_job(job_id)
        print("  Done")
    
    def wait(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
        callback=None,
    ) -> JobInfo:
        """
        等待任务完成
        
        Args:
            job_id: 任务 ID
            poll_interval: 轮询间隔 (秒)
            timeout: 超时时间 (秒)
            callback: 状态更新回调
            
        Returns:
            最终任务信息
        """
        return self.provider.wait_for_completion(
            job_id,
            poll_interval=poll_interval,
            timeout=timeout,
            callback=callback,
        )
    
    def estimate_cost(self, config: TrainingJobConfig) -> float:
        """
        估算成本
        
        Args:
            config: 任务配置
            
        Returns:
            预估成本 (USD)
        """
        return self.provider.estimate_cost(config)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """列出历史任务"""
        history = self._load_history()
        return history.get("jobs", [])
    
    # =========================================================================
    # 历史管理
    # =========================================================================
    
    def _load_history(self) -> Dict:
        """加载任务历史"""
        path = Path(self.HISTORY_FILE)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {"jobs": []}
    
    def _save_job(self, job_id: str, config: TrainingJobConfig) -> None:
        """保存任务记录"""
        history = self._load_history()
        
        history["jobs"].append({
            "job_id": job_id,
            "provider": self.provider_name,
            "created_at": datetime.now().isoformat(),
            "config": config.to_dict(),
        })
        
        # 只保留最近 100 条
        history["jobs"] = history["jobs"][-100:]
        
        with open(self.HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)


def quick_train(
    config_path: str = "configs/base.yaml",
    data_path: str = "data/train.jsonl",
    provider: str = "runpod",
    gpu_type: str = "RTX4090",
    stages: List[str] = None,
    wait: bool = True,
    download_path: str = "./outputs",
) -> str:
    """
    快速云训练便捷函数
    
    Args:
        config_path: 配置文件路径
        data_path: 数据文件路径
        provider: 云平台
        gpu_type: GPU 类型
        stages: 训练阶段
        wait: 是否等待完成
        download_path: 下载路径
        
    Returns:
        任务 ID
    """
    stages = stages or ["vqvae", "dynamics"]
    
    manager = JobManager(provider=provider)
    
    config = TrainingJobConfig(
        gpu_type=gpu_type,
        config_path=config_path,
        data_path=data_path,
        stages=stages,
    )
    
    job_id = manager.submit(config)
    
    if wait:
        def progress_callback(info: JobInfo):
            print(f"\r[{info.status.value}] {info.progress:.1%} ", end="")
            if info.current_stage:
                print(f"Stage: {info.current_stage} ", end="")
            if info.current_loss:
                print(f"Loss: {info.current_loss:.4f}", end="")
        
        print("Waiting for completion...")
        final_info = manager.wait(job_id, callback=progress_callback)
        print()
        
        if final_info.status.is_success:
            manager.download(job_id, download_path)
        else:
            print(f"Job failed: {final_info.error_message}")
    
    return job_id
