"""
Cloud Provider Base - 云平台抽象基类

定义所有云平台提供商必须实现的接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..job_manager import TrainingJobConfig


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"          # 等待中
    QUEUED = "queued"            # 排队中
    STARTING = "starting"        # 启动中
    RUNNING = "running"          # 运行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消
    TIMEOUT = "timeout"          # 超时
    
    @property
    def is_terminal(self) -> bool:
        """是否是终态"""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, 
                        JobStatus.CANCELLED, JobStatus.TIMEOUT)
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self == JobStatus.COMPLETED


@dataclass
class JobInfo:
    """任务信息"""
    job_id: str
    status: JobStatus
    provider: str
    
    # 时间信息
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 资源信息
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    
    # 状态信息
    progress: float = 0.0          # 0.0 - 1.0
    current_stage: Optional[str] = None
    current_epoch: int = 0
    current_loss: Optional[float] = None
    
    # 成本信息
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    
    # 日志
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # 产出
    output_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "provider": self.provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "error_message": self.error_message,
            "output_path": self.output_path,
        }


class CloudProvider(ABC):
    """
    云平台抽象基类
    
    所有云平台提供商必须实现此接口。
    
    使用示例:
        provider = RunPodProvider(api_key="xxx")
        
        # 提交任务
        job_id = provider.submit_job(config)
        
        # 查询状态
        info = provider.get_job_info(job_id)
        print(f"Status: {info.status}, Progress: {info.progress:.1%}")
        
        # 获取日志
        logs = provider.get_logs(job_id)
        
        # 下载结果
        provider.download_outputs(job_id, "./outputs")
        
        # 取消任务
        provider.terminate_job(job_id)
    """
    
    # 平台名称
    name: str = "base"
    
    # 支持的 GPU 类型
    supported_gpus: List[str] = []
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: API 密钥
            **kwargs: 平台特定参数
        """
        self.api_key = api_key
        self._validate_api_key()
    
    def _validate_api_key(self) -> None:
        """验证 API 密钥"""
        if not self.api_key:
            import os
            env_key = f"{self.name.upper()}_API_KEY"
            self.api_key = os.environ.get(env_key)
            
            if not self.api_key:
                raise ValueError(
                    f"API key required. Set {env_key} environment variable "
                    f"or pass api_key parameter."
                )
    
    # =========================================================================
    # 抽象方法 (必须实现)
    # =========================================================================
    
    @abstractmethod
    def submit_job(self, config: "TrainingJobConfig") -> str:
        """
        提交训练任务
        
        Args:
            config: 训练任务配置
            
        Returns:
            任务 ID
        """
        pass
    
    @abstractmethod
    def get_job_info(self, job_id: str) -> JobInfo:
        """
        获取任务信息
        
        Args:
            job_id: 任务 ID
            
        Returns:
            任务信息
        """
        pass
    
    @abstractmethod
    def get_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """
        获取任务日志
        
        Args:
            job_id: 任务 ID
            tail: 返回最后 N 行
            
        Returns:
            日志行列表
        """
        pass
    
    @abstractmethod
    def download_outputs(self, job_id: str, local_path: str) -> None:
        """
        下载任务产出
        
        Args:
            job_id: 任务 ID
            local_path: 本地保存路径
        """
        pass
    
    @abstractmethod
    def terminate_job(self, job_id: str) -> None:
        """
        终止任务
        
        Args:
            job_id: 任务 ID
        """
        pass
    
    # =========================================================================
    # 可选方法 (可覆盖)
    # =========================================================================
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobInfo]:
        """
        列出任务
        
        Args:
            status: 按状态过滤
            
        Returns:
            任务列表
        """
        raise NotImplementedError()
    
    def estimate_cost(self, config: "TrainingJobConfig") -> float:
        """
        估算成本
        
        Args:
            config: 训练任务配置
            
        Returns:
            预估成本 (USD)
        """
        return 0.0
    
    def get_available_gpus(self) -> List[Dict[str, Any]]:
        """
        获取可用 GPU 列表
        
        Returns:
            GPU 信息列表
        """
        return []
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否正常
        """
        return True
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def wait_for_completion(
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
        import time
        
        start_time = time.time()
        
        while True:
            info = self.get_job_info(job_id)
            
            if callback:
                callback(info)
            
            if info.status.is_terminal:
                return info
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
            
            time.sleep(poll_interval)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
