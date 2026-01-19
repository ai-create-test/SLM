"""
RunPod Provider - RunPod 云平台适配器

RunPod API 文档: https://docs.runpod.io/
"""

import os
import time
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

from .base import CloudProvider, JobStatus, JobInfo

if TYPE_CHECKING:
    from ..job_manager import TrainingJobConfig


class RunPodProvider(CloudProvider):
    """
    RunPod 云平台提供商
    
    支持:
    - Serverless Endpoints
    - GPU Pods
    
    GPU 价格参考 (2024):
    - RTX 4090: ~$0.34/hr
    - A100 40GB: ~$0.60/hr
    - A100 80GB: ~$1.19/hr
    - H100: ~$1.99/hr
    """
    
    name = "runpod"
    
    supported_gpus = [
        "RTX3090", "RTX4090", "A5000",
        "A40", "A100-40GB", "A100-80GB", 
        "H100-80GB", "H100-SXM",
    ]
    
    # GPU 价格 (USD/hour)
    GPU_PRICES = {
        "RTX3090": 0.20,
        "RTX4090": 0.34,
        "A5000": 0.25,
        "A40": 0.35,
        "A100-40GB": 0.60,
        "A100-80GB": 1.19,
        "H100-80GB": 1.99,
        "H100-SXM": 2.50,
    }
    
    API_BASE = "https://api.runpod.io/graphql"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: RunPod API Key (或设置 RUNPOD_API_KEY 环境变量)
        """
        if not api_key:
            api_key = os.environ.get("RUNPOD_API_KEY")
        
        self.api_key = api_key
        self._client = None
        
        # 延迟验证 (允许 dry-run)
        self._validated = False
    
    def _ensure_client(self):
        """确保 API 客户端可用"""
        if self._validated:
            return
        
        if not self.api_key:
            raise ValueError(
                "RunPod API key required. Set RUNPOD_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # 尝试导入 runpod SDK
        try:
            import runpod
            runpod.api_key = self.api_key
            self._client = runpod
            self._validated = True
        except ImportError:
            # 使用 HTTP 回退
            self._client = None
            self._validated = True
    
    def submit_job(self, config: "TrainingJobConfig") -> str:
        """提交训练任务到 RunPod"""
        self._ensure_client()
        
        if self._client is None:
            # Dry-run 模式
            job_id = f"dry-run-{int(time.time())}"
            print(f"[DRY RUN] Would submit job with config:")
            print(f"  GPU: {config.gpu_type} x {config.gpu_count}")
            print(f"  Image: {config.docker_image}")
            return job_id
        
        # 构建训练命令
        train_cmd = self._build_train_command(config)
        
        # 创建 Pod
        pod_config = {
            "name": config.name or f"nf-train-{int(time.time())}",
            "imageName": config.docker_image,
            "gpuTypeId": self._normalize_gpu_type(config.gpu_type),
            "gpuCount": config.gpu_count,
            "volumeInGb": config.disk_gb,
            "containerDiskInGb": 20,
            "dockerArgs": train_cmd,
            "env": config.env_vars,
        }
        
        try:
            pod = self._client.create_pod(**pod_config)
            return pod["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to create RunPod: {e}")
    
    def get_job_info(self, job_id: str) -> JobInfo:
        """获取任务信息"""
        self._ensure_client()
        
        if self._client is None or job_id.startswith("dry-run"):
            return JobInfo(
                job_id=job_id,
                status=JobStatus.PENDING,
                provider=self.name,
            )
        
        try:
            pod = self._client.get_pod(job_id)
            
            status = self._map_status(pod.get("desiredStatus", ""))
            
            return JobInfo(
                job_id=job_id,
                status=status,
                provider=self.name,
                gpu_type=pod.get("gpuType"),
                gpu_count=pod.get("gpuCount", 1),
                created_at=self._parse_time(pod.get("createdAt")),
            )
        except Exception as e:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.FAILED,
                provider=self.name,
                error_message=str(e),
            )
    
    def get_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """获取任务日志"""
        self._ensure_client()
        
        if self._client is None or job_id.startswith("dry-run"):
            return ["[DRY RUN] No logs available"]
        
        try:
            logs = self._client.get_pod_logs(job_id)
            lines = logs.split("\n") if logs else []
            return lines[-tail:]
        except Exception:
            return []
    
    def download_outputs(self, job_id: str, local_path: str) -> None:
        """下载任务产出"""
        self._ensure_client()
        
        if job_id.startswith("dry-run"):
            print("[DRY RUN] Would download outputs")
            return
        
        # RunPod 使用 rsync 或 scp
        # 实际实现需要 SSH 连接
        print(f"To download outputs, use:")
        print(f"  runpodctl receive {job_id}:/workspace/outputs {local_path}")
    
    def terminate_job(self, job_id: str) -> None:
        """终止任务"""
        self._ensure_client()
        
        if self._client is None or job_id.startswith("dry-run"):
            print("[DRY RUN] Would terminate job")
            return
        
        try:
            self._client.terminate_pod(job_id)
        except Exception as e:
            raise RuntimeError(f"Failed to terminate pod: {e}")
    
    def estimate_cost(self, config: "TrainingJobConfig") -> float:
        """估算成本"""
        gpu_price = self.GPU_PRICES.get(config.gpu_type, 0.50)
        return gpu_price * config.gpu_count * config.max_hours
    
    def get_available_gpus(self) -> List[Dict[str, Any]]:
        """获取可用 GPU 列表"""
        self._ensure_client()
        
        if self._client is None:
            return [{"type": gpu, "price": self.GPU_PRICES.get(gpu, 0)} 
                    for gpu in self.supported_gpus]
        
        try:
            gpus = self._client.get_gpus()
            return gpus
        except Exception:
            return []
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _build_train_command(self, config: "TrainingJobConfig") -> str:
        """构建训练命令"""
        stages = " ".join(config.stages)
        
        cmd = f"""
cd /workspace && 
git clone https://github.com/user/neuralflow.git . 2>/dev/null || true &&
pip install -r requirements.txt &&
python scripts/train.py \\
    --config {config.config_path} \\
    --data {config.data_path} \\
    --stages {stages} \\
    --output /workspace/outputs
"""
        return cmd.strip()
    
    def _normalize_gpu_type(self, gpu_type: str) -> str:
        """标准化 GPU 类型名称"""
        mapping = {
            "RTX4090": "NVIDIA GeForce RTX 4090",
            "A100-40GB": "NVIDIA A100-PCIE-40GB",
            "A100-80GB": "NVIDIA A100 80GB PCIe",
            "H100": "NVIDIA H100 80GB HBM3",
        }
        return mapping.get(gpu_type, gpu_type)
    
    def _map_status(self, runpod_status: str) -> JobStatus:
        """映射 RunPod 状态到 JobStatus"""
        mapping = {
            "CREATED": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "EXITED": JobStatus.COMPLETED,
            "STOPPED": JobStatus.CANCELLED,
        }
        return mapping.get(runpod_status.upper(), JobStatus.PENDING)
    
    def _parse_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """解析时间字符串"""
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except Exception:
            return None
