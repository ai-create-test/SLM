"""
Lambda Labs Provider - Lambda Labs 云平台适配器

Lambda Labs API 文档: https://cloud.lambdalabs.com/api/v1/docs
"""

import os
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

from .base import CloudProvider, JobStatus, JobInfo

if TYPE_CHECKING:
    from ..job_manager import TrainingJobConfig


class LambdaLabsProvider(CloudProvider):
    """
    Lambda Labs 云平台提供商
    
    特点:
    - 高性能 GPU (A100, H100)
    - 简单透明定价
    - 适合正式训练
    
    使用需要安装: pip install lambda-cloud-client
    """
    
    name = "lambda"
    
    supported_gpus = [
        "A6000", "A100-40GB", "A100-80GB", "H100-80GB",
    ]
    
    # GPU 价格 (USD/hour)
    GPU_PRICES = {
        "A6000": 0.80,
        "A100-40GB": 1.29,
        "A100-80GB": 1.79,
        "H100-80GB": 2.99,
    }
    
    API_BASE = "https://cloud.lambdalabs.com/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: Lambda Labs API Key (或设置 LAMBDA_API_KEY)
        """
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        self._validated = False
        self._instances: Dict[str, Dict] = {}
    
    def _get_headers(self) -> Dict[str, str]:
        """获取 API 请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _api_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """发送 API 请求"""
        import urllib.request
        import json
        
        url = f"{self.API_BASE}/{endpoint}"
        
        req = urllib.request.Request(url, method=method)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        
        if data:
            req.data = json.dumps(data).encode()
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            raise RuntimeError(f"Lambda Labs API error: {e}")
    
    def submit_job(self, config: "TrainingJobConfig") -> str:
        """提交训练任务到 Lambda Labs"""
        job_id = f"lambda-{int(time.time())}"
        
        if not self.api_key:
            print(f"[DRY RUN] Lambda Labs job would be submitted:")
            print(f"  GPU: {config.gpu_type} x {config.gpu_count}")
            print(f"  Config: {config.config_path}")
            self._instances[job_id] = {"status": "pending", "config": config}
            return job_id
        
        # 查找可用实例类型
        instance_type = self._map_gpu_to_instance(config.gpu_type)
        
        # 启动实例
        launch_data = {
            "region_name": "us-west-1",  # 默认区域
            "instance_type_name": instance_type,
            "ssh_key_names": [],
            "quantity": 1,
            "name": config.name or f"nf-train-{int(time.time())}",
        }
        
        try:
            response = self._api_request("POST", "instance-operations/launch", launch_data)
            instance_id = response.get("data", {}).get("instance_ids", [job_id])[0]
            
            self._instances[instance_id] = {
                "status": "starting",
                "config": config,
                "created_at": datetime.now(),
            }
            
            return instance_id
        except Exception as e:
            print(f"Failed to launch Lambda Labs instance: {e}")
            return job_id
    
    def get_job_info(self, job_id: str) -> JobInfo:
        """获取任务信息"""
        local_data = self._instances.get(job_id, {})
        
        if not self.api_key or job_id.startswith("lambda-"):
            status_str = local_data.get("status", "pending")
            status_mapping = {
                "pending": JobStatus.PENDING,
                "starting": JobStatus.STARTING,
                "running": JobStatus.RUNNING,
                "completed": JobStatus.COMPLETED,
            }
            return JobInfo(
                job_id=job_id,
                status=status_mapping.get(status_str, JobStatus.PENDING),
                provider=self.name,
            )
        
        try:
            response = self._api_request("GET", f"instances/{job_id}")
            instance = response.get("data", {})
            
            status = self._map_status(instance.get("status", ""))
            
            return JobInfo(
                job_id=job_id,
                status=status,
                provider=self.name,
                gpu_type=instance.get("instance_type", {}).get("name"),
                created_at=self._parse_time(instance.get("created_at")),
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
        return [
            f"[Lambda Labs] Instance {job_id}",
            "SSH into instance to view logs:",
            f"  ssh ubuntu@<instance-ip>",
            f"  cat /workspace/train.log | tail -{tail}",
        ]
    
    def download_outputs(self, job_id: str, local_path: str) -> None:
        """下载任务产出"""
        instance_info = self.get_job_info(job_id)
        print(f"Download outputs using SCP:")
        print(f"  scp -r ubuntu@<instance-ip>:/workspace/outputs {local_path}")
    
    def terminate_job(self, job_id: str) -> None:
        """终止任务"""
        if job_id in self._instances:
            self._instances[job_id]["status"] = "terminated"
        
        if not self.api_key:
            print(f"[DRY RUN] Would terminate instance {job_id}")
            return
        
        try:
            self._api_request("POST", "instance-operations/terminate", {
                "instance_ids": [job_id]
            })
            print(f"Terminated instance {job_id}")
        except Exception as e:
            print(f"Failed to terminate: {e}")
    
    def estimate_cost(self, config: "TrainingJobConfig") -> float:
        """估算成本"""
        gpu_price = self.GPU_PRICES.get(config.gpu_type, 1.50)
        return gpu_price * config.gpu_count * config.max_hours
    
    def get_available_gpus(self) -> List[Dict[str, Any]]:
        """获取可用 GPU 列表"""
        if not self.api_key:
            return [
                {"type": gpu, "price_per_hour": price, "available": True}
                for gpu, price in self.GPU_PRICES.items()
            ]
        
        try:
            response = self._api_request("GET", "instance-types")
            return response.get("data", {})
        except Exception:
            return []
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _map_gpu_to_instance(self, gpu_type: str) -> str:
        """映射 GPU 类型到实例类型"""
        mapping = {
            "A6000": "gpu_1x_a6000",
            "A100-40GB": "gpu_1x_a100",
            "A100-80GB": "gpu_1x_a100_sxm4",
            "H100-80GB": "gpu_1x_h100_sxm5",
        }
        return mapping.get(gpu_type, "gpu_1x_a100")
    
    def _map_status(self, lambda_status: str) -> JobStatus:
        """映射 Lambda Labs 状态"""
        mapping = {
            "active": JobStatus.RUNNING,
            "booting": JobStatus.STARTING,
            "terminated": JobStatus.COMPLETED,
            "unhealthy": JobStatus.FAILED,
        }
        return mapping.get(lambda_status.lower(), JobStatus.PENDING)
    
    def _parse_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """解析时间"""
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except Exception:
            return None
