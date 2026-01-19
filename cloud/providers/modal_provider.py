"""
Modal Provider - Modal 云平台适配器

Modal 是 Serverless 云计算平台，按秒计费。
Modal 文档: https://modal.com/docs
"""

import os
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

from .base import CloudProvider, JobStatus, JobInfo

if TYPE_CHECKING:
    from ..job_manager import TrainingJobConfig


class ModalProvider(CloudProvider):
    """
    Modal 云平台提供商
    
    特点:
    - Serverless 按秒计费
    - Python 原生集成
    - 适合短任务和调试
    
    使用需要安装: pip install modal
    """
    
    name = "modal"
    
    supported_gpus = [
        "T4", "L4", "A10G", "A100-40GB", "A100-80GB", "H100",
    ]
    
    # GPU 价格估算 (USD/hour)
    GPU_PRICES = {
        "T4": 0.20,
        "L4": 0.35,
        "A10G": 0.50,
        "A100-40GB": 1.10,
        "A100-80GB": 1.60,
        "H100": 3.00,
    }
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: Modal token (或设置 MODAL_TOKEN_ID/MODAL_TOKEN_SECRET)
        """
        self.api_key = api_key or os.environ.get("MODAL_TOKEN_ID")
        self._client = None
        self._validated = False
        self._jobs: Dict[str, Dict] = {}
    
    def _ensure_client(self):
        """确保客户端可用"""
        if self._validated:
            return
        
        try:
            import modal
            self._client = modal
            self._validated = True
        except ImportError:
            self._client = None
            self._validated = True
    
    def submit_job(self, config: "TrainingJobConfig") -> str:
        """提交训练任务到 Modal"""
        self._ensure_client()
        
        job_id = f"modal-{int(time.time())}"
        
        if self._client is None:
            print(f"[DRY RUN] Modal job would be submitted:")
            print(f"  GPU: {config.gpu_type} x {config.gpu_count}")
            print(f"  Config: {config.config_path}")
            self._jobs[job_id] = {"status": "pending", "config": config}
            return job_id
        
        # Modal 实际实现需要定义 App
        # 这里展示结构，实际需要 modal.App
        print(f"Modal job submitted: {job_id}")
        print("Note: Full Modal integration requires modal.App definition")
        
        self._jobs[job_id] = {
            "status": "submitted",
            "config": config,
            "created_at": datetime.now(),
        }
        
        return job_id
    
    def get_job_info(self, job_id: str) -> JobInfo:
        """获取任务信息"""
        self._ensure_client()
        
        job_data = self._jobs.get(job_id, {})
        status_str = job_data.get("status", "unknown")
        
        status_mapping = {
            "pending": JobStatus.PENDING,
            "submitted": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED,
        }
        
        return JobInfo(
            job_id=job_id,
            status=status_mapping.get(status_str, JobStatus.PENDING),
            provider=self.name,
            created_at=job_data.get("created_at"),
        )
    
    def get_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """获取任务日志"""
        return [f"[Modal] Logs for job {job_id} - use 'modal logs' CLI"]
    
    def download_outputs(self, job_id: str, local_path: str) -> None:
        """下载任务产出"""
        print(f"Modal outputs are typically stored in Modal Volumes")
        print(f"Use: modal volume get <volume-name> <path> {local_path}")
    
    def terminate_job(self, job_id: str) -> None:
        """终止任务"""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "cancelled"
        print(f"Modal job {job_id} termination requested")
    
    def estimate_cost(self, config: "TrainingJobConfig") -> float:
        """估算成本"""
        gpu_price = self.GPU_PRICES.get(config.gpu_type, 0.50)
        return gpu_price * config.gpu_count * config.max_hours
    
    def get_modal_app_template(self) -> str:
        """
        获取 Modal App 模板代码
        
        实际使用时需要用户创建 Modal App
        """
        return '''
import modal

app = modal.App("neuralflow-training")

# 定义训练镜像
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "safetensors",
    "pyyaml",
    "faiss-cpu",
)

# 创建数据卷
volume = modal.Volume.from_name("neuralflow-data", create_if_missing=True)

@app.function(
    gpu="A100",
    timeout=28800,  # 8 hours
    image=image,
    volumes={"/data": volume},
)
def train(config_path: str, data_path: str, stages: list):
    import sys
    sys.path.insert(0, "/workspace")
    
    from app.model import NeuralFlowModel
    from app.training import UnifiedTrainer
    
    model = NeuralFlowModel.from_preset("base")
    trainer = UnifiedTrainer(model, output_dir="/data/outputs")
    
    # ... training logic
    
    return {"status": "completed"}

# 使用: modal run train.py::train
'''
