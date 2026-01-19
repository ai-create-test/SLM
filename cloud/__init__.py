"""
Cloud Training Module - 云训练支持

提供多平台云 GPU 训练能力:
- RunPod
- Modal
- Lambda Labs
"""

from .job_manager import JobManager, TrainingJobConfig, JobStatus
from .providers import get_provider, CloudProvider

__all__ = [
    "JobManager",
    "TrainingJobConfig",
    "JobStatus",
    "get_provider",
    "CloudProvider",
]
