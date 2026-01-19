"""
Sync Utils - 数据同步工具

用于本地和云端之间的数据传输。
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class SyncManager:
    """
    数据同步管理器
    
    支持:
    - 本地文件打包/解包
    - 远程复制 (SSH/SCP)
    - 检查点同步
    """
    
    def __init__(self, work_dir: str = "."):
        self.work_dir = Path(work_dir)
    
    def prepare_upload_package(
        self,
        config_path: str,
        data_path: str,
        output_path: str = "upload_package.tar.gz",
        include_code: bool = True,
    ) -> str:
        """
        准备上传包
        
        Args:
            config_path: 配置文件路径
            data_path: 数据文件路径
            output_path: 输出包路径
            include_code: 是否包含代码
            
        Returns:
            包文件路径
        """
        import tarfile
        
        output_path = Path(output_path)
        
        with tarfile.open(output_path, "w:gz") as tar:
            # 添加配置
            if Path(config_path).exists():
                tar.add(config_path, arcname=f"configs/{Path(config_path).name}")
            
            # 添加数据
            if Path(data_path).exists():
                tar.add(data_path, arcname=f"data/{Path(data_path).name}")
            
            # 添加代码
            if include_code:
                for subdir in ["app", "scripts"]:
                    subpath = self.work_dir / subdir
                    if subpath.exists():
                        tar.add(str(subpath), arcname=subdir)
                
                # requirements
                req_file = self.work_dir / "requirements.txt"
                if req_file.exists():
                    tar.add(str(req_file), arcname="requirements.txt")
        
        print(f"Created upload package: {output_path}")
        return str(output_path)
    
    def extract_download_package(
        self,
        package_path: str,
        output_dir: str = "./downloaded",
    ) -> str:
        """
        解压下载包
        
        Args:
            package_path: 包文件路径
            output_dir: 输出目录
            
        Returns:
            输出目录路径
        """
        import tarfile
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(package_path, "r:gz") as tar:
            tar.extractall(output_dir)
        
        print(f"Extracted to: {output_dir}")
        return str(output_dir)
    
    def sync_checkpoints(
        self,
        remote_host: str,
        remote_path: str,
        local_path: str,
        ssh_key: Optional[str] = None,
    ) -> None:
        """
        同步检查点 (使用 rsync)
        
        Args:
            remote_host: 远程主机
            remote_path: 远程路径
            local_path: 本地路径
            ssh_key: SSH 密钥路径
        """
        import subprocess
        
        ssh_opts = ""
        if ssh_key:
            ssh_opts = f"-e 'ssh -i {ssh_key}'"
        
        cmd = f"rsync -avz {ssh_opts} {remote_host}:{remote_path}/ {local_path}/"
        
        print(f"Syncing checkpoints: {remote_host}:{remote_path} -> {local_path}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Sync failed: {e}")
    
    def list_checkpoints(self, path: str) -> List[Dict[str, Any]]:
        """
        列出检查点
        
        Args:
            path: 检查点目录
            
        Returns:
            检查点信息列表
        """
        path = Path(path)
        checkpoints = []
        
        if not path.exists():
            return checkpoints
        
        for subdir in path.iterdir():
            if subdir.is_dir():
                config_file = subdir / "config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                    
                    checkpoints.append({
                        "name": subdir.name,
                        "path": str(subdir),
                        "size_mb": sum(
                            f.stat().st_size for f in subdir.rglob("*") if f.is_file()
                        ) / (1024 * 1024),
                        "created": datetime.fromtimestamp(
                            subdir.stat().st_mtime
                        ).isoformat(),
                        "config": config,
                    })
        
        return sorted(checkpoints, key=lambda x: x["created"], reverse=True)
    
    def cleanup_old_checkpoints(
        self,
        path: str,
        keep_last: int = 5,
    ) -> int:
        """
        清理旧检查点
        
        Args:
            path: 检查点目录
            keep_last: 保留最近 N 个
            
        Returns:
            删除的数量
        """
        checkpoints = self.list_checkpoints(path)
        
        to_delete = checkpoints[keep_last:]
        
        for ckpt in to_delete:
            shutil.rmtree(ckpt["path"])
            print(f"Deleted: {ckpt['name']}")
        
        return len(to_delete)


def create_training_script(config: Dict[str, Any]) -> str:
    """
    创建云端训练脚本
    
    Args:
        config: 训练配置
        
    Returns:
        脚本内容
    """
    stages = " ".join(config.get("stages", ["vqvae", "dynamics"]))
    config_path = config.get("config_path", "configs/base.yaml")
    data_path = config.get("data_path", "data/train.jsonl")
    
    script = f'''#!/bin/bash
set -e

echo "=== NeuralFlow Cloud Training ==="
echo "Starting at $(date)"

# 安装依赖
cd /workspace
pip install -r requirements.txt -q

# 运行训练
python scripts/train.py \\
    --config {config_path} \\
    --data {data_path} \\
    --stages {stages} \\
    --output /workspace/outputs

echo "Training completed at $(date)"

# 打包输出
cd /workspace
tar -czf outputs.tar.gz outputs/
echo "Outputs packaged"
'''
    return script
