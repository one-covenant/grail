#!/usr/bin/env python3
"""
Local service orchestrator for grail - replaces docker-compose functionality.
This script manages multiple services similar to docker-compose.
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages multiple services with dependencies."""
    
    def __init__(self, env_file: str = ".env"):
        self.services: Dict[str, subprocess.Popen] = {}
        self.env_file = env_file
        self.base_env = self._load_environment()
        
    def _load_environment(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env = os.environ.copy()
        
        if Path(self.env_file).exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env[key.strip()] = value.strip().strip('"\'')
        
        return env
    
    def start_minio_mock(self):
        """Start a mock S3 service (moto) instead of MinIO."""
        logger.info("Starting mock S3 service...")
        
        # Check if moto is installed
        try:
            import moto
        except ImportError:
            logger.error("moto not installed. Run: uv pip install moto[server]")
            return False
        
        env = self.base_env.copy()
        env['MOTO_PORT'] = '9000'
        
        cmd = [sys.executable, '-m', 'moto.server', 's3', '-p', '9000']
        self.services['s3'] = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for service to be ready
        time.sleep(2)
        logger.info("Mock S3 service started on port 9000")
        return True
    
    def start_miner(self, miner_id: int = 1):
        """Start a miner service."""
        logger.info(f"Starting miner-{miner_id}...")
        
        env = self.base_env.copy()
        # Override with miner-specific settings
        env.update({
            'NODE_TYPE': 'miner',
            'BT_WALLET_COLD': f'miner{chr(64 + miner_id)}',  # minerA, minerB, etc.
            'BT_WALLET_HOT': f'hot{chr(64 + miner_id)}',
            'R2_ENDPOINT_URL': 'http://localhost:9000',
            'R2_FORCE_PATH_STYLE': 'true',
            'GRAIL_WINDOW_LENGTH': '3',
            'CUDA_VISIBLE_DEVICES': str(miner_id - 1),
        })
        
        cmd = ['uv', 'run', 'grail', '-vv', 'mine', '--no-drand']
        
        self.services[f'miner-{miner_id}'] = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Miner-{miner_id} started")
        return True
    
    def start_validator(self):
        """Start a validator service."""
        logger.info("Starting validator...")
        
        env = self.base_env.copy()
        env.update({
            'NODE_TYPE': 'validator',
            'BT_WALLET_COLD': 'validator',
            'BT_WALLET_HOT': 'hotV',
            'R2_ENDPOINT_URL': 'http://localhost:9000',
            'R2_FORCE_PATH_STYLE': 'true',
            'GRAIL_WINDOW_LENGTH': '3',
            'CUDA_VISIBLE_DEVICES': '2',
        })
        
        cmd = ['uv', 'run', 'grail', '-vv', 'validate', '--no-drand']
        
        self.services['validator'] = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info("Validator started")
        return True
    
    def monitor_service(self, name: str, process: subprocess.Popen):
        """Monitor a service and log its output."""
        def read_output(pipe, prefix):
            for line in iter(pipe.readline, b''):
                if line:
                    logger.info(f"[{name}] {prefix}: {line.decode().strip()}")
        
        # Start threads to read stdout and stderr
        threading.Thread(
            target=read_output, 
            args=(process.stdout, "OUT"),
            daemon=True
        ).start()
        
        threading.Thread(
            target=read_output,
            args=(process.stderr, "ERR"),
            daemon=True
        ).start()
    
    def start_all(self):
        """Start all services with proper dependencies."""
        # Start S3 mock first
        if not self.start_minio_mock():
            return False
        
        # Wait a bit for S3 to be ready
        time.sleep(3)
        
        # Start miners
        self.start_miner(1)
        self.start_miner(2)
        
        # Wait for miners to initialize
        time.sleep(5)
        
        # Start validator
        self.start_validator()
        
        # Monitor all services
        for name, process in self.services.items():
            self.monitor_service(name, process)
        
        return True
    
    def stop_all(self):
        """Stop all running services."""
        logger.info("Stopping all services...")
        
        for name, process in self.services.items():
            if process.poll() is None:  # Still running
                logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
        
        self.services.clear()
        logger.info("All services stopped")
    
    def status(self):
        """Check status of all services."""
        for name, process in self.services.items():
            if process.poll() is None:
                logger.info(f"{name}: RUNNING (PID: {process.pid})")
            else:
                logger.info(f"{name}: STOPPED (exit code: {process.poll()})")
    
    def wait(self):
        """Wait for all services to complete."""
        try:
            while any(p.poll() is None for p in self.services.values()):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupt received, stopping services...")
            self.stop_all()


def main():
    parser = argparse.ArgumentParser(description='Local service orchestrator for grail')
    parser.add_argument('command', choices=['up', 'down', 'status'], 
                       help='Command to execute')
    parser.add_argument('--detach', '-d', action='store_true',
                       help='Run in background (detached mode)')
    parser.add_argument('--env-file', default='.env',
                       help='Environment file to use')
    
    args = parser.parse_args()
    
    manager = ServiceManager(env_file=args.env_file)
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Signal received, stopping services...")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.command == 'up':
        if manager.start_all():
            if not args.detach:
                logger.info("Services started. Press Ctrl+C to stop.")
                manager.wait()
            else:
                logger.info("Services started in detached mode.")
                # In real detached mode, you'd write PIDs to a file
        else:
            logger.error("Failed to start services")
            sys.exit(1)
    
    elif args.command == 'down':
        manager.stop_all()
    
    elif args.command == 'status':
        manager.status()


if __name__ == '__main__':
    main()

