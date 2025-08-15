#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import click
import random
import asyncio
import logging
import hashlib
import traceback
import bittensor as bt
from dotenv import load_dotenv
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from aiobotocore.session import get_session
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file, save, load
from safetensors import safe_open
from trl import PPOTrainer, PPOConfig
from accelerate import Accelerator

__version__ = "0.0.0"

from .grail import Prover, Verifier, SATProblem, SATEnvironment, generate_sat_problem

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
WINDOW_LENGTH = 20  # Generate inferences every 20 blocks (increased for model downloads)
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")

# Model configuration
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama 1B model

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("grail")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    # GRAIL debug details only visible with -vv or higher
    if verbosity < 2:
        logging.getLogger("grail").setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    return v or default

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.trace("Connected")
    return SUBTENSOR


# --------------------------------------------------------------------------- #
#                   S3 helpers                                                #
# --------------------------------------------------------------------------- #
get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=f"https://{get_conf('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
    aws_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
    config=Config(max_pool_connections=256)
)

class TransferProgress:
    """Track upload/download progress and speed"""
    def __init__(self, total_size: int, operation: str):
        self.total_size = total_size
        self.operation = operation
        self.transferred = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_transferred = 0
        
    def update(self, bytes_transferred: int):
        self.transferred += bytes_transferred
        now = time.time()
        
        # Log progress every 2 seconds or on completion
        if now - self.last_log_time >= 2.0 or self.transferred >= self.total_size:
            elapsed = now - self.start_time
            speed_mbps = (self.transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            progress_pct = (self.transferred / self.total_size) * 100 if self.total_size > 0 else 0
            
            logger.info(f"📊 {self.operation}: {progress_pct:.1f}% ({self.transferred}/{self.total_size} bytes) @ {speed_mbps:.2f} MB/s")
            self.last_log_time = now
            self.last_transferred = self.transferred

async def upload_file_chunked(key: str, data: bytes, chunk_size: int = 5 * 1024 * 1024, max_retries: int = 3) -> bool:
    """Upload file in chunks with retry logic and progress logging"""
    total_size = len(data)
    progress = TransferProgress(total_size, f"Upload {key}")
    
    # For small files, use single upload
    if total_size <= chunk_size:
        logger.info(f"📤 Uploading {key} ({total_size} bytes)")
        return await _upload_single_chunk(key, data, progress, max_retries)
    
    # For large files, use multipart upload
    logger.info(f"📤 Starting chunked upload of {key} ({total_size} bytes, {(total_size + chunk_size - 1) // chunk_size} chunks)")
    
    try:
        async with get_client_ctx() as client:
            # Initiate multipart upload
            response = await client.create_multipart_upload(
                Bucket=get_conf("R2_BUCKET_ID"),
                Key=key
            )
            upload_id = response['UploadId']
            
            # Upload chunks concurrently with limited concurrency
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent chunks
            tasks = []
            
            for i in range(0, total_size, chunk_size):
                chunk_data = data[i:i + chunk_size]
                part_number = (i // chunk_size) + 1
                task = _upload_chunk_with_semaphore(semaphore, client, key, upload_id, part_number, chunk_data, progress, max_retries)
                tasks.append(task)
            
            # Wait for all chunks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for failures
            parts = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i+1} failed: {result}")
                    await client.abort_multipart_upload(
                        Bucket=get_conf("R2_BUCKET_ID"),
                        Key=key,
                        UploadId=upload_id
                    )
                    return False
                parts.append(result)
            
            # Complete multipart upload
            await client.complete_multipart_upload(
                Bucket=get_conf("R2_BUCKET_ID"),
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            elapsed = time.time() - progress.start_time
            speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Upload completed: {key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s")
            return True
            
    except Exception as e:
        logger.error(f"❌ Upload failed for {key}: {e}")
        return False

async def _upload_single_chunk(key: str, data: bytes, progress: TransferProgress, max_retries: int) -> bool:
    """Upload single chunk with retry logic"""
    for attempt in range(max_retries):
        try:
            async with get_client_ctx() as client:
                await client.put_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key, Body=data)
            progress.update(len(data))
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Upload attempt {attempt + 1} failed for {key}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Upload failed after {max_retries} attempts for {key}: {e}")
    return False

async def _upload_chunk_with_semaphore(semaphore, client, key: str, upload_id: str, part_number: int, data: bytes, progress: TransferProgress, max_retries: int):
    """Upload a single chunk with concurrency control and retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.upload_part(
                    Bucket=get_conf("R2_BUCKET_ID"),
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data
                )
                progress.update(len(data))
                return {
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Chunk {part_number} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

async def download_file_chunked(key: str, max_retries: int = 3) -> Optional[bytes]:
    """Download file in chunks with retry logic and progress logging"""
    for attempt in range(max_retries):
        try:
            async with get_client_ctx() as client:
                # Get object info first to know the size
                head_response = await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
                total_size = head_response['ContentLength']
                
                logger.info(f"📥 Downloading {key} ({total_size} bytes)")
                progress = TransferProgress(total_size, f"Download {key}")
                
                # For small files, download in one go
                chunk_size = 5 * 1024 * 1024  # 5MB chunks
                if total_size <= chunk_size:
                    response = await client.get_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
                    data = await response["Body"].read()
                    progress.update(len(data))
                    elapsed = time.time() - progress.start_time
                    speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    logger.info(f"✅ Download completed: {key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s")
                    return data
                
                # For large files, download in chunks
                chunks = []
                semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
                tasks = []
                
                for start in range(0, total_size, chunk_size):
                    end = min(start + chunk_size - 1, total_size - 1)
                    task = _download_chunk_with_semaphore(semaphore, client, key, start, end, progress, max_retries)
                    tasks.append(task)
                
                # Wait for all chunks
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures and reassemble
                for i, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        logger.error(f"Download chunk {i} failed: {result}")
                        raise result
                    chunks.append(result)
                
                data = b''.join(chunks)
                elapsed = time.time() - progress.start_time
                speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                logger.info(f"✅ Download completed: {key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s")
                return data
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Download attempt {attempt + 1} failed for {key}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts for {key}: {e}")
                return None

async def _download_chunk_with_semaphore(semaphore, client, key: str, start: int, end: int, progress: TransferProgress, max_retries: int):
    """Download a single chunk with concurrency control and retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.get_object(
                    Bucket=get_conf("R2_BUCKET_ID"),
                    Key=key,
                    Range=f'bytes={start}-{end}'
                )
                data = await response["Body"].read()
                progress.update(len(data))
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Download chunk {start}-{end} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

async def sink_window_inferences(wallet: bt.wallet, window_start: int, inferences: List[dict]):
    """Upload window of inferences to S3 with improved logging"""
    key = f"grail/windows/{wallet.hotkey.ss58_address}-window-{window_start}.json"
    
    # Pack all inferences into window data
    window_data = {
        "wallet": wallet.hotkey.ss58_address,
        "window_start": window_start,
        "window_length": WINDOW_LENGTH,
        "inference_count": len(inferences),
        "inferences": inferences,
        "timestamp": time.time()
    }
    
    body = json.dumps(window_data).encode()
    logger.debug(f"[SINK] window={window_start} count={len(inferences)} → key={key}")
    
    success = await upload_file_chunked(key, body)
    if success:
        logger.info(f"📤 Uploaded window data for window {window_start} ({len(inferences)} inferences)")
    else:
        logger.error(f"❌ Failed to upload window data for window {window_start}")

async def file_exists(key: str) -> bool:
    """Check if a file exists in the bucket without downloading it"""
    try:
        async with get_client_ctx() as client:
            await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
        return True
    except Exception:
        return False

async def list_bucket_files(prefix: str) -> List[str]:
    """List files in bucket with given prefix"""
    try:
        async with get_client_ctx() as client:
            response = await client.list_objects_v2(
                Bucket=get_conf("R2_BUCKET_ID"), 
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
    except Exception:
        logger.error("Failed to list bucket files with prefix %s", prefix, exc_info=True)
        return []

async def get_file(key: str):
    """Download and parse JSON file with improved error handling"""
    try:
        data = await download_file_chunked(key)
        if data:
            return json.loads(data.decode())
        return None
    except Exception as e:
        logger.debug(f"Failed to get file {key}: {e}")
        return None

async def save_model_state(model: AutoModelForCausalLM, hotkey: str, window: int):
    """Save model state as safetensors to S3 with chunked upload and progress logging"""
    key = f"grail/models/{hotkey}-{window}.safetensors"
    
    # Save model state dict as safetensors bytes
    from safetensors.torch import save_file
    import tempfile
    import os
    
    # Create temporary file for safetensors
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        logger.info(f"💾 Preparing model state for {hotkey} window {window}")
        # Save to temporary file
        save_file(model.state_dict(), temp_path)
        
        # Read file content as bytes
        with open(temp_path, 'rb') as f:
            body = f.read()
        
        file_size_mb = len(body) / (1024 * 1024)
        logger.info(f"📦 Model state prepared: {file_size_mb:.1f} MB")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    logger.debug(f"[MODEL] Saving model state for {hotkey} window {window} → {key}")
    
    # Use chunked upload with retry logic
    success = await upload_file_chunked(key, body)
    
    if success:
        logger.info(f"✅ Successfully uploaded model state for window {window}")
    else:
        logger.error(f"❌ Failed to upload model state for window {window}")
    
    return success

async def load_model_state(model: AutoModelForCausalLM, hotkey: str, window: int) -> bool:
    """Load model state from S3 with chunked download and progress logging"""
    key = f"grail/models/{hotkey}-{window}.safetensors"
    
    logger.info(f"🔍 Loading model state for {hotkey} window {window}")
    
    # Use chunked download with retry logic
    data = await download_file_chunked(key)
    
    if data is None:
        logger.debug(f"Model state not found for {key}")
        return False
    
    try:
        # Load safetensors from bytes using temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(data)
        
        try:
            # Load from temporary file
            from safetensors.torch import load_file
            state_dict = load_file(temp_path)
            model.load_state_dict(state_dict)
            
            file_size_mb = len(data) / (1024 * 1024)
            logger.info(f"✅ Successfully loaded model state for window {window} ({file_size_mb:.1f} MB)")
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model state for window {window}: {e}")
        return False

async def model_state_exists(hotkey: str, window: int) -> bool:
    """Check if model state exists for given hotkey and window"""
    key = f"grail/models/{hotkey}-{window}.safetensors"
    return await file_exists(key)

async def upload_valid_inferences(window: int, valid_inferences: List[dict]):
    """Upload validated inferences for training with chunked upload and progress logging"""
    key = f"grail/valid_inferences/{window}.json"
    
    data = {
        "window": window,
        "count": len(valid_inferences),
        "inferences": valid_inferences,
        "timestamp": time.time()
    }
    
    body = json.dumps(data).encode()
    logger.debug(f"[VALID] Uploading {len(valid_inferences)} valid inferences for window {window}")
    
    success = await upload_file_chunked(key, body)
    
    if success:
        logger.info(f"📤 Uploaded {len(valid_inferences)} valid inferences for window {window}")
    else:
        logger.error(f"❌ Failed to upload valid inferences for window {window}")
    
    return success

async def get_valid_inferences(window: int) -> List[dict]:
    """Download valid inferences for training"""
    key = f"grail/valid_inferences/{window}.json"
    
    try:
        data = await get_file(key)
        if data and 'inferences' in data:
            return data['inferences']
        return []
    except Exception:
        logger.debug("No valid inferences found for window %s", window)
        return []

# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #
def generate_prompt(hotkey_address: str, block_hash: str, nonce: int) -> str:
    """Generate prompt in the required format"""
    return f"Hey my name is {hotkey_address} it is currently {block_hash} days since friday and my fav number is {nonce}, tell me a story about these three facts"

def parse_filename(filename: str) -> Tuple[str, int, int]:
    """Parse filename to extract wallet, block, nonce"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    parts = basename.split('-')
    if len(parts) >= 3:
        wallet = parts[0]
        block = int(parts[1])
        nonce = int(parts[2])
        return wallet, block, nonce
    return None, None, None

def parse_window_filename(filename: str) -> Tuple[str, int]:
    """Parse window filename to extract wallet and window_start"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    # Format: {wallet}-window-{window_start}
    parts = basename.split('-')
    if len(parts) >= 3 and parts[1] == 'window':
        wallet = parts[0]
        window_start = int(parts[2])
        return wallet, window_start
    return None, None

def sign_inference(inference_data: dict, wallet: bt.wallet) -> dict:
    """Sign an inference using the wallet hotkey"""
    # Create challenge string from key inference data
    challenge = f"{inference_data['prompt']}{inference_data['block_hash']}{inference_data['nonce']}"
    inference_data['challenge'] = challenge
    inference_data['hotkey'] = wallet.hotkey.ss58_address
    inference_data['signature'] = wallet.hotkey.sign(data=challenge).hex()
    return inference_data

def verify_inference_signature(inference_data: dict) -> bool:
    """Verify the signature of an inference"""
    try:
        challenge = inference_data.get('challenge')
        hotkey = inference_data.get('hotkey')
        signature = inference_data.get('signature')
        
        if not all([challenge, hotkey, signature]):
            return False
            
        keypair = bt.Keypair(ss58_address=hotkey)
        return keypair.verify(data=challenge, signature=bytes.fromhex(signature))
    except Exception:
        return False

def derive_secret_key(hotkey_address: str) -> bytes:
    """Derive deterministic secret key from hotkey for verification"""
    return hashlib.sha256(f"grail_secret_{hotkey_address}".encode()).digest()

# Global storage for miner state
miner_inference_counts = defaultdict(list)  # track inferences per block for weight calculation

# --------------------------------------------------------------------------- #
#                               TRAINER                                       #
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model_name=LLAMA_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare for training
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)
        
    async def train_window(self, hotkey: str, window: int) -> bool:
        """Train model on SAT rollouts from previous window using GRPO and upload for future window"""
        
        # Download valid inferences from the previous window  
        valid_inferences = await get_valid_inferences(window - WINDOW_LENGTH)
        
        if not valid_inferences:
            logger.warning(f"No valid inferences found for window {window - WINDOW_LENGTH}")
            # Still upload base model state if no training data
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
            
        logger.info(f"🎓 Training on {len(valid_inferences)} SAT rollouts from window {window - WINDOW_LENGTH}")
        
        # Prepare training data for GRPO
        texts = []
        rewards = []
        successful_count = 0
        
        for inference in valid_inferences:
            try:
                # Extract SAT problem and rollout data
                commit = inference.get('commit', {})
                tokens = commit.get('tokens', [])
                rollout = commit.get('rollout', {})
                sat_problem = commit.get('sat_problem', {})
                
                if not tokens or not rollout:
                    continue
                
                # Decode the full sequence (SAT problem + solution attempt)
                full_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                texts.append(full_text)
                
                # Calculate reward based on SAT solving performance
                # GRPO rewards: higher for successful solutions, partial credit for progress
                if rollout.get('success', False):
                    # High reward for successful solution
                    reward = 1.0
                    successful_count += 1
                else:
                    # Partial reward based on satisfied clauses
                    satisfied = rollout.get('satisfied_clauses', 0)
                    total = len(sat_problem.get('clauses', [1]))  # Avoid division by zero
                    reward = -0.5 + (satisfied / total) * 0.5  # Range: [-0.5, 0]
                
                # Add trajectory reward (bonus for efficiency)
                trajectory = rollout.get('trajectory', [])
                if trajectory and rollout.get('success', False):
                    # Bonus for solving quickly
                    efficiency_bonus = max(0, 0.2 * (1 - len(trajectory) / (sat_problem.get('num_vars', 10) * 2)))
                    reward += efficiency_bonus
                
                rewards.append(reward)
                
            except Exception as e:
                logger.debug(f"Skipping invalid SAT rollout: {e}")
                continue
        
        if not texts:
            logger.warning("No valid training texts extracted")
            # Still upload base model state
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
            
        logger.info(f"📚 Training on {len(texts)} SAT rollouts ({successful_count} successful)")
        logger.info(f"📊 Average reward: {sum(rewards)/len(rewards):.3f}")
        
        # GRPO-style training: reinforce successful trajectories
        try:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)  # Lower LR for stability
            
            for epoch in range(2):  # Two epochs for better learning
                total_loss = 0
                batch_size = min(4, len(texts))  # Small batch size
                
                # Sort by rewards to prioritize learning from successful rollouts
                sorted_indices = sorted(range(len(texts)), key=lambda i: rewards[i], reverse=True)
                
                for batch_idx in range(0, len(sorted_indices), batch_size):
                    batch_indices = sorted_indices[batch_idx:batch_idx+batch_size]
                    batch_texts = [texts[i] for i in batch_indices]
                    batch_rewards = [rewards[i] for i in batch_indices]
                    
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # GRPO reward weighting: emphasize high-reward trajectories
                    # Normalize rewards to [0, 1] range for this batch
                    min_reward = min(batch_rewards)
                    max_reward = max(batch_rewards)
                    if max_reward > min_reward:
                        normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in batch_rewards]
                    else:
                        normalized_rewards = [0.5] * len(batch_rewards)
                    
                    # Apply reward-weighted loss
                    avg_normalized_reward = sum(normalized_rewards) / len(normalized_rewards)
                    reward_weight = 0.5 + avg_normalized_reward  # Range: [0.5, 1.5]
                    weighted_loss = loss * reward_weight
                    
                    # Backward pass
                    optimizer.zero_grad()
                    self.accelerator.backward(weighted_loss)
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += weighted_loss.item()
                    
                avg_loss = total_loss / (len(texts) // batch_size + 1)
                logger.info(f"Epoch {epoch+1} completed - avg loss: {avg_loss:.4f}")
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Still try to upload base model
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
        
        # Upload trained model state for future window (window + WINDOW_LENGTH)
        future_window = window + WINDOW_LENGTH
        logger.info(f"💾 Uploading trained model for future window {future_window}")
        success = await save_model_state(self.model, hotkey, future_window)
        
        if success:
            logger.info(f"✅ Successfully trained and uploaded model for window {future_window}")
        else:
            logger.error(f"❌ Failed to upload trained model for window {future_window}")
            
        return success

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """GRAIL CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
            
# --------------------------------------------------------------------------- #
#                               MINER                                         #
# --------------------------------------------------------------------------- #
@cli.command("mine")
def mine():    
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize model and prover
    logger.info(f"Loading base model: {LLAMA_MODEL}")
    prover = Prover(model_name=LLAMA_MODEL)
    # Set deterministic secret key based on hotkey
    prover.secret_key = derive_secret_key(wallet.hotkey.ss58_address)
    
    async def _run():
        subtensor = None
        last_window_start = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()                
                current_block = await subtensor.get_current_block()
                
                # Calculate current window start (blocks divisible by WINDOW_LENGTH)
                window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                
                # Only process if we're in a new window
                if window_start <= last_window_start:
                    await asyncio.sleep(2)  # Wait for new window
                    continue
                
                # Check if model state exists for current window, wait if not
                model_available = await model_state_exists(wallet.hotkey.ss58_address, window_start)
                if not model_available:
                    logger.info(f"⏳ Waiting for model state for window {window_start}...")
                    await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                    continue
                
                # Load model state for current window
                logger.info(f"📥 Loading model state for window {window_start}")
                try:
                    success = await load_model_state(prover.model, wallet.hotkey.ss58_address, window_start)
                    if success:
                        logger.info(f"✅ Loaded model state for window {window_start}")
                        # Update prover with new model state
                        prover.model.eval()
                    else:
                        logger.warning(f"⚠️ Failed to load model state for window {window_start}, using base model")
                except Exception as e:
                    logger.warning(f"Error loading model state: {e}, using base model")
                    pass
                
                logger.info(f"🔥 Starting inference generation for window {window_start}-{window_start + WINDOW_LENGTH - 1}")
                window_block_hash = await subtensor.get_block_hash(window_start)
                
                # Generate as many inferences as possible during this window
                inferences = []
                start_time = time.time()
                inference_count = 0
                
                # Generate inferences until the window closes
                while True:
                    current_block = await subtensor.get_current_block()
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                    
                    # Stop if we've moved to the next window
                    if current_window > window_start:
                        break
                    
                    try:
                        inference_count += 1
                        print(f"\r⚡ Generating SAT rollout {inference_count}...", end="", flush=True)
                        
                        # Generate unique seed for SAT problem
                        nonce = random.randint(1000, 9999)
                        sat_seed = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{nonce}"
                        
                        # Generate SAT problem from seed
                        difficulty = min(0.9, 0.3 + (inference_count * 0.01))  # Gradually increase difficulty
                        sat_problem = generate_sat_problem(sat_seed, difficulty)
                        
                        # Generate rollout with GRAIL proof
                        commit_data = prover.commit_rollout(sat_problem, window_block_hash)
                        proof_data = prover.open(window_block_hash)
                        
                        # Prepare inference data
                        inference_data = {
                            "window_start": window_start,
                            "block": current_block,
                            "nonce": nonce,
                            "sat_seed": sat_seed,
                            "difficulty": difficulty,
                            "block_hash": window_block_hash,
                            "commit": commit_data,
                            "proof": proof_data,
                            "timestamp": time.time()
                        }
                        
                        # Sign the inference
                        inference_data = sign_inference(inference_data, wallet)
                        
                        # Log successful rollouts
                        if commit_data["rollout"]["success"]:
                            logger.info(f"✅ Successfully solved SAT problem (vars={sat_problem.num_vars}, clauses={len(sat_problem.clauses)})")
                        
                        inferences.append(inference_data)
                        
                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate inference {inference_count}: {e}")
                        continue
                
                # Clear the progress line
                print("\r" + " " * 50 + "\r", end="")
                
                elapsed_time = time.time() - start_time
                logger.info(f"🎯 Generated {len(inferences)} inferences in {elapsed_time:.1f}s for window {window_start}")
                
                if inferences:
                    # Upload all inferences as a single window file
                    await sink_window_inferences(wallet, window_start, inferences)
                    logger.info(f"📤 Uploaded window {window_start} with {len(inferences)} inferences")
                else:
                    logger.warning(f"No inferences generated for window {window_start}")
                
                last_window_start = window_start
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in miner loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
                
    async def main():
        await asyncio.gather(
            _run(),
            watchdog()
        )
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
@cli.command("validate")
def validate():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize verifier
    logger.info(f"Loading base model for validation: {LLAMA_MODEL}")
    verifier = Verifier(model_name=LLAMA_MODEL)
    
    # Storage for inference counts per miner
    inference_counts = defaultdict(lambda: defaultdict(int))  # {hotkey: {window: count}}
    
    async def _run():
        subtensor = None
        last_processed_window = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()

                meta = await subtensor.metagraph(NETUID)
                current_block = await subtensor.get_current_block()
                
                # Calculate current and previous windows
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                # Process the previous complete window
                target_window = current_window - WINDOW_LENGTH
                
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(5)  # Wait for new window
                    continue
                
                # Check if model state exists for target window, wait if not
                model_available = await model_state_exists(wallet.hotkey.ss58_address, target_window)
                if not model_available:
                    logger.info(f"⏳ Waiting for model state for window {target_window}...")
                    await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                    continue
                
                logger.info(f"🔍 Processing window {target_window}-{target_window + WINDOW_LENGTH - 1}")
                
                # Load model state for target window
                logger.info(f"📥 Loading model state for window {target_window}")
                try:
                    success = await load_model_state(verifier.model, wallet.hotkey.ss58_address, target_window)
                    if success:
                        logger.info(f"✅ Loaded model state for window {target_window}")
                        verifier.model.eval()
                    else:
                        logger.warning(f"⚠️ Failed to load model state for window {target_window}, using base model")
                except Exception as e:
                    logger.warning(f"Error loading model state: {e}, using base model")
                    pass
                
                # Get block hash for the window start
                target_window_hash = await subtensor.get_block_hash(target_window)
                
                # Check for files from active hotkeys only
                logger.info(f"Checking files for {len(meta.hotkeys)} active hotkeys in window {target_window}")
                
                # Download and process files only from registered hotkeys
                total_valid_inferences = 0
                window_inference_counts = defaultdict(int)
                files_found = 0
                all_valid_inferences = []  # Store all valid inferences for uploading
                
                for wallet_addr in meta.hotkeys:
                    try:
                        # Construct expected filename for this hotkey and window
                        filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"
                        
                        # Check if file exists before downloading
                        if not await file_exists(filename):
                            continue
                        
                        files_found += 1
                        logger.debug(f"Found file for hotkey {wallet_addr}")
                        
                        window_data = await get_file(filename)
                        if not window_data:
                            logger.warning(f"Could not download {filename}")
                            continue
                        
                        file_wallet_addr = window_data.get("wallet")
                        window_start = window_data.get("window_start")
                        inferences = window_data.get("inferences", [])
                        
                        # Basic window validation
                        if file_wallet_addr != wallet_addr:
                            logger.warning(f"Wallet mismatch in {filename}: expected {wallet_addr}, got {file_wallet_addr}")
                            continue
                        
                        if window_start != target_window:
                            logger.warning(f"Window mismatch in {filename}: expected {target_window}, got {window_start}")
                            continue
                        
                        # Verify all rollouts in the window
                        valid_count = 0
                        successful_rollouts = 0
                        unique_solutions = set()  # Track unique successful solutions
                        nonces_seen = set()
                        
                        for inference in inferences:
                            try:
                                # Check required fields for SAT rollouts
                                required_fields = ["window_start", "nonce", "sat_seed", "block_hash", "commit", "proof", "challenge", "hotkey", "signature"]
                                if not all(field in inference for field in required_fields):
                                    logger.debug(f"Missing required fields in inference from {wallet_addr}")
                                    continue
                                
                                # Check window consistency
                                if inference["window_start"] != target_window:
                                    logger.debug(f"Window mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check block hash matches
                                if inference["block_hash"] != target_window_hash:
                                    logger.debug(f"Block hash mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check nonce uniqueness within window
                                nonce = inference["nonce"]
                                if nonce in nonces_seen:
                                    logger.debug(f"Duplicate nonce {nonce} in window from {wallet_addr}")
                                    continue
                                nonces_seen.add(nonce)
                                
                                # Verify signature
                                if not verify_inference_signature(inference):
                                    logger.debug(f"Invalid signature for inference from {wallet_addr}")
                                    continue
                                
                                # Verify SAT seed format
                                expected_seed = f"{wallet_addr}-{target_window_hash}-{nonce}"
                                if inference.get("sat_seed") != expected_seed:
                                    logger.debug(f"Invalid SAT seed in inference from {wallet_addr}")
                                    continue
                                
                                # Verify GRAIL proof and SAT rollout (spot checking for efficiency)
                                if random.random() < 0.3:  # 30% spot check for SAT problems
                                    try:
                                        logger.debug(f"Spot checking SAT rollout from {wallet_addr}")
                                        prover_secret_key = derive_secret_key(wallet_addr)
                                        is_valid = verifier.verify_rollout(inference["commit"], inference["proof"], prover_secret_key)
                                        if not is_valid:
                                            logger.debug(f"SAT rollout verification failed for {wallet_addr}")
                                            continue
                                    except Exception as e:
                                        logger.debug(f"Rollout verification error for {wallet_addr}: {e}")
                                        continue
                                
                                valid_count += 1
                                
                                # Track successful unique solutions
                                rollout = inference.get("commit", {}).get("rollout", {})
                                if rollout.get("success", False):
                                    successful_rollouts += 1
                                    # Create hash of solution for uniqueness
                                    assignment = rollout.get("assignment", [])
                                    solution_hash = hashlib.sha256(str(assignment).encode()).hexdigest()
                                    unique_solutions.add(solution_hash)
                                
                                # Add to collection of all valid inferences
                                all_valid_inferences.append(inference)
                                
                            except Exception as e:
                                logger.debug(f"Error processing inference from {wallet_addr}: {e}")
                                continue
                        
                        # Store metrics for this miner
                        window_inference_counts[wallet_addr] = {
                            "valid": valid_count,
                            "successful": successful_rollouts,
                            "unique": len(unique_solutions)
                        }
                        total_valid_inferences += valid_count
                        
                        logger.info(f"✅ {wallet_addr}: {valid_count} valid, {successful_rollouts} successful, {len(unique_solutions)} unique solutions")
                        
                    except Exception as e:
                        logger.warning(f"Error processing window file {filename}: {e}")
                        continue
                
                logger.info(f"📁 Found {files_found} window files from {len(meta.hotkeys)} active hotkeys")
                logger.info(f"🏁 Total valid inferences in window {target_window}: {total_valid_inferences}")
                
                # Upload all valid inferences for training
                if all_valid_inferences:
                    upload_success = await upload_valid_inferences(target_window, all_valid_inferences)
                    if upload_success:
                        logger.info(f"📤 Uploaded {len(all_valid_inferences)} valid inferences for training")
                    else:
                        logger.warning(f"⚠️ Failed to upload valid inferences for training")
                
                # Update global inference counts for weight calculation
                for hotkey, metrics in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = metrics
                
                # Compute weights based on unique successful rollouts
                weights = []
                for uid, hotkey in enumerate(meta.hotkeys):
                    # Calculate score over last 3 windows
                    recent_windows = range(max(0, target_window - 2*WINDOW_LENGTH), target_window + 1, WINDOW_LENGTH)
                    
                    total_unique = 0
                    total_successful = 0
                    total_valid = 0
                    
                    for w in recent_windows:
                        metrics = inference_counts[hotkey].get(w, {})
                        if isinstance(metrics, dict):
                            total_unique += metrics.get("unique", 0)
                            total_successful += metrics.get("successful", 0)
                            total_valid += metrics.get("valid", 0)
                        else:
                            # Backward compatibility
                            total_valid += metrics if isinstance(metrics, (int, float)) else 0
                    
                    # Scoring formula: prioritize unique solutions, then successful, then valid
                    # Weight = 0.6 * unique_ratio + 0.3 * success_ratio + 0.1 * valid_ratio
                    unique_score = min(1.0, total_unique / 10.0) if total_unique > 0 else 0
                    success_score = min(1.0, total_successful / 20.0) if total_successful > 0 else 0
                    valid_score = min(1.0, total_valid / 50.0) if total_valid > 0 else 0
                    
                    weight = 0.6 * unique_score + 0.3 * success_score + 0.1 * valid_score
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [0.0] * len(meta.hotkeys)
                
                # Log non-zero weights
                non_zero_weights = [(meta.hotkeys[i], weights[i]) for i in range(len(weights)) if weights[i] > 0]
                if non_zero_weights:
                    logger.info(f"⚖️  Setting weights for {len(non_zero_weights)} miners")
                    for hotkey, weight in non_zero_weights[:5]:  # Show top 5
                        logger.info(f"   {hotkey}: {weight:.4f}")
                else:
                    logger.info("⚖️  No miners received weights this window")
                
                # Set weights on network
                await subtensor.set_weights(
                    wallet=wallet,
                    netuid=NETUID,
                    uids=meta.uids,
                    weights=weights,
                    wait_for_inclusion=False
                )
                
                last_processed_window = target_window
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                               TRAINER CLI                                   #
# --------------------------------------------------------------------------- #
@cli.command("train")
def train():
    """Run the training process"""
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize trainer
    logger.info(f"Initializing trainer with model: {LLAMA_MODEL}")
    trainer = Trainer(model_name=LLAMA_MODEL)
    
    async def _run():
        subtensor = None
        last_processed_window = -1
        
        # Upload initial base model state on startup
        logger.info("🏁 Uploading initial base model state...")
        current_block = 0
        if subtensor is None:
            subtensor = await get_subtensor()
            current_block = await subtensor.get_current_block()
        
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        initial_window = current_window + WINDOW_LENGTH
        
        # Upload base model for the next window
        success = await save_model_state(trainer.model, wallet.hotkey.ss58_address, initial_window)
        if success:
            logger.info(f"✅ Uploaded initial model state for window {initial_window}")
        else:
            logger.error("❌ Failed to upload initial model state")
            return
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()
                    
                current_block = await subtensor.get_current_block()
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                
                # Process previous complete window for training
                target_window = current_window - WINDOW_LENGTH
                
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(10)  # Wait for new window
                    continue
                
                logger.info(f"🎓 Processing training for window {target_window}")
                
                # Train on previous window's valid inferences and upload for future window
                success = await trainer.train_window(wallet.hotkey.ss58_address, target_window)
                
                if success:
                    logger.info(f"✅ Completed training cycle for window {target_window}")
                else:
                    logger.warning(f"⚠️ Training cycle had issues for window {target_window}")
                
                last_processed_window = target_window
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in trainer loop: {e}. Continuing...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(30)  # Wait before retrying
                continue
    
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout=(60 * 15))  # 15 minute timeout for training
        )
    
    asyncio.run(main())