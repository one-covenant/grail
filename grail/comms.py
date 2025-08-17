"""S3/R2 communication utilities for GRAIL."""

import os
import time
import json
import gzip
import asyncio
import logging
import tempfile
import hashlib
from datetime import datetime
from typing import Any, List, Dict, Optional
from botocore.config import Config
from aiobotocore.session import get_session
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi, HfFolder
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# Protocol version for dataset versioning
PROTOCOL_VERSION = "v1.0.0"

# --------------------------------------------------------------------------- #
#                   S3/R2 Configuration                                       #
# --------------------------------------------------------------------------- #

def get_conf(key, default=None) -> Any:
    """Get configuration from environment variables."""
    v = os.getenv(key)
    if not v and default is None:
        raise ValueError(f"{key} not set. Please set the environment variable.")
    return v or default

get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=f"https://{get_conf('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
    aws_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
    config=Config(max_pool_connections=256)
)

# --------------------------------------------------------------------------- #
#                   Progress Tracking                                         #
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#                   Upload Functions                                          #
# --------------------------------------------------------------------------- #

async def upload_file_chunked(key: str, data: bytes, chunk_size: int = 100 * 1024 * 1024, max_retries: int = 3, compress: bool = True) -> bool:
    """Upload file in chunks optimized for H100 high-bandwidth - 100MB chunks with compression"""
    # Compress data if enabled and it's JSON
    if compress and key.endswith('.json'):
        original_size = len(data)
        data = gzip.compress(data, compresslevel=1)  # Fast compression
        key = key + '.gz'
        logger.info(f"🗜️ Compressed {original_size} → {len(data)} bytes ({100*(1-len(data)/original_size):.1f}% reduction)")
    
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
            semaphore = asyncio.Semaphore(30)  # High concurrency for H100 bandwidth
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

# --------------------------------------------------------------------------- #
#                   Download Functions                                        #
# --------------------------------------------------------------------------- #

async def download_file_chunked(key: str, max_retries: int = 3) -> Optional[bytes]:
    """Download file in chunks with automatic decompression"""
    actual_key = key
    is_compressed = False
    
    for attempt in range(max_retries):
        try:
            async with get_client_ctx() as client:
                # Try compressed version first if not already .gz
                if not key.endswith('.gz'):
                    try:
                        compressed_key = key + '.gz'
                        head_response = await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=compressed_key)
                        actual_key = compressed_key
                        is_compressed = True
                        logger.debug(f"Found compressed version: {compressed_key}")
                    except:
                        # Fallback to uncompressed
                        head_response = await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
                        actual_key = key
                else:
                    head_response = await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
                    is_compressed = key.endswith('.gz')
                total_size = head_response['ContentLength']
                
                logger.info(f"📥 Downloading {actual_key} ({total_size} bytes){' (compressed)' if is_compressed else ''}")
                progress = TransferProgress(total_size, f"Download {actual_key}")
                
                # For small files, download in one go
                chunk_size = 100 * 1024 * 1024  # 100MB chunks for H100 bandwidth
                if total_size <= chunk_size:
                    response = await client.get_object(Bucket=get_conf("R2_BUCKET_ID"), Key=actual_key)
                    data = await response["Body"].read()
                    progress.update(len(data))
                    elapsed = time.time() - progress.start_time
                    speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    logger.info(f"✅ Download completed: {actual_key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s")
                    # Decompress if needed
                    if is_compressed:
                        data = gzip.decompress(data)
                        logger.debug(f"🗜️ Decompressed to {len(data)} bytes")
                    return data
                
                # For large files, download in chunks
                chunks = []
                semaphore = asyncio.Semaphore(30)  # High concurrency for H100 bandwidth
                tasks = []
                
                for start in range(0, total_size, chunk_size):
                    end = min(start + chunk_size - 1, total_size - 1)
                    task = _download_chunk_with_semaphore(semaphore, client, actual_key, start, end, progress, max_retries)
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
                logger.info(f"✅ Download completed: {actual_key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s")
                
                # Decompress if needed
                if is_compressed:
                    data = gzip.decompress(data)
                    logger.debug(f"🗜️ Decompressed to {len(data)} bytes")
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

# --------------------------------------------------------------------------- #
#                   File Operations                                           #
# --------------------------------------------------------------------------- #

async def file_exists(key: str) -> bool:
    """Check if a file exists in the bucket (checks both compressed and uncompressed)"""
    try:
        async with get_client_ctx() as client:
            # Check for compressed version first
            if not key.endswith('.gz'):
                try:
                    await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key + '.gz')
                    return True
                except:
                    pass
            
            # Check for original key
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

async def get_file(key: str) -> Optional[Dict[str, Any]]:
    """Download and parse JSON file with improved error handling"""
    try:
        data = await download_file_chunked(key)
        if data:
            return json.loads(data.decode())
        return None
    except Exception as e:
        logger.debug(f"Failed to get file {key}: {e}")
        return None

# --------------------------------------------------------------------------- #
#                   GRAIL-specific Storage Functions                          #
# --------------------------------------------------------------------------- #

async def sink_window_inferences(wallet, window_start: int, inferences: List[dict]):
    """Upload window of inferences to S3 with improved logging"""
    key = f"grail/windows/{wallet.hotkey.ss58_address}-window-{window_start}.json"
    
    # Pack all inferences into window data
    window_data = {
        "wallet": wallet.hotkey.ss58_address,
        "window_start": window_start,
        "window_length": 20,  # WINDOW_LENGTH constant
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

# TODO(v2): Re-enable model state management for training
'''
async def save_model_state(model: AutoModelForCausalLM, hotkey: str, window: int):
    # Save model state as safetensors to S3 with chunked upload and progress logging
    key = f"grail/models/{hotkey}-{window}.safetensors"
    
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
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(data)
        
        try:
            # Load from temporary file
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
    # Check if model state exists for given hotkey and window
    key = f"grail/models/{hotkey}-{window}.safetensors"
    return await file_exists(key)
'''  # End of commented model state functions

async def upload_valid_rollouts(window: int, valid_rollouts: List[dict]):
    """Upload validated SAT rollouts for training with chunked upload and progress logging"""
    key = f"grail/valid_rollouts/{window}.json"
    
    data = {
        "window": window,
        "count": len(valid_rollouts),
        "rollouts": valid_rollouts,
        "timestamp": time.time()
    }
    
    body = json.dumps(data).encode()
    logger.debug(f"[VALID] Uploading {len(valid_rollouts)} valid rollouts for window {window}")
    
    success = await upload_file_chunked(key, body)
    
    if success:
        logger.info(f"📤 Uploaded {len(valid_rollouts)} valid rollouts for window {window}")
    else:
        logger.error(f"❌ Failed to upload valid rollouts for window {window}")
    
    return success

async def get_valid_rollouts(window: int) -> List[dict]:
    """
    Download valid SAT rollouts for training.
    
    These rollouts have already been:
    - Verified by validators using verify_rollout()
    - Confirmed to have valid GRAIL proofs (model identity verified)
    - Checked for SAT problem correctness and solution validity
    
    The trainer can safely use these for GRPO training.
    """
    key = f"grail/valid_rollouts/{window}.json"
    
    try:
        data = await get_file(key)
        if data and 'rollouts' in data:
            logger.info(f"Downloaded {len(data['rollouts'])} verified rollouts for window {window}")
            return data['rollouts']
        # Backward compatibility: check old format
        elif data and 'inferences' in data:
            logger.info(f"Downloaded {len(data['inferences'])} verified rollouts (legacy format) for window {window}")
            return data['inferences']
        return []
    except Exception:
        logger.debug("No valid rollouts found for window %s", window)
        return []

# --------------------------------------------------------------------------- #
#                   Hugging Face Dataset Upload                               #
# --------------------------------------------------------------------------- #

def login_huggingface():
    """
    Login to Hugging Face using token from environment or cache.
    This should be called once at startup.
    """
    try:
        from huggingface_hub import login, HfFolder
        
        # Check if already logged in
        existing_token = HfFolder.get_token()
        if existing_token:
            logger.info("Already logged into Hugging Face")
            return True
        
        # Try to get token from environment
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
            logger.info("✅ Successfully logged into Hugging Face")
            return True
        else:
            logger.warning("No HF_TOKEN found in environment. Set HF_TOKEN to enable dataset uploads.")
            logger.info("Get your token at: https://huggingface.co/settings/tokens")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to login to Hugging Face: {e}")
        return False

async def upload_to_huggingface(rollouts: List[Dict], window: int, version: str = None):
    """
    Upload rollouts to unified Hugging Face dataset.
    
    Dataset: grail/sat-rollouts (single dataset for all windows)
    Each rollout is versioned and includes window metadata.
    
    Args:
        rollouts: List of validated rollout dictionaries
        window: Window number for temporal tracking
        version: Protocol version (defaults to PROTOCOL_VERSION)
    """
    if not rollouts:
        logger.debug("No rollouts to upload to Hugging Face")
        return False
    
    if version is None:
        version = PROTOCOL_VERSION
    
    dataset_name = "grail/sat-rollouts"
    
    try:
        # Prepare rollouts with metadata
        processed_rollouts = []
        for rollout in rollouts:
            # Generate unique ID for each rollout
            rollout_id = hashlib.sha256(
                f"{rollout.get('hotkey', '')}_{window}_{rollout.get('nonce', '')}".encode()
            ).hexdigest()[:16]
            
            # Extract key information
            commit = rollout.get('commit', {})
            sat_problem = commit.get('sat_problem', {})
            rollout_data = commit.get('rollout', {})
            proof = rollout.get('proof', {})
            
            # Create flattened structure for dataset
            processed_rollout = {
                "id": rollout_id,
                "version": version,
                "window": window,
                "timestamp": rollout.get('timestamp', time.time()),
                "uploaded_at": datetime.utcnow().isoformat(),
                
                # Miner info
                "miner": rollout.get('hotkey', ''),
                "nonce": rollout.get('nonce', 0),
                
                # SAT problem
                "sat_seed": sat_problem.get('seed', ''),
                "sat_num_vars": sat_problem.get('num_vars', 0),
                "sat_num_clauses": len(sat_problem.get('clauses', [])),
                "sat_difficulty": sat_problem.get('difficulty', 0.5),
                "sat_clauses": json.dumps(sat_problem.get('clauses', [])),  # Store as JSON string
                
                # Solution
                "solution_success": rollout_data.get('success', False),
                "solution_assignment": json.dumps(rollout_data.get('assignment', [])),
                "solution_trajectory": json.dumps(rollout_data.get('trajectory', [])),
                "solution_satisfied_clauses": rollout_data.get('satisfied_clauses', 0),
                "solution_total_reward": rollout_data.get('total_reward', 0.0),
                
                # GRAIL proof (store as JSON strings for complex fields)
                "grail_tokens": json.dumps(commit.get('tokens', [])),
                "grail_s_vals": json.dumps(commit.get('s_vals', [])),
                "grail_signature": commit.get('signature', ''),
                "grail_beacon": json.dumps(commit.get('beacon', {})),
                "grail_indices": json.dumps(proof.get('indices', [])),
                
                # Metrics
                "token_count": len(commit.get('tokens', [])),
                "inference_count": rollout.get('inference_count', 1),
            }
            
            processed_rollouts.append(processed_rollout)
        
        # Create dataset from rollouts
        dataset = Dataset.from_list(processed_rollouts)
        
        # Check if we're logged in
        token = HfFolder.get_token()
        if not token:
            # Try to login if not already
            if not login_huggingface():
                logger.debug("Cannot upload to Hugging Face without login")
                return False
            token = HfFolder.get_token()
        
        # Push to Hugging Face Hub
        # This will append to existing dataset or create new one
        dataset.push_to_hub(
            dataset_name,
            token=token,
            private=False,  # Make it public for community access
            append=True,    # Append to existing dataset
        )
        
        logger.info(f"📤 Successfully uploaded {len(processed_rollouts)} rollouts to HF dataset {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        return False

async def download_from_huggingface(version: str = None, window: int = None, limit: int = None) -> List[Dict]:
    """
    Download rollouts from Hugging Face dataset with optional filtering.
    
    Args:
        version: Filter by protocol version (e.g., "v1.0.0")
        window: Filter by specific window number
        limit: Maximum number of rollouts to return
    
    Returns:
        List of rollout dictionaries
    """
    dataset_name = "grail/sat-rollouts"
    
    try:
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        # Apply filters
        if version:
            dataset = dataset.filter(lambda x: x["version"] == version)
        
        if window is not None:
            dataset = dataset.filter(lambda x: x["window"] == window)
        
        # Convert to list of dicts
        rollouts = dataset.to_list()
        
        # Apply limit if specified
        if limit:
            rollouts = rollouts[:limit]
        
        # Decode JSON strings back to objects
        for rollout in rollouts:
            rollout["sat_clauses"] = json.loads(rollout.get("sat_clauses", "[]"))
            rollout["solution_assignment"] = json.loads(rollout.get("solution_assignment", "[]"))
            rollout["solution_trajectory"] = json.loads(rollout.get("solution_trajectory", "[]"))
            rollout["grail_tokens"] = json.loads(rollout.get("grail_tokens", "[]"))
            rollout["grail_s_vals"] = json.loads(rollout.get("grail_s_vals", "[]"))
            rollout["grail_beacon"] = json.loads(rollout.get("grail_beacon", "{}"))
            rollout["grail_indices"] = json.loads(rollout.get("grail_indices", "[]"))
        
        logger.info(f"Downloaded {len(rollouts)} rollouts from HF dataset")
        return rollouts
        
    except Exception as e:
        logger.error(f"Failed to download from Hugging Face: {e}")
        return []