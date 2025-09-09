# R2 Credentials Management Design for GRAIL

## Overview

This document outlines the design for implementing a dual-credential system for R2 (Cloudflare R2 or compatible S3 storage) in the GRAIL codebase. The system uses the same bucket and account but with different access credentials:

1. **Read Credentials**: Read-only access keys posted to the chain as commitments, shared with everyone for reading rollout data
2. **Write Credentials**: Read-write access keys stored locally, used by miners/validators to write to their buckets

## Current State Analysis

### Existing R2 Integration
- GRAIL currently uses environment variables for R2 credentials:
  - `R2_ACCOUNT_ID`: Account identifier for R2
  - `R2_WRITE_ACCESS_KEY_ID`: Write access key
  - `R2_WRITE_SECRET_ACCESS_KEY`: Write secret key
  - `R2_BUCKET_ID`: Bucket identifier
- All operations use the same write credentials
- No chain commitment mechanism exists currently

### Example Implementation Reference
The provided `chain.py` and `schema.py` files demonstrate:
- `ChainManager` class for managing chain interactions
- `Bucket` schema with fields: name, account_id, access_key_id, secret_access_key
- Commitment mechanism to post bucket info to chain
- Periodic fetching of commitments from chain
- Verification that local config matches chain commitment

## Proposed Design

### 1. Data Structures

#### Bucket Schema Extension
```python
# grail/shared/schemas.py
class BucketCredentials(BaseModel):
    """Dual credential configuration for R2 buckets
    
    Same bucket and account, but different access keys:
    - Read credentials: read-only access (shared on chain)
    - Write credentials: read-write access (kept private)
    """
    
    # Shared bucket configuration
    bucket_name: str
    account_id: str
    
    # Read-only credentials (shared on chain)
    read_access_key_id: str
    read_secret_access_key: str
    
    # Read-write credentials (local only, never shared)
    write_access_key_id: str
    write_secret_access_key: str
    
    @property
    def read_commitment(self) -> str:
        """Generate commitment string for chain (128 chars total)"""
        # Truncate/pad to exactly 32 chars each for 128 total
        return (
            self.bucket_name[:32].ljust(32) +
            self.read_access_key_id[:32].ljust(32) +
            self.read_secret_access_key[:64].ljust(64)
        )
```

### 2. Chain Management Integration

#### Enhanced ChainManager
```python
# grail/infrastructure/chain.py
class GrailChainManager:
    """GRAIL-specific chain manager extending the base ChainManager"""
    
    def __init__(self, config, wallet, credentials: BucketCredentials):
        self.credentials = credentials
        self.wallet = wallet
        self.config = config
        self.netuid = config.netuid
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(self.netuid)
        
        # Commitment tracking
        self.commitments = {}  # uid -> read_credentials
        self.fetch_interval = 600  # 10 minutes
        self._fetch_task = None
        
    async def initialize(self):
        """Initialize chain manager and verify/commit credentials"""
        # Fetch existing commitments
        await self.fetch_commitments()
        
        # Check if our commitment matches what's on chain
        try:
            uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            existing = self.commitments.get(uid)
            
            if not existing or existing != self.credentials.read_commitment:
                logger.info("Committing new read credentials to chain")
                self.commit_read_credentials()
        except ValueError:
            logger.warning("Hotkey not found in metagraph, will commit when registered")
            
        # Start periodic fetching
        self.start_commitment_fetcher()
```

### 3. Integration Points

#### Miner Integration (`grail/cli/mine.py`)
```python
def mine(...):
    # Load credentials from environment
    credentials = load_r2_credentials()
    
    # Initialize chain manager
    chain_manager = GrailChainManager(config, wallet, credentials)
    await chain_manager.initialize()
    
    # Use write credentials for uploading rollouts
    async with get_client_ctx(credentials.write_credentials) as client:
        # Upload rollout data
        ...
    
    # Share bucket info for validators to read
    # (This happens automatically via chain_manager.initialize())
```

#### Validator Integration (`grail/cli/validate.py`)
```python
def validate(...):
    # Load local write credentials
    credentials = load_r2_credentials()
    
    # Initialize chain manager
    chain_manager = GrailChainManager(config, wallet, credentials)
    await chain_manager.initialize()
    
    # Fetch miner read credentials from chain
    miner_buckets = chain_manager.get_all_buckets()
    
    for uid, read_creds in miner_buckets.items():
        if read_creds:
            # Use miner's read credentials to fetch their rollouts
            async with get_client_ctx(read_creds) as client:
                # Download and validate rollouts
                ...
```

### 4. Credential Loading

#### Environment Variable Structure
```bash
# Shared configuration (same for read and write)
R2_BUCKET_ID=<your_bucket_name>
R2_ACCOUNT_ID=<your_account_id>

# Read-only credentials (will be shared on chain)
R2_READ_ACCESS_KEY_ID=<your_read_only_key>
R2_READ_SECRET_ACCESS_KEY=<your_read_only_secret>

# Read-write credentials (local only, never shared)
R2_WRITE_ACCESS_KEY_ID=<your_write_key>
R2_WRITE_SECRET_ACCESS_KEY=<your_write_secret>
```

#### Credential Loader
```python
# grail/infrastructure/credentials.py
def load_r2_credentials() -> BucketCredentials:
    """Load R2 credentials from environment variables"""
    # Same bucket and account for both read and write
    bucket_id = get_conf("R2_BUCKET_ID")
    account_id = get_conf("R2_ACCOUNT_ID")
    
    return BucketCredentials(
        # Shared bucket configuration
        bucket_name=bucket_id,
        account_id=account_id,
        
        # Read-only credentials (to be shared on chain)
        read_access_key_id=get_conf("R2_READ_ACCESS_KEY_ID"),
        read_secret_access_key=get_conf("R2_READ_SECRET_ACCESS_KEY"),
        
        # Read-write credentials (private, never shared)
        write_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
        write_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
    )
```

### 5. Modified Communication Layer

#### Updated S3 Client Context
```python
# grail/infrastructure/comms.py
def get_client_ctx(credentials: Union[BucketCredentials, dict], use_write: bool = True):
    """Create S3 client with specified credentials
    
    Args:
        credentials: Either BucketCredentials object or dict with credential fields
        use_write: If True and credentials is BucketCredentials, use write creds
    """
    if isinstance(credentials, BucketCredentials):
        # Same account and bucket, different access keys
        account_id = credentials.account_id
        bucket_name = credentials.bucket_name
        if use_write:
            access_key = credentials.write_access_key_id
            secret_key = credentials.write_secret_access_key
        else:
            access_key = credentials.read_access_key_id
            secret_key = credentials.read_secret_access_key
    else:
        # Handle dict format (from chain commitments)
        account_id = credentials['account_id']
        access_key = credentials['access_key_id']
        secret_key = credentials['secret_access_key']
    
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    return get_session().create_client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(...)
    )
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `grail/infrastructure/chain.py` with `GrailChainManager` class
2. Add `BucketCredentials` schema to `grail/shared/schemas.py`
3. Create `grail/infrastructure/credentials.py` for credential loading
4. Update `grail/infrastructure/comms.py` to support dual credentials

### Phase 2: Miner Integration
1. Modify `grail/cli/mine.py` to:
   - Load dual credentials on startup
   - Initialize chain manager and commit read credentials
   - Use write credentials for uploading rollouts
2. Update rollout upload functions to use write credentials

### Phase 3: Validator Integration
1. Modify `grail/cli/validate.py` to:
   - Load local write credentials
   - Fetch miner read credentials from chain
   - Use appropriate credentials for reading miner rollouts
2. Update validation logic to handle per-miner bucket access

### Phase 4: Testing & Migration
1. Add unit tests for credential management
2. Add integration tests for chain commitment/retrieval
3. Create migration guide for existing deployments
4. Update documentation with new environment variables

## Security Considerations

1. **Credential Isolation**: Write credentials must NEVER be committed to chain
2. **Validation**: Always verify signatures on chain commitments
3. **Access Control**: Ensure read credentials have minimal permissions (read-only)
4. **Rotation**: Design should support credential rotation without breaking validation
5. **Error Handling**: Graceful fallback if chain commitments are unavailable

## Backwards Compatibility

To maintain compatibility during migration:
1. Check for new environment variables first
2. Fall back to old single-credential system if new vars not found
3. Log warnings about deprecated configuration
4. Provide clear migration timeline

## Benefits

1. **Security**: Write credentials remain private while sharing read-only access
2. **Simplicity**: Single bucket and account simplifies management
3. **Transparency**: Validators can verify miner data availability
4. **Decentralization**: No central authority needed for data access
5. **Auditability**: All participants can verify rollout data
6. **Cost Efficiency**: Single bucket reduces storage costs and complexity

## Future Extensions

1. **Multi-provider Support**: Allow different miners to use different storage providers
2. **Credential Rotation**: Automated rotation with grace period
3. **Access Analytics**: Track who's reading whose data
4. **Compression Negotiation**: Advertise supported compression formats in commitments
5. **Replication**: Support multiple read endpoints for redundancy