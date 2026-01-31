#!/usr/bin/env python3
"""Script to retrieve account_id and credentials for uid 0 from the chain."""

import asyncio
import os
import sys

import bittensor as bt
from dotenv import load_dotenv

from grail.infrastructure.chain import GrailChainManager
from grail.shared.schemas import BucketCredentials

load_dotenv()


async def retrieve_uid0_credentials() -> None:
    """Retrieve and display credentials for uid 0."""

    # Load configuration from environment
    netuid: int = int(os.getenv("GRAIL_NETUID", "1"))
    wallet_name: str = os.getenv("GRAIL_WALLET_NAME", "default")
    hotkey_name: str = os.getenv("GRAIL_HOTKEY_NAME", "default")
    network: str = os.getenv("BT_NETWORK", "finney")

    # Get credentials from environment
    account_id: str = os.getenv("BUCKET_ACCOUNT_ID", "")
    bucket_name: str = os.getenv("BUCKET_NAME", "")
    read_access_key_id: str = os.getenv("BUCKET_READ_ACCESS_KEY_ID", "")
    read_secret_access_key: str = os.getenv(
        "BUCKET_READ_SECRET_ACCESS_KEY", ""
    )
    write_access_key_id: str = os.getenv("BUCKET_WRITE_ACCESS_KEY_ID", "")
    write_secret_access_key: str = os.getenv(
        "BUCKET_WRITE_SECRET_ACCESS_KEY", ""
    )

    # Validate environment
    if not all(
        [
            account_id,
            bucket_name,
            read_access_key_id,
            read_secret_access_key,
            write_access_key_id,
            write_secret_access_key,
        ]
    ):
        msg = "❌ Error: Missing required environment variables"
        print(msg + " for credentials")
        print("   Required: BUCKET_ACCOUNT_ID, BUCKET_NAME,")
        print(
            "             BUCKET_READ_ACCESS_KEY_ID,"
        )
        print(
            "             BUCKET_READ_SECRET_ACCESS_KEY, "
            "BUCKET_WRITE_ACCESS_KEY_ID,"
        )
        print("             BUCKET_WRITE_SECRET_ACCESS_KEY")
        sys.exit(1)

    try:
        print(f"🔗 Connecting to network: {network}, netuid: {netuid}")

        # Initialize Bittensor connection
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        subtensor = bt.subtensor(network=network)
        metagraph = subtensor.metagraph(netuid)

        # Create credentials object
        credentials = BucketCredentials(
            bucket_name=bucket_name,
            account_id=account_id,
            read_access_key_id=read_access_key_id,
            read_secret_access_key=read_secret_access_key,
            write_access_key_id=write_access_key_id,
            write_secret_access_key=write_secret_access_key,
        )

        # Initialize chain manager
        chain_manager = GrailChainManager(
            config=type("Config", (), {"netuid": netuid})(),
            wallet=wallet,
            metagraph=metagraph,
            subtensor=subtensor,
            credentials=credentials,
            fetch_interval=300,
        )

        # Initialize and wait for commitments
        print("📡 Initializing chain manager and fetching commitments...")
        await chain_manager.initialize()

        # Retrieve bucket for uid 0
        uid_0_bucket = chain_manager.get_bucket(0)

        print("\n" + "=" * 70)
        print("📊 UID 0 CREDENTIALS FROM CHAIN")
        print("=" * 70)

        if uid_0_bucket:
            print("\n✅ Found commitment for UID 0")
            print(
                f"\n  📦 Bucket Name:              "
                f"{uid_0_bucket.name.strip()}"
            )
            print(
                f"  👤 Account ID:              "
                f"{uid_0_bucket.account_id.strip()}"
            )
            print(
                f"  🔑 Read Access Key ID:      "
                f"{uid_0_bucket.access_key_id.strip()}"
            )
            print(
                f"  🔐 Read Secret Access Key:  "
                f"{uid_0_bucket.secret_access_key.strip()}"
            )
        else:
            print("\n❌ No commitment found for UID 0 on chain")

        print("\n" + "=" * 70)
        print("📋 LOCAL WRITE CREDENTIALS (Not on chain)")
        print("=" * 70)
        print(f"\n  🔑 Write Access Key ID:     {write_access_key_id}")
        print(
            f"  🔐 Write Secret Access Key: {write_secret_access_key}"
        )

        print("\n" + "=" * 70)
        print("\n✨ Credentials retrieved successfully!")

        # Cleanup
        chain_manager.stop()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(retrieve_uid0_credentials())
