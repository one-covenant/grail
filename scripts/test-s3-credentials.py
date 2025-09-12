#!/usr/bin/env python3
"""Test script to verify S3/MinIO credentials and connectivity."""

import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def test_credentials() -> None:
    """Test S3/MinIO credentials from environment variables."""

    # Get credentials from environment
    endpoint_url = os.getenv("R2_ENDPOINT_URL", "http://localhost:9000")
    bucket_name = os.getenv("R2_BUCKET_ID", "grail")

    # Test both read and write credentials
    credentials = {
        "write": {
            "access_key": os.getenv("R2_WRITE_ACCESS_KEY_ID", "minioadmin"),
            "secret_key": os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "minioadmin"),
        },
        "read": {
            "access_key": os.getenv("R2_READ_ACCESS_KEY_ID", "minioadmin"),
            "secret_key": os.getenv("R2_READ_SECRET_ACCESS_KEY", "minioadmin"),
        },
    }

    print("🔍 Testing S3/MinIO credentials")
    print(f"   Endpoint: {endpoint_url}")
    print(f"   Bucket: {bucket_name}")
    print()

    for cred_type, creds in credentials.items():
        print(f"Testing {cred_type.upper()} credentials:")
        print(f"  Access Key ID: {creds['access_key']}")
        print(f"  Secret Key: {'*' * len(creds['secret_key']) if creds['secret_key'] else 'EMPTY'}")

        try:
            # Create S3 client
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=creds["access_key"],
                aws_secret_access_key=creds["secret_key"],
                region_name="us-east-1",
                use_ssl=False,
            )

            # Test connection by listing buckets
            response = s3_client.list_buckets()
            print("  ✅ Connection successful!")
            print(f"     Buckets found: {[b['Name'] for b in response['Buckets']]}")

            # Check if our bucket exists
            if bucket_name in [b["Name"] for b in response["Buckets"]]:
                print(f"  ✅ Bucket '{bucket_name}' exists")

                # Try to list objects (read test)
                try:
                    objects = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
                    obj_count = objects.get("KeyCount", 0)
                    print(f"     Objects in bucket: {obj_count}")
                except ClientError as e:
                    print(f"  ⚠️  Cannot list objects: {e}")

                # Try to upload a test object (write test) - only for write credentials
                if cred_type == "write":
                    try:
                        test_key = "test/credential-test.txt"
                        s3_client.put_object(
                            Bucket=bucket_name,
                            Key=test_key,
                            Body=b"Test upload from credential verification script",
                        )
                        print(f"  ✅ Write test successful (uploaded {test_key})")

                        # Clean up test file
                        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
                        print("     Cleaned up test file")
                    except ClientError as e:
                        print(f"  ❌ Write test failed: {e}")
            else:
                print(f"  ⚠️  Bucket '{bucket_name}' does not exist")

        except NoCredentialsError:
            print("  ❌ No credentials provided")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidAccessKeyId":
                print("  ❌ Invalid Access Key ID - not recognized by MinIO")
            elif error_code == "SignatureDoesNotMatch":
                print("  ❌ Invalid Secret Access Key - signature mismatch")
            else:
                print(f"  ❌ Connection failed: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")

        print()

    # Test from inside Docker network perspective
    print("Testing from Docker network perspective:")
    print("  If running inside Docker, the endpoint should be: http://s3:9000")
    print("  If running from host, the endpoint should be: http://localhost:9000")
    print()

    # Show what the miners/validators should use
    print("📝 Environment variables for miners/validators in docker-compose:")
    print("   R2_ENDPOINT_URL: http://s3:9000")
    print("   R2_BUCKET_ID: grail")
    print("   R2_WRITE_ACCESS_KEY_ID: minioadmin")
    print("   R2_WRITE_SECRET_ACCESS_KEY: minioadmin")
    print("   R2_READ_ACCESS_KEY_ID: minioadmin")
    print("   R2_READ_SECRET_ACCESS_KEY: minioadmin")


if __name__ == "__main__":
    test_credentials()
