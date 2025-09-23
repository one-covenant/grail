"""
Configuration management for the monitoring system.

This module provides utilities for loading and managing monitoring configuration
from environment variables and other sources.
"""

from __future__ import annotations

import os
from typing import Any


class MonitoringConfig:
    """Configuration management for monitoring system.

    This class provides static methods to load monitoring configuration
    from environment variables and create specialized configurations
    for different operation modes (mining, validation, training).
    """

    @staticmethod
    def from_environment() -> dict[str, Any]:
        """Load monitoring configuration from environment variables.

        Environment variables:
            GRAIL_MONITORING_BACKEND: Backend type ("wandb", "null")
            WANDB_PROJECT: WandB project name
            WANDB_ENTITY: WandB entity/team name
            WANDB_MODE: WandB mode ("online", "offline", "disabled")
            WANDB_TAGS: Comma-separated list of tags
            WANDB_NOTES: Description/notes for runs
            GRAIL_METRIC_BUFFER_SIZE: Size of metric buffer before flushing
            GRAIL_METRIC_FLUSH_INTERVAL: Interval in seconds between flushes

        Returns:
            Configuration dictionary
        """
        # Parse tags from comma-separated string
        tags_str = os.getenv("WANDB_TAGS", "grail,bittensor")
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

        return {
            "backend_type": os.getenv("GRAIL_MONITORING_BACKEND", "wandb"),
            "project": os.getenv("WANDB_PROJECT", "grail"),
            "entity": os.getenv("WANDB_ENTITY"),
            "mode": os.getenv("WANDB_MODE", "online"),
            "tags": tags,
            "notes": os.getenv("WANDB_NOTES", "GRAIL production monitoring"),
            "buffer_size": int(os.getenv("GRAIL_METRIC_BUFFER_SIZE", "100")),
            "flush_interval": float(os.getenv("GRAIL_METRIC_FLUSH_INTERVAL", "30.0")),
            "resume": os.getenv("WANDB_RESUME", "allow"),
        }

    @staticmethod
    def for_mining(wallet_name: str | None = None) -> dict[str, Any]:
        """Get configuration specific to mining operations.

        Args:
            wallet_name: Name of the Bittensor wallet (optional)

        Returns:
            Configuration dictionary for mining
        """
        config = MonitoringConfig.from_environment()

        # Generate run name
        wallet_part = wallet_name or os.getenv("BT_WALLET_NAME", "default")
        run_name = f"mining_{wallet_part}"

        # Add mining-specific configuration
        config.update(
            {
                "run_name": run_name,
                "tags": config["tags"] + ["mining"],
                "hyperparameters": {
                    "operation_type": "mining",
                    "wallet_name": wallet_part,
                    "model_name": os.getenv("GRAIL_MODEL_NAME", "microsoft/DialoGPT-medium"),
                },
            }
        )

        return config

    @staticmethod
    def for_validation(wallet_name: str | None = None) -> dict[str, Any]:
        """Get configuration specific to validation operations.

        Args:
            wallet_name: Name of the Bittensor wallet (optional)

        Returns:
            Configuration dictionary for validation
        """
        config = MonitoringConfig.from_environment()

        # Generate run name
        wallet_part = wallet_name or os.getenv("BT_WALLET_NAME", "default")
        run_name = f"validation_{wallet_part}"

        # Add validation-specific configuration
        config.update(
            {
                "run_name": run_name,
                "tags": config["tags"] + ["validation"],
                "hyperparameters": {
                    "operation_type": "validation",
                    "wallet_name": wallet_part,
                    "netuid": int(os.getenv("NETUID", "1")),
                },
            }
        )

        return config

    @staticmethod
    def for_training(wallet_name: str | None = None) -> dict[str, Any]:
        """Get configuration specific to training operations.

        Args:
            wallet_name: Name of the Bittensor wallet (optional)

        Returns:
            Configuration dictionary for training
        """
        config = MonitoringConfig.from_environment()

        # Generate run name
        wallet_part = wallet_name or os.getenv("BT_WALLET_NAME", "default")
        run_name = f"training_{wallet_part}"

        # Add training-specific configuration
        config.update(
            {
                "run_name": run_name,
                "tags": config["tags"] + ["training"],
                "hyperparameters": {
                    "operation_type": "training",
                    "wallet_name": wallet_part,
                    "model_name": os.getenv("GRAIL_MODEL_NAME", "microsoft/DialoGPT-medium"),
                    "rollouts_per_problem": int(os.getenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4")),
                },
            }
        )

        return config

    @staticmethod
    def get_wallet_name() -> str:
        """Get wallet name from environment or default.

        Returns:
            Wallet name for use in configurations
        """
        return os.getenv("BT_WALLET_NAME", "default")

    @staticmethod
    def is_monitoring_enabled() -> bool:
        """Check if monitoring is enabled.

        Returns:
            True if monitoring is enabled, False otherwise
        """
        backend = os.getenv("GRAIL_MONITORING_BACKEND", "wandb")
        mode = os.getenv("WANDB_MODE", "online")

        return backend != "null" and mode != "disabled"

    @staticmethod
    def get_debug_config() -> dict[str, Any]:
        """Get configuration for debugging/development.

        Returns:
            Configuration suitable for development and debugging
        """
        config = MonitoringConfig.from_environment()

        # Override settings for debugging
        config.update(
            {
                "mode": "online",  # Send to cloud during debugging as well
                "buffer_size": 10,  # Smaller buffer for faster feedback
                "flush_interval": 5.0,  # More frequent flushing
                "tags": ["debug", "development"] + config["tags"],
                "notes": "Debug/development session",
            }
        )

        return config

    @staticmethod
    def validate_config(config: dict[str, Any]) -> list[str]:
        """Validate monitoring configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = ["backend_type", "project"]
        for field in required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")

        # Validate backend type
        valid_backends = ["wandb", "null"]
        if config.get("backend_type") not in valid_backends:
            errors.append(f"Invalid backend_type. Must be one of: {valid_backends}")

        # Validate numeric fields
        numeric_fields = {
            "buffer_size": int,
            "flush_interval": float,
        }

        for field, expected_type in numeric_fields.items():
            if field in config:
                try:
                    expected_type(config[field])
                except (ValueError, TypeError):
                    errors.append(f"Invalid {field}: must be {expected_type.__name__}")

        # Validate buffer size
        if "buffer_size" in config and config["buffer_size"] <= 0:
            errors.append("buffer_size must be greater than 0")

        # Validate flush interval
        if "flush_interval" in config and config["flush_interval"] <= 0:
            errors.append("flush_interval must be greater than 0")

        return errors
