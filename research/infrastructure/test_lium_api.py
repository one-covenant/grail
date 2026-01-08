#!/usr/bin/env python3
"""Comprehensive test of lium.io SDK API to verify all parameter usage."""

import inspect
import json
from typing import Any

from lium import Lium, Config


def test_lium_methods() -> None:
    """Test all Lium SDK methods and print their signatures."""
    print("=" * 80)
    print("LIUM SDK METHOD SIGNATURES")
    print("=" * 80)

    # Get all public methods
    methods = [
        m for m in dir(Lium)
        if not m.startswith("_") and callable(getattr(Lium, m))
    ]

    for method_name in methods:
        method = getattr(Lium, method_name)
        try:
            sig = inspect.signature(method)
            print(f"\n{method_name}{sig}")
        except (ValueError, TypeError):
            print(f"\n{method_name}: (unable to get signature)")


def test_executor_fields() -> None:
    """Test executor fields and available attributes."""
    print("\n" + "=" * 80)
    print("EXECUTOR FIELDS")
    print("=" * 80)

    try:
        config = Config.load()
        lium = Lium(config)
        executors = lium.ls(gpu_type="A100")

        if executors:
            executor = executors[0]
            print(f"\nSample Executor: {executor.huid}")
            print(f"Available attributes:")
            for attr in dir(executor):
                if not attr.startswith("_"):
                    try:
                        value = getattr(executor, attr)
                        if not callable(value):
                            print(f"  {attr}: {value!r}")
                    except Exception:
                        pass

            print(f"\nExecutor specs structure:")
            print(json.dumps(executor.specs, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}")


def test_pod_info_fields() -> None:
    """Test PodInfo fields and available attributes."""
    print("\n" + "=" * 80)
    print("POD INFO FIELDS")
    print("=" * 80)

    try:
        config = Config.load()
        lium = Lium(config)
        pods = lium.ps()

        if pods:
            pod = pods[0]
            print(f"\nSample Pod: {pod.name}")
            print(f"Available attributes:")
            for attr in dir(pod):
                if not attr.startswith("_"):
                    try:
                        value = getattr(pod, attr)
                        if not callable(value):
                            print(f"  {attr}: {value!r}")
                    except Exception:
                        pass
        else:
            print("\nNo running pods found - skipping PodInfo test")

    except Exception as e:
        print(f"Error: {e}")


def test_up_method() -> None:
    """Test Lium.up() method signature."""
    print("\n" + "=" * 80)
    print("LIUM.UP() METHOD PARAMETERS")
    print("=" * 80)

    sig = inspect.signature(Lium.up)
    print(f"\nSignature: {sig}")

    # Get parameter info
    for param_name, param in sig.parameters.items():
        if param_name != "self":
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"  {param_name}{annotation}{default}")


def test_down_method() -> None:
    """Test Lium.down() method signature."""
    print("\n" + "=" * 80)
    print("LIUM.DOWN() METHOD PARAMETERS")
    print("=" * 80)

    sig = inspect.signature(Lium.down)
    print(f"\nSignature: {sig}")

    # Get parameter info
    for param_name, param in sig.parameters.items():
        if param_name != "self":
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"  {param_name}{annotation}{default}")


def test_wait_ready_method() -> None:
    """Test Lium.wait_ready() method signature."""
    print("\n" + "=" * 80)
    print("LIUM.WAIT_READY() METHOD PARAMETERS")
    print("=" * 80)

    sig = inspect.signature(Lium.wait_ready)
    print(f"\nSignature: {sig}")

    # Get parameter info
    for param_name, param in sig.parameters.items():
        if param_name != "self":
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"  {param_name}{annotation}{default}")


def test_schedule_termination_method() -> None:
    """Test Lium.schedule_termination() method signature."""
    print("\n" + "=" * 80)
    print("LIUM.SCHEDULE_TERMINATION() METHOD PARAMETERS")
    print("=" * 80)

    sig = inspect.signature(Lium.schedule_termination)
    print(f"\nSignature: {sig}")

    # Get parameter info
    for param_name, param in sig.parameters.items():
        if param_name != "self":
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"  {param_name}{annotation}{default}")


if __name__ == "__main__":
    try:
        test_lium_methods()
        test_executor_fields()
        test_pod_info_fields()
        test_up_method()
        test_down_method()
        test_wait_ready_method()
        test_schedule_termination_method()

        print("\n" + "=" * 80)
        print("✅ API TEST COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
