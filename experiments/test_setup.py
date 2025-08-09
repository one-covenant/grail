#!/usr/bin/env python3
"""
Test script to verify GRAIL experiments setup
"""

import sys
import importlib
import subprocess

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, f"✓ {module_name} installed"
    except ImportError as e:
        return False, f"✗ {module_name} not installed: {str(e)}"

def test_gpu():
    """Test GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"✓ GPU available: {device_name} ({memory:.1f} GB)"
        else:
            return True, "✓ No GPU detected, will use CPU"
    except Exception as e:
        return False, f"✗ GPU check failed: {str(e)}"

def test_grail_import():
    """Test if GRAIL module can be imported"""
    try:
        sys.path.append('..')
        from grail import grail
        return True, "✓ GRAIL module imported successfully"
    except ImportError as e:
        return False, f"✗ GRAIL module import failed: {str(e)}"

def main():
    print("GRAIL Experiments Setup Test")
    print("=" * 50)
    
    # Test required packages
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'matplotlib',
        'tqdm'
    ]
    
    all_passed = True
    print("\nTesting required packages:")
    for package in required_packages:
        passed, message = test_import(package)
        print(f"  {message}")
        all_passed &= passed
    
    # Test GPU
    print("\nTesting GPU availability:")
    passed, message = test_gpu()
    print(f"  {message}")
    all_passed &= passed
    
    # Test GRAIL import
    print("\nTesting GRAIL module:")
    passed, message = test_grail_import()
    print(f"  {message}")
    all_passed &= passed
    
    # Test experiment scripts exist
    print("\nTesting experiment scripts:")
    experiment_scripts = [
        'experiment_1_hyperplane_collision.py',
        'experiment_2_multi_position_constraints.py',
        'experiment_3_model_perturbation.py',
        'experiment_4_attack_simulation.py',
        'experiment_5_single_constraint_analysis.py',
        'run.py'
    ]
    
    import os
    for script in experiment_scripts:
        if os.path.exists(script):
            print(f"  ✓ {script} found")
        else:
            print(f"  ✗ {script} not found")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Ready to run experiments.")
        print("\nTo run all experiments:")
        print("  python run.py --all")
        print("\nTo run a specific experiment:")
        print("  python run.py --exp 1")
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())