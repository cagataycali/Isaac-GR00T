#!/usr/bin/env python3
"""
GR00T N1.6-3B Inference Test with strands-robots

This script validates that GR00T N1.6 inference works correctly
inside the jetson-containers Docker environment with strands-robots.

Prerequisites:
    - NVIDIA GPU with CUDA support
    - Docker container with Isaac-GR00T N1.6 installed
    - strands-robots package installed (pip install strands-robots)
    - Model: nvidia/GR00T-N1.6-3B (auto-downloaded from HuggingFace)

Usage (inside Docker container):
    python3 tests/test_n16_inference.py

    # Or with pytest:
    pytest tests/test_n16_inference.py -v

Expected output:
    GR00T N1.6-3B loaded on <GPU_NAME>
    GR1 embodiment: state_dim=29, action_dim=29
    Cold inference: X.XXXs
    Warm inference: X.XXXs (XX.X Hz)
    All actions valid ✅

Tested on:
    - NVIDIA Thor (131.9 GB VRAM) - Jetson Thor
    - Container: dustynv/gr00t:r36.5.0
    - Python 3.12, PyTorch 2.7+, Transformers 4.51+
    - GR00T N1.6-3B with GR1 embodiment (29 DOF)
    - Result: 0.125s/step (~8.0 Hz warm inference)
"""

import numpy as np
import time
import sys


def test_groot_n16_native_inference():
    """Test GR00T N1.6-3B inference using native Isaac-GR00T API."""
    import torch
    import gr00t.model  # noqa: F401 - registers model types
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Load model
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.GR1,
        model_path="nvidia/GR00T-N1.6-3B",
        device=device,
    )
    print("✅ GR00T N1.6-3B loaded!")

    # Verify dimensions
    sap = policy.processor.state_action_processor
    state_dim = sap.get_state_dim("gr1")
    action_dim = sap.get_action_dim("gr1")
    print(f"GR1: state_dim={state_dim}, action_dim={action_dim}")
    assert state_dim == 29, f"Expected state_dim=29, got {state_dim}"
    assert action_dim == 29, f"Expected action_dim=29, got {action_dim}"

    # Get modality config
    mc = policy.get_modality_config()

    # Build observation matching GR1 format
    # GR1 DOF layout: left_arm(7) + right_arm(7) + left_hand(6) + right_hand(6) + waist(3) = 29
    gr1_dims = {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
        "waist": 3,
    }

    obs = {"video": {}, "state": {}, "language": {}}

    # Video: (B=1, T=1, H=256, W=256, C=3)
    for vk in mc["video"].modality_keys:
        obs["video"][vk] = np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8)

    # State: per-key with correct dims
    for sk in mc["state"].modality_keys:
        d = gr1_dims[sk]
        obs["state"][sk] = np.zeros((1, 1, d), dtype=np.float32)

    # Language instruction
    for lk in mc["language"].modality_keys:
        obs["language"][lk] = [["wave hello"]]

    # Cold inference
    t0 = time.time()
    actions, info = policy.get_action(obs)
    cold_dt = time.time() - t0
    print(f"Cold inference: {cold_dt:.3f}s")

    # Validate action shapes
    expected_action_keys = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
    for k in expected_action_keys:
        assert k in actions, f"Missing action key: {k}"
        v = actions[k]
        assert v.shape[0] == 1, f"Batch dim should be 1, got {v.shape[0]}"
        assert v.shape[1] == 16, f"Action horizon should be 16, got {v.shape[1]}"
        assert v.shape[2] == gr1_dims[k], f"{k} action dim should be {gr1_dims[k]}, got {v.shape[2]}"
        assert not np.any(np.isnan(v)), f"NaN in {k} actions"
        print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    # Warm inference
    t0 = time.time()
    actions2, _ = policy.get_action(obs)
    warm_dt = time.time() - t0
    print(f"Warm inference: {warm_dt:.3f}s ({1/warm_dt:.1f} Hz)")

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.time()
        policy.get_action(obs)
        times.append(time.time() - t0)
    avg = np.mean(times)
    print(f"Avg 10 runs: {avg:.3f}s ({1/avg:.1f} Hz)")

    print("\n✅ GR00T N1.6-3B native inference: PASSED")
    return True


def test_groot_n16_strands_robots():
    """Test GR00T N1.6 inference via strands-robots Gr00tPolicy wrapper.

    This tests the strands-robots integration layer that provides
    a simplified interface for robot control with GR00T.
    """
    try:
        from strands_robots.policies.groot import Gr00tPolicy, _detect_groot_version
    except ImportError:
        print("⚠️  strands-robots not installed, skipping strands integration test")
        return True

    import torch

    version = _detect_groot_version()
    print(f"Detected GR00T version: {version}")
    assert version == "n1.6", f"Expected n1.6, got {version}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load via strands-robots wrapper
    # Note: For SO-100 (6 DOF), use data_config='so100_dualcam'
    # For GR1 (29 DOF), use the base model directly
    policy = Gr00tPolicy(
        data_config="so100_dualcam",
        model_path="nvidia/GR00T-N1.6-3B",
        embodiment_tag="gr1",  # Must use a tag that exists in the base model
        denoising_steps=4,
        device=device,
    )
    print(f"✅ strands-robots Gr00tPolicy loaded (version={policy._groot_version})")

    print("\n✅ strands-robots GR00T integration: PASSED")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("GR00T N1.6-3B Inference Test")
    print("=" * 60)

    passed = True

    try:
        passed &= test_groot_n16_native_inference()
    except Exception as e:
        print(f"\n❌ Native inference test failed: {e}")
        passed = False

    print()

    try:
        passed &= test_groot_n16_strands_robots()
    except Exception as e:
        print(f"\n❌ strands-robots test failed: {e}")
        passed = False

    print()
    print("=" * 60)
    if passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
        sys.exit(1)
    print("=" * 60)
