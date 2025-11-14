#!/usr/bin/env python3
"""
Test script to verify ACT policy works with temporal observations (n_obs_steps > 1) during inference.
This tests the observation queue mechanism for the model: Orellius/so101_sort_act_context16
"""

import torch
from lerobot.policies.act.modeling_act import ACTPolicy

def test_temporal_act_inference():
    """Test that ACT can load a temporal model and run inference correctly."""
    
    print("=" * 80)
    print("Testing ACT Temporal Observation Inference")
    print("=" * 80)
    
    # Load the model from HuggingFace
    model_path = "Orellius/so101_sort_act_context16"
    print(f"\n1. Loading model: {model_path}")
    
    try:
        policy = ACTPolicy.from_pretrained(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   - n_obs_steps: {policy.config.n_obs_steps}")
        print(f"   - chunk_size: {policy.config.chunk_size}")
        print(f"   - n_action_steps: {policy.config.n_action_steps}")
        print(f"   - image_features: {policy.config.image_features}")
        print(f"   - robot_state_feature: {policy.config.robot_state_feature}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Check that n_obs_steps > 1
    if policy.config.n_obs_steps <= 1:
        print(f"   ✗ Model doesn't have temporal observations (n_obs_steps={policy.config.n_obs_steps})")
        return False
    
    print(f"\n2. Initializing policy for inference")
    policy.eval()
    policy.reset()
    print(f"   ✓ Policy reset and in eval mode")
    
    # Create dummy observations matching the model's expected input shape
    print(f"\n3. Creating dummy observations")
    batch_size = 1
    device = policy.config.device
    
    # Get observation shapes from config
    batch = {}
    
    # Add images if the model expects them
    if policy.config.image_features:
        print(f"   - Creating images for cameras: {list(policy.config.image_features.keys())}")
        for cam_key, feature_info in policy.config.image_features.items():
            # Get the shape from the PolicyFeature
            img_shape = feature_info.shape
            # Create dummy image: (batch_size, C, H, W)
            batch[cam_key] = torch.randn(batch_size, *img_shape, device=device)
            print(f"     {cam_key}: shape {batch[cam_key].shape}")
    
    # Add robot state if the model expects it
    if policy.config.robot_state_feature:
        state_shape = policy.config.robot_state_feature.shape
        batch["observation.state"] = torch.randn(batch_size, *state_shape, device=device)
        print(f"   - observation.state: shape {batch['observation.state'].shape}")
    
    # Add environment state if the model expects it
    if policy.config.env_state_feature:
        env_state_shape = policy.config.env_state_feature.shape
        batch["observation.environment_state"] = torch.randn(batch_size, *env_state_shape, device=device)
        print(f"   - observation.environment_state: shape {batch['observation.environment_state'].shape}")
    
    print(f"\n4. Testing inference over multiple steps (simulating temporal queue)")
    n_steps = policy.config.n_obs_steps + 5  # Test beyond the queue fill phase
    
    try:
        for step in range(n_steps):
            # In real inference, observations would change each step
            # Here we just use random observations
            action = policy.select_action(batch)
            
            if step < policy.config.n_obs_steps:
                print(f"   Step {step}: Filling observation queue (queue length: {step+1}/{policy.config.n_obs_steps}) - action shape: {action.shape}")
            elif step == policy.config.n_obs_steps:
                print(f"   Step {step}: Queue full, now using temporal observations - action shape: {action.shape}")
            elif step == n_steps - 1:
                print(f"   Step {step}: Final step - action shape: {action.shape}")
        
        print(f"   ✓ Successfully ran {n_steps} inference steps with temporal observations")
        
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n5. Verifying action output")
    expected_action_dim = policy.config.action_feature.shape[0]
    if action.shape[-1] != expected_action_dim:
        print(f"   ✗ Action dimension mismatch: expected {expected_action_dim}, got {action.shape[-1]}")
        return False
    print(f"   ✓ Action dimension correct: {action.shape}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! ACT temporal inference is working correctly.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_temporal_act_inference()
    exit(0 if success else 1)

