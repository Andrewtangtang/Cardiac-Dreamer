#!/usr/bin/env python3
"""
Test script to compare patch token vs channel token approaches
"""

import torch
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.system import get_cardiac_dreamer_system


def test_token_approach(token_type: str, device: torch.device, batch_size: int = 4):
    """Test a specific token approach and return performance metrics"""
    print(f"\n{'='*50}")
    print(f"Testing {token_type.upper()} Token Approach")
    print(f"{'='*50}")
    
    # Create system
    system = get_cardiac_dreamer_system(
        token_type=token_type,
        d_model=768,
        nhead=12,
        num_layers=6,
        feature_dim=49,  # Only used for channel mode
        lr=1e-4,
        use_flash_attn=False,
        in_channels=1,
        use_pretrained=True
    ).to(device)
    
    # Create sample data
    image_t1 = torch.randn(batch_size, 1, 224, 224, device=device)
    a_hat_t1_to_t2_gt = torch.randn(batch_size, 6, device=device)
    at1_6dof_gt = torch.randn(batch_size, 6, device=device)
    at2_6dof_gt = torch.randn(batch_size, 6, device=device)
    
    print(f"Input shapes:")
    print(f"  Image: {image_t1.shape}")
    print(f"  Action: {a_hat_t1_to_t2_gt.shape}")
    
    # Test forward pass
    system.eval()
    with torch.no_grad():
        start_time = time.time()
        
        # Forward pass
        outputs = system(image_t1, a_hat_t1_to_t2_gt)
        
        forward_time = time.time() - start_time
        
        # Compute losses
        total_loss, loss_dict = system.compute_losses(
            outputs["predicted_action_composed"],
            at1_6dof_gt,
            outputs["a_prime_t2_hat"],
            at2_6dof_gt
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in system.parameters())
    trainable_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    
    print(f"\nResults:")
    print(f"  Forward pass time: {forward_time:.4f}s")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Main task loss: {loss_dict['main_task_loss'].item():.4f}")
    print(f"  Aux task loss: {loss_dict['aux_t2_action_loss'].item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Test memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        print(f"  GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"  GPU memory reserved: {memory_reserved:.1f} MB")
    
    return {
        'token_type': token_type,
        'forward_time': forward_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'main_task_loss': loss_dict['main_task_loss'].item(),
        'aux_task_loss': loss_dict['aux_t2_action_loss'].item(),
        'total_loss': total_loss.item(),
        'memory_allocated': torch.cuda.memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0,
        'memory_reserved': torch.cuda.memory_reserved(device) / 1024**2 if device.type == 'cuda' else 0
    }


def main():
    """Main comparison function"""
    print("🔬 Cardiac Dreamer: Patch Token vs Channel Token Comparison")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4
    print(f"Batch size: {batch_size}")
    
    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Test both approaches
    results = []
    
    # Test Channel Token approach
    try:
        channel_results = test_token_approach("channel", device, batch_size)
        results.append(channel_results)
        
        # Clear memory between tests
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Channel token test failed: {e}")
    
    # Test Patch Token approach
    try:
        patch_results = test_token_approach("patch", device, batch_size)
        results.append(patch_results)
        
    except Exception as e:
        print(f"❌ Patch token test failed: {e}")
    
    # Comparison summary
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        channel_res, patch_res = results
        
        print(f"{'Metric':<25} {'Channel':<15} {'Patch':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Forward time comparison
        time_diff = patch_res['forward_time'] - channel_res['forward_time']
        time_pct = (time_diff / channel_res['forward_time']) * 100
        print(f"{'Forward Time (s)':<25} {channel_res['forward_time']:<15.4f} {patch_res['forward_time']:<15.4f} {time_pct:+.1f}%")
        
        # Parameter comparison
        param_diff = patch_res['total_params'] - channel_res['total_params']
        param_pct = (param_diff / channel_res['total_params']) * 100
        print(f"{'Total Parameters':<25} {channel_res['total_params']:<15,} {patch_res['total_params']:<15,} {param_pct:+.1f}%")
        
        # Memory comparison (if GPU available)
        if device.type == 'cuda':
            mem_diff = patch_res['memory_allocated'] - channel_res['memory_allocated']
            mem_pct = (mem_diff / channel_res['memory_allocated']) * 100 if channel_res['memory_allocated'] > 0 else 0
            print(f"{'GPU Memory (MB)':<25} {channel_res['memory_allocated']:<15.1f} {patch_res['memory_allocated']:<15.1f} {mem_pct:+.1f}%")
        
        # Loss comparison
        loss_diff = patch_res['main_task_loss'] - channel_res['main_task_loss']
        print(f"{'Main Task Loss':<25} {channel_res['main_task_loss']:<15.4f} {patch_res['main_task_loss']:<15.4f} {loss_diff:+.4f}")
        
        print("\n🎯 Key Insights:")
        print(f"   • Patch tokens use {patch_res['total_params'] - channel_res['total_params']:,} more parameters")
        print(f"   • Forward pass is {time_pct:+.1f}% {'slower' if time_pct > 0 else 'faster'} with patch tokens")
        
        if device.type == 'cuda':
            print(f"   • GPU memory usage is {mem_pct:+.1f}% {'higher' if mem_pct > 0 else 'lower'} with patch tokens")
        
        print(f"   • Main task loss difference: {loss_diff:+.4f}")
        
        print("\n💡 Architectural Differences:")
        print("   • Channel tokens: [B, 512, 49] - each channel has 49 spatial values")
        print("   • Patch tokens: [B, 49, 512] - each spatial location has 512-dim features")
        print("   • Patch approach preserves spatial structure better")
        print("   • Channel approach processes feature channels independently")


if __name__ == "__main__":
    main() 