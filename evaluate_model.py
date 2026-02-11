#!/usr/bin/env python3
"""
Post-training evaluation script for Lattice Autoencoder.

Loads a trained model checkpoint and generates:
- Reconstruction visualizations for each digit (0-9)
- Per-digit MSE scores
- Overall test metrics
- Cell parameter statistics
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lattice_visualizer import LatticeVisualizer
from lightning_trainer import LightningLatticeAutoencoder, MNISTDataModule
from trainer import load_mnist


def load_model_from_checkpoint(
    checkpoint_path: str,
    layers: int = 2,
    hex_radius: int = 8,
    propagation_steps: int = 5
) -> LightningLatticeAutoencoder:
    """Load a trained model from checkpoint."""
    # Create lattice with same architecture
    lattice = LatticeVisualizer(layers=layers, hex_radius=hex_radius)
    
    # Load model from checkpoint
    model = LightningLatticeAutoencoder.load_from_checkpoint(
        checkpoint_path,
        lattice=lattice,
        input_dim=784,
        propagation_steps=propagation_steps
    )
    model.eval()
    
    return model, lattice


def evaluate_reconstructions(
    model: LightningLatticeAutoencoder,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: str,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Returns:
        Dictionary with per-digit and overall metrics
    """
    model = model.to(device)
    model.eval()
    
    metrics = {
        'per_digit_mse': {},
        'per_digit_count': {},
        'overall_mse': 0.0,
        'total_samples': len(test_images)
    }
    
    # Get reconstructions
    with torch.no_grad():
        test_batch = test_images.to(device)
        reconstructions, _, _ = model(test_batch)
        reconstructions = reconstructions.cpu()
    
    # Calculate per-digit MSE
    for digit in range(10):
        mask = test_labels == digit
        if mask.sum() > 0:
            digit_mse = ((test_images[mask] - reconstructions[mask]) ** 2).mean().item()
            metrics['per_digit_mse'][digit] = digit_mse
            metrics['per_digit_count'][digit] = mask.sum().item()
    
    # Overall MSE
    metrics['overall_mse'] = ((test_images - reconstructions) ** 2).mean().item()
    
    # Generate visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Grid of all digits (one example per digit)
    fig, axes = plt.subplots(10, 3, figsize=(8, 24))
    
    for digit in range(10):
        mask = test_labels == digit
        if mask.sum() > 0:
            idx = torch.where(mask)[0][0].item()
            
            # Original
            axes[digit, 0].imshow(test_images[idx].reshape(28, 28).numpy(), cmap='gray')
            axes[digit, 0].set_title(f'Original: {digit}')
            axes[digit, 0].axis('off')
            
            # Reconstructed
            axes[digit, 1].imshow(reconstructions[idx].reshape(28, 28).numpy(), cmap='gray')
            axes[digit, 1].set_title(f'Reconstructed')
            axes[digit, 1].axis('off')
            
            # Difference
            diff = (test_images[idx] - reconstructions[idx]).abs().reshape(28, 28).numpy()
            axes[digit, 2].imshow(diff, cmap='hot')
            axes[digit, 2].set_title(f'MSE: {metrics["per_digit_mse"].get(digit, 0):.4f}')
            axes[digit, 2].axis('off')
    
    plt.suptitle('Reconstruction Results - One Per Digit', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_digits_reconstruction.png'), dpi=150)
    plt.close()
    
    # 2. Multiple examples per digit (3x3 grid for each digit)
    for digit in range(10):
        mask = test_labels == digit
        indices = torch.where(mask)[0][:9]  # Up to 9 examples
        
        if len(indices) > 0:
            n_examples = len(indices)
            n_cols = min(3, n_examples)
            n_rows = (n_examples + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 2.5))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, idx in enumerate(indices):
                row = i // n_cols
                col = (i % n_cols) * 2
                
                # Original
                axes[row, col].imshow(test_images[idx].reshape(28, 28).numpy(), cmap='gray')
                axes[row, col].set_title('Original')
                axes[row, col].axis('off')
                
                # Reconstructed
                axes[row, col + 1].imshow(reconstructions[idx].reshape(28, 28).numpy(), cmap='gray')
                axes[row, col + 1].set_title('Reconstructed')
                axes[row, col + 1].axis('off')
            
            # Hide unused axes
            for i in range(n_examples, n_rows * n_cols):
                row = i // n_cols
                col = (i % n_cols) * 2
                axes[row, col].axis('off')
                axes[row, col + 1].axis('off')
            
            plt.suptitle(f'Digit {digit} Reconstructions (MSE: {metrics["per_digit_mse"].get(digit, 0):.4f})', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'digit_{digit}_reconstructions.png'), dpi=150)
            plt.close()
    
    # 3. MSE bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    digits = list(metrics['per_digit_mse'].keys())
    mses = [metrics['per_digit_mse'][d] for d in digits]
    
    bars = ax.bar(digits, mses, color='steelblue', edgecolor='black')
    ax.axhline(y=metrics['overall_mse'], color='red', linestyle='--', label=f'Overall MSE: {metrics["overall_mse"]:.4f}')
    ax.set_xlabel('Digit')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction MSE by Digit')
    ax.set_xticks(digits)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{mse:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_by_digit.png'), dpi=150)
    plt.close()
    
    return metrics


def evaluate_cell_parameters(
    model: LightningLatticeAutoencoder,
    output_dir: str
) -> dict:
    """Analyze and visualize learned cell parameters."""
    
    with torch.no_grad():
        means = model.diff_lattice.means.cpu().numpy()
        std_devs = model.diff_lattice.std_devs.cpu().numpy()
        bounce_angles = model.diff_lattice.bounce_angles.cpu().numpy()
    
    stats = {
        'mean': {'min': means.min(), 'max': means.max(), 'avg': means.mean(), 'std': means.std()},
        'std_dev': {'min': std_devs.min(), 'max': std_devs.max(), 'avg': std_devs.mean(), 'std': std_devs.std()},
        'bounce_angles': {
            'x': {'min': bounce_angles[:, 0].min(), 'max': bounce_angles[:, 0].max(), 'avg': bounce_angles[:, 0].mean()},
            'y': {'min': bounce_angles[:, 1].min(), 'max': bounce_angles[:, 1].max(), 'avg': bounce_angles[:, 1].mean()},
            'z': {'min': bounce_angles[:, 2].min(), 'max': bounce_angles[:, 2].max(), 'avg': bounce_angles[:, 2].mean()}
        }
    }
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean distribution
    axes[0, 0].hist(means, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(means.mean(), color='red', linestyle='--', label=f'Mean: {means.mean():.3f}')
    axes[0, 0].set_xlabel('Mean (μ)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Cell Means')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std dev distribution
    axes[0, 1].hist(std_devs, bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(std_devs.mean(), color='red', linestyle='--', label=f'Mean: {std_devs.mean():.3f}')
    axes[0, 1].set_xlabel('Std Dev (σ)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Cell Std Devs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bounce angles distribution
    axes[1, 0].hist(bounce_angles[:, 0], bins=30, alpha=0.5, label='X', color='red')
    axes[1, 0].hist(bounce_angles[:, 1], bins=30, alpha=0.5, label='Y', color='green')
    axes[1, 0].hist(bounce_angles[:, 2], bins=30, alpha=0.5, label='Z', color='blue')
    axes[1, 0].set_xlabel('Angle (degrees)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Bounce Angles')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean vs Std dev scatter
    axes[1, 1].scatter(means, std_devs, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Mean (μ)')
    axes[1, 1].set_ylabel('Std Dev (σ)')
    axes[1, 1].set_title('Mean vs Std Dev per Cell')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_parameters.png'), dpi=150)
    plt.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Lattice Autoencoder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./images/evaluation_results', help='Output directory')
    parser.add_argument('--layers', type=int, default=2, help='Lattice layers')
    parser.add_argument('--hex_radius', type=int, default=8, help='Hexagonal radius')
    parser.add_argument('--propagation_steps', type=int, default=5, help='Propagation steps')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.checkpoint}")
    model, lattice = load_model_from_checkpoint(
        args.checkpoint,
        layers=args.layers,
        hex_radius=args.hex_radius,
        propagation_steps=args.propagation_steps
    )
    
    print(f"Lattice: {len(lattice.cells)} cells")
    print(f"Loading MNIST test data...")
    
    test_images, test_labels = load_mnist(train=False, n_samples=args.test_samples)
    print(f"Loaded {len(test_images)} test samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate reconstructions
    print(f"\nEvaluating reconstructions...")
    recon_metrics = evaluate_reconstructions(
        model, test_images, test_labels, 
        args.output_dir, args.device
    )
    
    # Evaluate cell parameters
    print(f"Analyzing cell parameters...")
    cell_stats = evaluate_cell_parameters(model, args.output_dir)
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nOverall Test MSE: {recon_metrics['overall_mse']:.6f}")
    print(f"Total test samples: {recon_metrics['total_samples']}")
    
    print(f"\nPer-Digit MSE:")
    for digit in range(10):
        if digit in recon_metrics['per_digit_mse']:
            mse = recon_metrics['per_digit_mse'][digit]
            count = recon_metrics['per_digit_count'][digit]
            print(f"  Digit {digit}: MSE={mse:.6f} (n={count})")
    
    print(f"\nCell Parameter Statistics:")
    print(f"  Mean (μ): min={cell_stats['mean']['min']:.3f}, max={cell_stats['mean']['max']:.3f}, avg={cell_stats['mean']['avg']:.3f}")
    print(f"  Std Dev (σ): min={cell_stats['std_dev']['min']:.3f}, max={cell_stats['std_dev']['max']:.3f}, avg={cell_stats['std_dev']['avg']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - all_digits_reconstruction.png")
    print(f"  - digit_X_reconstructions.png (for each digit)")
    print(f"  - mse_by_digit.png")
    print(f"  - cell_parameters.png")
    
    # Save metrics to file
    import json
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'reconstruction': recon_metrics,
            'cell_parameters': {k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v) 
                               for k, v in cell_stats.items()}
        }, f, indent=2, default=float)
    print(f"  - metrics.json")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
