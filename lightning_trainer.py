"""
PyTorch Lightning DDP wrapper for Lattice Autoencoder training.

Provides:
- LightningLatticeAutoencoder: LightningModule wrapper
- MNISTDataModule: LightningDataModule for MNIST
- Training utilities with DDP support
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any, List
import os

from cell import DEVICE
from lattice_visualizer import LatticeVisualizer
from trainer import (
    LatticeEncoder,
    LatticeDecoder, 
    DifferentiableLattice,
    load_mnist
)


class LightningLatticeAutoencoder(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Lattice Autoencoder.
    
    Supports DDP training across multiple GPUs.
    Features learnable propagation steps optimized via reconstruction loss.
    """
    
    def __init__(
        self,
        lattice: LatticeVisualizer,
        input_dim: int = 784,
        propagation_steps: int = 5,
        max_steps: int = 10,
        learn_steps: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        diversity_weight: float = 0.01
    ):
        """
        Initialize the Lightning Lattice Autoencoder.
        
        Args:
            lattice: LatticeVisualizer instance
            input_dim: Input/output dimension (784 for MNIST)
            propagation_steps: Initial/fixed number of propagation steps
            max_steps: Maximum steps for learnable step mode
            learn_steps: Whether to learn optimal propagation steps
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            diversity_weight: Weight for std_dev diversity loss
        """
        super().__init__()
        self.save_hyperparameters(ignore=['lattice'])
        
        self.lattice = lattice
        self.propagation_steps = propagation_steps
        self.max_steps = max_steps
        self.learn_steps = learn_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.diversity_weight = diversity_weight
        
        n_cells = len(lattice.cells)
        
        # Model components
        self.encoder = LatticeEncoder(input_dim, n_cells, device=None)
        self.diff_lattice = DifferentiableLattice(lattice, max_steps=max_steps)
        self.decoder = LatticeDecoder(n_cells, input_dim, device=None)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: encode -> propagate -> decode.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            reconstruction: Reconstructed input
            final_state: Final lattice state
            history: Propagation history
        """
        # Encode
        initial_activations = self.encoder(x)
        
        # Propagate through lattice with learnable or fixed steps
        n_steps = None if self.learn_steps else self.propagation_steps
        final_state, history = self.diff_lattice(initial_activations, n_steps)
        
        # Decode
        reconstruction = self.decoder(final_state)
        
        return reconstruction, final_state, history
    
    def _compute_loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction and diversity loss."""
        reconstruction, _, _ = self(batch)
        
        # Reconstruction loss
        recon_loss = self.criterion(reconstruction, batch)
        
        # Diversity loss (encourage variance in std_devs)
        std_dev_var = self.diff_lattice.std_devs.var()
        diversity_loss = -self.diversity_weight * std_dev_var
        
        total_loss = recon_loss + diversity_loss
        
        return total_loss, recon_loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        total_loss, recon_loss = self._compute_loss(batch)
        
        # Logging
        self.log('train_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def on_after_backward(self):
        """Ensure std_devs stay >= 0.5 after gradient update."""
        with torch.no_grad():
            self.diff_lattice.std_devs.data = torch.clamp(
                torch.abs(self.diff_lattice.std_devs.data), min=0.5
            )
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        _, recon_loss = self._compute_loss(batch)
        
        self.log('val_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return recon_loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        _, recon_loss = self._compute_loss(batch)
        
        self.log('test_loss', recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return recon_loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def sync_to_lattice(self):
        """Sync learned parameters back to original lattice cells."""
        self.diff_lattice.sync_to_lattice()
    
    def on_train_epoch_end(self):
        """Log cell and step parameter statistics at end of each epoch."""
        with torch.no_grad():
            means = self.diff_lattice.means
            
            # Log learned step parameters
            if self.learn_steps:
                effective_steps = self.diff_lattice.get_effective_steps()
                decay_rate = torch.clamp(self.diff_lattice.decay_rate, 0.5, 0.99).item()
                self.log('effective_steps', effective_steps, sync_dist=True)
                self.log('decay_rate', decay_rate, sync_dist=True)
            std_devs = self.diff_lattice.std_devs
            
            self.log('mean_avg', means.mean(), sync_dist=True)
            self.log('std_dev_avg', std_devs.mean(), sync_dist=True)


class MNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MNIST.
    
    Handles data loading, splitting, and batching for distributed training.
    """
    
    def __init__(
        self,
        data_path: str = None,
        batch_size: int = 64,
        num_workers: int = 4,
        train_samples: int = None,
        val_split: float = 0.1,
        test_samples: int = None,
        pin_memory: bool = True
    ):
        """
        Initialize MNIST DataModule.
        
        Args:
            data_path: Path to MNIST data
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            train_samples: Number of training samples (None = all)
            val_split: Fraction of training data to use for validation
            test_samples: Number of test samples (None = all)
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Use relative path if not specified
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist')
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_split = val_split
        self.test_samples = test_samples
        self.pin_memory = pin_memory
        
        # Data placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download/prepare data (called only on rank 0 in DDP).
        
        Since we're using local files, just verify they exist.
        """
        train_file = os.path.join(self.data_path, 'train-images.idx3-ubyte')
        test_file = os.path.join(self.data_path, 't10k-images.idx3-ubyte')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"MNIST training data not found at {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"MNIST test data not found at {test_file}")
        
        print(f"MNIST data found at {self.data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets (called on every GPU in DDP).
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Load training data
            train_images, train_labels = load_mnist(
                train=True,
                n_samples=self.train_samples,
                data_path=self.data_path
            )
            
            # Split into train and validation
            full_dataset = TensorDataset(train_images, train_labels)
            n_val = int(len(full_dataset) * self.val_split)
            n_train = len(full_dataset) - n_val
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            
            print(f"Train samples: {n_train}, Validation samples: {n_val}")
        
        if stage == 'test' or stage is None:
            # Load test data
            test_images, test_labels = load_mnist(
                train=False,
                n_samples=self.test_samples,
                data_path=self.data_path
            )
            self.test_dataset = TensorDataset(test_images, test_labels)
            print(f"Test samples: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )


def create_lightning_trainer(
    max_epochs: int = 100,
    accelerator: str = 'auto',
    devices: int = 'auto',
    strategy: str = 'auto',
    precision: str = '32',
    log_every_n_steps: int = 10,
    checkpoint_dir: str = './checkpoints',
    enable_progress_bar: bool = True,
    early_stopping_patience: int = 20,
    gradient_clip_val: float = 1.0
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer with DDP support.
    
    Args:
        max_epochs: Maximum training epochs
        accelerator: 'cpu', 'gpu', 'mps', 'auto'
        devices: Number of devices or 'auto'
        strategy: 'ddp', 'ddp_spawn', 'auto', etc.
        precision: '32', '16-mixed', 'bf16-mixed'
        log_every_n_steps: Logging frequency
        checkpoint_dir: Directory for checkpoints
        enable_progress_bar: Show progress bar
        early_stopping_patience: Epochs to wait before early stopping
        gradient_clip_val: Gradient clipping value
        
    Returns:
        Configured Lightning Trainer
    """
    # Callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='lattice-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min',
            verbose=True
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Configure strategy for multi-GPU
    # Always use find_unused_parameters=True because DifferentiableLattice
    # has parameters (positions buffer, some bounce angles) that may not be
    # used in every forward pass
    if devices != 1 and devices != 'auto':
        # Multi-GPU: use DDP with find_unused_parameters
        strategy_obj = DDPStrategy(find_unused_parameters=True)
    elif strategy == 'ddp' or strategy == 'ddp_find_unused_parameters_true':
        strategy_obj = DDPStrategy(find_unused_parameters=True)
    else:
        strategy_obj = strategy
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy_obj,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=gradient_clip_val,
        deterministic=False,
        enable_checkpointing=True
    )
    
    return trainer


def setup_lightning_training(
    lattice: LatticeVisualizer,
    train_samples: int = None,
    test_samples: int = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    propagation_steps: int = 5,
    max_epochs: int = 100,
    accelerator: str = 'auto',
    devices: int = 'auto',
    strategy: str = 'auto',
    num_workers: int = 4
) -> Tuple[LightningLatticeAutoencoder, MNISTDataModule, pl.Trainer]:
    """
    Set up complete Lightning training pipeline.
    
    Args:
        lattice: LatticeVisualizer instance
        train_samples: Number of training samples (None = all 60k)
        test_samples: Number of test samples (None = all 10k)
        batch_size: Batch size
        learning_rate: Learning rate
        propagation_steps: Lattice propagation steps
        max_epochs: Maximum epochs
        accelerator: Device accelerator
        devices: Number of devices
        strategy: Training strategy (e.g., 'ddp')
        num_workers: Data loading workers
        
    Returns:
        model: LightningLatticeAutoencoder
        datamodule: MNISTDataModule
        trainer: Lightning Trainer
    """
    # Create model
    model = LightningLatticeAutoencoder(
        lattice=lattice,
        input_dim=784,
        propagation_steps=propagation_steps,
        learning_rate=learning_rate
    )
    
    # Create data module
    datamodule = MNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_samples=train_samples,
        test_samples=test_samples
    )
    
    # Create trainer
    trainer = create_lightning_trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy
    )
    
    return model, datamodule, trainer


def train_with_lightning(
    lattice: LatticeVisualizer,
    train_samples: int = None,
    test_samples: int = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    propagation_steps: int = 5,
    max_epochs: int = 100,
    accelerator: str = 'auto',
    devices: int = 'auto',
    strategy: str = 'auto',
    num_workers: int = 4
) -> Tuple[LightningLatticeAutoencoder, Dict[str, float]]:
    """
    Train the lattice autoencoder using PyTorch Lightning.
    
    Convenience function that sets up and runs training.
    
    Args:
        lattice: LatticeVisualizer instance
        train_samples: Number of training samples
        test_samples: Number of test samples
        batch_size: Batch size
        learning_rate: Learning rate
        propagation_steps: Lattice propagation steps
        max_epochs: Maximum epochs
        accelerator: Device accelerator
        devices: Number of devices
        strategy: Training strategy
        num_workers: Data loading workers
        
    Returns:
        model: Trained model
        results: Test results dictionary
    """
    model, datamodule, trainer = setup_lightning_training(
        lattice=lattice,
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=batch_size,
        learning_rate=learning_rate,
        propagation_steps=propagation_steps,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_workers=num_workers
    )
    
    # Train
    print(f"Starting Lightning training...")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices}")
    print(f"  Strategy: {strategy}")
    print(f"  Max epochs: {max_epochs}")
    
    trainer.fit(model, datamodule=datamodule)
    
    # Test
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Sync parameters back to lattice
    model.sync_to_lattice()
    
    return model, test_results[0] if test_results else {}


# CLI entry point for distributed training
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Lattice Autoencoder with Lightning DDP')
    parser.add_argument('--layers', type=int, default=2, help='Lattice layers')
    parser.add_argument('--hex_radius', type=int, default=8, help='Hexagonal radius')
    parser.add_argument('--train_samples', type=int, default=None, help='Training samples')
    parser.add_argument('--test_samples', type=int, default=None, help='Test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--propagation_steps', type=int, default=5, help='Propagation steps')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    parser.add_argument('--strategy', type=str, default='auto', help='Strategy (ddp, auto)')
    parser.add_argument('--num_workers', type=int, default=4, help='Data workers')
    
    args = parser.parse_args()
    
    # Create lattice
    lattice = LatticeVisualizer(layers=args.layers, hex_radius=args.hex_radius)
    print(f"Created lattice with {len(lattice.cells)} cells")
    
    # Train
    model, results = train_with_lightning(
        lattice=lattice,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        propagation_steps=args.propagation_steps,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_workers=args.num_workers
    )
    
    print(f"\nTraining complete!")
    print(f"Test results: {results}")
