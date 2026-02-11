#!/usr/bin/env python3
"""
PyTorch Lightning Training with Hyperparameter Tuning for Toroidal Token Generation
Optimized for 8x H100 GPUs with Optuna HPO

Usage:
    # Single run
    python train_toroidal_lightning.py --embeddings-path /path/to/embeddings.txt
    
    # Hyperparameter tuning
    python train_toroidal_lightning.py --embeddings-path /path/to/embeddings.txt --hpo --n-trials 50
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Optuna for hyperparameter tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback


# ============================================================================
# Model Components
# ============================================================================

class BidirectionalToroidalLattice(nn.Module):
    """Toroidal lattice with bidirectional propagation."""
    
    def __init__(self, n_theta: int, n_phi: int, max_steps: int = 10):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.max_steps = max_steps
        self._total_cells = n_theta * n_phi
        
        # Learnable parameters
        self.forward_step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.forward_decay = nn.Parameter(torch.tensor(0.85))
        self.reverse_step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.reverse_decay = nn.Parameter(torch.tensor(0.85))
        self.interaction_weight = nn.Parameter(torch.tensor(0.5))
        self.bounce_angles = nn.Parameter(torch.randn(self._total_cells, 3) * 0.1)
        
        # Build adjacency matrices
        self._build_adjacency_matrices()
    
    def _build_adjacency_matrices(self):
        """Build dense adjacency matrices."""
        n = self._total_cells
        forward_adj = torch.zeros(n, n)
        reverse_adj = torch.zeros(n, n)
        
        for idx in range(n):
            ti = idx // self.n_phi
            pi = idx % self.n_phi
            
            for dti, dpi in [(1, 0), (0, 1), (1, 1)]:
                ni = ((ti + dti) % self.n_theta) * self.n_phi + ((pi + dpi) % self.n_phi)
                forward_adj[idx, ni] = 1.0
            
            for dti, dpi in [(-1, 0), (0, -1), (-1, -1)]:
                ni = ((ti + dti) % self.n_theta) * self.n_phi + ((pi + dpi) % self.n_phi)
                reverse_adj[idx, ni] = 1.0
        
        forward_adj = forward_adj / (forward_adj.sum(dim=1, keepdim=True) + 1e-8)
        reverse_adj = reverse_adj / (reverse_adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self.register_buffer('forward_adj', forward_adj)
        self.register_buffer('reverse_adj', reverse_adj)
    
    def _propagate(self, entry_probs: torch.Tensor, adj: torch.Tensor, 
                   decay: torch.Tensor, step_weights: torch.Tensor) -> torch.Tensor:
        state = entry_probs
        states = [state]
        decay_val = torch.clamp(decay, 0.5, 0.99)
        
        for _ in range(self.max_steps):
            propagated = torch.matmul(state, adj)
            angle_mag = self.bounce_angles.abs().mean(dim=1)
            angle_factor = torch.cos(angle_mag)
            propagated = propagated * (0.5 + 0.5 * angle_factor.unsqueeze(0))
            state = (0.3 * state + 0.7 * propagated) * decay_val
            states.append(state)
        
        weights = torch.softmax(step_weights, dim=0)
        final = sum(w * s for w, s in zip(weights, states[1:]))
        return final
    
    def forward(self, entry_probs: torch.Tensor) -> torch.Tensor:
        forward_state = self._propagate(
            entry_probs, self.forward_adj, self.forward_decay, self.forward_step_weights
        )
        reverse_state = self._propagate(
            entry_probs, self.reverse_adj, self.reverse_decay, self.reverse_step_weights
        )
        
        interaction = forward_state * reverse_state
        interaction_w = torch.sigmoid(self.interaction_weight)
        combined = forward_state + reverse_state + interaction_w * interaction
        
        return combined


class ToroidalTokenGenerator(nn.Module):
    """Token generation model with bidirectional toroidal propagation."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_theta: int = 20,
        n_phi: int = 40,
        max_steps: int = 12,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        encoder_layers: int = 2,
        output_layers: int = 2,
        pretrained_embeddings: torch.Tensor = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_cells = n_theta * n_phi
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.token_embedding.weight.data.copy_(pretrained_embeddings)
        
        # Signal encoder (dynamic layers)
        self.signal_encoder = self._build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=n_theta * n_phi,
            n_layers=encoder_layers,
            dropout=dropout
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
        # Bidirectional propagation
        self.propagation = BidirectionalToroidalLattice(n_theta, n_phi, max_steps)
        
        # Output projection (dynamic layers)
        self.output_projection = self._build_mlp(
            input_dim=n_theta * n_phi,
            hidden_dim=hidden_dim,
            output_dim=vocab_size,
            n_layers=output_layers,
            dropout=dropout
        )
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int, 
                   n_layers: int, dropout: float) -> nn.Module:
        """Build MLP with configurable number of layers."""
        if n_layers < 1:
            # Direct projection
            return nn.Linear(input_dim, output_dim)
        
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        if input_tokens.dim() == 2:
            emb = self.token_embedding(input_tokens).mean(dim=1)
        else:
            emb = self.token_embedding(input_tokens)
        
        logits = self.signal_encoder(emb)
        temp = self.temperature.abs() + 0.1
        entry_probs = torch.softmax(logits / temp, dim=-1)
        
        combined = self.propagation(entry_probs)
        output_logits = self.output_projection(combined)
        
        return output_logits


# ============================================================================
# Lightning Module
# ============================================================================

class ToroidalLightningModule(LightningModule):
    """PyTorch Lightning module for toroidal token generation."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_theta: int = 24,
        n_phi: int = 48,
        max_steps: int = 12,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        encoder_layers: int = 2,
        output_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        pretrained_embeddings: torch.Tensor = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_embeddings'])
        
        self.model = ToroidalTokenGenerator(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_theta=n_theta,
            n_phi=n_phi,
            max_steps=max_steps,
            hidden_dim=hidden_dim,
            dropout=dropout,
            encoder_layers=encoder_layers,
            output_layers=output_layers,
            pretrained_embeddings=pretrained_embeddings
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
    
    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_tokens)
    
    def _shared_step(self, batch, batch_idx, prefix: str):
        input_tokens, target_tokens = batch
        logits = self(input_tokens)
        loss = F.cross_entropy(logits, target_tokens)
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == target_tokens).float().mean()
        
        # Perplexity
        ppl = torch.exp(loss)
        
        self.log(f'{prefix}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_acc', acc, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_ppl', ppl, prog_bar=False, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=True  # H100 optimization
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * (step - self.warmup_steps) / 
                                      max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


# ============================================================================
# Data Module
# ============================================================================

class ToroidalDataModule(LightningDataModule):
    """Data module for toroidal token generation."""
    
    def __init__(
        self,
        embeddings_path: str,
        max_embeddings: int = 50000,
        n_pairs: int = 500000,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42
    ):
        super().__init__()
        self.embeddings_path = embeddings_path
        self.max_embeddings = max_embeddings
        self.n_pairs = n_pairs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.vocab_size = None
        self.embedding_dim = None
        self.embeddings_tensor = None
    
    def prepare_data(self):
        """Load embeddings (called only on rank 0)."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        # Load embeddings
        words, embeddings = self._load_embeddings()
        self.vocab_size = len(words)
        self.embedding_dim = embeddings.shape[1]
        self.embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        self.words = words
        
        # Create token pairs
        np.random.seed(self.seed)
        n_pairs = min(self.n_pairs, self.vocab_size * 10)
        input_tokens = torch.randint(0, self.vocab_size, (n_pairs,))
        target_tokens = torch.randint(0, self.vocab_size, (n_pairs,))
        
        # Split
        n_train = int(0.9 * n_pairs)
        
        self.train_dataset = TensorDataset(
            input_tokens[:n_train], target_tokens[:n_train]
        )
        self.val_dataset = TensorDataset(
            input_tokens[n_train:], target_tokens[n_train:]
        )
    
    def _load_embeddings(self):
        """Load word embeddings from file."""
        words = []
        embeddings = []
        
        with open(self.embeddings_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                
                word = parts[0]
                try:
                    vector = [float(x) for x in parts[1:]]
                    words.append(word)
                    embeddings.append(vector)
                except ValueError:
                    continue
                
                if self.max_embeddings and len(words) >= self.max_embeddings:
                    break
        
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return words, embeddings
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# ============================================================================
# Hyperparameter Tuning with Optuna
# ============================================================================

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function for hyperparameter tuning."""
    
    # Sample hyperparameters
    # Lattice size
    n_theta = trial.suggest_categorical('n_theta', [12, 16, 20, 24, 32, 40])
    n_phi = trial.suggest_categorical('n_phi', [24, 32, 40, 48, 64, 80])
    
    # Lattice size constraint: keep total cells reasonable (avoid OOM)
    max_cells = 3200
    if n_theta * n_phi > max_cells:
        # Prune this trial if lattice is too large
        raise optuna.TrialPruned(f"Lattice too large: {n_theta}x{n_phi}={n_theta*n_phi} > {max_cells}")
    
    hparams = {
        # Lattice architecture
        'n_theta': n_theta,
        'n_phi': n_phi,
        'max_steps': trial.suggest_int('max_steps', 6, 20),
        
        # Network architecture
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 384, 512, 768, 1024]),
        'encoder_layers': trial.suggest_int('encoder_layers', 1, 5),
        'output_layers': trial.suggest_int('output_layers', 1, 5),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        
        # Optimization
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
    }
    
    # Data module
    data_module = ToroidalDataModule(
        embeddings_path=args.embeddings_path,
        max_embeddings=args.max_embeddings,
        n_pairs=args.n_pairs,
        batch_size=hparams['batch_size'],
        num_workers=args.num_workers
    )
    data_module.setup()
    
    # Model
    model = ToroidalLightningModule(
        vocab_size=data_module.vocab_size,
        embedding_dim=data_module.embedding_dim,
        n_theta=hparams['n_theta'],
        n_phi=hparams['n_phi'],
        max_steps=hparams['max_steps'],
        hidden_dim=hparams['hidden_dim'],
        encoder_layers=hparams['encoder_layers'],
        output_layers=hparams['output_layers'],
        dropout=hparams['dropout'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        warmup_steps=hparams['warmup_steps'],
        pretrained_embeddings=data_module.embeddings_tensor
    )
    
    # Callbacks
    callbacks = [
        PyTorchLightningPruningCallback(trial, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='hpo_logs',
        name=f'trial_{trial.number}'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.hpo_epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else 'auto',
        precision='bf16-mixed',
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    try:
        trainer.fit(model, data_module)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')
    
    return trainer.callback_metrics.get('val_loss', float('inf')).item()


def run_hpo(args: argparse.Namespace):
    """Run hyperparameter optimization."""
    
    # Create Optuna study
    study = optuna.create_study(
        study_name='toroidal_hpo',
        direction='minimize',
        storage=f'sqlite:///hpo_results/optuna_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    print("="*60)
    print("Starting Hyperparameter Optimization")
    print(f"Number of trials: {args.n_trials}")
    print("="*60)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,  # Sequential for DDP
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Hyperparameter Optimization Complete!")
    print("="*60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    best_params_path = Path('hpo_results') / 'best_params.json'
    import json
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\nBest params saved to: {best_params_path}")
    
    return study.best_trial.params


# ============================================================================
# Main Training
# ============================================================================

def train(args: argparse.Namespace, hparams: Optional[Dict] = None):
    """Main training function."""
    
    # Use HPO params or defaults
    if hparams is None:
        hparams = {
            'n_theta': args.n_theta,
            'n_phi': args.n_phi,
            'max_steps': args.max_steps,
            'hidden_dim': args.hidden_dim,
            'encoder_layers': args.encoder_layers,
            'output_layers': args.output_layers,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'batch_size': args.batch_size,
        }
    
    # Data module
    data_module = ToroidalDataModule(
        embeddings_path=args.embeddings_path,
        max_embeddings=args.max_embeddings,
        n_pairs=args.n_pairs,
        batch_size=hparams['batch_size'],
        num_workers=args.num_workers
    )
    data_module.setup()
    
    print(f"\nData loaded:")
    print(f"  Vocab size: {data_module.vocab_size}")
    print(f"  Embedding dim: {data_module.embedding_dim}")
    print(f"  Train samples: {len(data_module.train_dataset)}")
    print(f"  Val samples: {len(data_module.val_dataset)}")
    
    # Model
    model = ToroidalLightningModule(
        vocab_size=data_module.vocab_size,
        embedding_dim=data_module.embedding_dim,
        n_theta=hparams['n_theta'],
        n_phi=hparams['n_phi'],
        max_steps=hparams['max_steps'],
        hidden_dim=hparams['hidden_dim'],
        encoder_layers=hparams.get('encoder_layers', args.encoder_layers),
        output_layers=hparams.get('output_layers', args.output_layers),
        dropout=hparams['dropout'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
        warmup_steps=hparams['warmup_steps'],
        pretrained_embeddings=data_module.embeddings_tensor
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Torus size: {hparams['n_theta']} x {hparams['n_phi']} = {hparams['n_theta'] * hparams['n_phi']} cells")
    print(f"Encoder layers: {hparams.get('encoder_layers', args.encoder_layers)}")
    print(f"Output layers: {hparams.get('output_layers', args.output_layers)}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints_lightning',
            filename=f'toroidal_{timestamp}_{{epoch:02d}}_{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar()
    ]
    
    # Loggers
    loggers = [
        TensorBoardLogger(save_dir='logs_lightning', name='toroidal'),
        CSVLogger(save_dir='logs_lightning', name='toroidal_csv')
    ]
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else 'auto',
        precision='bf16-mixed',
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad,
        val_check_interval=0.25,  # Validate 4 times per epoch
        log_every_n_steps=10,
    )
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.fit(model, data_module)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("="*60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lightning Training for Toroidal Token Generation")
    
    # Data
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--max-embeddings", type=int, default=50000)
    parser.add_argument("--n-pairs", type=int, default=500000)
    
    # Lattice architecture
    parser.add_argument("--n-theta", type=int, default=24, help="Torus theta dimension")
    parser.add_argument("--n-phi", type=int, default=48, help="Torus phi dimension")
    parser.add_argument("--max-steps", type=int, default=12, help="Max propagation steps")
    
    # Network architecture
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--encoder-layers", type=int, default=2, help="Number of encoder MLP layers")
    parser.add_argument("--output-layers", type=int, default=2, help="Number of output MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--accumulate-grad", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=8)
    
    # HPO
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--hpo-epochs", type=int, default=10, help="Epochs per HPO trial")
    parser.add_argument("--timeout", type=int, default=None, help="HPO timeout in seconds")
    parser.add_argument("--use-best-params", action="store_true", help="Use saved best params")
    
    args = parser.parse_args()
    
    # Create directories
    Path('checkpoints_lightning').mkdir(exist_ok=True)
    Path('logs_lightning').mkdir(exist_ok=True)
    Path('hpo_results').mkdir(exist_ok=True)
    
    if args.hpo:
        # Run hyperparameter optimization
        best_params = run_hpo(args)
        
        print("\nTraining with best hyperparameters...")
        args.epochs = 100  # Full training
        train(args, best_params)
    
    elif args.use_best_params:
        # Load and use best params
        import json
        best_params_path = Path('hpo_results') / 'best_params.json'
        if best_params_path.exists():
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            print(f"Using best params from: {best_params_path}")
            train(args, best_params)
        else:
            print("No saved best params found. Running with defaults.")
            train(args)
    
    else:
        # Normal training
        train(args)


if __name__ == "__main__":
    main()
