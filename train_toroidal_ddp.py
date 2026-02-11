#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) Training for Toroidal Token Generation
Optimized for 8x H100 GPUs

Usage:
    torchrun --nproc_per_node=8 train_toroidal_ddp.py [args]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
import json

# Local imports
from toroidal_lattice import (
    ToroidalLatticeVisualizer,
    ToroidalCell,
    create_toroidal_lattice
)


def setup_ddp():
    """Initialize distributed process group."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Clean up distributed process group."""
    destroy_process_group()


def get_rank():
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    return get_rank() == 0


def print_rank0(msg):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


# ============================================================================
# Model Components (same as notebook but optimized for DDP)
# ============================================================================

class BidirectionalToroidalLattice(nn.Module):
    """
    Toroidal lattice with bidirectional propagation.
    Optimized for multi-GPU training.
    """
    
    def __init__(self, n_theta: int, n_phi: int, max_steps: int = 10, device='cuda'):
        super().__init__()
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.max_steps = max_steps
        self.device = device
        self._total_cells = n_theta * n_phi
        
        # Learnable parameters
        self.forward_step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.forward_decay = nn.Parameter(torch.tensor(0.85))
        self.reverse_step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.reverse_decay = nn.Parameter(torch.tensor(0.85))
        self.interaction_weight = nn.Parameter(torch.tensor(0.5))
        self.threshold = nn.Parameter(torch.tensor(0.01))
        self.bounce_angles = nn.Parameter(torch.randn(self._total_cells, 3) * 0.1)
        
        # Pre-compute adjacency matrices (more efficient for DDP)
        self._build_adjacency_matrices()
    
    def _build_adjacency_matrices(self):
        """Build dense adjacency matrices for efficient GPU computation."""
        n = self._total_cells
        forward_adj = torch.zeros(n, n)
        reverse_adj = torch.zeros(n, n)
        
        for idx in range(n):
            ti = idx // self.n_phi
            pi = idx % self.n_phi
            
            # Forward neighbors
            for dti, dpi in [(1, 0), (0, 1), (1, 1)]:
                ni = ((ti + dti) % self.n_theta) * self.n_phi + ((pi + dpi) % self.n_phi)
                forward_adj[idx, ni] = 1.0
            
            # Reverse neighbors
            for dti, dpi in [(-1, 0), (0, -1), (-1, -1)]:
                ni = ((ti + dti) % self.n_theta) * self.n_phi + ((pi + dpi) % self.n_phi)
                reverse_adj[idx, ni] = 1.0
        
        # Normalize
        forward_adj = forward_adj / (forward_adj.sum(dim=1, keepdim=True) + 1e-8)
        reverse_adj = reverse_adj / (reverse_adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self.register_buffer('forward_adj', forward_adj)
        self.register_buffer('reverse_adj', reverse_adj)
    
    def _propagate(self, entry_probs: torch.Tensor, adj: torch.Tensor, 
                   decay: torch.Tensor, step_weights: torch.Tensor) -> torch.Tensor:
        """Efficient batched propagation using dense matrix ops."""
        state = entry_probs
        states = [state]
        
        decay_val = torch.clamp(decay, 0.5, 0.99)
        
        for _ in range(self.max_steps):
            # Propagate: mix self-retention with neighbor spread
            propagated = torch.matmul(state, adj)
            
            # Apply bounce transform
            angle_mag = self.bounce_angles.abs().mean(dim=1)
            angle_factor = torch.cos(angle_mag)
            propagated = propagated * (0.5 + 0.5 * angle_factor.unsqueeze(0))
            
            state = (0.3 * state + 0.7 * propagated) * decay_val
            states.append(state)
        
        # Weighted combination
        weights = torch.softmax(step_weights, dim=0)
        final = sum(w * s for w, s in zip(weights, states[1:]))
        return final
    
    def forward(self, entry_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional propagation."""
        forward_state = self._propagate(
            entry_probs, self.forward_adj, self.forward_decay, self.forward_step_weights
        )
        reverse_state = self._propagate(
            entry_probs, self.reverse_adj, self.reverse_decay, self.reverse_step_weights
        )
        
        interaction = forward_state * reverse_state
        interaction_w = torch.sigmoid(self.interaction_weight)
        combined = forward_state + reverse_state + interaction_w * interaction
        
        return combined, interaction


class ToroidalTokenGenerator(nn.Module):
    """
    Token generation model with bidirectional toroidal propagation.
    Optimized for DDP training on H100 GPUs.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_theta: int = 20,
        n_phi: int = 40,
        max_steps: int = 12,
        hidden_dim: int = 512,
        dropout: float = 0.1,
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
        
        # Signal encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_theta * n_phi),
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
        # Bidirectional propagation
        self.propagation = BidirectionalToroidalLattice(n_theta, n_phi, max_steps)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(n_theta * n_phi, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens → logits."""
        # Embed tokens
        if input_tokens.dim() == 2:
            emb = self.token_embedding(input_tokens).mean(dim=1)
        else:
            emb = self.token_embedding(input_tokens)
        
        # Encode to entry probabilities
        logits = self.signal_encoder(emb)
        temp = self.temperature.abs() + 0.1
        entry_probs = torch.softmax(logits / temp, dim=-1)
        
        # Bidirectional propagation
        combined, _ = self.propagation(entry_probs)
        
        # Project to vocabulary
        output_logits = self.output_projection(combined)
        
        return output_logits


# ============================================================================
# Data Loading
# ============================================================================

def load_embeddings(filepath: str, max_embeddings: int = None):
    """Load word embeddings from file."""
    words = []
    embeddings = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
            
            if max_embeddings and len(words) >= max_embeddings:
                break
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return words, embeddings


def create_token_dataset(vocab_size: int, n_pairs: int, seed: int = 42):
    """Create token pairs for training."""
    np.random.seed(seed)
    input_tokens = torch.randint(0, vocab_size, (n_pairs,))
    target_tokens = torch.randint(0, vocab_size, (n_pairs,))
    return TensorDataset(input_tokens, target_tokens)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, device):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for input_tokens, target_tokens in loader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_tokens)
            loss = F.cross_entropy(logits, target_tokens)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == target_tokens).sum().item()
            total_tokens += target_tokens.size(0)
        
        total_loss += loss.item()
    
    return total_loss / len(loader), total_correct / total_tokens


@torch.no_grad()
def validate(model, loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for input_tokens, target_tokens in loader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_tokens)
            loss = F.cross_entropy(logits, target_tokens)
        
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == target_tokens).sum().item()
        total_tokens += target_tokens.size(0)
        total_loss += loss.item()
    
    return total_loss / len(loader), total_correct / total_tokens


# ============================================================================
# Main Training Loop
# ============================================================================

def main(args):
    # Setup DDP
    setup_ddp()
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    print_rank0(f"Training with {world_size} GPUs")
    print_rank0(f"Args: {args}")
    
    # Load embeddings (only on rank 0, then broadcast)
    if is_main_process():
        print("Loading embeddings...")
        words, embeddings = load_embeddings(args.embeddings_path, args.max_embeddings)
        vocab_size = len(words)
        embedding_dim = embeddings.shape[1]
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        print(f"Loaded {vocab_size} embeddings of dim {embedding_dim}")
    else:
        vocab_size = None
        embedding_dim = None
        embeddings_tensor = None
    
    # Broadcast vocab info (must be on GPU for NCCL)
    if world_size > 1:
        info = torch.tensor([vocab_size or 0, embedding_dim or 0], device=device)
        torch.distributed.broadcast(info, src=0)
        vocab_size, embedding_dim = int(info[0].item()), int(info[1].item())
        
        # Move embeddings to GPU for NCCL broadcast
        if is_main_process():
            embeddings_tensor = embeddings_tensor.to(device)
        else:
            embeddings_tensor = torch.zeros(vocab_size, embedding_dim, device=device)
        
        torch.distributed.broadcast(embeddings_tensor, src=0)
        
        # Move back to CPU to save GPU memory during model creation
        embeddings_tensor = embeddings_tensor.cpu()
    
    # Create dataset
    n_pairs = min(args.n_pairs, vocab_size * 10)
    dataset = create_token_dataset(vocab_size, n_pairs)
    
    # Split train/val
    n_train = int(0.9 * len(dataset))
    train_dataset = TensorDataset(dataset.tensors[0][:n_train], dataset.tensors[1][:n_train])
    val_dataset = TensorDataset(dataset.tensors[0][n_train:], dataset.tensors[1][n_train:])
    
    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Data loaders with optimized settings for H100
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print_rank0(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print_rank0(f"Batch size per GPU: {args.batch_size}, Effective batch: {args.batch_size * world_size}")
    
    # Create model
    model = ToroidalTokenGenerator(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pretrained_embeddings=embeddings_tensor
    ).to(device)
    
    # Wrap with DDP (find_unused_parameters for threshold param that may not get gradients)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, fused=True)

    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # Grad scaler for mixed precision
    scaler = GradScaler('cuda')
    
    # Training loop
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print_rank0("="*60)
    print_rank0("Starting DDP Training")
    print_rank0("="*60)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        scheduler.step()
        
        # Aggregate metrics across GPUs
        metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.AVG)
        train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        # Save best model (rank 0 only)
        if is_main_process():
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(args.checkpoint_dir).mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'args': vars(args),
                }, f'{args.checkpoint_dir}/toroidal_ddp_{timestamp}_best.pt')
            
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"LR: {lr:.2e}")
    
    print_rank0("="*60)
    print_rank0("Training Complete!")
    print_rank0(f"Best validation loss: {best_val_loss:.4f}")
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training for Toroidal Token Generation")
    
    # Data
    parser.add_argument("--embeddings-path", type=str, required=True,
                        help="Path to word embeddings file")
    parser.add_argument("--max-embeddings", type=int, default=50000,
                        help="Maximum embeddings to load")
    parser.add_argument("--n-pairs", type=int, default=500000,
                        help="Number of token pairs to generate")
    
    # Model
    parser.add_argument("--n-theta", type=int, default=24,
                        help="Cells around torus tube")
    parser.add_argument("--n-phi", type=int, default=48,
                        help="Cells around torus")
    parser.add_argument("--max-steps", type=int, default=12,
                        help="Propagation steps")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers per GPU")
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    
    args = parser.parse_args()
    main(args)
