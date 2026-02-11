import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from cell import DEVICE
from lattice_visualizer import LatticeVisualizer

# Try to import custom kernels (optional)
try:
    from kernels import create_kernels, PulsePropagationKernels
    CUSTOM_KERNELS_AVAILABLE = True
except ImportError:
    CUSTOM_KERNELS_AVAILABLE = False
    PulsePropagationKernels = None


class LatticeEncoder(nn.Module):
    """
    Encodes input data into lattice cell activations.
    Maps input features to initial pulse powers for each cell.
    """
    
    def __init__(self, input_dim: int, n_cells: int, device: torch.device = None):
        super().__init__()
        self.device = device if device is not None else DEVICE
        self.input_dim = input_dim
        self.n_cells = n_cells
        
        # Linear projection from input to cell activations
        self.encoder = nn.Linear(input_dim, n_cells, device=self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to cell activations.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Cell activations of shape (batch_size, n_cells)
        """
        x = x.to(self.device)
        activations = torch.sigmoid(self.encoder(x))  # [0, 1] activations
        return activations


class LatticeDecoder(nn.Module):
    """
    Decodes lattice cell states back to reconstructed input.
    """
    
    def __init__(self, n_cells: int, output_dim: int, device: torch.device = None):
        super().__init__()
        self.device = device if device is not None else DEVICE
        self.n_cells = n_cells
        self.output_dim = output_dim
        
        # Linear projection from cell states to output
        self.decoder = nn.Linear(n_cells, output_dim, device=self.device)
        
    def forward(self, cell_states: torch.Tensor) -> torch.Tensor:
        """
        Decode cell states to reconstruction.
        
        Args:
            cell_states: Cell states of shape (batch_size, n_cells)
            
        Returns:
            Reconstructed output of shape (batch_size, output_dim)
        """
        return self.decoder(cell_states)


class DifferentiableLattice(nn.Module):
    """
    A differentiable version of the lattice for backpropagation.
    
    Implements soft attention-based pulse propagation that maintains gradients
    through the forward pass for training.
    
    Features learnable propagation steps:
    - step_weights: Learnable importance of each step (softmax normalized)
    - effective_steps: Learnable continuous value controlling computation depth
    - Final output is weighted combination of states at each step
    
    Supports custom CUDA/MPS kernels for accelerated propagation.
    """
    
    def __init__(
        self, 
        lattice: LatticeVisualizer, 
        max_steps: int = 10,
        use_custom_kernels: bool = True
    ):
        super().__init__()
        self.lattice = lattice
        self.device = lattice.device
        self.n_cells = len(lattice.cells)
        self.max_steps = max_steps
        
        # Initialize custom kernels if available and requested
        self.use_custom_kernels = use_custom_kernels and CUSTOM_KERNELS_AVAILABLE
        self.kernels: Optional[PulsePropagationKernels] = None
        if self.use_custom_kernels:
            try:
                self.kernels = create_kernels(self.device)
                print(f"DifferentiableLattice using {self.kernels.backend} kernels")
            except Exception as e:
                print(f"Custom kernels unavailable: {e}, using PyTorch fallback")
                self.use_custom_kernels = False
        
        # Extract learnable parameters from cells
        self._init_parameters()
        
        # Precompute adjacency matrix for differentiable propagation
        self._build_adjacency_matrix()
        
        # Initialize learnable propagation step parameters
        self._init_step_parameters()
    
    def _init_parameters(self):
        """Initialize learnable parameters from lattice cells."""
        # Stack cell parameters into tensors
        means = torch.stack([cell.mean for cell in self.lattice.cells]).squeeze()
        std_devs = torch.stack([cell.std_dev for cell in self.lattice.cells]).squeeze()
        bounce_angles = torch.stack([cell.bounce_angles for cell in self.lattice.cells])
        split_probs = torch.stack([cell.split_prob for cell in self.lattice.cells]).squeeze()
        join_probs = torch.stack([cell.join_prob for cell in self.lattice.cells]).squeeze()
        
        # Register as learnable parameters
        self.means = nn.Parameter(means)
        self.std_devs = nn.Parameter(std_devs)
        self.bounce_angles = nn.Parameter(bounce_angles)  # (n_cells, 3)
        self.split_probs = nn.Parameter(split_probs)  # (n_cells,)
        self.join_probs = nn.Parameter(join_probs)  # (n_cells,)
    
    def _init_step_parameters(self):
        """Initialize learnable step parameters for adaptive computation."""
        # Learnable step weights - controls importance of each step
        # Initialized to favor middle steps (like a Gaussian)
        init_weights = torch.zeros(self.max_steps, device=self.device)
        for i in range(self.max_steps):
            # Gaussian-like initialization centered at middle
            center = self.max_steps / 2
            init_weights[i] = np.exp(-0.5 * ((i - center) / (self.max_steps / 4)) ** 2)
        self.step_weights = nn.Parameter(init_weights)
        
        # Learnable effective steps - continuous value for soft step count
        # Initialized to half of max_steps
        self.effective_steps = nn.Parameter(torch.tensor([float(self.max_steps // 2)], device=self.device))
        
        # Learnable decay rate per step
        self.decay_rate = nn.Parameter(torch.tensor([0.95], device=self.device))
        
    def _build_adjacency_matrix(self):
        """Build sparse adjacency matrix for message passing."""
        n = self.n_cells
        # Create adjacency matrix
        adj = torch.zeros(n, n, device=self.device)
        
        for i, neighbors in self.lattice.adjacency.items():
            for j in neighbors:
                adj[i, j] = 1.0
        
        # Normalize rows (average over neighbors)
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer('adjacency', adj / row_sum)
        
        # Store positions for direction computation
        self.register_buffer('positions', self.lattice.positions.clone())
    
    def compute_propagation_weights(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute soft propagation weights based on cell parameters.
        
        Uses the normal distribution parameters (mean, std_dev) and split/join
        probabilities to determine how activations spread to neighbors.
        
        Args:
            activations: Current cell activations (batch_size, n_cells)
            
        Returns:
            Propagation weights (n_cells, n_cells)
        """
        # Use custom kernel if available
        if self.use_custom_kernels and self.kernels is not None:
            return self.kernels.compute_propagation_weights(
                self.std_devs, self.adjacency, threshold=0.5
            )
        
        # Fallback: PyTorch implementation
        # Probability of going straight (vs bouncing) per cell
        # Higher std_dev = more spread = more bouncing
        # P(straight) ≈ erf(threshold / std_dev)
        threshold = 0.5
        std_devs_safe = torch.clamp(torch.abs(self.std_devs), min=2.0)
        straight_prob = torch.erf(threshold / (std_devs_safe * np.sqrt(2)))
        
        # Clamp split/join probabilities to [0, 1]
        split_probs_safe = torch.clamp(self.split_probs, min=0.0, max=1.0)
        join_probs_safe = torch.clamp(self.join_probs, min=0.0, max=1.0)
        
        # Create propagation matrix
        # Straight: energy stays more concentrated
        # Bounce: energy spreads to neighbors
        
        # Identity component (self-retention based on straight probability)
        # Higher split_prob = less self-retention (energy spreads out)
        self_retention = straight_prob * 0.3 * (1 - split_probs_safe * 0.5)
        identity = torch.diag(self_retention)
        
        # Neighbor spreading (based on 1 - straight_prob)
        # Split probability increases spread to neighbors
        spread_factor = (1 - straight_prob) + split_probs_safe * 0.3
        spread_factor = spread_factor.unsqueeze(1)  # (n_cells, 1)
        neighbor_spread = self.adjacency * spread_factor
        
        # Join probability: cells with high join_prob accumulate more from neighbors
        # Apply join boost to incoming connections (columns of the matrix)
        join_boost = 1.0 + join_probs_safe * 0.5  # (n_cells,)
        neighbor_spread = neighbor_spread * join_boost.unsqueeze(0)  # Boost columns
        
        # Combine: some energy stays, some spreads to neighbors
        propagation = identity + neighbor_spread * 0.7
        
        # Normalize to conserve total energy (approximately)
        propagation = propagation / propagation.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        return propagation
    
    def apply_bounce_transform(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply bounce angle transformation to activations.
        
        The bounce angles modulate how much each cell's activation
        is transformed during propagation.
        
        Args:
            activations: Cell activations (batch_size, n_cells)
            
        Returns:
            Transformed activations (batch_size, n_cells)
        """
        # Clamp bounce angles to valid range [0, 2] radians
        constrained_angles = torch.clamp(self.bounce_angles, min=0.0, max=2.0)
        
        # Use custom kernel if available
        if self.use_custom_kernels and self.kernels is not None:
            return self.kernels.apply_bounce_transform(activations, constrained_angles)
        
        # Fallback: PyTorch implementation
        # Bounce angles affect activation magnitude (angles are in radians, max 2 rad each)
        # Large angles = more transformation = slightly reduced activation
        angle_magnitude = constrained_angles.mean(dim=1)  # (n_cells,) - average of 3 axes
        angle_factor = torch.cos(angle_magnitude)  # Already in radians, no conversion needed
        
        # Apply soft transformation (keep values positive)
        transformed = activations * (0.5 + 0.5 * angle_factor.unsqueeze(0))
        return transformed
    
    def forward(
        self, 
        initial_activations: torch.Tensor, 
        n_steps: int = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Differentiable forward pass through the lattice with learnable steps.
        
        Args:
            initial_activations: Initial cell activations (batch_size, n_cells)
            n_steps: Number of propagation steps (None = use learnable effective_steps)
            
        Returns:
            final_state: Final cell states (weighted by learnable step_weights)
            history: List of intermediate states
        """
        state = initial_activations.to(self.device)
        history = [state]
        
        # Determine actual number of steps
        if n_steps is None:
            # Use learnable effective_steps, clamped to valid range
            effective = torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps))
            actual_steps = int(torch.ceil(effective).item())
        else:
            actual_steps = min(n_steps, self.max_steps)
        
        # Get learnable decay rate (clamped to valid range)
        decay = torch.clamp(self.decay_rate, min=2.0, max=0.99)
        decay_val = decay.item()
        
        # Compute propagation weights once (reused across steps)
        prop_weights = self.compute_propagation_weights(state)
        
        # Propagate for actual_steps
        for step in range(actual_steps):
            state_max = state.max().item()
            
            # Use fused kernel if available (combines all operations)
            if self.use_custom_kernels and self.kernels is not None:
                state = self.kernels.fused_propagation_step(
                    state, prop_weights, self.bounce_angles, decay_val, state_max
                )
            else:
                # Fallback: separate operations
                # Propagate: state @ prop_weights^T
                # Each cell receives weighted sum from neighbors
                state = torch.matmul(state, prop_weights.T)
                
                # Apply bounce transformation
                state = self.apply_bounce_transform(state)
                
                # Apply learnable decay
                state = state * decay
                
                # Apply nonlinearity to keep values bounded
                state = torch.sigmoid(state * 2 - 1) * max(state_max, 0.1)
            
            history.append(state)
        
        # Compute weighted output using learnable step_weights
        # This allows gradient flow to all steps based on their importance
        if len(history) > 1:
            # Get normalized step weights (softmax over steps actually computed)
            weights = torch.softmax(self.step_weights[:len(history)], dim=0)
            
            # Soft step masking based on effective_steps
            # Steps beyond effective_steps get progressively less weight
            if n_steps is None:
                effective = torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps))
                step_mask = torch.zeros(len(history), device=self.device)
                for i in range(len(history)):
                    # Smooth sigmoid mask: full weight up to effective_steps, then decay
                    step_mask[i] = torch.sigmoid(5.0 * (effective - i))
                weights = weights * step_mask
                weights = weights / (weights.sum() + 1e-8)
            
            # Weighted combination of all states
            stacked_states = torch.stack(history, dim=0)  # (n_steps, batch, n_cells)
            weighted_state = torch.einsum('s,sbc->bc', weights, stacked_states)
        else:
            weighted_state = state
        
        return weighted_state, history
    
    def get_effective_steps(self) -> float:
        """Get the current learned effective number of steps."""
        return torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps)).item()
    
    def get_step_weights(self) -> torch.Tensor:
        """Get the normalized step weights."""
        return torch.softmax(self.step_weights, dim=0)
    
    def sync_to_lattice(self):
        """Sync learned parameters back to the original lattice cells."""
        with torch.no_grad():
            for i, cell in enumerate(self.lattice.cells):
                cell.mean = self.means[i:i+1].clone()
                cell.std_dev = torch.clamp(torch.abs(self.std_devs[i:i+1].clone()), min=2.0)
                cell.bounce_angles = torch.clamp(self.bounce_angles[i].clone(), min=0.0, max=2.0)
                cell.split_prob = torch.clamp(self.split_probs[i:i+1].clone(), min=0.0, max=1.0)
                cell.join_prob = torch.clamp(self.join_probs[i:i+1].clone(), min=0.0, max=1.0)
    
    def get_learned_parameters_summary(self) -> dict:
        """Get a summary of all learned parameters including step parameters."""
        with torch.no_grad():
            # Get constrained bounce angles for accurate reporting
            constrained_angles = torch.clamp(self.bounce_angles, min=0.0, max=2.0)
            return {
                'effective_steps': self.get_effective_steps(),
                'decay_rate': torch.clamp(self.decay_rate, min=2.0, max=0.99).item(),
                'step_weights': self.get_step_weights().cpu().numpy(),
                'mean_avg': self.means.mean().item(),
                'mean_std': self.means.std().item(),
                'std_dev_avg': self.std_devs.mean().item(),
                'std_dev_std': self.std_devs.std().item(),
                'bounce_angles_avg': constrained_angles.mean().item(),
                'bounce_angles_std': constrained_angles.std().item(),
                'split_prob_avg': self.split_probs.mean().item(),
                'split_prob_std': self.split_probs.std().item(),
                'join_prob_avg': self.join_probs.mean().item(),
                'join_prob_std': self.join_probs.std().item(),
                'backend': self.kernels.backend if self.kernels else 'pytorch',
            }


class LatticeAutoencoder(nn.Module):
    """
    Full autoencoder using the lattice for information processing.
    
    Architecture:
    1. Encoder: Input -> Cell activations
    2. Lattice: Propagate activations through cells (learnable steps)
    3. Decoder: Final cell states -> Reconstructed output
    
    Training minimizes reconstruction loss to learn:
    - Encoder/decoder mappings
    - Cell parameters (mean, std_dev, bounce_angles)
    - Propagation steps (effective_steps, step_weights, decay_rate)
    """
    
    def __init__(
        self,
        lattice: LatticeVisualizer,
        input_dim: int,
        propagation_steps: int = 5,
        max_steps: int = 10,
        learn_steps: bool = True,
        use_custom_kernels: bool = True
    ):
        super().__init__()
        self.device = lattice.device
        self.propagation_steps = propagation_steps
        self.max_steps = max_steps
        self.learn_steps = learn_steps
        
        n_cells = len(lattice.cells)
        
        # Components
        self.encoder = LatticeEncoder(input_dim, n_cells, self.device)
        self.diff_lattice = DifferentiableLattice(
            lattice, max_steps=max_steps, use_custom_kernels=use_custom_kernels
        )
        self.decoder = LatticeDecoder(n_cells, input_dim, self.device)
        
        # Store reference to original lattice
        self.lattice = lattice
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: encode -> propagate -> decode.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            reconstruction: Reconstructed input (batch_size, input_dim)
            final_state: Final lattice state (batch_size, n_cells)
            history: Propagation history
        """
        # Encode input to initial activations
        initial_activations = self.encoder(x)
        
        # Propagate through lattice
        # Use None for n_steps to enable learnable steps, or fixed value if disabled
        n_steps = None if self.learn_steps else self.propagation_steps
        final_state, history = self.diff_lattice(initial_activations, n_steps)
        
        # Decode to reconstruction
        reconstruction = self.decoder(final_state)
        
        return reconstruction, final_state, history
    
    def sync_to_lattice(self):
        """Sync learned parameters back to original lattice."""
        self.diff_lattice.sync_to_lattice()
    
    def get_learned_steps_info(self) -> dict:
        """Get information about learned propagation steps."""
        return self.diff_lattice.get_learned_parameters_summary()


class LatticeTrainer:
    """
    Training loop for the lattice autoencoder.
    
    Uses reconstruction loss (MSE) to train:
    - Encoder/decoder weights
    - Cell parameters (mean, std_dev, bounce_angles)
    """
    
    def __init__(
        self,
        autoencoder: LatticeAutoencoder,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.autoencoder = autoencoder
        self.device = autoencoder.device
        
        # Optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.loss_history = []
        self.val_loss_history = []
        self.epoch = 0
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            batch: Input batch (batch_size, input_dim)
            
        Returns:
            Loss value
        """
        self.autoencoder.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass
        reconstruction, final_state, _ = self.autoencoder(batch)
        
        # Compute reconstruction loss
        loss = self.criterion(reconstruction, batch)
        
        # Optional: Add regularization on cell parameters
        # Encourage diverse std_devs (some cells straight, some bouncy)
        std_dev_var = self.autoencoder.diff_lattice.std_devs.var()
        diversity_loss = -0.01 * std_dev_var  # Encourage variance in std_devs
        
        total_loss = loss + diversity_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Ensure std_devs stay >= 2.0
        with torch.no_grad():
            self.autoencoder.diff_lattice.std_devs.data = torch.clamp(
                torch.abs(self.autoencoder.diff_lattice.std_devs.data), min=2.0
            )
        
        return loss.item()
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with training data
            
        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle (data, label) tuples
            
            loss = self.train_step(batch)
            total_loss += loss
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        self.loss_history.append(avg_loss)
        self.scheduler.step(avg_loss)
        self.epoch += 1
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Run validation on the validation set.
        
        Args:
            val_dataloader: DataLoader with validation data
            
        Returns:
            Average validation loss
        """
        self.autoencoder.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(self.device)
            reconstruction, _, _ = self.autoencoder(batch)
            loss = self.criterion(reconstruction, batch)
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader = None,
        n_epochs: int = 100,
        print_every: int = 10,
        sync_every: int = 50
    ) -> Tuple[List[float], List[float]]:
        """
        Full training loop with optional validation.
        
        Args:
            dataloader: DataLoader with training data
            val_dataloader: Optional DataLoader with validation/test data
            n_epochs: Number of epochs
            print_every: Print loss every N epochs
            sync_every: Sync parameters to lattice every N epochs
            
        Returns:
            Tuple of (train_loss_history, val_loss_history)
        """
        print(f"Starting training on device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.autoencoder.parameters())}")
        if val_dataloader:
            print(f"Validation enabled with {len(val_dataloader.dataset)} samples")
        
        for epoch in range(n_epochs):
            # Training
            avg_loss = self.train_epoch(dataloader)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_loss_history.append(val_loss)
                # Use validation loss for scheduler
                self.scheduler.step(val_loss)
            
            if (epoch + 1) % print_every == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_loss:.6f}")
            
            if (epoch + 1) % sync_every == 0:
                self.autoencoder.sync_to_lattice()
                print("  -> Synced parameters to lattice")
        
        # Final sync
        self.autoencoder.sync_to_lattice()
        final_msg = f"Training complete. Final train loss: {self.loss_history[-1]:.6f}"
        if self.val_loss_history:
            final_msg += f", Final val loss: {self.val_loss_history[-1]:.6f}"
        print(final_msg)
        
        return self.loss_history, self.val_loss_history
    
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, torch.Tensor]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            avg_loss: Average reconstruction loss
            reconstructions: Sample reconstructions
        """
        self.autoencoder.eval()
        total_loss = 0.0
        n_batches = 0
        all_reconstructions = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(self.device)
            reconstruction, _, _ = self.autoencoder(batch)
            
            loss = self.criterion(reconstruction, batch)
            total_loss += loss.item()
            n_batches += 1
            
            all_reconstructions.append(reconstruction.cpu())
        
        avg_loss = total_loss / max(n_batches, 1)
        reconstructions = torch.cat(all_reconstructions, dim=0)
        
        return avg_loss, reconstructions
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.autoencoder.state_dict(),
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'learned_params': self.autoencoder.get_learned_steps_info(),
            'lattice_config': {
                'n_cells': len(self.autoencoder.lattice.cells),
                'layers': self.autoencoder.lattice.layers,
                'hex_radius': self.autoencoder.lattice.hex_radius,
            }
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss_history = checkpoint.get('loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Sync loaded parameters to lattice
        self.autoencoder.sync_to_lattice()
        
        print(f"Checkpoint loaded from: {filepath}")
        print(f"  Epoch: {self.epoch}")
        if 'learned_params' in checkpoint:
            params = checkpoint['learned_params']
            print(f"  Effective steps: {params.get('effective_steps', 'N/A')}")
            print(f"  Decay rate: {params.get('decay_rate', 'N/A')}")
        
        return checkpoint


class LatticeEvaluator:
    """
    Evaluator for propagating inputs through the trained lattice and
    analyzing the pulse propagation through saved neuron states.
    """
    
    def __init__(self, autoencoder: LatticeAutoencoder):
        """
        Initialize evaluator with a trained autoencoder.
        
        Args:
            autoencoder: Trained LatticeAutoencoder instance
        """
        self.autoencoder = autoencoder
        self.device = autoencoder.device
        self.lattice = autoencoder.lattice
    
    @torch.no_grad()
    def propagate_input(
        self, 
        input_data: torch.Tensor,
        return_all_states: bool = False
    ) -> dict:
        """
        Propagate an input through the lattice and return detailed results.
        
        Args:
            input_data: Input tensor (batch_size, input_dim) or (input_dim,)
            return_all_states: Whether to return all intermediate states
            
        Returns:
            Dictionary with propagation results
        """
        self.autoencoder.eval()
        
        # Handle single sample
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        
        input_data = input_data.to(self.device)
        
        # Encode to cell activations
        initial_activations = self.autoencoder.encoder(input_data)
        
        # Propagate through lattice
        n_steps = None if self.autoencoder.learn_steps else self.autoencoder.propagation_steps
        final_state, history = self.autoencoder.diff_lattice(initial_activations, n_steps)
        
        # Decode to output
        reconstruction = self.autoencoder.decoder(final_state)
        
        # Get cell-level statistics
        cell_activations = {
            'initial': initial_activations.cpu(),
            'final': final_state.cpu(),
        }
        
        if return_all_states:
            cell_activations['history'] = [h.cpu() for h in history]
        
        # Get learned parameters
        learned_params = self.autoencoder.get_learned_steps_info()
        
        return {
            'input': input_data.cpu(),
            'reconstruction': reconstruction.cpu(),
            'mse': ((input_data.cpu() - reconstruction.cpu()) ** 2).mean().item(),
            'cell_activations': cell_activations,
            'n_steps_used': len(history) - 1,  # -1 because history includes initial state
            'effective_steps': learned_params['effective_steps'],
            'decay_rate': learned_params['decay_rate'],
            'step_weights': learned_params['step_weights'],
        }
    
    @torch.no_grad()
    def get_cell_states(self) -> dict:
        """
        Get the current state of all cells in the lattice.
        
        Returns:
            Dictionary with cell parameters
        """
        diff_lattice = self.autoencoder.diff_lattice
        
        return {
            'means': diff_lattice.means.cpu().numpy(),
            'std_devs': diff_lattice.std_devs.cpu().numpy(),
            'bounce_angles': diff_lattice.bounce_angles.cpu().numpy(),
            'step_weights': diff_lattice.get_step_weights().cpu().numpy(),
            'effective_steps': diff_lattice.get_effective_steps(),
            'decay_rate': torch.clamp(diff_lattice.decay_rate, 0.5, 0.99).cpu().item(),
            'n_cells': diff_lattice.n_cells,
            'max_steps': diff_lattice.max_steps,
        }
    
    @torch.no_grad()
    def visualize_propagation(
        self,
        input_data: torch.Tensor,
        title: str = "Pulse Propagation"
    ):
        """
        Visualize the propagation of an input through the lattice.
        
        Args:
            input_data: Single input sample (input_dim,) or (1, input_dim)
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        # Get propagation results with all states
        results = self.propagate_input(input_data, return_all_states=True)
        history = results['cell_activations']['history']
        
        n_steps = len(history)
        n_cells = history[0].shape[1]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Plot 1: Input vs Reconstruction
        ax1 = axes[0, 0]
        input_flat = results['input'].squeeze().numpy()
        recon_flat = results['reconstruction'].squeeze().numpy()
        
        # Check if it's an image (e.g., MNIST)
        if len(input_flat) == 784:
            ax1.imshow(input_flat.reshape(28, 28), cmap='gray')
            ax1.set_title(f'Input (MSE: {results["mse"]:.4f})')
        else:
            ax1.plot(input_flat, label='Input')
            ax1.plot(recon_flat, label='Reconstruction', alpha=0.7)
            ax1.set_title(f'Input vs Reconstruction (MSE: {results["mse"]:.4f})')
            ax1.legend()
        ax1.axis('off') if len(input_flat) == 784 else None
        
        # Plot 2: Reconstruction (if image)
        ax2 = axes[0, 1]
        if len(input_flat) == 784:
            ax2.imshow(recon_flat.reshape(28, 28), cmap='gray')
            ax2.set_title('Reconstruction')
            ax2.axis('off')
        else:
            ax2.plot(recon_flat)
            ax2.set_title('Reconstruction')
        
        # Plot 3: Cell activation heatmap over steps
        ax3 = axes[0, 2]
        activation_matrix = np.array([h.squeeze().numpy() for h in history])
        # Show subset of cells for visibility
        n_show = min(100, n_cells)
        cell_indices = np.linspace(0, n_cells-1, n_show, dtype=int)
        im = ax3.imshow(activation_matrix[:, cell_indices].T, aspect='auto', cmap='viridis')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cell (sampled)')
        ax3.set_title('Cell Activations Over Steps')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Mean activation per step
        ax4 = axes[1, 0]
        mean_activations = [h.mean().item() for h in history]
        ax4.plot(mean_activations, 'b-o')
        ax4.axvline(x=results['effective_steps'], color='r', linestyle='--', 
                   label=f'Effective steps: {results["effective_steps"]:.1f}')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Mean Activation')
        ax4.set_title('Mean Cell Activation per Step')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Step weights
        ax5 = axes[1, 1]
        step_weights = results['step_weights'][:n_steps]
        ax5.bar(range(len(step_weights)), step_weights, color='steelblue')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Weight')
        ax5.set_title('Learned Step Weights')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Initial vs Final cell activations
        ax6 = axes[1, 2]
        initial_act = results['cell_activations']['initial'].squeeze().numpy()
        final_act = results['cell_activations']['final'].squeeze().numpy()
        ax6.scatter(initial_act[:100], final_act[:100], alpha=0.5, s=10)
        ax6.plot([0, max(initial_act.max(), final_act.max())], 
                [0, max(initial_act.max(), final_act.max())], 'r--', alpha=0.5)
        ax6.set_xlabel('Initial Activation')
        ax6.set_ylabel('Final Activation')
        ax6.set_title('Cell Activation: Initial vs Final')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title}\nSteps: {results["n_steps_used"]}, '
                    f'Effective: {results["effective_steps"]:.1f}, '
                    f'Decay: {results["decay_rate"]:.3f}', fontsize=12)
        plt.tight_layout()
        
        return fig, results
    
    @classmethod
    def load_from_checkpoint(
        cls,
        filepath: str,
        lattice: 'LatticeVisualizer',
        input_dim: int = 784,
        max_steps: int = 10,
        device: str = None
    ) -> 'LatticeEvaluator':
        """
        Load an evaluator from a saved checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            lattice: LatticeVisualizer with same architecture as saved model
            input_dim: Input dimension (784 for MNIST)
            max_steps: Maximum propagation steps (auto-detected from checkpoint if None)
            device: Device to load model on
            
        Returns:
            LatticeEvaluator instance
        """
        if device is None:
            device = lattice.device
        
        # Load checkpoint first to detect max_steps
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        # Auto-detect max_steps from checkpoint if not specified or mismatched
        if 'model_state_dict' in checkpoint and 'diff_lattice.step_weights' in checkpoint['model_state_dict']:
            checkpoint_max_steps = checkpoint['model_state_dict']['diff_lattice.step_weights'].shape[0]
            if max_steps != checkpoint_max_steps:
                print(f"Note: Adjusting max_steps from {max_steps} to {checkpoint_max_steps} to match checkpoint")
                max_steps = checkpoint_max_steps
        
        # Check for n_cells mismatch
        if 'model_state_dict' in checkpoint and 'diff_lattice.means' in checkpoint['model_state_dict']:
            checkpoint_n_cells = checkpoint['model_state_dict']['diff_lattice.means'].shape[0]
            if len(lattice.cells) != checkpoint_n_cells:
                raise ValueError(
                    f"Lattice cell count mismatch: checkpoint has {checkpoint_n_cells} cells, "
                    f"but provided lattice has {len(lattice.cells)} cells. "
                    f"Create a lattice with matching configuration."
                )
        
        # Create autoencoder with same architecture
        autoencoder = LatticeAutoencoder(
            lattice=lattice,
            input_dim=input_dim,
            max_steps=max_steps,
            learn_steps=True
        )
        
        # Load state dict
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.eval()
        
        # Sync to lattice
        autoencoder.sync_to_lattice()
        
        print(f"Loaded evaluator from: {filepath}")
        if 'learned_params' in checkpoint:
            params = checkpoint['learned_params']
            print(f"  Effective steps: {params.get('effective_steps', 'N/A')}")
            print(f"  Decay rate: {params.get('decay_rate', 'N/A')}")
        
        return cls(autoencoder)


def create_training_data(
    n_samples: int = 1000,
    input_dim: int = 64,
    pattern_type: str = 'random'
) -> torch.Tensor:
    """
    Create synthetic training data.
    
    Args:
        n_samples: Number of samples
        input_dim: Dimension of each sample
        pattern_type: Type of patterns ('random', 'sinusoidal', 'sparse')
        
    Returns:
        Training data tensor (n_samples, input_dim)
    """
    if pattern_type == 'random':
        # Random patterns with some structure
        data = torch.randn(n_samples, input_dim)
        data = torch.sigmoid(data)
        
    elif pattern_type == 'sinusoidal':
        # Sinusoidal patterns at various frequencies
        t = torch.linspace(0, 4 * np.pi, input_dim).unsqueeze(0)
        freqs = torch.rand(n_samples, 1) * 3 + 0.5
        phases = torch.rand(n_samples, 1) * 2 * np.pi
        data = (torch.sin(t * freqs + phases) + 1) / 2
        
    elif pattern_type == 'sparse':
        # Sparse patterns (few active elements)
        data = torch.zeros(n_samples, input_dim)
        for i in range(n_samples):
            n_active = torch.randint(3, 10, (1,)).item()
            indices = torch.randperm(input_dim)[:n_active]
            data[i, indices] = torch.rand(n_active)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    return data


# Convenience function to set up training
def setup_training(
    lattice: LatticeVisualizer,
    input_dim: int = 64,
    n_samples: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    propagation_steps: int = 5,
    pattern_type: str = 'sinusoidal'
) -> Tuple[LatticeAutoencoder, LatticeTrainer, torch.utils.data.DataLoader]:
    """
    Set up the complete training pipeline.
    
    Args:
        lattice: LatticeVisualizer instance
        input_dim: Dimension of input/output
        n_samples: Number of training samples
        batch_size: Training batch size
        learning_rate: Learning rate
        propagation_steps: Number of lattice propagation steps
        pattern_type: Type of training patterns
        
    Returns:
        autoencoder: LatticeAutoencoder model
        trainer: LatticeTrainer instance
        dataloader: Training DataLoader
    """
    # Create model
    autoencoder = LatticeAutoencoder(
        lattice=lattice,
        input_dim=input_dim,
        propagation_steps=propagation_steps
    )
    
    # Create trainer
    trainer = LatticeTrainer(
        autoencoder=autoencoder,
        learning_rate=learning_rate
    )
    
    # Create training data
    data = create_training_data(n_samples, input_dim, pattern_type)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return autoencoder, trainer, dataloader


def load_mnist(
    train: bool = True,
    n_samples: int = None,
    flatten: bool = True,
    normalize: bool = True,
    data_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST dataset from local IDX files.
    
    Args:
        train: Whether to load training set (True) or test set (False)
        n_samples: Number of samples to load (None = all)
        flatten: Whether to flatten images to 1D (784,) or keep 2D (28, 28)
        normalize: Whether to normalize pixel values to [0, 1]
        data_path: Path to local MNIST data folder (defaults to ./mnist relative to this file)
        
    Returns:
        images: Image tensor
        labels: Label tensor
    """
    import struct
    import os
    
    # Use relative path if not specified
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist')
    
    # Select train or test files
    if train:
        images_file = os.path.join(data_path, 'train-images.idx3-ubyte')
        labels_file = os.path.join(data_path, 'train-labels.idx1-ubyte')
    else:
        images_file = os.path.join(data_path, 't10k-images.idx3-ubyte')
        labels_file = os.path.join(data_path, 't10k-labels.idx1-ubyte')
    
    # Read images
    with open(images_file, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    
    # Read labels
    with open(labels_file, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Convert to torch tensors
    images = torch.from_numpy(images.copy()).float()
    labels = torch.from_numpy(labels.copy()).long()
    
    # Normalize to [0, 1]
    if normalize:
        images = images / 255.0
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(images):
        indices = torch.randperm(len(images))[:n_samples]
        images = images[indices]
        labels = labels[indices]
    
    # Flatten if requested (28x28 -> 784)
    if flatten:
        images = images.view(images.shape[0], -1)
    
    return images, labels


def setup_mnist_training(
    lattice: LatticeVisualizer,
    n_train_samples: int = 5000,
    n_test_samples: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    propagation_steps: int = 5
) -> Tuple[LatticeAutoencoder, LatticeTrainer, torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader, torch.Tensor, torch.Tensor]:
    """
    Set up training pipeline with MNIST data including train and test sets.
    
    MNIST images are 28x28 = 784 dimensions.
    
    Args:
        lattice: LatticeVisualizer instance
        n_train_samples: Number of MNIST training samples to use
        n_test_samples: Number of MNIST test samples for validation
        batch_size: Training batch size
        learning_rate: Learning rate
        propagation_steps: Number of lattice propagation steps
        
    Returns:
        autoencoder: LatticeAutoencoder model
        trainer: LatticeTrainer instance
        train_dataloader: Training DataLoader
        test_dataloader: Test/validation DataLoader
        train_labels: MNIST labels for training samples
        test_labels: MNIST labels for test samples
    """
    # MNIST is 28x28 = 784 dimensions
    input_dim = 784
    
    # Load MNIST train set
    print("Loading MNIST dataset...")
    train_images, train_labels = load_mnist(train=True, n_samples=n_train_samples)
    print(f"Loaded {len(train_images)} MNIST training images")
    
    # Load MNIST test set for validation
    test_images, test_labels = load_mnist(train=False, n_samples=n_test_samples)
    print(f"Loaded {len(test_images)} MNIST test images for validation")
    
    # Create model
    autoencoder = LatticeAutoencoder(
        lattice=lattice,
        input_dim=input_dim,
        propagation_steps=propagation_steps
    )
    
    # Create trainer
    trainer = LatticeTrainer(
        autoencoder=autoencoder,
        learning_rate=learning_rate
    )
    
    # Create train dataloader
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create test dataloader for validation
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return autoencoder, trainer, train_dataloader, test_dataloader, train_labels, test_labels
