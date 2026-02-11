import torch
import numpy as np
from typing import Union, Optional

# Set up device - use MPS if available (Apple Silicon), else CPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()


class Cell:
    """
    A neural network cell with normal distribution parameters for pulse propagation.
    Uses MPS (Apple Silicon GPU) when available.
    
    Each cell has:
    - mean (weight): Mean of normal distribution - center of "straight forward" probability
    - std_dev (bias): Std deviation - how likely to bounce vs go straight
      - Low std_dev = pulse more likely to go straight
      - High std_dev = pulse more likely to bounce in different directions
    - bounce_angles: 3-element tensor for X, Y, Z axis bounce angles
    - All axis outputs are multiplied by -1
    
    The normal distribution determines:
    - Probability of going straight forward (near mean)
    - Probability of bouncing at an angle (further from mean)
    """
    
    def __init__(
        self, 
        mean: Union[float, np.ndarray, torch.Tensor] = None,
        std_dev: Union[float, np.ndarray, torch.Tensor] = None,
        bounce_angles: Union[list, np.ndarray, torch.Tensor] = None,
        split_prob: Union[float, torch.Tensor] = None,
        join_prob: Union[float, torch.Tensor] = None,
        state: int = 0,
        device: torch.device = None,
        requires_grad: bool = True,
        alternate_mode: str = None  # 'closer' or 'further' from axis
    ):
        """
        Initialize a Cell.
        
        Args:
            mean: Mean of normal distribution (μ). Controls center of direction probability.
                  If None, initialized to 0 (straight forward most likely).
            std_dev: Standard deviation (σ). Controls spread of direction probability.
                     Low = go straight, High = more bouncing. If None, initialized to 0.5.
            bounce_angles: 3-element list/array for X, Y, Z bounce angles. If None, initialized randomly.
            split_prob: Probability [0,1] that pulse splits into multiple directions at this cell.
                       Higher = more likely to branch. If None, initialized randomly.
            join_prob: Probability [0,1] that incoming pulses merge/combine at this cell.
                      Higher = more absorption/combination. If None, initialized randomly.
            state: State number identifier for the cell.
            device: Torch device (defaults to MPS if available).
            requires_grad: Whether to track gradients for training.
            alternate_mode: 'closer' or 'further' - determines if angles tend toward or away from axis.
        """
        self.state = state
        self.device = device if device is not None else DEVICE
        self.requires_grad = requires_grad
        self.alternate_mode = alternate_mode
        
        # Initialize mean (μ) - center of distribution
        # mean=0 means straight forward is most likely
        if mean is None:
            self.mean = torch.tensor([0.0], device=self.device, dtype=torch.float32)
        elif isinstance(mean, torch.Tensor):
            self.mean = mean.to(device=self.device, dtype=torch.float32).flatten()
        else:
            self.mean = torch.tensor([mean] if np.isscalar(mean) else mean, 
                                     device=self.device, dtype=torch.float32).flatten()
        
        # Initialize std_dev (σ) - spread of distribution (min 2.0)
        # Low std_dev = more likely to go straight
        # High std_dev = more spread out, more bouncing
        if std_dev is None:
            if alternate_mode == 'closer':
                # Closer mode: lower std_dev, more likely to go straight (2.0 to 2.5)
                self.std_dev = torch.rand(1, device=self.device, dtype=torch.float32) * 0.5 + 2.0
            elif alternate_mode == 'further':
                # Further mode: higher std_dev, more bouncing (2.5 to 3.5)
                self.std_dev = torch.rand(1, device=self.device, dtype=torch.float32) * 1.0 + 2.5
            else:
                self.std_dev = torch.rand(1, device=self.device, dtype=torch.float32) * 1.0 + 2.0
        elif isinstance(std_dev, torch.Tensor):
            self.std_dev = std_dev.to(device=self.device, dtype=torch.float32).flatten()
        else:
            self.std_dev = torch.tensor([std_dev] if np.isscalar(std_dev) else std_dev,
                                        device=self.device, dtype=torch.float32).flatten()
        
        # Ensure std_dev is at least 2.0
        self.std_dev = torch.clamp(torch.abs(self.std_dev), min=2.0)
        
        # Initialize 3-axis bounce angles (X, Y, Z) in RADIANS
        # Maximum bounce angle is 2 radians (~114.6 degrees)
        if bounce_angles is None:
            if alternate_mode == 'closer':
                # Closer to axis: smaller angles (0-0.5 radians)
                self.bounce_angles = torch.rand(3, device=self.device, dtype=torch.float32) * 0.5
            elif alternate_mode == 'further':
                # Further from axis: larger angles (1.0-2.0 radians)
                self.bounce_angles = torch.rand(3, device=self.device, dtype=torch.float32) * 1.0 + 1.0
            else:
                # Random across full range (0-2 radians)
                self.bounce_angles = torch.rand(3, device=self.device, dtype=torch.float32) * 2.0
        elif isinstance(bounce_angles, torch.Tensor):
            self.bounce_angles = bounce_angles.to(device=self.device, dtype=torch.float32)
        else:
            self.bounce_angles = torch.tensor(bounce_angles, device=self.device, dtype=torch.float32)
        
        # Ensure bounce_angles has 3 elements
        if self.bounce_angles.numel() == 1:
            self.bounce_angles = self.bounce_angles.repeat(3)
        elif self.bounce_angles.numel() != 3:
            raise ValueError(f"bounce_angles must have 3 elements, got {self.bounce_angles.numel()}")
        
        # Initialize split probability [0, 1]
        # Higher = more likely to split pulse into multiple directions
        if split_prob is None:
            if alternate_mode == 'closer':
                # Closer mode: lower split probability (more focused)
                self.split_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.2
            elif alternate_mode == 'further':
                # Further mode: higher split probability (more branching)
                self.split_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.3 + 0.2
            else:
                # Random across full range
                self.split_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.4
        elif isinstance(split_prob, torch.Tensor):
            self.split_prob = split_prob.to(device=self.device, dtype=torch.float32).flatten()
        else:
            self.split_prob = torch.tensor([split_prob], device=self.device, dtype=torch.float32)
        
        # Clamp split_prob to [0, 1]
        self.split_prob = torch.clamp(self.split_prob, min=0.0, max=1.0)
        
        # Initialize join probability [0, 1]
        # Higher = more likely to absorb/combine incoming pulses
        if join_prob is None:
            if alternate_mode == 'closer':
                # Closer mode: higher join probability (more combining)
                self.join_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.3 + 0.2
            elif alternate_mode == 'further':
                # Further mode: lower join probability (more pass-through)
                self.join_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.2
            else:
                # Random across full range
                self.join_prob = torch.rand(1, device=self.device, dtype=torch.float32) * 0.4
        elif isinstance(join_prob, torch.Tensor):
            self.join_prob = join_prob.to(device=self.device, dtype=torch.float32).flatten()
        else:
            self.join_prob = torch.tensor([join_prob], device=self.device, dtype=torch.float32)
        
        # Clamp join_prob to [0, 1]
        self.join_prob = torch.clamp(self.join_prob, min=0.0, max=1.0)
        
        # Enable gradients if requested
        if requires_grad:
            self.mean.requires_grad_(True)
            self.std_dev.requires_grad_(True)
            self.bounce_angles.requires_grad_(True)
            self.split_prob.requires_grad_(True)
            self.join_prob.requires_grad_(True)
        
        # Store last samples for analysis
        self._last_sample: Optional[torch.Tensor] = None
        self._last_probability: Optional[float] = None
    
    # ==================== Distribution Sampling ====================
    
    def sample_direction_factor(self) -> torch.Tensor:
        """
        Sample from the normal distribution to determine direction factor.
        
        Returns:
            A sample from N(mean, std_dev). Values near mean = go straight,
            values far from mean = bounce more.
        """
        # Sample from normal distribution
        sample = torch.normal(self.mean, self.std_dev)
        self._last_sample = sample
        return sample
    
    def get_straight_probability(self, threshold: float = 0.5) -> float:
        """
        Get probability of going straight (sample within threshold of mean).
        
        Uses the CDF of normal distribution to calculate P(|X - μ| < threshold).
        
        Args:
            threshold: Distance from mean considered "straight"
            
        Returns:
            Probability in [0, 1]
        """
        # P(μ - t < X < μ + t) = 2 * Φ(t/σ) - 1
        z = threshold / (self.std_dev.item() + 1e-6)
        # Use error function approximation: Φ(z) ≈ 0.5 * (1 + erf(z / sqrt(2)))
        prob = torch.erf(torch.tensor(z / np.sqrt(2))).item()
        self._last_probability = prob
        return prob
    
    def should_go_straight(self, threshold: float = 0.5) -> tuple:
        """
        Sample and determine if pulse should go straight or bounce.
        
        Args:
            threshold: Distance from mean considered "straight"
            
        Returns:
            (go_straight: bool, sample_value: float, bounce_factor: float)
            - go_straight: True if sample is within threshold of mean
            - sample_value: The actual sample from distribution
            - bounce_factor: How much to scale bounce angles (0 if straight, scaled by distance if bouncing)
        """
        sample = self.sample_direction_factor()
        distance_from_mean = torch.abs(sample - self.mean)
        
        go_straight = distance_from_mean.item() < threshold
        
        if go_straight:
            bounce_factor = 0.0
        else:
            # Scale bounce by how far from mean (normalized by std_dev)
            bounce_factor = (distance_from_mean / self.std_dev).item()
            bounce_factor = min(bounce_factor, 2.0)  # Cap at 2x
        
        return go_straight, sample.item(), bounce_factor
    
    def get_bounce_angles(self) -> torch.Tensor:
        """
        Get the 3-axis bounce angles [X, Y, Z] multiplied by -1.
        All axis outputs are negated.
        """
        return self.bounce_angles * -1
    
    def get_bounce_angles_raw(self) -> torch.Tensor:
        """Get the raw 3-axis bounce angles [X, Y, Z] without negation."""
        return self.bounce_angles
    
    def set_bounce_angles(self, angles: Union[list, torch.Tensor]):
        """Set the 3-axis bounce angles [X, Y, Z]."""
        if isinstance(angles, torch.Tensor):
            self.bounce_angles = angles.to(device=self.device, dtype=torch.float32)
        else:
            self.bounce_angles = torch.tensor(angles, device=self.device, dtype=torch.float32)
    
    # ==================== Split/Join Methods ====================
    
    def should_split(self) -> tuple:
        """
        Determine if pulse should split into multiple directions at this cell.
        
        Returns:
            (should_split: bool, n_splits: int, split_weights: list)
            - should_split: True if split occurs
            - n_splits: Number of directions to split into (2-4)
            - split_weights: Relative power distribution for each split
        """
        # Sample random value and compare to split probability
        sample = torch.rand(1, device=self.device).item()
        should_split = sample < self.split_prob.item()
        
        if should_split:
            # Number of splits based on how far below threshold (2-4 splits)
            split_intensity = (self.split_prob.item() - sample) / max(self.split_prob.item(), 0.01)
            n_splits = 2 + int(split_intensity * 2)  # 2-4 splits
            n_splits = min(n_splits, 4)
            
            # Generate weights (sum to 1) with some randomness
            weights = torch.softmax(torch.randn(n_splits, device=self.device), dim=0)
            split_weights = weights.tolist()
        else:
            n_splits = 1
            split_weights = [1.0]
        
        return should_split, n_splits, split_weights
    
    def should_join(self, n_incoming: int = 1) -> tuple:
        """
        Determine if incoming pulses should join/combine at this cell.
        
        Args:
            n_incoming: Number of incoming pulse activations
            
        Returns:
            (should_join: bool, join_factor: float)
            - should_join: True if pulses should combine
            - join_factor: How much to boost combined signal (1.0-2.0)
        """
        if n_incoming <= 1:
            return False, 1.0
        
        # More incoming pulses = higher chance to join
        effective_prob = self.join_prob.item() * min(n_incoming / 2, 1.5)
        effective_prob = min(effective_prob, 0.9)  # Cap at 90%
        
        sample = torch.rand(1, device=self.device).item()
        should_join = sample < effective_prob
        
        if should_join:
            # Join factor: combined pulses get a boost (constructive interference)
            join_factor = 1.0 + 0.5 * self.join_prob.item() * (n_incoming - 1) / n_incoming
            join_factor = min(join_factor, 2.0)  # Cap at 2x
        else:
            join_factor = 1.0
        
        return should_join, join_factor
    
    def get_split_prob(self) -> float:
        """Get the split probability."""
        return self.split_prob.item()
    
    def get_join_prob(self) -> float:
        """Get the join probability."""
        return self.join_prob.item()
    
    def set_split_prob(self, prob: float):
        """Set the split probability [0, 1]."""
        self.split_prob = torch.tensor([max(0.0, min(1.0, prob))], 
                                        device=self.device, dtype=torch.float32)
    
    def set_join_prob(self, prob: float):
        """Set the join probability [0, 1]."""
        self.join_prob = torch.tensor([max(0.0, min(1.0, prob))], 
                                       device=self.device, dtype=torch.float32)
    
    # ==================== Properties for backward compatibility ====================
    
    @property
    def weight(self) -> torch.Tensor:
        """Alias for mean (backward compatibility)."""
        return self.mean
    
    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self.mean = value.to(device=self.device, dtype=torch.float32).flatten()
        else:
            self.mean = torch.tensor([value] if np.isscalar(value) else value,
                                     device=self.device, dtype=torch.float32).flatten()
    
    @property
    def bias(self) -> torch.Tensor:
        """Alias for std_dev (backward compatibility)."""
        return self.std_dev
    
    @bias.setter
    def bias(self, value):
        if isinstance(value, torch.Tensor):
            self.std_dev = torch.clamp(torch.abs(value.to(device=self.device, dtype=torch.float32).flatten()), min=2.0)
        else:
            self.std_dev = torch.clamp(torch.abs(torch.tensor([value] if np.isscalar(value) else value,
                                     device=self.device, dtype=torch.float32).flatten()), min=2.0)
    
    # ==================== Arithmetic Operations ====================
    
    def _to_tensor(self, other):
        """Convert other to tensor on same device."""
        if isinstance(other, Cell):
            return other
        if isinstance(other, torch.Tensor):
            return other.to(device=self.device)
        return torch.tensor(other, device=self.device, dtype=torch.float32)
    
    def __add__(self, other: Union['Cell', float, torch.Tensor]) -> 'Cell':
        """Add: cell + other"""
        result = Cell(
            mean=self.mean.clone().detach(),
            std_dev=self.std_dev.clone().detach(),
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
        if isinstance(other, Cell):
            result.mean = self.mean + other.mean
            result.std_dev = self.std_dev + other.std_dev
            result.split_prob = torch.clamp((self.split_prob + other.split_prob) / 2, 0, 1)
            result.join_prob = torch.clamp((self.join_prob + other.join_prob) / 2, 0, 1)
        else:
            other_t = self._to_tensor(other)
            result.mean = self.mean + other_t
            result.std_dev = self.std_dev + other_t
        return result
    
    def __radd__(self, other) -> 'Cell':
        """Reverse add: other + cell"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Cell', float, torch.Tensor]) -> 'Cell':
        """Subtract: cell - other"""
        result = Cell(
            mean=self.mean.clone().detach(),
            std_dev=self.std_dev.clone().detach(),
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
        if isinstance(other, Cell):
            result.mean = self.mean - other.mean
            result.std_dev = torch.clamp(torch.abs(self.std_dev - other.std_dev), min=2.0)
            result.split_prob = torch.clamp(torch.abs(self.split_prob - other.split_prob), 0, 1)
            result.join_prob = torch.clamp(torch.abs(self.join_prob - other.join_prob), 0, 1)
        else:
            other_t = self._to_tensor(other)
            result.mean = self.mean - other_t
            result.std_dev = torch.clamp(torch.abs(self.std_dev - other_t), min=2.0)
        return result
    
    def __mul__(self, other: Union['Cell', float, torch.Tensor]) -> 'Cell':
        """Multiply: cell * other"""
        result = Cell(
            mean=self.mean.clone().detach(),
            std_dev=self.std_dev.clone().detach(),
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
        if isinstance(other, Cell):
            result.mean = self.mean * other.mean
            result.std_dev = self.std_dev * other.std_dev
            result.split_prob = torch.clamp(self.split_prob * other.split_prob, 0, 1)
            result.join_prob = torch.clamp(self.join_prob * other.join_prob, 0, 1)
        else:
            other_t = self._to_tensor(other)
            result.mean = self.mean * other_t
            result.std_dev = torch.clamp(torch.abs(self.std_dev * other_t), min=2.0)
        return result
    
    def __rmul__(self, other) -> 'Cell':
        """Reverse multiply: other * cell"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Cell', float, torch.Tensor]) -> 'Cell':
        """Divide: cell / other"""
        result = Cell(
            mean=self.mean.clone().detach(),
            std_dev=self.std_dev.clone().detach(),
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
        if isinstance(other, Cell):
            result.mean = self.mean / (other.mean + 1e-6)
            result.std_dev = torch.clamp(self.std_dev / (other.std_dev + 1e-6), min=2.0)
            result.split_prob = torch.clamp(self.split_prob / (other.split_prob + 1e-6), 0, 1)
            result.join_prob = torch.clamp(self.join_prob / (other.join_prob + 1e-6), 0, 1)
        else:
            other_t = self._to_tensor(other) + 1e-6
            result.mean = self.mean / other_t
            result.std_dev = torch.clamp(torch.abs(self.std_dev / other_t), min=2.0)
        return result
    
    def __neg__(self) -> 'Cell':
        """Negate: -cell"""
        return Cell(
            mean=-self.mean,
            std_dev=self.std_dev.clone().detach(),  # std_dev stays positive
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),  # split/join stay positive
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
    
    # ==================== Utility Methods ====================
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.mean.grad is not None:
            self.mean.grad.zero_()
        if self.std_dev.grad is not None:
            self.std_dev.grad.zero_()
        if self.bounce_angles.grad is not None:
            self.bounce_angles.grad.zero_()
        if self.split_prob.grad is not None:
            self.split_prob.grad.zero_()
        if self.join_prob.grad is not None:
            self.join_prob.grad.zero_()
    
    def parameters(self) -> list:
        """Return list of parameters (mean, std_dev, bounce_angles, split_prob, join_prob)."""
        return [self.mean, self.std_dev, self.bounce_angles, self.split_prob, self.join_prob]
    
    def clone(self) -> 'Cell':
        """Create a copy of this cell."""
        return Cell(
            mean=self.mean.clone().detach(),
            std_dev=self.std_dev.clone().detach(),
            bounce_angles=self.bounce_angles.clone().detach(),
            split_prob=self.split_prob.clone().detach(),
            join_prob=self.join_prob.clone().detach(),
            state=self.state,
            device=self.device,
            requires_grad=self.requires_grad,
            alternate_mode=self.alternate_mode
        )
    
    def to(self, device: Union[str, torch.device]) -> 'Cell':
        """Move cell to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.mean = self.mean.to(device)
        self.std_dev = self.std_dev.to(device)
        self.bounce_angles = self.bounce_angles.to(device)
        self.split_prob = self.split_prob.to(device)
        self.join_prob = self.join_prob.to(device)
        return self
    
    def cpu(self) -> 'Cell':
        """Move cell to CPU."""
        return self.to(torch.device("cpu"))
    
    def mps(self) -> 'Cell':
        """Move cell to MPS (Apple Silicon GPU)."""
        return self.to(torch.device("mps"))
    
    def numpy(self) -> tuple:
        """Return mean, std_dev, bounce_angles, split_prob, join_prob as numpy arrays."""
        return (
            self.mean.detach().cpu().numpy(),
            self.std_dev.detach().cpu().numpy(),
            self.bounce_angles.detach().cpu().numpy(),
            self.split_prob.detach().cpu().numpy(),
            self.join_prob.detach().cpu().numpy()
        )
    
    def __repr__(self) -> str:
        angles = self.get_bounce_angles().detach().cpu().numpy()
        prob = self.get_straight_probability()
        return (f"Cell(state={self.state}, μ={self.mean.item():.3f}, σ={self.std_dev.item():.3f}, "
                f"P(straight)={prob:.1%}, split={self.split_prob.item():.2f}, "
                f"join={self.join_prob.item():.2f}, mode={self.alternate_mode})")
    
    def __str__(self) -> str:
        angles = self.get_bounce_angles().detach().cpu().numpy()
        raw_angles = self.bounce_angles.detach().cpu().numpy()
        prob = self.get_straight_probability()
        return (f"Cell[state={self.state}, mode={self.alternate_mode}, device={self.device}]\n"
                f"  mean (μ): {self.mean.item():.4f}\n"
                f"  std_dev (σ): {self.std_dev.item():.4f}\n"
                f"  P(straight): {prob:.1%}\n"
                f"  split_prob: {self.split_prob.item():.4f}\n"
                f"  join_prob: {self.join_prob.item():.4f}\n"
                f"  bounce_angles (raw): [X={raw_angles[0]:.3f} rad, Y={raw_angles[1]:.3f} rad, Z={raw_angles[2]:.3f} rad]\n"
                f"  bounce_angles (*-1): [X={angles[0]:.3f} rad, Y={angles[1]:.3f} rad, Z={angles[2]:.3f} rad]")
