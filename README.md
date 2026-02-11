# Cross-Dimensional Architecture: Pulse Propagation Pattern Search

A neural architecture that searches for optimal **3D lattice patterns** combined with **pulse propagation types** to create powerful signal processing networks.

## Abstract

This project explores how different **3D geometric patterns** interact with various **pulse propagation behaviors** in lattice-based neural networks. The core insight: by searching over combinations of pattern initializations (sphere, shell, gradients, waves, etc.) and propagation types (sharp, diffuse, wave, resonant, etc.), we can discover configurations that dramatically outperform uniform or random initialization.

**Key Result**: Grid search over 105 pattern × propagation combinations found that **shell pattern + wave propagation** achieves optimal reconstruction (MSE: 0.0047), significantly outperforming uniform baselines.

---

## Pulse Propagation Pattern Search (Primary Workflow)

The main workflow searches for optimal pattern × propagation type combinations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PULSE PROPAGATION PATTERN SEARCH                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐      ┌─────────────────┐                              │
│   │  15 PATTERNS    │  ×   │  7 PROPAGATION  │  =  105 Combinations         │
│   │                 │      │     TYPES       │                              │
│   │  • sphere       │      │  • sharp        │                              │
│   │  • shell        │      │  • diffuse      │                              │
│   │  • cube         │      │  • wave         │                              │
│   │  • gradients    │      │  • resonant     │                              │
│   │  • waves        │      │  • persistent   │                              │
│   │  • gyroid       │      │  • explosive    │                              │
│   │  • noise        │      │  • static       │                              │
│   └────────┬────────┘      └────────┬────────┘                              │
│            │                        │                                        │
│            └──────────┬─────────────┘                                        │
│                       ↓                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  GRID SEARCH EVALUATION                                              │   │
│   │    For each (pattern, propagation) pair:                             │   │
│   │    1. Initialize lattice with pattern weights                        │   │
│   │    2. Run propagation with type parameters                           │   │
│   │    3. Train autoencoder for N epochs                                 │   │
│   │    4. Record MSE, variance, sparsity                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  RESULTS ANALYSIS                                                    │   │
│   │    • Heatmap of MSE by pattern × propagation                         │   │
│   │    • Best pattern per propagation type                               │   │
│   │    • Best propagation per pattern                                    │   │
│   │    • Propagation trajectory visualization                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                       ↓                                                      │
│              OPTIMAL CONFIGURATION                                           │
│           (shell × wave = MSE 0.0047)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Propagation Types

| Type | Steps | Decay | Diffusion | Self-Retention | Use Case |
|------|-------|-------|-----------|----------------|----------|
| **sharp** | 3 | 0.95 | 0.2 | 0.8 | Fast, localized features |
| **diffuse** | 5 | 0.85 | 0.5 | 0.5 | Balanced spread |
| **wave** | 8 | 0.9 | 0.6 | 0.4 | Long-range dependencies |
| **resonant** | 6 | 0.92 | 0.4 | 0.6 | Oscillating patterns |
| **persistent** | 10 | 0.98 | 0.3 | 0.7 | Signal preservation |
| **explosive** | 4 | 0.7 | 0.8 | 0.2 | Rapid global spread |
| **static** | 0 | 1.0 | 0.0 | 1.0 | No propagation (baseline) |

### Pattern Categories

| Category | Patterns | Best For |
|----------|----------|----------|
| **Geometric** | sphere, shell, cube | Localized activation, boundaries |
| **Gradients** | z-axis, radial, inverted radial | Directional flow |
| **Waves** | z-wave, xyz-wave, spherical, spiral | Periodic structures |
| **Structured** | grid, checkerboard | Channeled propagation |
| **Mathematical** | gyroid | Complex surfaces |
| **Stochastic** | noise | Regularization |
| **Baseline** | uniform | Control comparison |

### Quick Start: Run Pattern Search

```bash
# Run the pulse propagation pattern search
jupyter notebook search_pulse_patterns.ipynb
```

**Outputs:**
- `pulse_search_results.json` — Full 105-combination results
- `best_pulse_pattern.npy` — Optimal pattern array
- `best_pulse_model.pt` — Trained model with best config

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT EMBEDDING                                    │
│                                 ↓                                            │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  PropagationSignalEncoder                                            │  │
│    │    Input → Hidden Layers → Softmax Temperature → PROPAGATION SIGNAL  │  │
│    │    Encodes the "address" or "query", not the content itself          │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                            │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  3D Hexagonal Lattice Propagation                                    │  │
│    │    Signal propagates through cells based on learned parameters:      │  │
│    │    • mean (μ): probability of straight propagation                   │  │
│    │    • std_dev (σ): variance in propagation direction                  │  │
│    │    • bounce_angles: deflection angles on X, Y, Z axes                │  │
│    │    • split_prob: probability of pulse splitting                      │  │
│    │    • join_prob: probability of pulse merging                         │  │
│    │    Creates unique ACTIVATION PATTERN per input                       │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                            │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  SparseEmbeddingMemory                                               │  │
│    │    Activation pattern queries stored embeddings via attention:       │  │
│    │    • cell_embeddings: direct embedding storage per cell              │  │
│    │    • memory_keys: attention keys for slot selection                  │  │
│    │    • memory_values: content stored in memory slots                   │  │
│    │    Returns RETRIEVED embedding from distributed memory               │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                            │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  PropagationRetriever                                                │  │
│    │    Final transformation: Propagated State → Output Embedding         │  │
│    │    Combines memory read with learned projection                      │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                 ↓                                            │
│                          RETRIEVED EMBEDDING                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Cell: The Fundamental Unit

Each cell in the lattice is a learnable unit with the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean` (μ) | Tensor | Center of normal distribution; higher values = more likely to propagate straight |
| `std_dev` (σ) | Tensor | Spread of distribution; determines bounce probability (constrained ≥ 2.0) |
| `bounce_angles` | Tensor[3] | Deflection angles for X, Y, Z axes in radians (0 to 2π) |
| `split_prob` | Tensor | Probability that a pulse splits into multiple paths |
| `join_prob` | Tensor | Probability that converging pulses merge and amplify |

**Propagation Decision:**
```
sample ~ N(μ, σ²)
if sample > threshold:
    propagate straight
else:
    apply bounce_angles rotation to direction vector
```

### 2. Hexagonal Lattice Structure

The 3D lattice uses hexagonal packing for optimal neighbor connectivity:

- **Layers**: Stacked hexagonal grids
- **Hex Radius**: Controls density within each layer
- **Cell Spacing**: 0.9 units between adjacent cells (configurable)
- **Adjacency**: Each cell connects to ~6 horizontal neighbors + ~2 vertical neighbors

**Coordinate System:**
```python
# Hexagonal grid positioning
x = col * hex_spacing + (row % 2) * hex_spacing / 2
y = row * hex_spacing * sqrt(3) / 2
z = layer * layer_spacing
```

### 3. Sparse Lattice (Memory Efficiency)

The `SparseLatticeVisualizer` implements **true on-demand cell generation**:

- **Blueprint Only**: Stores positions and adjacency matrix for all potential cells
- **Zero Pre-Initialization**: No cells exist until propagation touches them
- **Propagation-Driven Creation**: Cells created as the signal reaches them
- **Persistent Storage**: Modified cells saved to disk, loaded on demand
- **Dynamic Growth**: Network topology emerges from the data itself

**Memory Efficiency:**
```
Traditional:     O(total_cells × cell_params)
Pre-initialized: O(active_cells × cell_params)  
On-Demand:       O(touched_cells × cell_params) where touched << active << total
```

**Key Insight**: The network structure is not predefined—it grows organically based on which cells the propagation signal reaches. This means:
1. Different inputs may activate completely different cell subsets
2. The effective network capacity scales with data complexity
3. Unused regions of the lattice never consume memory

### 4. Toroidal Lattice (Alternative Topology)

The `ToroidalLatticeVisualizer` provides an alternative **torus-shaped** arrangement:

```
         ___________
       /             \
      |    Major R    |  ← Cells wrap around here (φ direction)
      |   ┌─────┐     |
      |   │  r  │     |  ← Cells wrap around here (θ direction)
      |   └─────┘     |
       \             /
         ‾‾‾‾‾‾‾‾‾‾‾
```

**Key Properties:**
- **No edge effects**: Wrap-around connectivity in both directions
- **Uniform neighbors**: Every cell has exactly 8 neighbors
- **Cyclic topology**: Natural for periodic/cyclic data representations
- **On-demand creation**: Same sparse computation as hexagonal lattice

**Torus Parameters:**
- `n_theta`: Cells around the tube (minor circle)
- `n_phi`: Cells around the torus (major circle)
- `major_radius`: Distance from center to tube center (R)
- `minor_radius`: Radius of the tube (r)

**When to use each:**

| Feature | Hexagonal Lattice | Toroidal Lattice |
|---------|-------------------|------------------|
| Edge effects | Yes (boundary cells) | No (wrap-around) |
| Neighbors | Variable (4-8) | Fixed (8) |
| Best for | Spatial data, images | Cyclic data, embeddings |
| Propagation | Can end at edges | Loops continuously |

### 5. Tensor3D Lattice (Generalized 3D)

The `Tensor3DLattice` provides a **flexible 3D tensor-shaped** lattice that accepts arbitrary shapes:

```python
from tensor_lattice import Tensor3DLattice, DifferentiableTensor3DLattice

# Create a 24×24×24 bounded lattice with 6-connectivity
lattice = Tensor3DLattice(
    shape=(24, 24, 24),
    topology='bounded',    # 'bounded', 'toroidal', 'cylindrical'
    connectivity=6         # 6, 18, or 26 neighbors
)
```

**Topology Options:**
| Topology | Description |
|----------|-------------|
| `bounded` | Hard edges, no wrap-around |
| `toroidal` | Wraps in all 3 dimensions |
| `cylindrical` | Wraps in Z-axis only |

**Connectivity Options:**
| Connectivity | Neighbors | Pattern |
|--------------|-----------|---------|
| 6 | Face neighbors only | ±x, ±y, ±z |
| 18 | + Edge neighbors | + diagonals on faces |
| 26 | + Corner neighbors | Full 3D diagonal connectivity |

### 6. Pattern Generation

The `PatternGenerator` class creates 3D patterns for lattice initialization:

```python
from lattice_patterns import PatternGenerator, Pattern3D

pg = PatternGenerator(shape=(24, 24, 24))

# Geometric primitives
sphere = pg.sphere(center=(12, 12, 12), radius=8)
shell = pg.shell(center=(12, 12, 12), inner_radius=6, outer_radius=10)

# Gradients
gradient = pg.gradient(axis='z', start=0.0, end=1.0)
radial = pg.radial_gradient(center=(12, 12, 12), max_radius=12)

# Waves and noise
wave = pg.sine_wave(frequency=(2, 0, 0), phase=0)
noise = pg.noise(scale=0.5, octaves=3)

# Composition
combined = pg.blend(sphere, gradient, blend_factor=0.5)
```

**Pattern Categories:**

| Category | Patterns |
|----------|----------|
| Geometric | `sphere`, `shell`, `cube`, `cylinder`, `cone`, `torus` |
| Gradients | `gradient`, `radial_gradient` |
| Waves | `sine_wave`, `spherical_wave`, `spiral` |
| Noise | `noise`, `perlin_noise` |
| Structured | `checkerboard`, `grid_lines`, `dots` |
| Mathematical | `gyroid`, `voronoi` |

**Composition Operations:**
- `compose(patterns, operation)` — Combine with 'add', 'multiply', 'max', 'min'
- `blend(p1, p2, factor)` — Linear interpolation
- `mask(pattern, mask_pattern)` — Apply mask to pattern

### 7. Differentiable Propagation

The `SparseDifferentiableLattice` enables gradient-based learning:

```python
# Propagation weights based on cell parameters
straight_prob = sigmoid(means / (std_devs + ε))
bounce_prob = 1 - straight_prob

# Self-retention vs spreading
self_retention = straight_prob * 0.3 * (1 - split_probs * 0.5)
neighbor_spread = bounce_prob * adjacency_weights

# Apply bounce transformation
propagation_matrix = self_retention + neighbor_spread * bounce_transform
```

### 8. Embedding Memory

The `SparseEmbeddingMemory` stores embeddings in the lattice structure:

**Storage:**
- `cell_embeddings`: Shape `(n_cells, embedding_dim)` - Direct embedding per cell
- `memory_keys`: Shape `(n_cells, memory_slots)` - Attention keys
- `memory_values`: Shape `(n_cells, memory_slots, embedding_dim)` - Slot contents

**Read Operation:**
```python
# Attention over memory slots
attention = softmax(propagated_state @ memory_keys)

# Weighted retrieval
slot_contribution = attention @ memory_values
direct_contribution = propagated_state @ cell_embeddings

retrieved = direct_contribution + slot_contribution
```

---

## Training

### Loss Functions

1. **Retrieval Loss**: MSE + Cosine similarity between input and retrieved embedding
2. **Write Loss**: Encourages proper storage distribution across cells

```python
total_loss = retrieval_loss + λ * write_loss
```

### Regularization

- **Dropout**: Applied in encoder/decoder networks
- **Weight Decay**: L2 regularization on all parameters
- **Gradient Clipping**: Prevents exploding gradients in deep propagation
- **Early Stopping**: Halts when validation loss plateaus

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `n_active_cells` | 150-300 | Number of cells used in computation |
| `max_steps` | 8-15 | Propagation iterations |
| `memory_slots` | 32-64 | Storage slots per cell |
| `hidden_dim` | 256-512 | MLP hidden layer size |
| `learning_rate` | 1e-4 to 5e-4 | Adam/AdamW learning rate |
| `weight_decay` | 1e-3 to 1e-2 | L2 regularization strength |
| `dropout` | 0.2-0.3 | Dropout probability |

---

## Theoretical Foundations

### Relation to Associative Memory

This architecture shares principles with:

1. **Hopfield Networks**: Content-addressable memory via energy minimization
2. **Neural Turing Machines**: Differentiable read/write with attention
3. **Memory Networks**: Key-value storage with soft addressing

**Key Difference**: Memory addressing is performed via *physical propagation* through a structured network, not just attention weights. The lattice topology constrains and shapes the retrieval process.

### Relation to Graph Neural Networks

The hexagonal lattice can be viewed as a GNN with:
- Fixed, structured topology (vs. arbitrary graphs)
- Probabilistic message passing (vs. deterministic aggregation)
- Learnable edge weights encoded in cell parameters

### Information Flow Properties

1. **Locality**: Nearby inputs in embedding space → similar initial signals → overlapping propagation patterns → similar retrieval
2. **Sparsity**: Only touched cells participate → efficient computation
3. **Capacity**: Total storage = `n_cells × memory_slots × embedding_dim`
4. **Interference**: Propagation patterns can overlap, enabling generalization but risking confusion

---

## Implementation Files

| File | Description |
|------|-------------|
| `cell.py` | `Cell` class with learnable parameters |
| `lattice_visualizer.py` | `LatticeVisualizer` with propagation and 3D visualization |
| `sparse_lattice.py` | `SparseLatticeVisualizer` and `SparseDifferentiableLattice` |
| `toroidal_lattice.py` | `ToroidalLatticeVisualizer` - torus-shaped lattice with wrap-around |
| `tensor_lattice.py` | `Tensor3DLattice` and `DifferentiableTensor3DLattice` for arbitrary 3D shapes |
| `lattice_patterns.py` | `PatternGenerator` and `Pattern3D` for 3D pattern creation |
| `trainer.py` | `DifferentiableLattice`, `LatticeAutoencoder`, `LatticeEvaluator` |
| `kernels/` | Custom CUDA and Metal kernels for GPU acceleration |

---

## Notebooks Guide

### 1. `search_pulse_patterns.ipynb` — Pulse Propagation Pattern Search (Primary)

**Purpose**: Performs a **grid search** over pattern × propagation type combinations to find optimal configurations for signal flow through the lattice.

**Architecture**:
```
Pattern (15 types) × Propagation Type (7 types)
       ↓
  Grid Search (105 combinations)
       ↓
  PropagationAutoencoder
    • Encode → Lattice activation
    • Pattern-modulated propagation
    • Skip connection (initial + final state)
    • Decode → Reconstruction
       ↓
  Evaluate by MSE + State metrics
       ↓
  Best Configuration Selection (shell × wave = MSE 0.0047)
```

**Propagation Types**:

| Type | Steps | Decay | Diffusion | Self-Retention | Description |
|------|-------|-------|-----------|----------------|-------------|
| `sharp` | 3 | 0.95 | 0.2 | 0.8 | Fast, minimal spreading |
| `diffuse` | 5 | 0.85 | 0.5 | 0.5 | Medium spread, moderate decay |
| `wave` | 8 | 0.9 | 0.6 | 0.4 | Long-range propagation |
| `resonant` | 6 | 0.92 | 0.4 | 0.6 | Oscillating with high bounce |
| `persistent` | 10 | 0.98 | 0.3 | 0.7 | Slow decay, signal preserving |
| `explosive` | 4 | 0.7 | 0.8 | 0.2 | Rapid spread, quick decay |
| `static` | 0 | 1.0 | 0.0 | 1.0 | No propagation (baseline) |

**Patterns for Propagation**:
- Geometric: sphere, shell, cube
- Gradients: z-axis, radial, inverted radial
- Waves: z-wave, xyz-wave, spherical, spiral
- Structured: grid, checkerboard
- Special: gyroid, noise, uniform

**Key Features**:
- Full grid search evaluates all 105 combinations
- Heatmap visualization of MSE by pattern × propagation
- Best pattern per propagation type analysis
- Best propagation per pattern analysis
- Propagation history visualization through lattice

**Outputs**:
- `pulse_search_results.json` — Full grid search results
- `best_pulse_pattern.npy` — Optimal pattern for propagation
- `best_pulse_model.pt` — Trained model with best configuration

**What You Learn**:
- Which patterns work best with which propagation types
- Trade-offs between propagation speed and reconstruction quality
- How diffusion rate affects pattern effectiveness

---

### 2. `search_optimal_patterns.ipynb` — Pattern Composition Search

**Purpose**: Implements a **beam search algorithm** to find optimal 3D patterns for initializing lattice weights that connect to a decoder network.

**Architecture**:
```
Pattern Primitives (21 types)
       ↓
  Beam Search with Composition
    • add, multiply, max, blend operations
    • Prune by reconstruction MSE
       ↓
  Top-K Patterns at each depth
       ↓
  Multi-Layer Lattice Model
    • Top-5 patterns as separate lattice layers
    • Learnable layer importance weights
       ↓
  3D Visualization (Plotly isosurfaces)
```

**Pattern Primitives**:
| Category | Patterns |
|----------|----------|
| Geometric | sphere, shell, cube |
| Gradients | linear (x/y/z), radial, inverted radial |
| Waves | sine, spherical, spiral |
| Structured | checkerboard, grid, dots |
| Special | gyroid, noise (fine/coarse) |
| Baselines | ones, zeros |

**Search Configuration**:
- **Beam Width**: 5 (top candidates expanded per level)
- **Max Depth**: 3 (composition depth)
- **Evaluation**: Quick training (10 epochs) per candidate

**Key Features**:
- Composition operations create complex patterns from primitives
- Multi-layer model uses top-5 discovered patterns
- Diversity regularization encourages different layer activations
- Interactive 3D visualization of patterns and activations

**Outputs**:
- `pattern_search_results.json` — Full search results and top configurations
- `best_pattern.npy` — Best discovered pattern array
- `best_pattern_model.pt` — Single-pattern model weights
- `multi_layer_model.pt` — Model using top-5 patterns as layers
- `top5_patterns.npz` — All top-5 pattern arrays

**What You Learn**:
- Which pattern shapes work best for reconstruction tasks
- How pattern compositions improve over primitives
- Layer importance weights reveal pattern complementarity

---

### 3. `classify_mnist.ipynb` — MNIST Classification

**Purpose**: Demonstrates the lattice architecture's ability to perform **classification** (not just reconstruction).

**Architecture**:
```
MNIST Image (28×28) → Flatten → LatticeEncoder → Propagation → Classification Head → 10 classes
```

**Key Features**:
- Uses standard hexagonal lattice with 2 layers, hex_radius=14 (~1,262 cells)
- Classification head converts propagation state to class logits
- Compares propagation-based vs. traditional neural network classification
- Visualizes how different digits create different activation patterns

**What You Learn**:
- How propagation patterns encode class-discriminative features
- Relationship between lattice structure and classification accuracy
- Effect of propagation steps on decision boundaries

---

### 4. `train_sparse_embeddings.ipynb` — Sparse Hexagonal Embedding Memory

**Purpose**: Trains an **on-demand sparse lattice** as associative memory for word embeddings (Dolma corpus).

**Architecture**:
```
Word Embedding (300D)
       ↓
  OnDemandSignalEncoder → N entry points (N = embedding_dim)
       ↓
  OnDemandDifferentiableLattice
    • Cells created ONLY when propagation touches them
    • No pre-initialization — network grows with data
       ↓
  DynamicEmbeddingMemory
    • Memory allocated on-demand per cell
    • Attention-based read from cell_embeddings + memory_slots
       ↓
  PropagationRetriever → Retrieved Embedding
```

**Key Features**:
- **True on-demand**: Zero cells exist initially; they're created during propagation
- **1:1 entry mapping**: Input dimension = number of entry points (input values = entry strengths)
- **Dynamic memory**: Cell embeddings only allocated when needed
- **Regularization**: Dropout, weight decay, early stopping, LR scheduling

**Data**: 50,000 Dolma word embeddings (300-dimensional, normalized)

**Training Details**:
- Loss: MSE + Cosine similarity
- Epochs: 50-100 with early stopping (patience=15)
- Learning rate: 5e-4 with ReduceLROnPlateau

**What You Learn**:
- How sparse computation scales to high-dimensional embeddings
- On-demand cell creation patterns
- Memory efficiency: only ~5-15% of cells typically touched

---

### 5. `test_toroidal_lattice.ipynb` — Toroidal Lattice Testing

**Purpose**: Validates the **toroidal (donut-shaped) lattice** implementation and its unique properties.

**Tests Included**:

1. **Blueprint Verification**
   - Correct cell positioning on torus surface
   - Parametric equations: `x = (R + r·cos(θ))·cos(φ)`
   
2. **Wrap-Around Connectivity**
   - Edge cells connect to opposite edge (no boundaries)
   - Every cell has exactly 8 neighbors
   - Visualize neighbor connections for edge/corner cases

3. **On-Demand Cell Creation**
   - Activate patterns (ring, spiral) and verify cells created
   - Check memory before/after activation
   
4. **Differentiable Propagation**
   - Test `ToroidalDifferentiableLattice` gradient flow
   - Verify `entry_probs.grad` is populated
   - Confirm propagation matrix is valid

5. **Persistence**
   - Save modified cells to disk
   - Load and verify cell parameters match

**Visualizations**:
- Empty torus blueprint (all positions)
- Activated patterns on torus surface
- Neighbor connectivity maps

**What You Learn**:
- Torus topology properties (uniform neighbors, wrap-around)
- Debugging on-demand architectures
- Gradient flow through sparse structures

---

### 6. `train_toroidal_embeddings.ipynb` — Bidirectional Toroidal Training

**Purpose**: Trains a toroidal lattice with **simultaneous forward and reverse propagation** on Dolma embeddings.

**Architecture**:
```
Word Embedding (300D)
       ↓
  ToroidalSignalEncoder → Entry points on torus surface
       ↓
  ┌─────────────────────────────────────────────────────┐
  │          BIDIRECTIONAL PROPAGATION                   │
  │                                                      │
  │   Forward Wave (+θ, +φ)  ────→  clockwise on torus   │
  │   Reverse Wave (-θ, -φ)  ←────  counter-clockwise    │
  │                                                      │
  │   Where waves meet: interaction = fwd × rev          │
  │   Combined = forward + reverse + α·interaction       │
  └─────────────────────────────────────────────────────┘
       ↓
  DynamicToroidalMemory → Attention read from cells
       ↓
  ToroidalRetriever → Retrieved Embedding
```

**Key Innovation — Bidirectional Propagation**:

| Aspect | Forward Only | Bidirectional |
|--------|--------------|---------------|
| Coverage | One direction | Full torus exploration |
| Interaction | None | Reinforcement where waves meet |
| Activation | Linear spread | Complex interference patterns |
| Redundancy | Single path | Multiple paths to each cell |

**Learnable Parameters per Direction**:
- `step_weights`: Weight each propagation step's contribution
- `decay`: How quickly signal attenuates
- `interaction_weight`: How much forward×reverse boosts combined state

**Torus Configuration**: 
- 20×40 cells (800 total)
- Major radius R=4.0, minor radius r=1.5
- 8-connected with wrap-around

**What You Learn**:
- How bidirectional propagation creates richer patterns
- Interaction zones where signals reinforce
- Torus topology advantages for embedding spaces

---

### 7. `test.ipynb` — Main Experimentation Notebook

**Purpose**: General experimentation sandbox with various lattice configurations and visualizations.

**Contents**: 
- Lattice creation and parameter exploration
- Propagation visualization
- Various training experiments
- Interactive 3D plotting with Plotly

---

## Notebook Comparison

| Notebook | Lattice Type | Propagation | Focus | Data |
|----------|--------------|-------------|-------|------|
| **`search_pulse_patterns`** | Tensor3D | **7 propagation types** | Pattern × propagation grid search | MNIST |
| **`search_optimal_patterns`** | Tensor3D | Pattern-modulated | Beam search for pattern composition | Synthetic |
| `classify_mnist` | Hexagonal | Unidirectional | Classification | MNIST |
| `train_sparse_embeddings` | Sparse Hexagonal | Unidirectional | Embedding memory | Dolma 300D |
| `test_toroidal_lattice` | Toroidal | Both tested | Topology validation | Synthetic |
| `train_toroidal_embeddings` | Toroidal | **Bidirectional** | Bidirectional propagation | Dolma 300D |

---

## Quick Start

**Primary workflow — Pulse propagation pattern search**:
```bash
# Search for optimal pattern × propagation combinations (recommended start)
jupyter notebook search_pulse_patterns.ipynb
```

**Pattern composition search** (beam search for pattern combinations):
```bash
jupyter notebook search_optimal_patterns.ipynb
```

**For classification**: 
```bash
jupyter notebook classify_mnist.ipynb
```

**For embedding memory** (requires Dolma embeddings):
```bash
jupyter notebook train_sparse_embeddings.ipynb
```

**For toroidal/bidirectional**:
```bash
jupyter notebook test_toroidal_lattice.ipynb
jupyter notebook train_toroidal_embeddings.ipynb
```

---

## Usage Example

```python
from sparse_lattice import SparseLatticeVisualizer, SparseDifferentiableLattice

# Create sparse lattice
lattice = SparseLatticeVisualizer(
    layers=8, 
    hex_radius=6, 
    storage_path="./lattice_storage",
    lazy_init=True
)

# Create propagation memory model
model = SparsePropagationMemory(
    sparse_lattice=lattice,
    embedding_dim=256,
    n_active_cells=200,
    max_steps=10,
    memory_slots=48
)

# Forward pass: input → propagate → retrieve
input_embedding = torch.randn(32, 256)  # batch of embeddings
retrieved, propagated_state, active_indices = model(input_embedding)

# Training
loss = F.mse_loss(retrieved, input_embedding)
loss.backward()
```

---

## Visualization

The architecture supports interactive 3D visualization via Plotly:

```python
lattice.visualize_full_plotly(
    pulse_records=propagation_history,
    show_cells=True,
    cell_opacity=0.3,
    pulse_size=3
)
```

Features:
- Cell positions colored by activation
- Pulse paths showing propagation flow
- Split/join events highlighted
- Hover info with cell parameters

---

## Future Directions

1. **Hierarchical Lattices**: Multi-scale propagation with coarse-to-fine retrieval
2. **Dynamic Topology**: Learnable adjacency structure
3. **Temporal Memory**: Sequence modeling via propagation dynamics
4. **Cross-Modal Retrieval**: Store text embeddings, query with images
5. **Biological Plausibility**: Connections to cortical column structure
6. **Multi-Wave Propagation**: More than 2 simultaneous propagation directions

---

## Citation

```bibtex
@misc{crossdimensional2026,
  title={Cross-Dimensional Architecture: Propagation-Based Sparse Embedding Memory},
  author={[Author Name]},
  year={2026},
  note={Work in progress}
}
```

---

## License

[To be determined]
