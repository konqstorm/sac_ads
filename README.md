# SAC-ADS: Soft Actor-Critic for Asteroid Defense System

A reinforcement learning implementation using the Soft Actor-Critic (SAC) algorithm to train an agent for an Asteroid Defense game environment.

## Project Overview

This project implements a deep reinforcement learning agent that learns to defend against incoming asteroids using a cannon. The agent is trained using the Soft Actor-Critic algorithm, which is a state-of-the-art off-policy reinforcement learning method that emphasizes exploration through entropy regularization.

### Key Features

- **SAC Algorithm**: State-of-the-art entropy-regularized off-policy RL method
- **Custom Gymnasium Environment**: AsteroidDefenseEnv with configurable parameters
- **Visualizations**: PyGame-based rendering for real-time visualization
- **Training Modes**: 
  - Supervised training mode (`train.py`)
  - Agent evaluation mode (`agent_mode.py`)
  - Baseline policy mode (`baseline_mode.py`)
  - Manual control mode (`manual_mode.py`)

## Project Structure

```
sac_ads/
├── train.py                # Main training script
├── env.py                  # Asteroid Defense Environment
├── models.py              # Actor and Critic neural networks
├── sac.py                 # SAC algorithm implementation
├── agent_mode.py          # Agent evaluation mode
├── baseline_mode.py       # Baseline policy evaluation
├── manual_mode.py         # Manual control mode
├── run_modes.py           # Mode runner
├── visual_pygame.py       # PyGame visualization
├── config.yaml            # Configuration file
├── pyproject.toml         # Project dependencies and metadata
├── weights/               # Saved model weights
│   ├── actor.pt
│   ├── critic1.pt
│   └── critic2.pt
└── plots/                 # Training plots and results
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sac_ads
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   python -m pip install -e .
   ```

## Dependencies

The project requires the following Python packages:

- **numpy** (>=1.26): Numerical computing
- **torch** (>=2.0): Deep learning framework
- **matplotlib** (>=3.8): Plotting and visualization
- **pygame** (>=2.5): Game rendering and visualization
- **pyyaml** (>=6.0): Configuration file parsing
- **gymnasium** (>=0.29): Reinforcement learning environment API

All dependencies are automatically installed when running `pip install -e .`

## Usage

### Training the Agent

Train a new SAC agent from scratch:

```bash
python train.py
```

Configuration is loaded from `config.yaml`. Modify this file to adjust:
- Environment parameters (asteroid spawn rate, field of view, etc.)
- Training hyperparameters (learning rates, network sizes, etc.)
- Episode count and training duration

### Running Modes

After training, you can evaluate your agent or test different modes:

```bash
# Run different modes based on configuration
python run_modes.py

# Or run specific modes directly:
python agent_mode.py      # Evaluate trained agent with visualization
python baseline_mode.py   # Run baseline policy
python manual_mode.py     # Manual keyboard control
```

### Configuration

Edit `config.yaml` to customize:

```yaml
env:
  max_steps: 1000
  dt: 0.033
  fov: 1.5708  # π/2 radians
  max_ang_vel: 2.0
  fire_threshold: 0.0
  # ... other environment parameters

agent:
  hidden_dim: 256
  learning_rate: 0.0003
  discount: 0.99
  # ... other SAC parameters

train:
  episodes: 1000
  seed: 42
```

## Architecture

### Agent (Actor-Critic)

- **Actor Network**: Outputs mean and variance for continuous actions
  - Input: Observation space (normalized asteroid positions and velocities)
  - Output: Continuous action (cannon angle velocity and fire command)
  
- **Critic Networks**: Two independent networks (Q-functions)
  - Input: Observation and action
  - Output: Q-value estimate

### Environment

The AsteroidDefenseEnv simulates:
- Incoming asteroids with configurable spawn patterns
- Cannon defensive mechanism with limited field of view
- Reward signal based on asteroids destroyed and hull integrity

## Training Results

Model weights are saved in the `weights/` directory:
- `actor.pt`: Trained actor network
- `critic1.pt`: First critic network
- `critic2.pt`: Second critic network

Training metrics and plots are saved to the `plots/` directory.

## Performance Metrics

The agent is evaluated on:
- **Reward**: Cumulative episode reward
- **Hull Integrity**: Remaining health of the defense system
- **Asteroids Destroyed**: Count of successfully destroyed asteroids

## Development

### Adding New Features

1. Modify the environment in `env.py`
2. Update network architectures in `models.py` if needed
3. Adjust SAC algorithm parameters in `config.yaml`

### Testing Changes

```bash
python train.py  # Train with new configuration
python agent_mode.py  # Visualize results
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

### CUDA/GPU Support
For GPU acceleration with PyTorch, install a CUDA-enabled version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## References

- **SAC Algorithm**: [Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- **Gymnasium**: [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please create an issue in the repository.
