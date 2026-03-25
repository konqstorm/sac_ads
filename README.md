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
python run_agent.py        # Evaluate trained agent with visualization
python run_baseline.py     # Run baseline policy
python run_manual.py       # Manual keyboard control
```

## Evaluation

### Running Evaluation

Evaluate trained agents using `evaluate.py`:

```bash
python evaluate.py --config configs/config_eval.yaml
```

### Evaluation Metrics

The evaluation system tracks:
- **Episode Reward**: Cumulative reward across evaluation episodes
- **Asteroids Destroyed**: Count of successfully destroyed asteroids
- **Hull Integrity**: Remaining health of the defense system
- **Run Time**: Total duration of evaluation runs

Results are saved to the `results/` directory with optional GIF recording for visualization.

## Baselines

The project includes two baseline strategies for comparison:

### 1. BaselineController (Heuristic Baseline)

A deterministic rule-based controller that uses physics-based target interception:

**Strategy:**
1. Select Target: Choose closest asteroid
2. Compute Intercept Point: Calculate where projectile meets asteroid
3. Track Target: Use proportional control to align cannon
4. Fire: When pointing error is small (< 0.02 rad)

**Advantages**: Interpretable, fast, requires no training  
**Limitations**: No learning, fixed strategy, suboptimal in complex scenarios

### 2. TwoStageAgent (Learned Baseline)

A two-stage agent combining a learned selector with a frozen aimer:

**Stage 1 (Selector)**: obs → discrete target slot selection  
**Stage 2 (Aimer)**: obs + aim_info → (yaw_velocity, pitch_velocity, fire_command)

The selector is trained via SAC to choose which asteroid to target, while the aimer is frozen to provide stable low-level control.

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

### Actor-Critic Networks

#### Actor Network

The actor network maps observations to continuous control actions using a stochastic policy:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$

where the mean $\mu_\theta$ and standard deviation $\sigma_\theta$ are learnable outputs parameterized by neural network weights $\theta$.

**Network Architecture:**
```
Input layer:           obs ∈ ℝᴰ  (observation dimension)
Hidden layer 1:        ReLU(W₁·obs + b₁) ∈ ℝᴴ
Hidden layer 2:        ReLU(W₂·h₁ + b₂) ∈ ℝᴴ
Hidden layer 3:        ReLU(W₃·h₂ + b₃) ∈ ℝᴴ
Output mean:           μ(s) = W_μ·h₃ + b_μ ∈ ℝᴬ
Output log-std:        log σ(s) = W_σ·h₃ + b_σ ∈ ℝᴬ
```

where D ≈ 50 (observation dim), H ≈ 256 (hidden dim), A ≈ 3 (action dim).

#### Critic Networks (Dual Q-functions)

Two independent Q-value networks to mitigate overestimation bias:

$$Q_\phi^{(i)}(s, a) \approx \mathbb{E}[R_t | s_t = s, a_t = a] \quad \text{for } i = 1, 2$$

**Network Architecture:**
```
Input:                 (obs, action) ∈ ℝᴰ⁺ᴬ
Concatenation:         concat(obs, a) ∈ ℝᴰ⁺ᴬ
Hidden layer 1:        ReLU(W₁·concat + b₁) ∈ ℝᴴ
Hidden layer 2:        ReLU(W₂·h₁ + b₂) ∈ ℝᴴ
Hidden layer 3:        ReLU(W₃·h₂ + b₃) ∈ ℝᴴ
Output (Q-value):      Q(s,a) = W_q·h₃ + b_q ∈ ℝ
```

### Soft Actor-Critic Algorithm

The objective function combines reward maximization with entropy maximization:

$$J(\pi) = \mathbb{E}_{s \sim \mathcal{D}}[V(s)] = \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a) - \alpha \log \pi(a|s)]$$

where $\mathcal{D}$ is the replay buffer, $\alpha$ is the entropy coefficient, and $H(\pi) = -\mathbb{E}_a[\log \pi(a|s)]$ is policy entropy.

**Value Function Estimate:**
$$V(s) = \mathbb{E}_{a \sim \pi}[\min(Q_1(s,a), Q_2(s,a)) - \alpha \log \pi(a|s)]$$

**Critic Loss** (with double Q-learning to reduce overestimation):
$$\mathcal{L}_Q = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[\left(Q(s,a) - y\right)^2\right]$$

where the target is:
$$y = r + \gamma(1-d)\left(\min(Q_1'(s',a'), Q_2'(s',a')) - \alpha \log \pi(a'|s')\right), \quad a' \sim \pi(\cdot|s')$$

and $Q'$ and $\pi'$ are target networks updated via exponential moving average.

**Actor Loss** (Policy Gradient with reparameterization):
$$\mathcal{L}_\pi = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[\alpha \log \pi(a_s|s) - \min(Q_1(s,a_s), Q_2(s,a_s))\right]$$

where $a_s = \mu(s) + \sigma(s) \odot \epsilon$ is the reparameterized action.

**Automatic Entropy Tuning** (optional, temperature learning):
$$\alpha^* = \arg\min_\alpha \mathbb{E}_{s} [-\alpha(\log \pi(a|s) + H_0)]$$

where $H_0 = -|A|$ is target entropy (dimensionality of action space).

### Environment

The AsteroidDefenseEnv simulates:
- Incoming asteroids with configurable spawn patterns
- Cannon defensive mechanism with limited field of view
- Reward signal based on asteroids destroyed and hull integrity

## Training & Results

### Training Model Checkpoints

During training, the following model weights are saved:

**Selector Agent** (`results/weights_selector/`):
- `actor.pt`: Trained actor network (target selection policy)
- `critic1.pt`: First Q-value critic network
- `critic2.pt`: Second Q-value critic network

**Aimer Agent** (`results/weights_aimer/`):
- `actor.pt`: Trained aimer network (low-level control)
- `critic1.pt`, `critic2.pt`: Critic networks

### Training Metrics

Training progress is logged and saved to `results/metrics/`:
- Episode rewards (mean, std, max, min)
- Asteroids destroyed per episode
- Hull integrity over time
- Loss curves (actor, critic, entropy)

Visualizations of training progress are saved to `plots/` directory.

## Performance Metrics

The agent is evaluated on multiple dimensions:

### Task Performance
- **Episode Reward**: Cumulative discounted reward per episode ($\sum_t \gamma^t r_t$)
- **Asteroids Destroyed**: Total count of successfully destroyed asteroids per episode
- **Hull Integrity**: Final remaining health of defense system (0-100%)
- **Survival Time**: Episode length in seconds before shield failure

### Learning Efficiency  
- **Sample Efficiency**: Reward per environment interaction (steps taken)
- **Training Time**: Wall-clock time to convergence
- **Convergence Speed**: Episodes required to reach target performance

### Robustness
- **Generalization**: Performance on held-out test seeds
- **Variance**: Standard deviation of reward across evaluation runs
- **Performance on Baselines**: Comparison vs. heuristic and learned baselines

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
