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
├── core/                           # Core RL algorithms and environment
│   ├── env.py                      # Asteroid Defense Environment
│   ├── models_discrete.py          # Discrete action networks (selector)
│   ├── models_continuous.py        # Continuous action networks (aimer)
│   ├── sac_discrete.py             # SAC for discrete actions (selector)
│   ├── sac_continuous.py           # SAC for continuous actions (aimer)
│   ├── aimer.py                    # Frozen aimer controller
│   ├── baseline.py                 # Heuristic baseline controller
│   ├── two_stage_agent.py          # Combined selector + aimer agent
│   ├── aim_utils.py                # Utility functions for aiming
│   ├── runtime_options.py          # Configuration loading utilities
│   ├── vec_env.py                  # Vectorized environment wrapper
│   └── __init__.py
├── visuals/                        # Visualization and rendering
│   ├── visual_pygame.py            # PyGame rendering
│   ├── visual_ursina.py            # Ursina 3D rendering
│   ├── gif_recorder.py             # GIF recording utility
│   └── __init__.py
├── configs/                        # Configuration files
│   ├── config_eval.yaml            # Evaluation configuration
│   ├── config_select.yaml          # Selector agent configuration
│   └── config_aim.yaml             # Aimer agent configuration
├── train_selector.py               # Train selector agent (target selection)
├── train_aimer.py                  # Train aimer agent (low-level control)
├── evaluate.py                     # Evaluate trained agents
├── run_agent.py                    # Evaluate trained agent with visualization
├── run_baseline.py                 # Run baseline policy
├── run_manual.py                   # Manual keyboard control mode
├── pyproject.toml                  # Project dependencies and metadata
├── requirements.txt                # Pinned dependency versions
├── results/                        # Training results and outputs
│   ├── weights_selector/           # Selector agent weights
│   │   ├── actor.pt
│   │   ├── critic1.pt
│   │   └── critic2.pt
│   ├── weights_aimer/              # Aimer agent weights
│   │   ├── actor.pt
│   │   ├── critic1.pt
│   │   └── critic2.pt
│   ├── plots_selector/             # Selector training plots
│   ├── plots_aimer/                # Aimer training plots
│   └── metrics/                    # Training metrics logs
├── important_gif/                  # Saved evaluation GIFs
└── .venv/                          # Python virtual environment
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

The project uses a **two-stage training approach**:

**Stage 1: Train Selector Agent** (discrete actions - which asteroid to target)
```bash
python train_selector.py --config configs/config_select.yaml
```

**Stage 2: Train Aimer Agent** (continuous actions - low-level cannon control)
```bash
python train_aimer.py --config configs/config_aim.yaml
```

Configuration is loaded from YAML files in `configs/`. Modify these files to adjust:
- Environment parameters (asteroid spawn rate, field of view, etc.)
- Training hyperparameters (learning rates, network sizes, batch sizes, etc.)
- Episode count and training duration

### Running Evaluation & Modes

Evaluate trained agents with different modes:

```bash
# Evaluate full TwoStageAgent (selector + frozen aimer) with visualization
python run_agent.py --config configs/config_eval.yaml --renderer pygame

# Run baseline heuristic controller (no learning required)
python run_baseline.py --config configs/config_eval.yaml

# Manual keyboard control (human plays the defense game)
python run_manual.py --config configs/config_eval.yaml

# Batch evaluation (no visualization)
python evaluate.py --config configs/config_eval.yaml
```

### Configuration Files

Edit configuration YAML files in `configs/` to customize:

```yaml
env:
  max_steps: 1000
  dt: 0.033                # Time step
  fov: 1.5708              # π/2 radians, field of view
  max_ang_vel: 2.0         # Max cannon rotation speed
  max_asteroids: 5         # Max simultaneous targets

agent:
  hidden_dim: 256          # Actor/Critic network hidden dimension
  learning_rate: 0.0003    # Optimizer learning rate
  discount: 0.99           # γ (gamma) in SAC formulas
  start_steps: 10000       # Random exploration steps before learning

train:
  episodes: 1000
  seed: 42
  batch_size: 256
  replay_buffer_size: 100000

visual:
  seeds: [0, 42, 123]      # Deterministic test seeds
  enabled: true
  fps: 30
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

Located in `core/baseline.py`, a deterministic rule-based controller that uses physics-based target interception:

**Strategy:**
1. **Select Target**: Choose closest asteroid by Euclidean distance
2. **Compute Intercept Point**: Solve ballistic interception equation
3. **Track Target**: Use proportional control to point cannon
4. **Fire**: When pointing error < 0.02 rad, fire projectile

**Advantages**: 
- Interpretable, deterministic, fast
- No training required
- Good baseline for comparison

**Limitations**: 
- Fixed strategy (no learning)
- Suboptimal in complex scenarios
- Cannot adapt to new situations

### 2. TwoStageAgent (Learned Agent)

Located in `core/two_stage_agent.py`, combines learned selector with frozen aimer:

**Architecture:**
- **Selector**: Discrete policy (which asteroid to target)
  - Input: observation
  - Output: target slot index ∈ {0, 1, ..., num_asteroids}
  - Training: `train_selector.py` with SAC (discrete actions)
  
- **Aimer**: Continuous policy (how to aim and fire)
  - Input: observation + aim features
  - Output: (yaw_velocity, pitch_velocity, fire)
  - Training: `train_aimer.py` with SAC (continuous actions) - then frozen

**Advantages**:
- Data-driven learning from experience
- Modular architecture (train components separately)
- Interpretable two-stage decomposition

**Limitations**:
- Requires training two agents
- Performance depends on aimer quality
- Frozen aimer may not be optimal for all selector choices

## Architecture

### Actor-Critic Networks

#### Actor Network

The actor network maps observations to continuous control actions using a stochastic policy:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$

**Variable meanings:**
- $s$ — current observation state (e.g., asteroid positions, velocities)
- $a$ — continuous action sampled from the distribution
- $\pi_\theta(a|s)$ — probability density of action $a$ given state $s$
- $\mu_\theta(s)$ — mean action output from network (what action to take on average)
- $\sigma_\theta(s)$ — standard deviation controlling exploration (higher = more random)
- $\theta$ — neural network weights/parameters
- $\mathcal{N}$ — Gaussian/Normal probability distribution

**Network Architecture** with weight matrices $W$ and biases $b$:
```
Input layer:           obs ∈ ℝᴰ  (observation dimension)
Hidden layer 1:        ReLU(W₁·obs + b₁) ∈ ℝᴴ
Hidden layer 2:        ReLU(W₂·h₁ + b₂) ∈ ℝᴴ
Hidden layer 3:        ReLU(W₃·h₂ + b₃) ∈ ℝᴴ
Output mean:           μ(s) = W_μ·h₃ + b_μ ∈ ℝᴬ
Output log-std:        log σ(s) = W_σ·h₃ + b_σ ∈ ℝᴬ
```

**Dimension sizes:**
- D ≈ 50 (observation: asteroid positions, velocities, etc.)
- H ≈ 256 (hidden: internal representation learned from data)
- A ≈ 3 (actions: yaw velocity, pitch velocity, fire command)

#### Critic Networks (Dual Q-functions)

Two independent Q-value networks to mitigate overestimation bias:

$$Q_\phi^{(i)}(s, a) \approx \mathbb{E}[R_t | s_t = s, a_t = a] \quad \text{for } i = 1, 2$$

**Variable meanings:**
- $s$ — current observation state at time $t$
- $a$ — action taken at time $t$
- $Q_\phi^{(i)}(s,a)$ — learned Q-value (expected cumulative future reward with discount)
- $R_t$ — cumulative discounted reward from time $t$ onward
- $\phi$ — neural network parameters for the critic network
- $(i)$ — index showing we have two networks (1 and 2) for stability
- $\mathbb{E}[·|·]$ — conditional expectation over all possible futures

**Network Architecture** combining both state and action:
```
Input:                 (obs, action) ∈ ℝᴰ⁺ᴬ
Concatenation:         concat(obs, a) ∈ ℝᴰ⁺ᴬ
Hidden layer 1:        ReLU(W₁·concat + b₁) ∈ ℝᴴ
Hidden layer 2:        ReLU(W₂·h₁ + b₂) ∈ ℝᴴ
Hidden layer 3:        ReLU(W₃·h₂ + b₃) ∈ ℝᴴ
Output (Q-value):      Q(s,a) = W_q·h₃ + b_q ∈ ℝ
```

### Soft Actor-Critic Algorithm

The objective function balances reward maximization with entropy (exploration) maximization:

$$J(\pi) = \mathbb{E}_{s \sim \mathcal{D}}[V(s)] = \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a) - \alpha \log \pi(a|s)]$$

**Variable meanings:**
- $J(\pi)$ — objective (what we want to maximize)
- $s$ — state sampled from environment experience
- $a$ — action sampled from policy
- $\mathcal{D}$ — replay buffer (stored past experiences)
- $Q(s,a)$ — estimated cumulative reward for this state-action
- $\alpha$ — temperature coefficient (temperature = entropy weight)
- $\log \pi(a|s)$ — log probability of action (higher = more likely action)
- $\mathbb{E}[·]$ — expectation (average over samples)

**Value Function Estimate:** What is this state worth?

$$V(s) = \mathbb{E}_{a \sim \pi}[\min(Q_1(s,a), Q_2(s,a)) - \alpha \log \pi(a|s)]$$

**Variable meanings:**
- $V(s)$ — value (expected return from this state)
- $\min(Q_1, Q_2)$ — use smaller Q-value to be conservative (pessimism bias reduction)
- $a \sim \pi$ — sample action from current policy
- $-\alpha \log \pi(a|s)$ — entropy bonus (encourage diverse actions)

**Critic Loss** (minimize prediction error with temporal-difference learning):

$$\mathcal{L}_Q = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[\left(Q(s,a) - y\right)^2\right]$$

Target value:
$$y = r + \gamma(1-d)\left(\min(Q_1'(s',a'), Q_2'(s',a')) - \alpha \log \pi(a'|s')\right), \quad a' \sim \pi(\cdot|s')$$

**Variable meanings:**
- $\mathcal{L}_Q$ — critic loss (MSE of Q-prediction error)
- $(s, a, r, s', d)$ — one stored experience (state, action, reward, next state, done flag)
- $r$ — immediate reward from environment
- $s'$ — next state after taking action
- $d$ — done flag (1 if episode ended, 0 otherwise)
- $\gamma$ ≈ 0.99 — discount factor (how much to value future rewards)
- $Q'$ and $\pi'$ — target networks (polyak-averaged copies for stability)
- $a'$ — best action in next state (sampled from policy)

**Actor Loss** (maximize expected Q-value and entropy):

$$\mathcal{L}_\pi = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[\alpha \log \pi(a_s|s) - \min(Q_1(s,a_s), Q_2(s,a_s))\right]$$

where $a_s = \mu(s) + \sigma(s) \odot \epsilon$ is the reparameterized action.

**Variable meanings:**
- $\mathcal{L}_\pi$ — actor loss (how to improve policy)
- $\epsilon \sim \mathcal{N}$ — Gaussian noise for gradient estimation
- $a_s$ — sampled action using reparameterization trick
- $\mu(s)$ — mean action output
- $\sigma(s)$ — standard deviation (controls noise)
- $\odot$ — element-wise multiplication (Hadamard product)
- Goal: Maximize Q-values while maintaining entropy

**Automatic Entropy Tuning** (learn temperature $\alpha$ automatically):

$$\alpha^* = \arg\min_\alpha \mathbb{E}_{s} [-\alpha(\log \pi(a|s) + H_0)]$$

**Variable meanings:**
- $\alpha^*$ — optimal temperature parameter
- $H_0 = -|A|$ — target entropy (negative of action dimension)
- Objective: Keep entropy at desired level automatically

### Environment

The AsteroidDefenseEnv simulates:
- Incoming asteroids with configurable spawn patterns
- Cannon defensive mechanism with limited field of view
- Reward signal based on asteroids destroyed and hull integrity

## Training & Results

### Two-Stage Training

The project trains two agents independently:

**1. Selector Agent** (`train_selector.py`)
- **Action Space**: Discrete (which asteroid slot to target: 0-4)
- **Training Algorithm**: SAC with discrete actions (`core/sac_discrete.py`)
- **Network**: `core/models_discrete.py` (Actor outputs Q-values, Critic evaluates)
- **Output**: Saved to `results/weights_selector/`
  - `actor.pt` — Target selection policy
  - `critic1.pt`, `critic2.pt` — Q-value estimators
- **Plots**: Training curves saved to `results/plots_selector/`

**2. Aimer Agent** (`train_aimer.py`)
- **Action Space**: Continuous (cannon yaw velocity, pitch velocity, fire command)
- **Training Algorithm**: SAC with continuous actions (`core/sac_continuous.py`)
- **Network**: `core/models_continuous.py` (Gaussian policy network)
- **Output**: Saved to `results/weights_aimer/`
  - `actor.pt` — Aiming policy (frozen during selector training)
  - `critic1.pt`, `critic2.pt` — Q-value estimators
- **Plots**: Training curves saved to `results/plots_aimer/`

### Model Checkpoints

Models are saved in `results/weights_*/`:

```
results/
├── weights_selector/
│   ├── actor.pt           # Selector policy (discrete)
│   ├── critic1.pt         # Q-function 1
│   └── critic2.pt         # Q-function 2
├── weights_aimer/
│   ├── actor.pt           # Aimer policy (continuous, frozen)
│   ├── critic1.pt         # Q-function 1
│   └── critic2.pt         # Q-function 2
├── plots_selector/
│   ├── reward.png         # Reward learning curves
│   ├── hull_damage.png    # Hull integrity over training
│   └── kills.png          # Asteroids destroyed per episode
├── plots_aimer/
│   ├── reward.png
│   ├── hull_damage.png
│   └── kills.png
└── metrics/               # Training logs
    ├── selector_metrics.log
    └── aimer_metrics.log
```

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

1. **Modify environment**: Edit `core/env.py` to add new mechanics or observations
2. **Update networks**: Edit `core/models_discrete.py` or `core/models_continuous.py` for new architectures
3. **Extend SAC**: Modify `core/sac_discrete.py` or `core/sac_continuous.py` for algorithm changes
4. **Update configurations**: Edit `configs/config_select.yaml` or `configs/config_aim.yaml`

### Project Organization

- **Core algorithms**: `core/` — SAC implementations, environment, networks
- **Visualization**: `visuals/` — PyGame and Ursina renderers, GIF recording
- **Training scripts**: `train_selector.py`, `train_aimer.py` — entry points
- **Evaluation scripts**: `run_agent.py`, `run_baseline.py`, `run_manual.py` — testing
- **Configurations**: `configs/` — YAML-based hyperparameter control

### Testing Changes

```bash
# Train selector agent with modified config
python train_selector.py --config configs/config_select.yaml

# Visualize results
python run_agent.py --config configs/config_eval.yaml --renderer pygame

# Compare against baseline
python run_baseline.py --config configs/config_eval.yaml
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
