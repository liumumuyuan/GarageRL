# GarageRL

**GarageRL** is a modular, efficient, and highly customizable reinforcement learning framework designed for both rapid experimentation and deployment.

### Currently Supported Algorithms:

- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **TD3 + BC** (TD3 with Behavior Cloning)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)

---

## Quick Setup

Set up dependencies easily using Conda:

```bash
conda env create -f environment.yaml
conda activate rl-dev
```

---

## Usage & Running Experiments

Experiments are organized through `.py` scripts and `.yaml` configuration files within the `experiments/` directory.

For instance, to execute the PPO algorithm on the CartPole environment:

```bash
python run.py --exp experiments/ppo/ppo_cartpole.py
```

Customize experiments by modifying the corresponding `.py` and `.yaml` files.

---

## Project Structure

```
GarageRL/
├── agents/                 # RL algorithm implementations
├── config/                 # Configuration management
├── experiments/            # Experiment scripts and YAML configs
├── networks/               # Neural network architectures
├── utils/                  # Utility scripts (buffers, noise, etc.)
├── run.py                  # Entry-point script for experiments
├── environment.yaml        # Conda environment definition
├── LICENSE                 # MIT License file
└── README.md               # This file
```

---

## License

This project is licensed under the [MIT License](LICENSE).
