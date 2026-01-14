# Energy Management RL: Offline & Online Learning for Building Control

A reinforcement learning study combining Conservative Q-Learning (CQL) and Soft Actor-Critic (SAC) for cost-effective building energy management with battery storage and dynamic pricing.

## Quick Start (5 minutes)

### Prerequisites
- Docker Desktop installed and running
- Visual Studio Code with Dev Containers extension

### Setup

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd energy-rl-project
   ```

2. Open in VS Code and reopen in container:
   - Open folder in VS Code
   - Click "Reopen in Container" (bottom-right popup)
   - Wait for Docker build (~10 min on first run)

3. Run main experiment:
   ```bash
   python src/run_experiment.py
   ```

Results will be saved to `results/logs/` and `results/figures/`.

## Project Structure

```
├── src/
│   ├── environments/       # Building environment wrapper with battery + pricing
│   ├── algorithms/         # CQL and SAC implementations
│   ├── utils/              # Logging and helpers
│   └── run_experiment.py   # Main entry point
├── data/
│   ├── offline_dataset/    # Generated offline trajectories
│   └── raw/                # Sample data (if needed)
├── results/
│   ├── logs/               # Training logs (CSV/JSON)
│   └── figures/            # Generated plots
├── tests/                  # Unit tests
└── notebooks/              # Analysis notebooks
```

## Algorithms

- **CQL (Conservative Q-Learning)**: Offline RL approach
- **SAC (Soft Actor-Critic)**: Online baseline
- **Rule-Based Controller**: Non-learned baseline

## Environment Extensions

- Battery energy storage system (BESS)
- Time-varying electricity pricing
- Photovoltaic production estimation
- Multi-objective reward (cost + comfort + cycling penalty)

## License

MIT License - See LICENSE file for details.