# Connect 4 — RL vs Minimax

Train a tabular Q-learning agent for Connect 4 and compare it against rule-based opponents: Random, Greedy, and Minimax (α–β). Includes post-training win-rate plots.

## Features
- Tabular Q-learning agent with ε-greedy policy
- Three training modes: vs **Random**, vs **Greedy**, and **Self-Play** (frozen snapshot cycles)
- Evaluation vs Random, Greedy, and **Minimax** at a chosen depth
- **Post-training** rolling win-rate plot (no live plotting during training)

## Requirements
- Python 3.9+ (3.10+ recommended)
- `numpy`, `matplotlib`, `torch`

## Install:
git clone https://github.com/hathcoat/CS441_Connect4.git
cd CS441_Connect4
git switch master
pip install numpy matplotlib torch

## Usage
- Run using python main.py or python3 main.py, depending on your version
- Upon running, you will be prompted with several options
- Likely you will need to start by traning a specific model, which will be saved into the current directory as random.pkl, greedy.pkl, and self-play.pkl, which can later be loaded
-From there, the model can be either evaluated against an algorithm, or you can play against it yourself