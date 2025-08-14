import numpy as np
from env import Connect4Env, P1, P2, EMPTY, ROWS, COLS
from opponents import RandomOpponent, GreedyOpponent, FrozenQOpponent
from rl import QLearningAgent

WIN_REWARD = 10.0
LOSS_REWARD = -10.0
DRAW_REWARD = 1.0
MOVE_PENALTY = -0.01 #Small negative reward for every move to encourage faster wins

def play_episode(agent, opponent, render=False):
    """
    Plays a single Connect 4 epoch (full game) where the Q-learning agent (P1) plays
    opponent (P2)

    Alternates turns until game ends. Only the Q-learning agent updates Q-values.

    Args:
        agent: Thelearning agent (P1)
        opponent: A non-learning opponent (Random or minimax)
        render(bool): If True, prints the board after each move.

    Rets:
        A float corresponding to thereward from the RL agent's perspective
    """
    env = Connect4Env(starting_player=P1) #RL agent starts first.
    state = env.reset()
    agent_moves = []
    while True:
        #RL agent'sturn
        legal = env.legal_actions()
        action = agent.select_action(state, legal)
        agent_moves.append((state, action))

        #last_agent_state = state
        #last_agent_action = action

        next_state, done = env.step(action)
        if render: env.render()

        if done:
            reward = terminal_reward(env.winner, for_player=P1)
            update_episode_rewards(agent, agent_moves, reward)
            agent.decay_epsilon()
            return reward

        #Update for non-terminal step 
        agent.update(state, action, MOVE_PENALTY, next_state, env.legal_actions())
        state = next_state

        #Opponent's turn
        opp_action = opponent.act(
            env_state=state,
            board_array=board_from_state(state),
            legal_actions=env.legal_actions(),
            player_token=P2
        )
        next_state, done = env.step(opp_action)
        if render: env.render()

        if done:
            #Don't update reward with opponent's action
            reward = terminal_reward(env.winner, for_player=P1)
            if agent_moves:
                last_state, last_action = agent_moves[-1]
                agent.update(last_state, last_action, reward, None, [])
            agent.decay_epsilon()
            return reward

        state = next_state

def update_episode_rewards(agent, moves, final_reward):
    if moves:
        last_state, last_action = moves[-1]
        agent.update(last_state, last_action, final_reward, None, [])

def terminal_reward(winner, for_player):
    """
    Returns a reward from the perspecitve of 'for_player'

    Args:
        winner (int): Who won (1, -1, or 0)
        for_layer (int): The player being assigned the reward

    Rets:
        float: +1 for win, -1 for loss,  0 for draw 
    """
    #Changed from 1 to 10
    if winner == for_player: return WIN_REWARD
    if winner == -for_player: return LOSS_REWARD
    else: return DRAW_REWARD#Draw

def board_from_state(state):
    """
    Convers a flat state tuple back to a 2D board array.

    Args:
        state tuple: Flattended board state

    Rets:
        np.ndarray: 6x7 
    """
    return np.array(state, dtype=np.int8).reshape(6, 7)

def train(num_episodes=20000, opponent_type="random"):
    """
    Trains the Q-learning agent over a number of episodes against the chosen opponent type

    Args:
        num_episodes: Number of games
        opponent_type: either "random" or "minimax" 

    Returns:
        The trained agent
    """

    if opponent_type == "random":
        opponent = RandomOpponent()
    elif opponent_type == "greedy":
        opponent = GreedyOpponent()
    #ADD CONDITION HERE TO CREATE opponent AS MINIMAX TO TRAIN
    else:
        raise ValueError("opponent_type must be 'random' or 'minimax'.")
    
    agent = QLearningAgent(
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
        alpha=0.15, gamma = 0.95
    )

    results = []

    for ep in range(1, num_episodes + 1):
        r = play_episode(agent, opponent, render=False)
        results.append(r)

        if ep % 1000 == 0:
            window = np.array(results[-1000:])
            win = (window > 0).mean()
            draw = (window == 0).mean()
            print(f"Ep {ep}: vs {opponent_type} | win%={win:.2f}, draw%={draw:.2f}, epsilon={agent.epsilon:.3f}")
    return agent, results


#Attempted rewrite
def play_episode_selfplay(agent, opponent, starting_player=P1, render=False):
    env = Connect4Env(starting_player=starting_player)
    state = env.reset()

    learner_token = P1 if starting_player == P1 else P2
    learner_moves = []

    while True:
        current_player = env.current_player

        if current_player == learner_token:
            legal = env.legal_actions()
            action = agent.select_action(state, legal)
            learner_moves.append((state, action))

            next_state, done = env.step(action)
            if render: env.render()

            if done:
                reward = terminal_reward(env.winner, for_player=learner_token)
                update_episode_rewards(agent, learner_moves, reward)
                agent.decay_epsilon()
                return reward
            
            # Small penalty for continuing the game
            agent.update(state, action, MOVE_PENALTY, next_state, env.legal_actions())
            state = next_state

        else:
            opp_action = opponent.act(
                env_state=state,
                board_array=board_from_state(state),
                legal_actions=env.legal_actions(),
                player_token=env.current_player
            )
            next_state, done = env.step(opp_action)
            if render: env.render()

            if done:
                reward = terminal_reward(env.winner, for_player=learner_token)
                if learner_moves:
                    last_state, last_action = learner_moves[-1]
                    agent.update(last_state, last_action, reward, None, [])
                agent.decay_epsilon()
                return reward

            state = next_state

def train_self_play(
    cycles=500,
    episodes_per_cycle=10000,
    epsilon_start=1.0,
    epsilon_min=0.15,
    epsilon_decay=0.9995,
    alpha=0.15,
    gamma=0.9,
    fallback_opponent="greedy"):

    agent = QLearningAgent(
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=alpha,
        gamma=gamma
    )
    fallback = GreedyOpponent() if fallback_opponent == "greedy" else None

    all_results = []
    for cycle in range(1, cycles+1):
        print(f"\n--- Cycle {cycle}/{cycles} ---")

        frozen_Q = dict(agent.Q)
        opponent = FrozenQOpponent(frozen_Q, fallback=fallback)
        cycle_results = []

        for ep in range(1, episodes_per_cycle+1):
            starting_player = P1 if (ep % 2 == 1) else P2
            reward = play_episode_selfplay(agent, opponent, starting_player=starting_player, render=False)
            cycle_results.append(reward) 

            # Progress within cycle
            if ep % 2000 == 0:
                window = cycle_results[-2000:]
                win_rate = (np.array(window) > 0).mean()
                print(f"  Episode {ep:5d}/{episodes_per_cycle}: Win={win_rate:.3f} Eps={agent.epsilon:.4f}")
        
        all_results.extend(cycle_results)

        # Cycle summary
        win_rate = (np.array(cycle_results) > 0).mean()
        draw_rate = (np.array(cycle_results) == DRAW_REWARD).mean()
        loss_rate = (np.array(cycle_results) < 0).mean()
        avg_reward = np.array(cycle_results).mean()

        print(f"Cycle {cycle} Summary: Win={win_rate:.3f} Draw={draw_rate:.3f} Loss={loss_rate:.3f} "
              f"AvgReward={avg_reward:6.2f}")
        print(f"Q-table size: {len(agent.Q):,} entries")

        agent.epsilon = 1.0

    print(f"\nTraining complete! Final Q-table size: {len(agent.Q):,}")
    return agent, all_results


#*****ADDED*****
def terminal_reward(winner, for_player):
    if winner == for_player: return WIN_REWARD
    if winner == -for_player: return LOSS_REWARD
    return 0.0

def board_from_state(state):
    return np.array(state, dtype=np.int8).reshape(ROWS, COLS)

def _is_win(b, player):
    # horiz
    for r in range(ROWS):
        for c in range(COLS-3):
            if np.all(b[r, c:c+4] == player): return True
    # vert
    for r in range(ROWS-3):
        for c in range(COLS):
            if np.all(b[r:r+4, c] == player): return True
    # diag \
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(b[r+i, c+i] == player for i in range(4)): return True
    # diag /
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(b[r-i, c+i] == player for i in range(4)): return True
    return False

def _drop(board, col, player):
    for r in range(ROWS-1, -1, -1):
        if board[r, col] == EMPTY:
            board[r, col] = player
            return True
    return False

def _legal_from_board(b):
    return [c for c in range(COLS) if b[0, c] == EMPTY]

def opponent_has_immediate_win(b, opp):
    """Return True if opponent can win in one move on board b."""
    for oa in _legal_from_board(b):
        bb = b.copy()
        _drop(bb, oa, opp)
        if _is_win(bb, opp):
            return True
    return False
