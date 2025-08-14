import pickle #Allows to save Q table
import torch
import matplotlib.pyplot as plt
import numpy as np
from rl import QLearningAgent
from train import train, terminal_reward, board_from_state, train_self_play
from opponents import RandomOpponent, GreedyOpponent, MinimaxOpponent
from env import Connect4Env, P1, P2

PATHS = {"Random":"random.pkl", "Greedy":"greedy.pkl", "Self-play": "self-play.pkl"}

def plot_final_winrate(results, window=1000, title="Win rate over episodes"):
    if not results:
        print("No results to plot.")
        return
    arr = np.array(results)
    wins = (arr > 0).astype(float)
    rolling = []
    xs = []
    for i in range(1, len(wins)+1):
        start = max(0, i - window)
        rolling.append(wins[start:i].mean())
        xs.append(i)
    plt.figure()
    plt.plot(xs, rolling)
    plt.xlabel("Episodes")
    plt.ylabel(f"Win rate (rolling {window})")
    plt.title(title)
    plt.grid(True)
    plt.show()

def save_agent(agent, kind):
    path = PATHS[kind]
    with open(path, "wb") as f:
        pickle.dump(agent.Q, f)
    print(f"Agent saved to {path}")

def load_agent(kind):
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, alpha=0.15, gamma = 0.9)
    path = PATHS[kind]
    with open(path, "rb") as f:
        agent.Q = pickle.load(f)
    print(f"Agent loaded from {path}")
    return agent

def evaluate(agent, opponent, games=100):
    wins = draws = losses = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(games):
        env = Connect4Env(starting_player=P1)
        state = env.reset()
        while True:
            #Agent Turn
            action = agent.select_action(state, env.legal_actions())
            state, done = env.step(action)
            if done:
                if env.winner == P1: wins += 1
                elif env.winner == 0: draws += 1
                else: losses += 1
                break

            #Opponents turn
            opp_action = opponent.act(
                env_state=state,
                board_array=board_from_state(state),
                legal_actions=env.legal_actions(),
                player_token=P2
            )
            state, done = env.step(opp_action)
            if done:
                if env.winner == P1: wins += 1
                elif env.winner == 0: draws += 1
                else: losses += 1
                break

    agent.set_epsilon(original_epsilon)
    print(f"Results vs {type(opponent).__name__} in {games} games:")
    print(f" Wins: {wins} ({wins/games:.2%})")
    print(f" Draws: {draws} ({draws/games:.2%})")
    print(f" Losses: {losses} ({losses/games:.2%})")


#This function is so we as people can test the growth
def human_vs_agent(agent):
    env = Connect4Env(starting_player=P2)  # Human goes first
    state = env.reset()
    print("You are player 2 (O). RL Agent is player 1 (X).") 

    while True:
        env.render()

        # Human move
        legal = env.legal_actions()
        print(f"Legal moves: {legal}")
        move = int(input("Your move (0–6): "))
        while move not in legal:
            move = int(input("Illegal move. Try again (0–6): "))
        state, done = env.step(move)
        if done:
            env.render()
            if env.winner == P2: print("You win!")
            elif env.winner == 0: print("Draw!")
            else: print("Agent wins!")
            break

        # Agent move
        agent_move = agent.select_action(state, env.legal_actions())
        print(f"Agent plays column {agent_move}")
        state, done = env.step(agent_move)
        if done:
            env.render()
            if env.winner == P2: print("You win!")
            elif env.winner == 0: print("Draw!")
            else: print("Agent wins!")
            break

def main():
    print("Q-Learning Connect 4\nOptions:")
    print("1. Train Random Agent")
    print("2. Train Greedy Agent")
    print("3. Train Self-Play Opponent")
    print("4. Load Random Agent")
    print("5. Load Greedy Agent")
    print("6. Load Self-Play Agent")
    print("7. Play against agent")
    print("8. Evaluate agent vs Random")
    print("9. Evaluate agent vs Greedy")
    print("10. Evaluate agent vs Minimax with Alpha-Beta Pruning")
    print("0. Exit")

    agent = None

    while True:
        choice = input("Enter choice: ").strip()

        if choice == "1":
            print("Training agent vs Random Opponent...")
            kind = "Random"
            agent, results = train(num_episodes=200000, opponent_type="random")
            save_agent(agent, kind)
            plot_final_winrate(results, window=5000, title="Win rate vs Random")

        if choice == "2":
            print("Training agent vs Greedy Opponent...")
            kind = "Greedy"
            agent, results = train(num_episodes=200000, opponent_type="greedy")
            save_agent(agent, kind)
            plot_final_winrate(results, window=5000, title="Win rate vs Greedy")


        elif choice == "3":
            print("Training with self-play (frozen snapshot cycles)...")
            kind = "Self-play"
            agent, results = train_self_play(
                cycles=250,
                episodes_per_cycle=20000,
                epsilon_start=1,
                epsilon_min=0.15,
                epsilon_decay=0.99978,
                alpha=0.15,
                gamma=0.9,
                fallback_opponent="greedy"
            )
            save_agent(agent, kind)
            plot_final_winrate(results, window=20000, title="Self-play win rate")

        elif choice == "4":
            kind = "Random"
            agent = load_agent(kind)

        elif choice == "5":
            kind = "Greedy"
            agent = load_agent(kind)

        elif choice == "6":
            kind = "Self-play"
            agent = load_agent(kind)

        elif choice == "7":
            if agent is None:
                print("Load or train an agent first.")
            else:
                human_vs_agent(agent)            

        elif choice == "8":
            if agent is None:
                print("Load or train an agent first.")
            else:
                evaluate(agent, RandomOpponent())        

        elif choice == "9":
            if agent is None:
                print("Load or train an agent first.")
            else:
                evaluate(agent, GreedyOpponent())

        elif choice == "10":
            if agent is None:
                print("Load or train an agent first.")
            else:
                try:
                    depth_in = input("Minimax depth (default 4): ").strip()
                    depth = int(depth_in) if depth_in else 4
                except Exception:
                    depth = 4
                print(f"Evaluating RL Agnet vs Minimax(depth={depth})...")
                evaluate(agent, MinimaxOpponent(depth=depth))

        elif choice == "0":
            break

        else:
            print("Invalid input") 

if __name__ == "__main__":
    main()