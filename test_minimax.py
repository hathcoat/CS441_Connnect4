#test_minimax.py
from env import Connect4Env, P1, P2
from players import RandomAgent, GreedyAgent, MinimaxAgent

def play_game(p1_agent, p2_agent, verbose=False):
    env = Connect4Env(starting_player=P1)
    agents = {P1: p1_agent, P2: p2_agent}
    while not env.done:
        col = agents[env.current_player].move(env)
        env.step(col)
        if verbose:
            env.render(); print()
    return env.winner

def run_series(p1, p2, n=50, label=""):
    results = {P1:0, P2:0, 0:0}
    for _ in range(n):
        w = play_game(p1, p2, verbose=False)
        results[w] += 1
    print(f"{label} over {n} games -> {results}")

if __name__ == "__main__":
    #Minimax depth to try
    depth = 4

    #1) Minimax vs Random
    run_series(MinimaxAgent(P1, depth), RandomAgent(P2), n=50,
               label=f"Minimax(d={depth}, P1) vs Random(P2)")

    #2) Minimax vs Greedy
    run_series(MinimaxAgent(P1, depth), GreedyAgent(P2), n=50,
               label=f"Minimax(d={depth}, P1) vs Greedy(P2)")
