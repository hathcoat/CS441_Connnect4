#players.py
import random
from minimax import choose_move

class RandomAgent:
    def __init__(self, pid):
        self.pid = pid
    def move(self, env):
        return random.choice(env.legal_actions())

class GreedyAgent:
    """
    1-ply lookahead:
    - If I can win in one, play it.
    - Else if opponent can win next, block it.
    - Else random valid move.
    """
    def __init__(self, pid):
        self.pid = pid

    def move(self, env):
        me = self.pid
        opp = 1 if me == -1 else -1
        valid = env.legal_actions()

        #try winning move
        for c in valid:
            e = _clone(env); e.step(c)
            if e.winner == me and e.done:
                return c

        #try block opponent's immediate win
        for c in valid:
            e = _clone(env); e.step(c)
            #simulate opponent’s best response: if opp can win anywhere next, block now
            for oc in e.legal_actions():
                e2 = _clone(e); e2.step(oc)
                if e2.winner == opp and e2.done:
                    return c

        return random.choice(valid)

class MinimaxAgent:
    def __init__(self, pid, depth=4):
        self.pid = pid
        self.depth = depth
    def move(self, env):
        return choose_move(env, self.depth, me=self.pid)

#--- small helper to avoid code dup
import copy
def _clone(env):
    return copy.deepcopy(env)
