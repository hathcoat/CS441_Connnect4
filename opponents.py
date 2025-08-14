import math
import random
import numpy as np
from env import ROWS, COLS, EMPTY, P1, P2, Connect4Env
from minimax import choose_move

class RandomOpponent():
    def act(self, env_state, board_array, legal_actions, player_token):
        return np.random.choice(legal_actions)

class GreedyOpponent():
    def act(self, env_state, board_array, legal_actions, player_token):
        #1. Try to win this move
        for a in legal_actions:
            temp_board = board_array.copy()
            self._drop(temp_board, a, player_token)
            if self._is_win(temp_board, player_token):
                return a
            
        #2. Try to block opponent's win
        opponent = -player_token
        for a in legal_actions:
            temp_board = board_array.copy()
            self._drop(temp_board, a, opponent)
            if self._is_win(temp_board, opponent):
                return a
            
        #3. Pick center column if possible
        center = 3
        if center in legal_actions:
            return center

        #4. Pick random fallback
        return random.choice(legal_actions)      
    
    def _drop(self, board, col, player):
        for r in range(ROWS-1, -1, -1):
            if board[r, col] == EMPTY:
                board[r, col] = player
                break

    def _is_win(self, b, player):
        #Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(b[r, c+i] == player for i in range(4)): return True
        #Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(b[r+i, c] == player for i in range(4)): return True
        #Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r+i, c+i] == player for i in range(4)): return True
        #Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(b[r-i, c+i] == player for i in range(4)): return True
        return False
    
class FrozenQOpponent:
    """
    An opponent that plays greedily from a fronzen Q-table snapshot
    Falls back to random when a state action par is unseen.
    """

    def __init__(self, frozen_Q, fallback =None):
        self.frozen_Q = frozen_Q # Plain dict {(state, action): qvalue}
        self.fallback = fallback # GreedyOpponent or RandomOpponent

    def act(self, env_state, board_array, legal_actions, player_token):
        best_a, best_q = None, float("-inf")

        for a in legal_actions:
            q = self.frozen_Q.get((env_state, a), None)
            if q is not None and q > best_q:
                best_q, best_a = q, a

        if best_a is not None:
            return best_a
        if self.fallback:
            return self.fallback.act(env_state, board_array, legal_actions, player_token)

        return random.choice(legal_actions)
    
class MinimaxOpponent:
    """
    Opponent with the same .act(...) interface as others, driven by minimax.
    """
    def __init__(self, depth=4):
        self.depth = depth

    def act(self, env_state, board_array, legal_actions, player_token):
        # Rebuild a temporary env from the current board & player
        env = Connect4Env(starting_player=player_token)
        env.board = board_array.copy()
        env.current_player = player_token
        env.done = False
        env.winner = 0

        col = choose_move(env, self.depth, me=player_token)
        if col in legal_actions:
            return col

        # Safety fallback
        try:
            return GreedyOpponent().act(env_state, board_array, legal_actions, player_token)
        except Exception:
            return random.choice(legal_actions)