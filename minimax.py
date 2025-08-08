# minimax.py
import copy
import math
import numpy as np
from env import Connect4Env, P1, P2, EMPTY, ROWS, COLS

def choose_move(env: Connect4Env, depth: int, me: int) -> int:
    """
    Top-level helper: returns the best column for player `me` using minimax+alpha-beta.
    """
    _, col = _minimax(env, depth, -math.inf, math.inf, maximizing=(env.current_player == me), me=me)
    return col

def _minimax(env: Connect4Env, depth: int, alpha: float, beta: float, maximizing: bool, me: int):
    """
    Returns (score, column). Uses deep copies of env to avoid needing an undo().
    """
    #terminal or depth limit: evaluate
    if depth == 0 or env.done:
        return _evaluate(env, me), None

    valid_moves = env.legal_actions()
    if not valid_moves:
        #draw if no moves
        return _evaluate(env, me), None

    best_col = valid_moves[0]

    if maximizing:
        value = -math.inf
        for col in valid_moves:
            child = copy.deepcopy(env)
            child.step(col)  #applies move & flips current_player internally
            score, _ = _minimax(child, depth - 1, alpha, beta, False, me)
            if score > value:
                value, best_col = score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  #beta cut-off
        return value, best_col
    else:
        value = math.inf
        for col in valid_moves:
            child = copy.deepcopy(env)
            child.step(col)
            score, _ = _minimax(child, depth - 1, alpha, beta, True, me)
            if score < value:
                value, best_col = score, col
            beta = min(beta, value)
            if alpha >= beta:
                break  #alpha cut-off
        return value, best_col

# -------------------------
# Evaluation
# -------------------------

def _evaluate(env: Connect4Env, me: int) -> int:
    """
    Heuristic score from the perspective of player `me`.
    Positive is good for `me`, negative is good for the opponent.
    """
    board = env.board  #shape (ROWS, COLS)
    opp = P1 if me == P2 else P2

    score = 0

    #Center column preference
    center_col = COLS // 2
    center_array = board[:, center_col]
    score += 3 * int(np.sum(center_array == me))

    #score all possible 4-length windows
    for r in range(ROWS):
        for c in range(COLS - 3):
            window = list(board[r, c:c+4])
            score += _score_window(window, me, opp)

    for r in range(ROWS - 3):
        for c in range(COLS):
            window = list(board[r:r+4, c])
            score += _score_window(window, me, opp)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i, c+i] for i in range(4)]
            score += _score_window(window, me, opp)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r-i, c+i] for i in range(4)]
            score += _score_window(window, me, opp)

    #Big terminal bonuses if game is over
    if env.done:
        if env.winner == me:
            score += 100000
        elif env.winner != 0:  #opponent win
            score -= 100000
        #draw (winner == 0) → no extra

    return score

def _score_window(window, me, opp) -> int:
    """
    Score a 4-cell window.
    """
    me_count = window.count(me)
    opp_count = window.count(opp)
    empty_count = window.count(EMPTY)

    #Wins / threats
    if me_count == 4:
        return 10000
    if me_count == 3 and empty_count == 1:
        return 120
    if me_count == 2 and empty_count == 2:
        return 12

    #Block opponent
    if opp_count == 3 and empty_count == 1:
        return -150
    if opp_count == 4:
        return -10000

    return 0
