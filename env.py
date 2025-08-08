#CS 441         August 4th, 2025

import numpy as np

ROWS, COLS = 6, 7

EMPTY, P1, P2 = 0, 1, -1 #RL defaults to P1, opponent is P2

class Connect4Env:
    def __init__(self, starting_player=P1):
        self.starting_player = starting_player
        self.reset()

    #Fresh board, sets the term, clears clags.
    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = self.starting_player
        self.done = False 
        self.winner = 0
        return self._state()
    
    #Action is legal if the top cell is empty.
    def legal_actions(self):
        return [c for c in range(COLS) if self.board[0, c] == EMPTY]
    
    #Applies current player's action, switches player, compute terminal
    def step(self, action):
        #Make sure game is in play and action is legal.
        assert not self.done, "Game is over"
        assert action in self.legal_actions(), "Illegal action"

        #Apply gravity
        row = self._drop_row(action)
        self.board[row, action] = self.current_player

        #Terminal Check for winner or draw, else switch player
        if self._is_win(self.current_player):
            self.done, self.winner = True, self.current_player
        elif not self.legal_actions():
            self.done, self.winner = True, 0 #Draw
        else:
            self.current_player = P1 if self.current_player == P2 else P2

        return self._state(), self.done #Return signature

    #Find where checker lands, raise error if col is full 
    def _drop_row(self, col):
        for r in range(ROWS -1, -1, -1):
            if self.board[r, col] == EMPTY:
                return r
        raise ValueError("Column Full")
    
    def _is_win(self, player):
        b = self.board

        #Check for horizontal win
        for r in range(ROWS):
            for c in range (COLS -3):
                if np.all(b[r, c:c+4] == player): 
                    return True

        #Check for vertical win 
        for r in range(ROWS - 3):
            for c in range(COLS):
                if np.all(b[r:r+4, c] == player):
                    return True
                
        #Check for diagonal down-right win
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r+i, c+i] == player for i in range(4)):
                    return True

        #Chec for diagonal up-right win
        for r in range(3, ROWS):
            for c in range(COLS -3):
                if all(b[r-i, c+i] == player for i in range(4)):
                    return True
        
        return False

    #Return a hashable state key for Q-learning 
    def _state(self):
        return tuple(self.board.reshape(-1).tolist())
    
    def render(self):
        print(np.where(self.board==P1, 'X', np.where(self.board==P2,'O','.')))