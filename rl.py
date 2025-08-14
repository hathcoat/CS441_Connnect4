import random
import pickle
from collections import defaultdict
from env import COLS #For action space size

class QLearningAgent:
    """
    Tabular Q-learning agent

    This agent learns an action-value function Q(s, a) through trial and
    error. It updates its estimates using the temporal difference method.

    Attributes:
        Q: Maps state and action tuples to Q values.
        epsilon: Probability of selecting an explorative random action.
        epsilon_min: Floor value to avoid complete exploitation.
        epsilon_decay: Decay factor of epsilon per episode.
        alpha: Learning rate (how much to update Q-vals per step)
        gamma: Discount factor
    """

    def __init__(self, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 alpha=0.15, gamma=0.9):
        #Initilize Q-vals to 0
        self.Q = defaultdict(float)

        #greedy exploratino params
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        #Learning params
        self.alpha = alpha
        self.gamma = gamma

        self.total_updates = 0

    def select_action(self, state, legal_actions):
        """
            Selects an action using epsilon-greedy policy

            Args:
                state: Tuple with the current game state
                legal_actions: List of valid column indicies

            Rets:
                Select action (int for col index)
        """
        if random.random() < self.epsilon:
            #Exploration
            return random.choice(legal_actions)
        
        #Exploitation
        best_q = float("-inf")
        best_actions = []

        for a in legal_actions:
            q = self.Q[(state, a)]
            if q > best_q:
                best_q = q
                best_actions = [a]
            elif q == best_q:
                best_actions.append(a)
        return random.choice(best_actions) if best_actions else random.choice(legal_actions)
    
    def update(self, state, action, reward, next_state, legal_next):
        """
        Peforms the Q-learning TD update for a single step.

        Args:
            state: current tuple state
            action: action taken in the current move (col number)
            reward: Reward received.
            next_state: Next state (None if terminal)
            legal_next: list of legal actions in the next state.
        """
        if action is None:
            return
        
        current_q = self.Q[(state, action)]
        
        if next_state is None:
            # Terminal state
            target = reward
        else:
            # Find maximum Q-value for next state
            if legal_next:
                max_next_q = max(self.Q[(next_state, a)] for a in legal_next)
            else:
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q
        
        # Q-learning update with clipping to prevent extreme values
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error
        
        # Optional: clip Q-values to reasonable range
        self.Q[(state, action)] = max(-50.0, min(50.0, new_q))
        
        self.total_updates += 1

    def decay_epsilon(self):
        """
        Decay the exploration rate to gradually shift from exploration to exploitation 
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        """
        Get training statistics.
        """
        return {
            'q_table_size': len(self.Q),
            'total_updates': self.total_updates,
            'current_epsilon': self.epsilon,
            'avg_q_value': sum(self.Q.values()) / len(self.Q) if self.Q else 0
        }
    
    def get_action_values(self, state, legal_actions):
        """
        Get Q-values for all legal actions in a given state.
        Useful for analysis and debugging.
        """
        return {action: self.Q[(state, action)] for action in legal_actions}
    
    def set_epsilon(self, epsilon):
        """
        Manually set epsilon (useful for evaluation).
        """
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def save_q_table(self, filepath):
        """
        Save Q-table to a file (alternative to pickle in main.py).
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q), f)
    
    def load_q_table(self, filepath):
        """
        Load Q-table from a file.
        """
        import pickle
        with open(filepath, 'rb') as f:
            q_dict = pickle.load(f)
            self.Q = defaultdict(lambda: 0.1)
            self.Q.update(q_dict)