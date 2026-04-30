import numpy as np
import random

random.seed(0)

class Agent:
    def __init__(self, env, alpha=0.1, gamma=0.5, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # self.q_table[x][y][action] = value
        self.q_table = {}

    def get_q(self, x, y, action):
        """Helper to get Q-value, defaulting to 0.0 if not yet visited."""
        return self.q_table.get(x, {}).get(y, {}).get(action, 0.0)

    def set_q(self, x, y, action, value):
        """Helper to set Q-value in the nested dictionary structure."""
        if x not in self.q_table:
            self.q_table[x] = {}
        if y not in self.q_table[x]:
            self.q_table[x][y] = {}
        self.q_table[x][y][action] = value

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose random action
            return random.choice(list(self.env.actions.keys()))
        else:
            # Exploitation: choose best action from Q-table
            q_values = [self.get_q(x, y, a) for a in self.env.actions.keys()]
            return np.argmax(q_values)


    def update(self,state, action, reward, next_state, next_action, method="Q-Learning"):
        x, y = state
        next_x, next_y = next_state

        current_q = self.get_q(x, y, action)

        if method == "Q-Learning":
            max_future_q = max([self.get_q(next_x, next_y, a) for a in self.env.actions.keys()])
            new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        else:
            #SARSA
            next_q = self.get_q(next_x, next_y, next_action)
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)


        self.set_q(x, y, action, new_q)