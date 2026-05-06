import random

import numpy as np

random.seed(0)


class Agent:
    """
    Represents a Reinforcement Learning agent.
    Manages the Q-table and implements action selection (epsilon-greedy) 
    and Q-value updates for both SARSA and Q-Learning algorithms.
    """
    def __init__(self, env, alpha=0.1, gamma=0.5, epsilon=0.1):
        """Initializes the agent with the given environment and learning parameters."""
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # self.q_table[x][y][action] = value
        self.q_table = {}

    def get_q(self, x, y, action):
        """Retrieves the Q-value for a state-action pair, defaulting to 0.0 if unvisited."""
        return self.q_table.get(x, {}).get(y, {}).get(action, 0.0)

    def set_q(self, x, y, action, value):
        """Updates the Q-value for a specific state-action pair in the nested dictionary structure."""
        if x not in self.q_table:
            self.q_table[x] = {}
        if y not in self.q_table[x]:
            self.q_table[x][y] = {}
        self.q_table[x][y][action] = value

    def choose_action(self, state):
        """
        Selects an action based on an epsilon-greedy policy.
        Explores randomly with probability epsilon, otherwise exploits the best known action.
        """
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose random action
            return random.choice(list(self.env.actions.keys()))
        else:
            # Exploitation: choose best action from Q-table
            q_values = [self.get_q(x, y, a) for a in self.env.actions.keys()]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, method="Q-Learning"):
        """
        Updates the Q-table based on the observed reward and subsequent state.
        Supports both Q-Learning (off-policy) and SARSA (on-policy) update rules.
        """
        x, y = state
        next_x, next_y = next_state

        current_q = self.get_q(x, y, action)

        if method == "Q-Learning":
            max_future_q = max([self.get_q(next_x, next_y, a) for a in self.env.actions.keys()])
            new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        else:
            # SARSA
            next_q = self.get_q(next_x, next_y, next_action)
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

        self.set_q(x, y, action, new_q)
