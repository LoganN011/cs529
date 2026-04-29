import numpy as np
from tqdm import tqdm

from Final.Agent import Agent
from Final.Environment import abstract_map, Environment


def train(env, agent, episodes, method="Q-Learning"):
    history = {"steps": [], "rewards": []}

    for episode in tqdm(range(episodes), desc=f"Training {method}"):
        state = (0, 0)
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Take action and observe environment
            next_state, reward = env.step(state, action)

            # Choose next action (needed for SARSA update and next step)
            next_action = agent.choose_action(next_state)

            # Update Q-table using the Agent's internal logic
            agent.update(state, action, reward, next_state, next_action, method=method)

            # Transition
            state = next_state
            action = next_action  # For SARSA, we use the action we just picked

            total_reward += reward
            steps += 1

            if state == env.target or steps > 200:
                done = True

        history["steps"].append(steps)
        history["rewards"].append(total_reward)

    return history



if __name__ == '__main__':
    grid = abstract_map("./Input_Maps/map2.bmp", size=(40, 40))
    target_pos = (39, 39)
    env = Environment(grid, target=target_pos)

    agent = Agent(env, alpha=0.1, gamma=0.5, epsilon=0.1)

    num_episodes = 20000

    q_history = train(env, agent, episodes=num_episodes, method="Q-Learning")

    avg_start = sum(q_history["steps"][:50]) / 50
    avg_end = sum(q_history["steps"][-50:]) / 50

    print(f"\nTraining Complete!")
    print(f"Average steps (first 50 episodes): {avg_start}")
    print(f"Average steps (last 50 episodes): {avg_end}")