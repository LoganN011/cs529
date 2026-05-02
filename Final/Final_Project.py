import csv
import os
import time

import numpy as np
from tqdm import tqdm

from Agent import Agent
from Environment import abstract_map, Environment


def train(env, agent, episodes, method="Q-Learning",strategy="S1", patience=1000):
    history = {"steps": [], "rewards": []}
    consecutive_successes = 0

    for episode in tqdm(range(episodes), desc=f"Training {method}"):
        state = env.get_random_start()
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        env.reset()
        done = False

        while not done:
            next_state, reward = env.step(state, action,strategy=strategy)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, method=method)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if state == env.target or steps > 200:
                done = True

        history["steps"].append(steps)
        history["rewards"].append(total_reward)

        # if state == env.target:
        #     consecutive_successes += 1
        # else:
        #     consecutive_successes = 0
        #
        # if consecutive_successes >= patience:
        #     tqdm.write(f" Converged at episode {episode}")
        #     break

    return history


def run_experiment(map_path, method, epsilon, gamma, test_name, strategy="S1"):
    """Runs a single training/testing session and returns performance metrics."""
    # Setup environment
    grid_size = (20, 20) if "map1" in map_path else (40, 40)
    target = (19,19) if "map1" in map_path else (39, 39)
    grid = abstract_map(map_path, size=grid_size)
    env = Environment(grid, target=target)
    agent = Agent(env, alpha=0.1, gamma=gamma, epsilon=epsilon)

    # Measure Training Time
    start_time = time.time()
    history = train(env, agent, episodes=50000, method=method,strategy=strategy)
    end_time = time.time()

    actual_episodes = len(history["rewards"])

    # Performance Evaluation (Accuracy logic)
    valid_paths = 0
    total_starts = 0
    for y in range(env.height):
        for x in range(env.width):
            if env.grid[y, x] != 0:  # Check white pixels only
                total_starts += 1
                if is_path_valid(env, agent, (x, y)):
                    valid_paths += 1

    accuracy = (valid_paths / total_starts) * 100
    return {
        "Test_Name": test_name,
        "Map": os.path.basename(map_path),
        "Method": method,
        "Epsilon": epsilon,
        "Gamma": gamma,
        "Strategy": strategy,
        "Episodes": actual_episodes,
        "Time": round(end_time - start_time, 2),
        "Accuracy": round(accuracy, 2)
    }


def is_path_valid(env, agent, start_pos, max_steps=200):
    """Greedy evaluation to see if the agent reaches the goal without hitting obstacles."""
    state = start_pos
    for _ in range(max_steps):
        if state == env.target:
            return True
        # Choose best action (epsilon=0 for testing)
        x, y = state
        q_vals = [agent.get_q(x, y, a) for a in env.actions.keys()]
        action = np.argmax(q_vals)

        next_state, _ = env.step(state, action)
        if next_state == state:  # Hit obstacle
            return False
        state = next_state
    return False


if __name__ == "__main__":
    results = []
    maps = ["./Input_Maps/map1.bmp", "./Input_Maps/map2.bmp", "./Input_Maps/map3.bmp", "./Input_Maps/map4.bmp"]
    grids = []

    # --- COMPARISON 1: SARSA vs Q-Learning (Varying Complexity) ---
    print("\n--- Task 6.1: SARSA vs Q-Learning ---")
    for m in maps:
        for method in ["SARSA", "Q-Learning"]:
            res = run_experiment(m, method, 0.1, 0.5, "Complexity_Test")
            results.append(res)

    # --- COMPARISON 2: Varying Epsilon (Use map1 for control) ---
    print("\n--- Task 6.2: Varying Epsilon ---")
    for eps in [0.0, 0.5, 1.0]:
        for method in ["SARSA", "Q-Learning"]:
            res = run_experiment(maps[-1], method, eps, 0.5, "Epsilon_Test")
            results.append(res)

    # --- COMPARISON 3: Varying Gamma ---
    print("\n--- Task 6.3: Varying Gamma ---")
    for gam in [0.1, 0.5, 1.0]:
        for method in ["SARSA", "Q-Learning"]:
            res = run_experiment(maps[-1], method, 0.5, gam, "Gamma_Test")
            results.append(res)

    # --- COMPARISON 4: Varying Strategy ---
    print("\n--- Task 6.4: Varying Reward Strategy ---")
    best_eps = 0.5
    best_gam = 1

    for method in ["SARSA", "Q-Learning"]:
        for strat in ["S1", "S2"]:
            res = run_experiment(maps[-1],method,best_eps,best_gam,"Strategy_Comparison",strategy=strat)
            results.append(res)

    # --- SAVE RESULTS ---
    keys = results[0].keys()
    with open('task6_results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print("\nTesting complete. Results saved to task6_results.csv")
