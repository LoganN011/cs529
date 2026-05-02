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


def run_experiment(map_path, method, epsilon, gamma, test_name, strategy="S1",save_model=False):
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
    efficiency_scores = []
    for y in range(env.height):
        for x in range(env.width):
            if env.grid[y, x] != 0:  # Check white pixels only
                total_starts += 1
                valid , path_len =is_path_valid(env, agent, (x, y))
                if valid:
                    valid_paths += 1
                    if path_len > 0:
                        efficiency_scores.append((env.get_optimal_path_length((x,y)) / path_len) * 100)


    accuracy = (valid_paths / total_starts) * 100
    avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0.0
    if save_model:
        return {
            "Test_Name": test_name,
            "Map": os.path.basename(map_path),
            "Method": method,
            "Epsilon": epsilon,
            "Gamma": gamma,
            "Strategy": strategy,
            "Episodes": actual_episodes,
            "Average Efficiency": round(avg_efficiency, 2),
            "Time": round(end_time - start_time, 2),
            "Accuracy": round(accuracy, 2)
        }, agent

    return {
        "Test_Name": test_name,
        "Map": os.path.basename(map_path),
        "Method": method,
        "Epsilon": epsilon,
        "Gamma": gamma,
        "Strategy": strategy,
        "Episodes": actual_episodes,
        "Average Efficiency": round(avg_efficiency, 2),
        "Time": round(end_time - start_time, 2),
        "Accuracy": round(accuracy, 2)
    }


def is_path_valid(env, agent, start_pos, max_steps=200):
    """Greedy evaluation. Returns (Success, Path_Length)."""
    state = start_pos
    path_length = 0
    for _ in range(max_steps):
        if state == env.target:
            return True, path_length

        x, y = state
        q_vals = [agent.get_q(x, y, a) for a in env.actions.keys()]
        action = np.argmax(q_vals)

        next_state, _ = env.step(state, action)
        path_length += 1

        if next_state == state:  # Hit obstacle or stayed in place
            return False, path_length
        state = next_state

    return False, path_length


def plot_path(agent, max_steps=200, start=(0, 0),title=None):
    env = agent.env
    state = start # (x, y)
    path = [state]

    for _ in range(max_steps):
        if state == env.target:
            break

        x, y = state
        # Get Q-values using the agent's (x, y) training format
        q_values = [agent.get_q(x, y, a) for a in env.actions.keys()]
        action = np.argmax(q_values)

        next_state, _ = env.step(state, action)

        if next_state == state: # Hit a wall
            break

        state = next_state
        path.append(state)

    if title:
        env.plot_map(path=path,title=title)
    else:
        env.plot_map(path=path)


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
            res,a = run_experiment(maps[-1],method,best_eps,best_gam,"Strategy_Comparison",strategy=strat,save_model=True)
            results.append(res)

            plot_path(a, start=(0, 0),title=f"{strat}-{method}")

    # --- COMPARISON 5: Hard Maps ---
    print("\n--- Task 6.5: Harder Maps ---")

    advanced_maps = ["./Input_Maps/Advanced_Maps/maze1.bmp","./Input_Maps/Advanced_Maps/U-Map.bmp"]

    for strat in ["S1","S2"]:
        for method in ["SARSA", "Q-Learning"]:
            for m in advanced_maps:
                res,a = run_experiment(m,method,0.5,1,"Hard Maps",strategy=strat,save_model=True)
                results.append(res)
                plot_path(a, start=(0, 0),title=f"{strat}-{os.path.splitext(os.path.basename(m))[0]}-{method}")

    # --- SAVE RESULTS ---
    keys = results[0].keys()
    with open('task6_results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print("\nTesting complete. Results saved to task6_results.csv")
