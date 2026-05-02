import random
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt

random.seed(0)

def abstract_map(img_path, size=(40, 40)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    old_width, old_height = img.shape
    new_width, new_height = size

    abstract_img = np.zeros((new_height, new_width))

    block_height = old_height / new_height
    block_width = old_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            y_start = int(i * block_height)
            y_end = int((i + 1) * block_height) if i < new_height - 1 else old_height

            x_start = int(j * block_width)
            x_end = int((j + 1) * block_width) if j < new_width - 1 else old_width

            block = img[y_start:y_end, x_start:x_end]

            if np.any(block == 0):
                abstract_img[i, j] = 0
            else:
                abstract_img[i, j] = 1

    return abstract_img


class Environment:
    def __init__(self, grid, target=(39, 39)):
        self.grid = grid
        self.target = target
        self.width, self.height = self.grid.shape
        self.actions = {0: "up", 1: "down", 2: "right", 3: "left"}
        self.visited_states = set()

    def reset(self):
        """Resets the environment for a new episode."""
        self.visited_states = set()  # Clear memory for S2

    def step(self, state, action, strategy="S1"):
        x, y = state
        next_x, next_y = state

        new_state = (next_x, next_y)

        if action == 0:
            next_y += 1
        elif action == 1:
            next_y -= 1
        elif action == 2:
            next_x += 1
        elif action == 3:
            next_x -= 1

        hit = False
        if not (0 <= next_x < self.width and 0 <= next_y < self.height):
            hit = True
            new_state = (x, y)
        elif self.grid[next_y, next_x] == 0:  # Check y then x
            hit = True
            new_state = (x, y)
        else:
            new_state = (next_x, next_y)

        reward = self._get_reward(state,new_state, hit, strategy)
        self.visited_states.add(tuple(new_state))

        return new_state, reward

    def _get_reward(self,old, state, hit, strategy):
        if strategy == "S1":
            if state == self.target:
                return 100
            if hit:
                return -100
            else:
                return -1
        elif strategy == 'S2':
            if hit:
                return -100
            if state == self.target:
                return 100
            reward = -1

            # 2. Distance improvement (Manhattan)
            old_dist = abs(old[0] - self.target[0]) + abs(old[1] - self.target[1])
            new_dist = abs(state[0] - self.target[0]) + abs(state[1] - self.target[1])


            reward += (old_dist - new_dist)*10

            # 3. Visited state penalty
            # if state in self.visited_states:
            #     reward -= 2  # Heavy penalty for backtracking

            return reward
        return 0

    def plot_map(self, path=None,title="Abstracted Map Environment"):
        fig, ax = plt.subplots(figsize=(8, 8))


        ax.imshow(self.grid, cmap='gray', origin='lower', extent=(0, self.width, 0, self.height))

        ax.scatter(self.target[0] + 0.5, self.target[1] + 0.5, c='red', marker='*', s=150, label='Target', zorder=5)

        if path and len(path) > 0:
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]

            ax.plot(path_x, path_y, color='cyan', linewidth=2, label='Agent Path', zorder=3)
            ax.scatter(path_x[0], path_y[0], color='blue', marker='o', s=50, label='Start', zorder=4)

        ax.set_xticks(np.arange(0, self.width + 1, 5))
        ax.set_yticks(np.arange(0, self.height + 1, 5))

        ax.set_xticks(np.arange(0, self.width + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, self.height + 1, 1), minor=True)

        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        ax.set_title(title)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.legend(loc='best')

        plt.show()

    def get_random_start(self):
        """Function to get a random starting position that is valid"""
        while True:
            rx = random.randint(0, self.width - 1)
            ry = random.randint(0, self.height - 1)
            if self.grid[rx][ry] == 1:
                return (rx, ry)

    def get_optimal_path_length(self, start_pos):
        """
        Uses BFS to find the absolute shortest path length from start_pos to target.
        Returns float('inf') if the target is unreachable.
        """
        queue = deque([(start_pos, 0)])
        visited = {start_pos}

        while queue:
            (curr_x, curr_y), dist = queue.popleft()

            if (curr_x, curr_y) == self.target:
                return dist

            # Check all 4 possible actions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_x + dx, curr_y + dy

                # Boundary and obstacle check
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] == 1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), dist + 1))

        return float('inf')


if __name__ == '__main__':
    test = abstract_map("./Input_Maps/map4.bmp")
    env = Environment(test)
    env.plot_map()
