import random

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

        reward = self._get_reward(new_state, hit, strategy)

        return new_state, reward  # add more things here???

    def _get_reward(self, state, hit, strategy):
        if strategy == "S1":
            if state == self.target:
                return 100
            if hit:
                return -100
            else:
                return -1
        elif strategy == "S2":  # Todo make a better strategy for S2
            if state == self.target:
                return 100
            if hit:
                return -100
            else:
                return -1

        return 0

    def plot_map(self):
        plt.imshow(self.grid, cmap='gray', extent=(0, self.width, 0, self.height))
        plt.scatter(self.target[1] + .5, self.height - self.target[0] - .5, c='red', marker='*')
        plt.title("Abstracted Map Environment")

        plt.xticks(np.arange(-5, self.width + 6, 5))
        plt.yticks(np.arange(-5, self.height + 6, 5))
        plt.xticks(np.arange(0, self.width + 1, 1), minor=True)
        plt.yticks(np.arange(0, self.height + 1, 1), minor=True)
        plt.grid(which='both', color='gray', linestyle='-', linewidth=0.2)

        plt.show()

    def get_random_start(self):
        """Function to get a random starting position that is valid"""
        while True:
            rx = random.randint(0, self.width - 1)
            ry = random.randint(0, self.height - 1)
            if self.grid[rx][ry] == 1:
                return (rx, ry)


if __name__ == '__main__':
    test = abstract_map("./Input_Maps/map4.bmp")
    env = Environment(test)
    env.plot_map()
