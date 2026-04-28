import cv2
import numpy as np


def abstract_map(img_path, size = (40,40)):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    old_width, old_height = img.shape
    new_width, new_height = size

    abstract_img = np.zeros((new_height,new_width))

    block_height = old_height/new_height
    block_width = old_width/new_width

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



if __name__ == '__main__':
    test = abstract_map("./Input_Maps/map2.bmp")
    print(test)
