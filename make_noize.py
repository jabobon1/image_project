import re

import cv2
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt
from glob import glob
import random

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def ret_array(img):
    if type(img) != np.ndarray:
        img = np.array(img)
    return img


def make_noize(img, mode='gaussian'):
    return random_noise(ret_array(img), mode=mode, seed=None, clip=False)


def show_img(img):
    if isinstance(img, np.array):
        mult = 255 if img.max() == 255 else 1
        img = Image.fromarray((img * mult).astype(np.uint8))
    img.show()


def rnd_color(size=3):
    return tuple(random.randint(0, 255) for _ in range(size))


def add_line(y1, x1, y2, x2, img, color=None, thickness=None):
    color = color or rnd_color()
    thickness = thickness or random.randint(1, 2)

    return cv2.line(img, (y1, x1), (y2, x2), color, thickness)


def get_rnd_pos(v_max, v_min=0):
    v_min = random.randint(v_min, v_max - 1)
    return v_min, random.randint(v_min, v_max)


def add_random_lines(img, length=6):
    y2, x2, *_ = ret_array(img).shape
    for i in range(random.randint(2, length)):
        y1, yn2 = get_rnd_pos(y2)
        x1, xn2 = get_rnd_pos(x2)
        img = add_line(x1, y1, xn2, yn2, img)

    return img


def noise_both(img):
    return make_noize(add_random_lines(img))


def find_pos(file_name):
    x1, y1 = [int(i) for i in
              re.findall(r'pos_(\d+_\d+)_idx', file_name)[0].split('_')]
    x2, y2 = x1 + 28, y1 + 28
    return [y1, x1, y2, x2]


def generate_img(path, batch_size=10):
    files = glob(path)
    random.shuffle(files)
    func = lambda img: cv2.cvtColor(img.astype('uint16'), cv2.COLOR_RGB2GRAY)
    return ((func(noise_both(cv2.imread(im))).tolist(), find_pos(im))
            for
            im in files)
    # return (
    #     (cv2.cvtColor(noise_both(cv2.imread(im)), cv2.COLOR_RGB2GRAY),
    #      find_pos(im))
    #     for im in files)


def background_random(img):
    img[::] = np.random.randint(1, 255, 3)
    return img


if __name__ == '__main__':
    # 1/ -1: color mode; 0: gray mode
    img = cv2.imread('new_set/0_pos_0_10_idx_13583.png', 1)

    img = add_random_lines(img)
    img = make_noize(img)
    cv2.imshow('image', img)

    gen = generate_img('new_set/*.png')
    new = gen.__next__()
    print(new)

    img = new[0]
