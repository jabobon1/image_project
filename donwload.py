import os
import random
from multiprocessing import Pool, cpu_count

import cv2
import progressbar
from captcha.image import ImageCaptcha
from emnist import extract_training_samples
from emnist import list_datasets

from useful_functions import check_path

LETTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

fonts = ['fonts/' + i for i in os.listdir('fonts')]


def download_mnist_dataset():
    print(list_datasets())
    images, labels = extract_training_samples('byclass')

    check_path('emnist')

    bar = progressbar.ProgressBar(max_value=len(images))
    bar.start()
    cnt = 0
    for image, label in zip(images, labels):
        cv2.imwrite(f'emnist/idx_{cnt}_label_{label}.png', image)
        cnt += 1
        bar.update(cnt)


def captcha_generator(file_names: list, prefix: str, directory: str):
    bar = progressbar.ProgressBar(max_value=len(file_names))
    bar.start()

    image = ImageCaptcha(fonts=fonts, width=200, height=50)

    for i, file_name in enumerate(file_names):
        data = ''.join([LETTERS[i] for i in file_name])
        image.generate(data)
        image.write(data, os.path.join(directory, '_'.join(str(l) for l in [f"{i}{prefix}", *file_name]) + '.png'))
        bar.update(i)


if __name__ == '__main__':
    save_path = 'result3'
    check_path(save_path)

    cpu = cpu_count()

    file_names = [[] for _ in range(cpu)]

    it = 50000
    for i in range(it):
        file_names[i % cpu].append([random.randint(0, len(LETTERS) - 1) for _ in range(5)])

    with Pool(cpu) as p:
        p.starmap(captcha_generator,
                  [(symbols, prefix, save_path) for symbols, prefix in zip(file_names, LETTERS[10:])])
