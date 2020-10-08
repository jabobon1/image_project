import os
import random
import progressbar
from multiprocessing import Pool, cpu_count

from captcha.image import ImageCaptcha
from useful_functions import check_path

LETTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

fonts = ['fonts/' + i for i in os.listdir('fonts')]


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
    # directory to save captcha
    save_path = check_path('result3')

    cpu = cpu_count()

    file_names = [[] for _ in range(cpu)]
    # number of images to generate
    it = 50000
    for i in range(it):
        file_names[i % cpu].append([random.randint(0, len(LETTERS) - 1) for _ in range(5)])

    with Pool(cpu) as p:
        p.starmap(captcha_generator,
                  [(symbols, prefix, save_path) for symbols, prefix in zip(file_names, LETTERS[10:])])
