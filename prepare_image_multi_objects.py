import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

from image_project.make_noize import noise_both, background_random, rnd_color
from image_project.write_img import add_arr
import progressbar


def get_iou(val1, val2):
    x1, x2, y1, y2 = val1
    xi_1, xi_2, yi_1, yi_2 = val2

    x_left = max(x1, xi_1)
    y_top = max(y1, y1)
    x_right = min(x2, xi_2)
    y_bottom = min(y2, yi_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (xi_2 - xi_1) * (yi_2 - yi_1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def collide_images(base_img, second_img, pos=(0, 0), dir='mnist'):
    im1 = Image.open(base_img)
    im2 = Image.open(os.path.join(dir, second_img))
    im1.paste(im2, box=pos)
    return im1


def random_pos(x_max, y_max, ready: list, step=28):
    """Creates random position with intersections between old pos < 0.3"""
    x_m, y_m = x_max - step, y_max - step
    x = random.randint(0, x_m)
    y = random.randint(0, y_m)
    if ready:
        new = [(x1, x2, y1, y2) for y1, y2, x1, x2 in ready]
        iou = max([get_iou((x, x + step, y, y + step), posis) for posis in new])
    else:
        iou = 0
    if iou < 0.3:
        return y, y + step, x, x + step
    else:
        return random_pos(x_max, y_max, ready, step)


def random_img_generate(images, idx, img_dir=None,
                        shapes=(200, 50, 28, 28),
                        ):
    results = []
    base = np.zeros((50, 200))

    step = shapes[2]
    for im in images:
        pos = random_pos(shapes[0], shapes[1], results, step)
        results.append(pos)
        path = im if img_dir is None else os.path.join(img_dir, im)
        img = cv2.imread(path, -1)
        # img[img > 20] = random.randint(20, 255)
        base = add_arr(base, img, pos=pos)

    ind_sorted_x = np.argsort([x[2] for x in results])

    file_name = '_'.join([f"{images[ind].split('_')[0]}" for ind in
                          ind_sorted_x])
    file_name = str(idx) + '_' + file_name
    df = pd.DataFrame([[y1, y2, x1, x2] for y1, y2, x1, x2 in np.array(
        results)[ind_sorted_x]],
                      columns=['y1', 'y2',
                               'x1', 'x2'])

    bg_gray = cv2.cvtColor(background_random(np.zeros((50, 200, 3))).astype(
        'float32'), cv2.COLOR_BGR2GRAY)
    base = add_arr(base, bg_gray)

    return base, df, file_name


def make_dataset(input_dir,
                 res_path='result'):
    idx = 0
    files = os.listdir(input_dir)
    random.shuffle(files)
    max_value = len(files)
    bar = progressbar.ProgressBar(max_value=max_value)
    bar.start()
    while files:
        images = []
        for _ in range(5):
            try:
                images.append(files.pop())
            except IndexError:
                break

        image, csv, file_name = random_img_generate(images, idx,
                                                    img_dir=input_dir)

        final = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2RGB)
        final = noise_both(final)

        cv2.imwrite(img=final,
                    filename=os.path.join(res_path, file_name + '.png'))

        csv.to_csv(os.path.join(res_path + '_annot', file_name + '.csv'),
                   index=False)

        idx += 1
        bar.update(max_value - len(files))


if __name__ == '__main__':
    make_dataset('mnist_2', res_path='result')
