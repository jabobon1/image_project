import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw


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
        iou = max([get_iou((x, x + step, y, y + step), (xi, xi + step, yi,
                                                        yi + step))
                   for xi, yi in ready])
    else:
        iou = 0
    if iou < 0.3:
        return x, y
    else:
        return random_pos(x_max, y_max, ready, step)


def random_img_generate(base_img, images, idx, img_dir=None, shapes=(200, 50,
                                                                     28, 28),
                        res_path='result'):
    results = []
    base = Image.open(base_img)
    step = shapes[2]
    for im in images:
        pos = random_pos(shapes[0], shapes[1], results, step)
        results.append(pos)
        path = im if img_dir is None else os.path.join(img_dir, im)
        img = Image.open(path)
        base.paste(img, box=pos)

    ind_sorted_x = np.argsort([x for x, _ in results])

    file_name = '_'.join([f"{images[ind].split('_')[0]}" for ind in
                          ind_sorted_x])
    file_name = str(idx) + '_' + file_name
    base.save(os.path.join(res_path, file_name + '.png'))
    df = pd.DataFrame([[x, y, x + step, y + step] for x, y in np.array(
        results)[ind_sorted_x]],
                      columns=['x', 'y',
                               'x2', 'y2'])

    df.to_csv(os.path.join(res_path + '_annot', file_name + '.csv'),
              index=False)


def make_dataset(input_dir, base_img='black_rectangle.png', rand_limit=8,
                 res_path='result'):
    idx = 0
    files = os.listdir(input_dir)
    random.shuffle(files)

    while files:
        images = []
        for _ in range(random.randint(1, rand_limit)):
            try:
                images.append(files.pop())
            except IndexError:
                break

        random_img_generate(base_img, images, idx, img_dir=input_dir,
                            res_path=res_path)
        idx += 1


make_dataset('mnist_2', res_path='result')
