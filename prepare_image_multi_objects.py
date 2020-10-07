import os
import random
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import pandas as pd
import progressbar
from PIL import Image

from make_noize import noise_both, background_random
from useful_functions import check_path


def get_iou(val1, val2):
    y1, y2, x1, x2 = val1
    yi_1, yi_2, xi_1, xi_2 = val2

    x_left = max(x1, xi_1)
    y_top = max(y1, yi_1)
    x_right = min(x2, xi_2)
    y_bottom = min(y2, yi_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # intersection_area = (x_right - x_left) * (y_bottom - y_top)
    intersection_area = abs(x_right - x_left) * abs(y_bottom - y_top)

    bb1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bb2_area = (xi_2 - xi_1 + 1) * (yi_2 - yi_1 + 1)
    try:
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    except ZeroDivisionError:
        print('zerro')
        print(f'intersection_area:{intersection_area}, bb1_area:{bb1_area}, bb2_area:{bb2_area} ')
        print(x1, x2, y1, y2)
        print(xi_1, xi_2, yi_1, yi_2)
        raise ZeroDivisionError
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
        new = [(y1, y2, x1, x2) for y1, y2, x1, x2 in ready]
        iou = [get_iou((y, y + step, x, x + step), posis) for posis in new]
        iou = max(iou)

    else:
        iou = 0
    if iou < 0.2 and all(abs((x + step) - i[2]) > 8 or abs(x - i[3]) > 8 for i in ready):
        return y, y + step, x, x + step
    else:
        return random_pos(x_max, y_max, ready, step)


def add_arr(arr1, arr2, pos=None):
    """
    :param arr1: background image
    :param arr2: second image for superposition
    :param pos: y1, y2, x1, x2
    :return: array, background with arr2 on it
    """
    if pos is None:
        pos = 0, arr2.shape[0], 0, arr2.shape[1]
    y1, y2, x1, x2 = pos
    new_arr = arr1.copy()
    frame = new_arr[y1:y2, x1:x2]
    second = arr2.copy()

    null_indecies = np.argwhere(frame == 0)
    # if all pixels in frame are empty
    if np.multiply(*frame.shape) == null_indecies.shape[0]:
        frame[:] = second
    else:
        for idx in null_indecies:
            frame[tuple(idx)] = second[tuple(idx)]
    return new_arr


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

    file_name = [str(idx)]
    for ind in ind_sorted_x:
        file_name.append(images[ind].split('.')[0].split('_')[-1])

    file_name = '_'.join(file_name)

    df = pd.DataFrame([[y1, y2, x1, x2] for y1, y2, x1, x2 in np.array(
        results)[ind_sorted_x]],
                      columns=['y1', 'y2',
                               'x1', 'x2'])

    bg_gray = cv2.cvtColor(background_random(np.zeros((50, 200, 3))).astype(
        'float32'), cv2.COLOR_BGR2GRAY)
    base = add_arr(base, bg_gray)

    return base, df, file_name





def make_dataset(files, input_dir='emnist',
                 res_path='result'):
    idx = 0
    # files = os.listdir(input_dir)
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
                    filename=os.path.join(check_path(res_path), file_name + '.png'))

        csv.to_csv(os.path.join(check_path(res_path + '_annot'), file_name + '.csv'),
                   index=False)

        idx += 1
        bar.update(max_value - len(files))


if __name__ == '__main__':
    files = os.listdir('emnist')
    random.shuffle(files)

    args = [[] for _ in range(cpu_count())]

    while files:
        for i in range(cpu_count()):
            try:
                args[i].append(files.pop())
            except IndexError:
                pass
    with Pool(cpu_count()) as p:
        p.map(make_dataset, args)
    p.close()
