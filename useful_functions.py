import os
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

LETTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_result(lim, x_data, y_pred, y_true):
    res_p = [''.join([LETTERS[i.argmax()] for i in y_pred[::, x]]) for x in range(lim)]
    true_p = [''.join([LETTERS[i.argmax()] for i in y_true[::, x]]) for x in range(lim)]

    print(res_p)
    print(true_p)

    fig = plt.figure(figsize=(10, 10))
    columns = 5
    rows = lim // columns
    for i in range(1, columns * rows + 1):
        img = (x_data[:lim][i - 1] * 255).astype(np.uint8)
        fig.add_subplot(rows, columns, i).set_title('pred:' + res_p[i - 1] + '\ntrue:' + true_p[i - 1])
        plt.imshow(img)
    plt.savefig(f'result_{datetime.now().time()}.png')
    plt.show()


def plot_history(data):
    # plot some data
    plt.figure()
    plt.plot(data.history['loss'], label='loss')
    plt.plot(data.history['digit1_loss'], label='digit1_loss')
    plt.plot(data.history['digit2_loss'], label='digit2_loss')
    plt.plot(data.history['digit3_loss'], label='digit3_loss')
    plt.plot(data.history['digit4_loss'], label='digit4_loss')
    plt.plot(data.history['digit5_loss'], label='digit5_loss')
    plt.plot(data.history['val_digit1_loss'], label='val_digit1_loss')
    plt.plot(data.history['val_digit2_loss'], label='val_digit2_loss')
    plt.plot(data.history['val_digit3_loss'], label='val_digit3_loss')
    plt.plot(data.history['val_digit3_loss'], label='val_digit3_loss')
    plt.plot(data.history['val_digit4_loss'], label='val_digit4_loss')
    plt.plot(data.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(f'loss_{datetime.now().time()}.png')

    # accuracies
    plt.figure()
    plt.plot(data.history['digit1_accuracy'], label='digit1_accuracy')
    plt.plot(data.history['digit2_accuracy'], label='digit2_accuracy')
    plt.plot(data.history['digit3_accuracy'], label='digit3_accuracy')
    plt.plot(data.history['digit4_accuracy'], label='digit4_accuracy')
    plt.plot(data.history['digit5_accuracy'], label='digit5_accuracy')

    plt.plot(data.history['val_digit1_accuracy'], label='val_digit1_accuracy')
    plt.plot(data.history['val_digit2_accuracy'], label='val_digit2_accuracy')
    plt.plot(data.history['val_digit3_accuracy'], label='val_digit3_accuracy')
    plt.plot(data.history['val_digit4_accuracy'], label='val_digit4_accuracy')
    plt.plot(data.history['val_digit5_accuracy'], label='val_digit5_accuracy')
    plt.savefig(f'accuracy_{datetime.now().time()}.png')

    plt.legend()
    plt.show()


def one_hot(y_vals, base_len=len(LETTERS)):
    res = np.zeros((5, base_len))
    for idx, y in enumerate(y_vals):
        res[idx][int(y)] = 1
    return res


def create_train_test(dir, maximum=-1, batch_size=1, train_p=0.75, img_shape=(50, 200, 3)):
    """Split dataset on train/test and return generators for each"""
    files = os.listdir(dir)[:maximum] if maximum else os.listdir(dir)
    shuffle(files)

    t_start = int(len(files) * train_p)
    train_files = files[:t_start]
    test_files = files[t_start:]

    def gen_function(files):
        while True:
            batch_x = np.zeros((batch_size, *img_shape))
            batch_y = np.zeros((5, batch_size, len(LETTERS)))
            cnt = 0
            for file_name in files:
                img = cv2.imread(os.path.join(dir, file_name), 1)

                y = one_hot(file_name.split('.')[0].split('_')[1:])

                batch_x[cnt] = img / 255.

                for idx, y_ in enumerate(y):
                    batch_y[idx][cnt] = y_
                cnt += 1
                if cnt == batch_size:
                    yield batch_x.copy(), list(batch_y)
                    cnt = 0

    return gen_function(train_files), gen_function(test_files)
