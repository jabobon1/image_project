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


def plot_result(lim, x_data, y_pred, y_true, save=False):
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

    check_path('metrics')
    if save:
        plt.savefig(f'metrics/result_{datetime.now().time()}.png')
    plt.show()


def plot_history(data):
    check_path('metrics')
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
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'metrics/loss_{datetime.now()}.png')

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
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(f'metrics/accuracy_{datetime.now()}.png')
    plt.show()


def one_hot(y_vals, base_len=len(LETTERS)):
    res = np.zeros((5, base_len))
    for idx, y in enumerate(y_vals):
        res[idx][int(y)] = 1
    return res


def makesymlink(files, new_dir):
    os.makedirs(new_dir)
    new_files = []
    for file in files:
        new_path = os.path.join(new_dir, file.split('/')[1])
        os.symlink(os.path.abspath(file), os.path.abspath(new_path))
        new_files.append(new_path)
    return new_files


def create_train_test(directory, maximum=-1, batch_size=1, train_p=0.75, img_shape=(50, 200, 3)):
    """Split dataset on train/test and return generators for each"""

    if not os.path.exists('train/') and not os.path.exists('test/'):
        files = os.listdir(directory)[:maximum] if maximum else os.listdir(directory)
        shuffle(files)
        t_start = int(len(files) * train_p)

        train_files = makesymlink([os.path.join(directory, fn) for fn in files[:t_start]], 'train')
        test_files = makesymlink([os.path.join(directory, fn) for fn in files[t_start:]], 'test')
    else:
        train_files = os.listdir('train')[:maximum] if maximum else os.listdir('train')
        test_files = os.listdir('test')[:int(maximum * (1 - train_p))] if maximum else os.listdir('test')

    def gen_function(file_links, directory):
        while True:
            batch_x = np.zeros((batch_size, *img_shape))
            batch_y = np.zeros((5, batch_size, len(LETTERS)))
            cnt = 0
            for file_name in file_links:
                img = cv2.imread(os.path.join(directory, file_name), 1)

                y = one_hot(file_name.split('.')[0].split('_')[1:])

                batch_x[cnt] = img / 255.

                for idx, y_ in enumerate(y):
                    batch_y[idx][cnt] = y_
                cnt += 1
                if cnt == batch_size:
                    yield batch_x.copy(), list(batch_y)
                    cnt = 0

    return gen_function(train_files, 'train'), gen_function(test_files, 'test')
