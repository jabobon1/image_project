import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import random


def create_rectangle(file_name=os.path.join('new_set',
                                            'blank_rectangle.png'),
                     color='white'):
    img = Image.new('RGBA', (200, 50), color)
    ImageDraw.Draw(img)
    img.save(file_name)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def shower(idx):
    plt.figure()
    plt.imshow(x_train[idx])
    # plt.colorbar()
    # plt.grid(False)
    plt.show()


def table():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()


def writer(file_name, image: Image):
    with    open(file_name, 'wb') as f:
        image.save(f)


def collect_mnist(directory, samples_x, y_lbl, mark='A', inverse=True):
    """Saves images(np arrays) into directory"""
    for idx, sample in enumerate(samples_x):
        file_path = os.path.join(directory, f'{y_lbl[idx]}_idx_{idx}_m_'
                                            f'{mark}.png')
        image = ImageOps.invert(Image.fromarray(sample)) if inverse else \
            Image.fromarray(sample)
        writer(file_path, image)


# create_rectangle('black_rectangle.png',color='black')
# collect_mnist('mnist_2', x_train, y_train,mark='A',inverse=False)
# collect_mnist('mnist_2', x_test, y_test, mark='B',inverse=False)


def collide_images(base_img, second_img, res_path, pos=(0, 0), dir='mnist'):
    im1 = Image.open(base_img)
    im2 = Image.open(os.path.join(dir, second_img))
    im1.paste(im2, box=pos)
    im1.save(res_path)


def create_set(dir, samples, input='mnist_2', base_img='black_rectangle.png'):
    for sam in samples:
        # original img =200x50 px, inserted img 28,28
        pos = (random.randint(0, 172), random.randint(0, 22))
        name = sam.split('_')
        name = os.path.join(dir, f"{name[0]}_pos_{pos[0]}_{pos[1]}_idx" \
                                 f"_{name[2]}.png")

        collide_images(base_img, sam, name, pos, dir=input)


def deleter(directory='new_set'):
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))


# def make_random_set(x, y, amount):
#     length = [i for i in range(len(x))]
#     random.shuffle(length)
#     for i in range(amount):
#         yield x[i], y[i]

direct = 'mnist_2'
files = os.listdir(direct)
random.shuffle(files)

amount = 60000
dataset = [files[i] for i in range(amount)]

create_set('new_set', dataset, input='mnist_2')
