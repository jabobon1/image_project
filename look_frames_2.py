import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import os
import pandas as pd
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import np_utils


DIRECTORY = 'new_set'


def create_train(dir, maximum=200):
    X = []
    y = []
    for file_name in os.listdir(dir)[:maximum]:
        img = Image.open(os.path.join(dir, file_name))
        _ = file_name.split('.')[0].split('_')
        res = (int(_[2]), int(_[3]), int(_[2]) + 28, int(_[3]) + 28)
        # Converting Images to Grayscale
        X.append(np.asarray(img.convert('L')))
        y.append(res)
    return np.array(X), np.array(y)


def shower(x_train, idx):
    plt.figure()
    plt.imshow(x_train[idx])
    plt.colorbar()
    plt.grid(False)
    plt.show()


X, y = create_train('new_set', -1)
X, y = X / 250, y / 250
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 200)),
    keras.layers.Dense(700, activation='softmax'),
    keras.layers.Dense(300, activation='sigmoid'),
    keras.layers.Dense(4, activation='sigmoid')
])

print(model.summary())

model.compile(
    optimizer="adam",
    loss='mae',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10)


def show_result(x_img, y_pred):
    plt.figure(figsize=(10, 10))
    for i in range(len(x_img)):
        shape = [(y_pred[i][0], y_pred[i][1]),
                 (y_pred[i][2], y_pred[i][3])]
        image = Image.fromarray(x_img[i] * 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle(shape, outline="white")

        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(y_pred[i])
    plt.show()


samp = 25
pred = model.predict(X_test[:samp])

show_result(X_test[:samp], (pred[:samp] * 255).round())
