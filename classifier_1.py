import os
from random import shuffle

import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential

LETTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

model = Sequential([
    Conv2D(input_shape=(50, 200, 3), filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.3)
])

out = [Dense(len(LETTERS), name='digit1', activation='softmax')(model.output),
       Dense(len(LETTERS), name='digit2', activation='softmax')(model.output),
       Dense(len(LETTERS), name='digit3', activation='softmax')(model.output),
       Dense(len(LETTERS), name='digit4', activation='softmax')(model.output),
       Dense(len(LETTERS), name='digit5', activation='softmax')(model.output)]
model = Model(inputs=model.inputs, outputs=out)


def one_hot(y_vals, base_len=len(LETTERS)):
    res = np.zeros((5, base_len))
    for idx, y in enumerate(y_vals):
        vector = np.zeros(base_len)
        vector[int(y)] = 1
        res[idx] = vector
    return res


def create_train_test(dir, maximum=-1, batch_size=1, train_p=0.25, img_shape=(50, 200, 3)):
    """Split dataset on train/test and return generators for each"""
    files = os.listdir(dir)[:maximum] if maximum else os.listdir(dir)
    shuffle(files)
    t_start = len(files) - int(len(files) * train_p)
    train_files = files[:t_start]
    test_files = files[t_start:]

    def gen_function(files):
        while True:
            batch_x = np.zeros((batch_size, *img_shape))
            batch_y = np.zeros((batch_size, 5, len(LETTERS)))
            cnt = 0
            for file_name in files:
                img = cv2.imread(os.path.join(dir, file_name), -1)

                y = one_hot(file_name.split('.')[0].split('_')[1:])
                batch_x[cnt] = img / 255.
                batch_y[cnt] = y
                cnt += 1
                if cnt == batch_size:
                    yield batch_x.copy(), batch_y.copy()
                    cnt = 0

    return gen_function(train_files), gen_function(test_files)


filepath = "./model/imitate_5_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_letters_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_letters_acc', patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)

callbacks_list = [checkpoint, earlystop, tensorBoard, ]

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

directory = 'result'
total_files = len(os.listdir(directory))
train_percent = 0.1
batch_size = 32
train_dataset, test_dataset = create_train_test(directory, maximum=-1,
                                                batch_size=batch_size,
                                                train_p=train_percent)

model.fit(train_dataset,
          validation_data=test_dataset,
          validation_steps=int(total_files * train_percent),
          steps_per_epoch=total_files - int(total_files * train_percent),
          batch_size=32,
          epochs=60,
          callbacks=callbacks_list)
