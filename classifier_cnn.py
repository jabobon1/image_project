import os

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers import ZeroPadding2D

from demo import demo
from useful_functions import plot_history, create_train_test

LETTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def create_model():
    model = Sequential([
        ZeroPadding2D((1, 1), input_shape=(50, 200, 3)),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=516, kernel_size=(3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=516, kernel_size=(3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(filters=516, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5)
    ])

    out = [Dense(len(LETTERS), name='digit1', activation='softmax')(model.output),
           Dense(len(LETTERS), name='digit2', activation='softmax')(model.output),
           Dense(len(LETTERS), name='digit3', activation='softmax')(model.output),
           Dense(len(LETTERS), name='digit4', activation='softmax')(model.output),
           Dense(len(LETTERS), name='digit5', activation='softmax')(model.output)]

    return Model(inputs=model.inputs, outputs=out)


if __name__ == '__main__':
    directory = 'result3'
    maximum = 50000
    epochs = 80
    total_files = len(os.listdir(directory)[:maximum])
    train_percent = 0.8
    batch_size = 300

    checkpoint = ModelCheckpoint(f"model/classifier-10-conv-ep{epochs}-bs{batch_size}", monitor='val_loss',
                                 verbose=0, save_best_only=True,
                                 mode='auto', save_freq="epoch")
    # earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)

    callbacks_list = [
        checkpoint,
        # earlystop,
        tensorBoard,
    ]
    model = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    train_dataset, test_dataset = create_train_test(directory, maximum=maximum,
                                                    batch_size=batch_size,
                                                    train_p=train_percent)

    history = model.fit(train_dataset,
                        validation_data=test_dataset,
                        validation_steps=(total_files - int(total_files * train_percent)) // batch_size,
                        steps_per_epoch=int(total_files * train_percent) // batch_size,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks_list
                        )
    plot_history(history)

    eval = test_dataset.__next__()
    x_eval = eval[0]
    y_true = np.array(eval[1])

    demo(model, x_eval, y_true, save=True)
