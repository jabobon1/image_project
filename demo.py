from tensorflow.keras.models import load_model
import numpy as np
from useful_functions import create_train_test, plot_result
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, precision_recall_fscore_support


def demo(model, x_eval, y_true, save=False):
    y_pred = np.array(model.predict(x_eval))

    average_precision = [precision_recall_fscore_support([y.argmax() for y in y_true[i]],
                                                         [y.argmax() for y in y_pred[i]],
                                                         average='macro') for i in range(5)]
    for idx, vals in enumerate(average_precision):
        print(idx, [f'{name}: {val}' for name, val in zip(['presision', 'recall', 'f1'], vals)])

    plot_result(40, x_eval, y_pred, y_true, save)


if __name__ == '__main__':
    model = load_model('model/classifier-10-conv-ep80-bs300')

    directory = 'result3'

    train, test_dataset = create_train_test(directory, maximum=-1,
                                           batch_size=1000,
                                           train_p=0.8)
    eval = train.__next__()
    x_eval = eval[0]
    y_true = np.array(eval[1])

    demo(model, x_eval, y_true)
