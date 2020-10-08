from tensorflow.keras.models import load_model
import numpy as np
from useful_functions import create_train_test, plot_result
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, precision_recall_fscore_support


def demo(model, x_eval, y_true):
    y_pred = np.array(model.predict(x_eval))

    average_precision = [precision_recall_fscore_support([y.argmax() for y in y_true[i]],
                                                         [y.argmax() for y in y_pred[i]],
                                                         average='macro') for i in range(5)]
    for idx, val in enumerate(average_precision):
        print(idx, val)

    plot_result(40, x_eval, y_pred, y_true)


if __name__ == '__main__':
    model = load_model('model/classifier_standart')

    directory = 'result2'

    _, test_dataset = create_train_test(directory, maximum=-1,
                                        batch_size=100,
                                        train_p=0.7)
    eval = test_dataset.__next__()
    x_eval = eval[0]
    y_true = np.array(eval[1])

    demo(model, x_eval, y_true)
