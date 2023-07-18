import os
import numpy as np
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
import matplotlib.pyplot as plt


def roc_generate(pred, y_score, MODEL_FILE):

    pred = pred.ravel()
    print("a")
    y_score = y_score.ravel()
    print("a2")
    y_test = label_binarize(pred, classes=[0, 1, 2, 3, 4])
    # 设置种类
    n_classes = y_test.shape[1]

    print("b")

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print("c")

    # Compute micro-average ROC curve and ROC area（方法二）
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    roc_path = os.path.split(MODEL_FILE)[0] + '/roc_curve/'
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    plt.savefig(
        '{}/{}.png'.format(roc_path, 'roc_curve'))
    plt.close()
    # plt.show()
    return



if __name__ == '__main__':
    score = [
        np.array([
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 3]
    ]),
        np.array([
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 3]
        ])
    ]
    pred = [
        np.array([
            [1, 1, 2],
            [1, 2, 2],
            [1, 2, 3]
        ]),
        np.array([
            [1, 1, 2],
            [2, 2, 2],
            [1, 2, 2]
        ])
    ]
    MODEL_FILE = "path of modelfile"
    roc_generate(np.array(pred), np.array(score), MODEL_FILE)