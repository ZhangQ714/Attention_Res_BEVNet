import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


# 混淆矩阵
def paint_confusion_matrix(label_th_np, pred_np, opts, paint_unknown=True):
    # 采用的方法是对样本里每一个像素点的分类结果进行统计，如果数据量太大的话可能需要分开计算再加和

    if opts.test_env == 'kitti4':                              # 这个数据集数据量大，分开计算，但是如果分开的每一部分样本里类没有全部出现的话最后相加会出问题
        label_th_npi = np.array(label_th_np[:500]).ravel()
        pred_npi = np.array(pred_np[:500]).ravel()
        cm = confusion_matrix(label_th_npi, pred_npi)
        for i in range(1, len(pred_np)//500 + 1):
            if i == len(pred_np)//500:
                label_th_npi = np.array(label_th_np[i * 500:]).ravel()
                pred_npi = np.array(pred_np[i * 500:]).ravel()
                cm += confusion_matrix(label_th_npi, pred_npi)
            else:
                label_th_npi = np.array(label_th_np[i*500:(i+1)*500]).ravel()
                pred_npi = np.array(pred_np[i*500:(i+1)*500]).ravel()
                cm += confusion_matrix(label_th_npi, pred_npi)
    else:         # 直接全部计算
        label_th_npi = np.array(label_th_np).ravel()
        pred_npi = np.array(pred_np).ravel()
        print('confusion_matrix:')
        cm = confusion_matrix(label_th_npi, pred_npi)
    print(cm)
    index = ['0', '1', '2', '3', '4']
    cm = cm.astype(np.float64)
    cm_sum = cm.sum(axis=1)
    for i in range(len(cm_sum)):
        cm[i] /= cm_sum[i]
    # cm_normed = cm / cm.sum(axis=1)
    if paint_unknown is False:
        cm = cm[:-1, :-1]            # drop unknown
        index = index[:-1]

    plt.figure(figsize=(8, 8))
    da = pd.DataFrame(cm, index=index)
    sns.heatmap(da, vmin=0, vmax=1, annot=True, annot_kws={'size': 22}, cbar=None, cmap='Blues')
    plt.title('Ours', fontsize=28)  # atr
    # plt.tight_layout()yt
    # plt.ylabel('Actual', fontsize=28)
    plt.ylabel(' ', fontsize=28)
    plt.xlabel('Predicted', fontsize=28)
    plt.xticks(fontsize=22)  # x轴刻度的字体大小
    plt.yticks(fontsize=22)  # y轴刻度的字体大小

    # plt.show()
    plt.savefig(
        '{}/{}.png'.format(os.path.split(opts.model_file)[0], 'confusion_matrix'))  # 将混淆矩阵图片保存下来
    plt.close()