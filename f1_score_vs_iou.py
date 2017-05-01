import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

root = "/home/carlo/caffe/myexperiments/proposals/iou_vs_f1scores_"
file_list = ["baseline", "threshold", "multi_clusters", "no_outliers"]
ext = ".txt"
colors = ['r', 'g', 'b', 'black']


def get_iou_and_f1(fil):
    line1 = fil.readline()
    line2 = fil.readline()
    iou = [float(n) for n in line1.split(' ')]
    f1_scores = [float(n) for n in line2.split(' ')]
    return iou, f1_scores


for c, el in zip(colors, file_list):
    with open(root + el + ext, "r") as f:
        iou, f1_scores = get_iou_and_f1(f)
        print el + ': max f1-score: ' + str(np.max(f1_scores)) + ', mean: ' + str(np.mean(f1_scores)) + ', auc: ' + str(auc(iou, f1_scores))
        plt.plot(iou, f1_scores, c, label=el)

plt.title('F1-score vs IoU')
plt.suptitle('region proposal evaluation')
plt.legend(loc='upper right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('F1-score')
plt.xlabel('IoU')
plt.show()
