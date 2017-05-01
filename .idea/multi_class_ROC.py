import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def build_cont_tables(n_tables, thresholds, scores, labels, positive_class):
    pos, neg = 0, 1
    cont_tables = np.array([[0] * 2 * 2] * n_tables)
    for s, l in zip(scores, labels):
        confidence = round(s[positive_class], 4)
        actual = pos if positive_class == l else neg
        for [th, index] in zip(thresholds, range(n_tables)):
            prediction = pos if confidence >= th else neg
            cont_tables[index][actual * 2 + prediction] += 1
    return cont_tables


def fpr_tpr_points(cont_tables):
    # each table is stored like this: [TP, FN, FP, TN]
    Pos = cont_tables[0][0] + cont_tables[0][1]
    Neg = cont_tables[0][2] + cont_tables[0][3]
    print Pos, Neg
    fpr, tpr = [0] * len(cont_tables), [0] * len(cont_tables)
    for t, index in zip(cont_tables, range(len(cont_tables))):
        fpr[index] = t[2] / float(Neg)
        tpr[index] = t[0] / float(Pos)
    return fpr, tpr


file_labels = "/home/carlo/caffe/myexperiments/label_vectors_finetuned_last.txt"
file_scores = "/home/carlo/caffe/myexperiments/score_vectors_finetuned_last.txt"
labels = []
n_classes = 3
marginals = [0] * n_classes # [0, 0, 0]
scores = []

with open(file_labels, "r") as f:
    for line in f.readlines():
        num = int(line)
        marginals[num] += 1
        labels.append(num)

with open(file_scores, "rb") as f:
    for line in f.readlines():
       scores.append([float(el) for el in line.strip('\n').split(' ')])

lower_t = 0
upper_t = 1
delta_t = 0.001
n_th = 1001
thresholds = np.linspace(lower_t,upper_t, n_th)

# create contingency table for each class and for each threshold (one-vs-all)
gori_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=0)
chim_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=1)
back_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=2)

# compute points to plot
fpr1, tpr1 = fpr_tpr_points(gori_tables)
fpr2, tpr2 = fpr_tpr_points(chim_tables)
fpr3, tpr3 = fpr_tpr_points(back_tables)

# plot a ROC for each class
plt.plot(fpr1, tpr1, 'r', label='gori AUC = %.4f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, 'b', label='chim AUC = %.4f' % auc(fpr2, tpr2))
plt.plot(fpr3, tpr3, 'g', label='back AUC = %.4f' % auc(fpr3, tpr3))

plt.fill_between(fpr1, tpr1, facecolor='red', alpha=0.3)
plt.fill_between(fpr2, tpr2, facecolor='blue', alpha=0.3)
plt.fill_between(fpr3, tpr3, facecolor='green', alpha=0.3)

plt.title('multi-class ROC curve')
plt.suptitle('One-vs-All finetuned last')
plt.plot([0, 1], [0, 1], 'k--', label='random guess')
plt.legend(loc='lower right')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()