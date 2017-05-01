import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def quadratic_intersections(p, q):
    """Given two quadratics p and q, determines the points of intersection"""
    x = np.roots(np.asarray(p) - np.asarray(q))
    y = np.polyval(p, x)
    return x, y


def get_EER_point_and_threshold(xs, ys, th_array):
    np_fpr1 = np.array(xs)
    np_tpr1 = np.array(ys)
    #poly1 = np.polyfit(np_fpr1, np_tpr1, 2)
    #poly2 = np.polyfit([0, 1], [1, 0], 1)
    #x, y = quadratic_intersections(poly1, poly2)
    diff = np.abs((1.0 - np_fpr1) - np_tpr1)
    argmin = np.argmin(diff)
    x, y, threshold = 1 - np_tpr1[argmin], np_tpr1[argmin], th_array[argmin]
    return x, y, threshold


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


file_labels = "/home/carlo/caffe/myexperiments/label_vectors_reference.txt"
file_scores = "/home/carlo/caffe/myexperiments/score_vectors_reference.txt"
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
thresholds = np.linspace(lower_t, upper_t, n_th)

# create contingency table for each class and for each threshold (one-vs-all)
gori_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=0)
chim_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=1)
back_tables = build_cont_tables(n_tables=n_th, thresholds=thresholds, scores=scores, labels=labels, positive_class=2)

# compute points to plot
fpr1, tpr1 = fpr_tpr_points(gori_tables)
fpr2, tpr2 = fpr_tpr_points(chim_tables)
fpr3, tpr3 = fpr_tpr_points(back_tables)

#plt.figure(0)
# plot a ROC for each class
#plt.plot(fpr1, tpr1, 'r', label='gori AUC = %.4f' % auc(fpr1, tpr1))
#plt.plot(fpr2, tpr2, 'b', label='chim AUC = %.4f' % auc(fpr2, tpr2))
#plt.plot(fpr3, tpr3, 'g', label='back AUC = %.4f' % auc(fpr3, tpr3))

#plt.fill_between(fpr1, tpr1, facecolor='red', alpha=0.3)
#plt.fill_between(fpr2, tpr2, facecolor='blue', alpha=0.3)
#plt.fill_between(fpr3, tpr3, facecolor='green', alpha=0.3)

#plt.title('multi-class ROC curve')
#plt.suptitle('One-vs-All finetuned last')
#plt.plot([0, 1], [0, 1], 'k--', label='random guess')
#plt.legend(loc='lower right')
#plt.xlim([-0.01, 1.01])
#plt.ylim([-0.01, 1.01])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

plt.figure(1)
# Equal Error Rate
fig2 = plt.fill_between(fpr1, tpr1, facecolor='red', alpha=0.3)
plt.fill_between(fpr2, tpr2, facecolor='blue', alpha=0.3)
plt.plot(fpr1, tpr1, 'r', label='gori AUC = %.4f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, 'b', label='chim AUC = %.4f' % auc(fpr2, tpr2))
plt.plot([0, 1], [1, 0], 'k--', label='EER')
plt.legend(loc='lower right')
plt.xlim([0, 0.2])
plt.ylim([0.8, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('EER: selection of the operating point')


x1, y1, threshold_gori = get_EER_point_and_threshold(fpr1, tpr1, thresholds)
x2, y2, threshold_chim = get_EER_point_and_threshold(fpr2, tpr2, thresholds)
print x1, y1, threshold_gori
print x2, y2, threshold_gori

plt.scatter(x1, y1, marker='o', s=40, zorder=2, linewidth=2, color='black')
plt.scatter(x2, y2, marker='o', s=40, zorder=2, linewidth=2, color='black')

plt.annotate('(%.3f,%.3f)' % (x1, y1), xy=(x1, y1), xytext=(0.07, 0.95), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('(%.3f,%.3f)' % (x2, y2), xy=(x2, y2), xytext=(0.08, 0.9), arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
