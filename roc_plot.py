from sklearn.metrics import auc
import matplotlib.pyplot as plt

root = "../../caffe/myexperiments/"

dir1 = "own_model/" # erroneously finetuned each layer
dir2 = "reference_model/"
dir3 = "finetuned_last/" # just finetuned the last layer

chim_vs_all_0 = root + dir3 + "chim_vs_all_0.txt"
chim_vs_all_1 = root + dir3 + "chim_vs_all_1.txt"
chim_vs_all_2 = root + dir3 + "chim_vs_all_2.txt"
chim_vs_all_3 = root + dir3 + "chim_vs_all_3.txt"

chim_vs_all_reference_0123 = root + dir2 + "chim_vs_all_entireset.txt"
chim_vs_all_own_0123 = root + dir1 + "chim_vs_all_0123_own.txt"
chim_vs_all_finetuned_last_0123 = root + dir3 + "chim_vs_all_0123_finetuned_last.txt"

#file_list = [chim_vs_all_0, chim_vs_all_1, chim_vs_all_2, chim_vs_all_3]
file_list = [chim_vs_all_reference_0123, chim_vs_all_own_0123, chim_vs_all_finetuned_last_0123]

colors = ['b', 'g', 'r', 'm']


def get_fpr_tpr(f):
    fil = open(f, "r")
    line1 = fil.readline()
    line2 = fil.readline()
    fpr = [float(n) for n in line1.split(' ')]
    tpr = [float(n) for n in line2.split(' ')]
    fil.close()
    return fpr, tpr


def plot_roc(l):
    i = 0
    for elem, color in zip(l, colors):
        fpr = elem[0]
        tpr = elem[1]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color, label='fold %d (AUC = %.2f)' % (i, roc_auc))
        i += 1
    plt.title('ROC curve')
    plt.suptitle('Finetuned model (last layer only) - Chimpanzee vs All')
    plt.plot([0, 1], [0, 1], 'k--', label='random guess')
    plt.legend(loc='lower right')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def roc_k_fold_cross_validation(f_list):
    l = []
    for filename in f_list:
        l.append(get_fpr_tpr(filename))
    plot_roc(l)


roc_k_fold_cross_validation(file_list)

"""
l = []
for filename in file_list:
    l.append(get_fpr_tpr(filename))

fpr = l[0][0]
tpr = l[0][1]
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'c', label=' reference (AUC = %.2f)' % roc_auc)

fpr2 = l[1][0]
tpr2 = l[1][1]
roc_auc = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, 'r', label=' finetuned all (AUC = %.2f)' % roc_auc)

fpr3 = l[2][0]
tpr3 = l[2][1]
roc_auc = auc(fpr3, tpr3)
plt.plot(fpr3, tpr3, 'g', label=' finetuned last (AUC = %.2f)' % roc_auc)


plt.title('ROC curve')
plt.suptitle('Chimpanzee-vs-All')
plt.plot([0, 1], [0, 1], 'k--', label='random guess')
plt.legend(loc='lower right')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""