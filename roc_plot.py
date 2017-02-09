from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

chim_vs_all_0 = "../../caffe/myexperiments/chim_vs_all_0.txt"
chim_vs_all_1 = "../../caffe/myexperiments/chim_vs_all_1.txt"
chim_vs_all_2 = "../../caffe/myexperiments/chim_vs_all_2.txt"
chim_vs_all_3 = "../../caffe/myexperiments/chim_vs_all_3.txt"

file_list = [chim_vs_all_0, chim_vs_all_1, chim_vs_all_2, chim_vs_all_3]


def get_frp_tpr(f):
    fpr = []
    tpr = []
    f = open(f, "r")
    line1 = f.readline()
    line2 = f.readline()
    fpr = [ n for n in line1.split(' ')]
    tpr = [ n for n in line2.split(' ')]
    f.close()
    return {'fpr': fpr, 'tpr': tpr}


def roc_k_fold_cross_validation(dd):
    l = []
    nfolds = len(file_list)
    i = 0
    for filename in file_list:
        l.append(get_fpr_tpr(filename))


# Instantiate list
false_positive_rate = []
true_positive_rate = []


# Parse the file
fil = open(chim_vs_all_1, "r")
# read the line of FPR
line = fil.readline()
for num in line.split(' '):
    false_positive_rate.append(float(num))

#read the line of TPR
line = fil.readline()
for num in line.split(' '):
    true_positive_rate.append(float(num))
fil.close()

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('ROC curve')
plt.suptitle('trained model - Chimpanzee vs All')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
