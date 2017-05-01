import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


# get FPR and TPR from file
def get_fpr_tpr(f):
    fil = open(f, "r")
    line1 = fil.readline()
    line2 = fil.readline()
    fpr = [float(n) for n in line1.split(' ')]
    tpr = [float(n) for n in line2.split(' ')]
    fil.close()
    return fpr, tpr

#  where the data is
root = "../../caffe/myexperiments/"
dir1 = "own_model/"   # fine-tuned each layer
dir2 = "reference_model/"
dir3 = "finetuned_last/"  # just fine-tuned the last layer
chim_vs_all_0 = root + dir2 + "chim_vs_all_0.txt"
chim_vs_all_1 = root + dir2 + "chim_vs_all_1.txt"
chim_vs_all_2 = root + dir2 + "chim_vs_all_2.txt"
chim_vs_all_3 = root + dir2 + "chim_vs_all_3.txt"

f_list = [chim_vs_all_0, chim_vs_all_1, chim_vs_all_2, chim_vs_all_3]
#f_list = ["f0.txt", "f1.txt", "f2.txt"]


seq_x = []
seq_y = []
x_all = []
for filename in f_list:
    x_, y_ = get_fpr_tpr(filename)
    seq_x.append(x_)  # [x1, x2, x3, x4]
    seq_y.append(y_)
    x_all.extend(x_)  # x1 + x2 + x3 + x4

# constructs a numpy array composed of the unique elements (a set)
x_all = np.unique(np.array(x_all))
x_all = np.linspace(0, 1, 1000001)  # generate a sequence of num=1001 sample numbers from start=0 to stop=1 included
y_all = np.empty((len(x_all), len(f_list)))  # returns a new uninitialized array of shape len(x_all)*4

# interpolate y values on new x-array
for i, x, y in zip(range(len(f_list)), seq_x, seq_y):
    # perform linear interpolation x_all are the x-coordinates of the interpolated values. (x,y) are data points to fit.
    y_all[:, i] = np.interp(x_all, x, y)

# find out min and max values
ymin = y_all.min(axis=1)
ymax = y_all.max(axis=1)

with open('test.txt', "w") as f:
    for x in seq_y:
        f.write(str(x) + '\n')

# Area Between the Curves
ABC = auc(x_all, ymax) - auc(x_all, ymin)
plt.fill_between(x_all, ymin, ymax, facecolor='orange', alpha=0.4)
plt.plot([], [], 'orange', label='ABC = %.4f' % ABC)

colors = ['b', 'g', 'r', 'm']
for x, y, c in zip(seq_x, seq_y, colors):
    plt.plot(x, y, c, label='AUC = %.2f' % auc(x, y))


plt.xlim(-0.01, 1.01)
plt.ylim(0, 1.01)
plt.legend(loc='lower right')
plt.show()
