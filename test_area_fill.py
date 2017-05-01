import matplotlib.pyplot as plt
from sklearn.metrics import auc
import json
import numpy as np

x1 = [0,   0.1, 0.2, 0.6, 0.7, 0.75, 1]
y1 = [0.5, 0.6, 0.7, 0.8, 1,   1,    1]

x2 = [0,   0.01, 0.2, 0.4, 0.7, 0.75,  1]
y2 = [0.6, 0.5,  0.8, 0.7, 0.9, 1,     1]

x3 = [0,   0.2, 0.45, 0.5,  0.6, 0.9, 1]
y3 = [0.4, 0.5, 0.55, 0.8, 0.9, 0.9, 1]

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

auc1 = auc(x1, y1)
plt.plot(x1,y1 ,'r', label='(AUC = %.2f)' % auc1)

auc2 = auc(x2, y2)
plt.plot(x2,y2, 'b', label='(AUC = %.2f)' % auc2)

auc3 = auc(x3, y3)
plt.plot(x3,y3, 'g', label='(AUC = %.2f)' % auc3)

plt.legend(loc='lower right')
plt.show()

mydic = dict()
# insert x and y values in dict
XY_list = [(x1, y1), (x2, y2)]
for X_list, Y_list in XY_list:
    for x, y in zip(X_list, Y_list):
        if x in mydic.keys():
            mydic[x] = mydic[x], y
        else:
            mydic[x] = y

print json.dumps(mydic, indent=2, sort_keys=True)


lower_fpr_tpr = list()
upper_fpr_tpr = list()
# get lower-contour curve -- list of (fpr,tpr) points
for key, value in mydic.iteritems():
    lower_fpr_tpr.append((key, np.min(value)))

lower_fpr_tpr.sort(key=lambda x: x[0])
fpr = [ pair[0] for pair in lower_fpr_tpr] # get sorted fpr
lower_tpr = [ pair[1] for pair in lower_fpr_tpr] # get corresponding tpr
print fpr
print lower_tpr

for key, value in mydic.iteritems():
    upper_fpr_tpr.append((key, np.max(value)))

upper_fpr_tpr.sort(key=lambda x: x[0])
upper_tpr = [pair[1] for pair in upper_fpr_tpr]  # get corresponding tpr
fpr = [ pair[0] for pair in lower_fpr_tpr] # get sorted fpr
print fpr
print upper_tpr


#plt.fill_between(fpr, lower_tpr, upper_tpr, where=lower_tpr < upper_tpr, facecolor='green')
#plt.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
#plt.fill_between(fpr, lower_fpr_tpr, upper_fpr_tpr)

#print json.dumps(lower_dic, indent=2, sort_keys=True)

