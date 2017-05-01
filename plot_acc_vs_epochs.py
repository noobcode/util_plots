import matplotlib.pyplot as plt

test = "../../caffe/myexperiments/test_accuracy_vs_epochs.txt"
train = "../../caffe/myexperiments/train_accuracy_vs_epochs.txt"


def get_accuracy_points(f_name):
    fil = open(f_name, 'r')
    acc = [float(num) for num in fil.readlines()]
    fil.close()
    return acc


accuracy_test = get_accuracy_points(test)
accuracy_train = get_accuracy_points(train)

#plot accuracy_test vs epochs
plt.plot(range(1, len(accuracy_test) + 1, 1), accuracy_test, 'b', label='accuracy on test set')
plt.plot(range(1, len(accuracy_train) + 1, 1), accuracy_train, 'r', label='accuracy on train set')

plt.xlim([0, len(accuracy_test)])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('weighted Accuracy vs Epochs')
plt.suptitle('train set vs test set')
plt.ylabel('weighted Accuracy')
plt.xlabel('#Epochs')
plt.show()


