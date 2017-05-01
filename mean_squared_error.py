import numpy as np

#score_file = "scoretest.txt"
#label_file = "labeltest.txt"

label_file = "/home/carlo/caffe/myexperiments/label_vectors_finetuned_last.txt"
score_file = "/home/carlo/caffe/myexperiments/score_vectors_finetuned_last.txt"


# compute Square Error between 2 vectors of probabilities
def square_error(v1_, v2_):
    return sum((v1_ - v2_) ** 2) / 2


# create a vector of zeros and a 1 in the index specified by the label
def create_label_vector(index_label, length):
    l = [0] * length
    l[index_label] = 1
    return np.array(l)


# compute MSE between a vector<vector<probabilities>> and vecto<vector<labels>>
def mean_se(p_vector, l_vector):
    mse = 0
    for v, l in zip(p_vector, l_vector):
        mse += square_error(v, l)
    return mse / float(len(p_vector))


def max_se(p_vector, l_vector):
    max_ = 0
    for v, l in zip(p_vector, l_vector):
        tmp = square_error(v, l)
        if tmp > max_:
            max_ = tmp
    return max_


def min_se(p_vector, l_vector):
    min_ = 1
    for v, l in zip(p_vector, l_vector):
        tmp = square_error(v, l)
        if tmp < min_:
            min_ = tmp
    return min_

# read score file
fil = open(score_file, "r")
prob_vectors = []
for line in fil.readlines():
    prob_vectors.append(np.array([float(n) for n in line.split(' ')]))
fil.close()

# read label file
label_vectors = []
n_classes = len(prob_vectors[0])  # number of classes
fil = open(label_file, "r")
for line in fil.readlines():
    label_vectors.append(create_label_vector(int(line), n_classes))
fil.close()

print "mean_se %.4f" % mean_se(prob_vectors, label_vectors)
print "max se %.4f" % max_se(prob_vectors, label_vectors)
print "min se %.4f" % min_se(prob_vectors, label_vectors)
