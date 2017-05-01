import numpy as np

score_file = "/home/carlo/caffe/myexperiments/score_vectors_finetuned_all.txt"
label_file = "/home/carlo/caffe/myexperiments/label_vectors_finetuned_all.txt"
images_file = "/home/carlo/caffe/data/mydata/test_0123.txt"

scores = []
# read score file
with open(score_file, "rb") as f:
    for line in f.readlines():
       scores.append([float(el) for el in line.strip('\n').split(' ')])

# class index with the highest score
maxIndex = np.argmax(scores, axis=1)
# read label file
with open(label_file, "rb") as f:
    labels = [int(el) for el in f.readlines()]

# position is true when the image was correctly classified, false otherwise
correctly_classified = maxIndex == labels

with open("/home/carlo/caffe/myexperiments/missclassified_images_finetuned_all.txt", "w") as outf:
    with open(images_file, "r") as f:
        for index, line in zip(range(len(labels)), f.readlines()):
            if correctly_classified[index] == False:
                outf.write(str(scores[index]) + " " + line)


