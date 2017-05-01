import cv2
from matplotlib import pyplot as plt


th = 100  # computed externally
img = cv2.imread('/home/carlo/caffe/data/mydata/testing/3/Gorilla/gorilla_122.png', 0)
plt.hist(img.ravel(), 256, [0, 256])
plt.plot((100, 100), (100, 20000), 'r', label='threshold = %d' % th)
plt.xlabel('pixel intensity')
plt.ylabel('frequency')
plt.title('Image Histogram')
plt.xlim([0, 260])
plt.legend(loc='upper left')
plt.show()
