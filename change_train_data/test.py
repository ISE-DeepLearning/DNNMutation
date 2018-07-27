# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

if __name__ == '__main__':
    # rate_list = [0.001, 0.01, 0.02, 0.05, 0.1, 1]
    rate_list = [0.2, 0.4, 0.5, 0.6, 0.8]
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    # 比率
    image_labels = mnist.train.labels

    for rate in rate_list:

        save_path = 'modify_label_data/data/change_' + str(rate).replace('.', '') + '.npy'
        labels = np.load(save_path)
        print(rate, 1-accuracy_score(image_labels, labels))
