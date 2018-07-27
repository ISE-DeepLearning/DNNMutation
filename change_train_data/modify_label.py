#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Activation
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import numpy as np


def change_pic_label(data, percent):

    iteration = int(data.shape[0] * percent)
    print('修改的次数：' + str(iteration))
    change_data = data.copy()
    number = len(change_data)
    print(change_data.shape)

    count = 0
    location_map = {}
    while count < iteration:
        random_location = np.random.randint(number)
        if location_map.__contains__(random_location):
            continue
        random_label = np.random.randint(10)
        location_map[random_location] = random_label
        count += 1

    print(len(location_map.keys()))
    for location in location_map.keys():
        print('location:', location)
        random_label = location_map[location]
        temp = np.zeros(10)
        temp[random_label] = 1
        print('before', change_data[location], random_label, location)
        change_data[location] = temp
        print('after', change_data[location])

    save_path = 'modify_label_data/data/change_' + str(percent).replace('.', '')
    np.save(save_path, change_data)
    return change_data


if __name__ == '__main__':

    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    # 比率
    # rate = 0.001
    # rate_list = [0.001, 0.01, 0.02, 0.05, 0.1, 1]
    rate_list = [0.2, 0.4, 0.5, 0.6, 0.8]
    image_labels = mnist.train.labels

    for rate in rate_list:
        labels = np.copy(image_labels)
        result = change_pic_label(labels, rate)
        print(result.shape)





