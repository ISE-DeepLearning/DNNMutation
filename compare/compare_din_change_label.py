# -*- coding: utf-8 -*-
from keras.models import load_model
from keras import Model,Input
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data


def getActivationLayers(model):
    intermediate_layer_model_1 = Model(inputs=model.input, outputs=model.layers[2].output)
    intermediate_layer_model_2 = Model(inputs=model.input, outputs=model.layers[4].output)
    intermediate_layer_model_3 = Model(inputs=model.input, outputs=model.layers[6].output)
    return intermediate_layer_model_1, intermediate_layer_model_2, intermediate_layer_model_3


if __name__ == '__main__':

    # 读取数据 全部图片数据
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    images_data = mnist.test.images
    labels_data = mnist.test.labels
    count = len(images_data)

    rate_list = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    # for rate in rate_list:
    #     train_model_path = '../change_train_data/modify_label_data/model/model_' + str(rate).replace('.', '') + '.hdf5'
    #     save_path = 'detail_data/change_label_' + str(rate).replace('.', '') + '.npz'
    #     train_model = load_model(train_model_path)
    #
    #     predict = train_model.predict(images_data)
    #     predict = list(map(lambda x: np.argmax(x), predict))
    #     test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    #     accuracy = accuracy_score(test_label, predict)
    #
    #     ba1, ba2, ba3 = getActivationLayers(train_model)
    #     ba1_output = ba1.predict(images_data)
    #     # hidden_activation_1_output[hidden_activation_1_output > 0.5] = 1
    #     # hidden_activation_1_output[hidden_activation_1_output < 0.5] = 0
    #     ba2_output = ba2.predict(images_data)
    #     # hidden_activation_2_output[hidden_activation_2_output > 0.5] = 1
    #     # hidden_activation_2_output[hidden_activation_2_output < 0.5] = 0
    #     ba3_output = ba3.predict(images_data)
    #
    #     np.savez(save_path, accuracy=accuracy, predict=predict, level1=ba1_output, level2=ba2_output, level3=ba3_output)

    for index in range(784):
        print("当前维度为：".format(index))
        mutation_model_path = '../mutation_model/DIN/model/din_' + str(index) + '.hdf5'
        save_path = 'detail_data/din_' + str(index) + '.npz'
        mutation_model = load_model(mutation_model_path)

        predict = mutation_model.predict(images_data)
        predict = list(map(lambda x: np.argmax(x), predict))
        test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
        accuracy = accuracy_score(test_label, predict)

        ba1, ba2, ba3 = getActivationLayers(mutation_model)
        ba1_output = ba1.predict(images_data)
        # hidden_activation_1_output[hidden_activation_1_output > 0.5] = 1
        # hidden_activation_1_output[hidden_activation_1_output < 0.5] = 0
        ba2_output = ba2.predict(images_data)
        # hidden_activation_2_output[hidden_activation_2_output > 0.5] = 1
        # hidden_activation_2_output[hidden_activation_2_output < 0.5] = 0
        ba3_output = ba3.predict(images_data)

        np.savez(save_path, accuracy=accuracy, predict=predict, level1=ba1_output, level2=ba2_output, level3=ba3_output)






