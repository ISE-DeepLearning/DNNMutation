from keras import Model,Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import numpy as np

def accuracy_mnist(model, mnist, index):
    pred = model.predict(mnist.test.images)
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    return accuracy_score(test_label, pred)

def train_model(data, label, save_model_path, save_predict_path):
    input_data = Input((28 * 28,))
    temp_data1 = Dense(128)(input_data)
    temp_data2 = Activation('relu')(temp_data1)
    temp_data3 = Dense(64)(temp_data2)
    temp_data4 = Activation('relu')(temp_data3)
    temp_data5 = Dense(10)(temp_data4)
    output_data = Activation('softmax')(temp_data5)
    model = Model(inputs=[input_data], outputs=[output_data])
    modelcheck = ModelCheckpoint(save_model_path, monitor='loss', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit([data], [label], batch_size=256, epochs=40, callbacks=[modelcheck], validation_data=(mnist.test.images, mnist.test.labels))

    pred = model.predict(mnist.test.images)
    np.save(save_predict_path, pred)
    # 准确率
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    print(accuracy_score(test_label, pred))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    # rate_list = [0.001, 0.01, 0.02, 0.05, 0.1, 1]
    # rate_list = [0.2, 0.4, 0.5, 0.6, 0.8]
    rate = 1000
    rate_list = []
    for i in range(1000, 1001):
        rate_list.append(rate / 10000)
        rate += 1
    print(rate_list)
    train_data = mnist.train.images

    for rate in rate_list:
        print('rate:' + str(rate))
        rate_str = str(rate).replace('.', '')
        predict_path = 'modify_label_data/predict/predict_cl_' + rate_str + '.npy'
        model_path = 'modify_label_data/model/model_cl_' + rate_str + '.hdf5'
        train_label = 'modify_label_data/data/change_label_' + rate_str + '.npy'
        train_label_data = np.load(train_label)

        train_model(train_data, train_label_data, model_path, predict_path)





