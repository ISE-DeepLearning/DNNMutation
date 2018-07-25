#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Activation
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

input_data = Input((28*28,))
temp_data1 = Dense(128)(input_data)
temp_data2 = Activation('relu')(temp_data1)
temp_data3 = Dense(64)(temp_data2)
temp_data4 = Activation('relu')(temp_data3)
temp_data5 = Dense(10)(temp_data4)
output_data = Activation('softmax')(temp_data5)
model = Model(inputs=[input_data],outputs=[output_data])
modelcheck = ModelCheckpoint('basic_model.hdf5', monitor='loss', verbose=1,save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit([mnist.train.images], [mnist.train.labels], batch_size=256, epochs=40, callbacks=[modelcheck], validation_data=(mnist.test.images, mnist.test.labels))


