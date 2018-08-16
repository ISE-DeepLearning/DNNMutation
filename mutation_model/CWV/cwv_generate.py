# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import save_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import csv


def HDF5_structure(data):

    root = data.keys()
    root = list(root)

    final_path = []
    data_path = []
    while True:
        if len(root) == 0:
            break
        else:
            for item in root:
                # print(data[item].items())
                if isinstance(data[item], h5py._hl.dataset.Dataset) or len(data[item].items()) == 0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item], h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item, sub_item[0]))
                    root.remove(item)
    return data_path


def model_mutation_single_neuron(model,cls='kernel',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model
    cls: 'kernel' or 'bias'
    extent:ratio
    '''
    json_string = model.to_json()
    model.save_weights('my_model_weight.h5')
    data = h5py.File('my_model_weight.h5', 'r+')
    data_path = HDF5_structure(data)
    lst = []

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        try:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        except:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change

def model_mutation_single_neuron_cnn(model,cls='kernel',layers='dense',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model or CNN_model
    cls: 'kernel' or 'bias'
    layers: 'dense' or 'conv'
    extent:ratio
    '''
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    data=h5py.File('my_model_weight.h5','r+')
    data_path=HDF5_structure(data)
    lst=[]

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if layers not in path.split('/')[0]:
            continue

        if len(data[path].shape)==4:
            a,b,c,d=data[path].shape
            lst.extend([(path,a_index,b_index,c_index,d_index) for a_index in range(a) for b_index in range(b) for c_index in range(c) for d_index in range(d)])
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        if len(path)==3:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==2:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==5:
            arr=data[path[0]][int(path[1])][int(path[2])][int(path[3])].copy()
            arr[int(path[4])]*=extent
            data[path[0]][int(path[1])][int(path[2])][int(path[3])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change



class model_mutation_del_neuron(object):
    '''
    1、初始化是模型
    2、首先看有几个全链接层，以及全连接层每层有多少个神经元
    3、del_neuron的输入是第几层神经元和第几个神经元
    4、可反复变异
    '''

    def __init__(self, model):
        self.model = model

    def get_neuron(self):
        neuron_num = 0
        layer_num = []
        self.model.save_weights('my_model_weight.h5')
        data = h5py.File('my_model_weight.h5', 'r+')

        data_path = HDF5_structure(data)
        # print data_path
        self.data_path = []
        for path in data_path:
            if os.path.basename(path.split(':')[0]) != 'kernel':
                continue
            self.data_path.append(path)
            neuron_num += data[path].shape[0]
            layer_num.append(data[path].shape[0])
        #print self.data_path
        #print self.data_path[2]
        #print data[self.data_path[2]].shape
        #print type(data[self.data_path[2]])
        #print data[self.data_path[2]][783].shape
        #print data[self.data_path[2]][456][67]
        data.close()
        print("get_neuron", neuron_num, layer_num)
        return neuron_num, layer_num

    def del_neuron(self, data, neuron_index):
        '''
        neuron_index:(layer_num,index)
        '''
        layer_num, index = neuron_index
        #print layer_num,index
        json_string = self.model.to_json()
        path = self.data_path[layer_num]
        #print data[path].shape
        data_change = data
        arr = data[path][index].copy()
        arr *= 0
        data_change[path][index] = arr
        
        #print('parameter:{}'.format(model.count_params()))
        #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
        #print('extend :{}'.format(extent))
        return

    def mask_input(self, ndim, index):
        '''
        ndim:总维数
        index:需要删除的维
        '''
        json_string = self.model.to_json()
        self.model.save_weights('my_model_weight.h5')
        data = h5py.File('my_model_weight.h5', 'r+')
        for i in range(len(self.data_path)):
            if data[self.data_path[i]].shape[0] == ndim:
                for j in range(data[self.data_path[i]].shape[1]):
                    arr = data[self.data_path[i]][index].copy()
                    arr[j] *= 0
                    data[self.data_path[i]][index] = arr
        data.close()
        model_change = model_from_json(json_string)
        model_change.load_weights('my_model_weight.h5')
        return model_change
    
    def del_neuron_random(self, ndim, num, loopnum):
    #num:每次删除的神经元个数
    #loopnum:循环的次数
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.model.save_weights('my_model_weight.h5')
        data = h5py.File('my_model_weight.h5','r+')
        lst = []
        statis = []
        for i in range(len(self.data_path)):
            if data[self.data_path[i]].shape[0] == ndim: #记录一下是不是输入层神经元
                break
            for index in range(data[self.data_path[i]].shape[0]):
                temp = (i, index)
                lst.append(temp)
        
        #print lst
        json_string = self.model.to_json()
        model_temp = model_from_json(json_string)
        for loop in range(loopnum):
            random_choice = np.random.choice(len(lst), num)
            print(random_choice)
            for j in range(num):
                #print 'num',num
                #print lst[random_choice[j]] 
                self.del_neuron(data, lst[random_choice[j]])
            model_temp.load_weights('my_model_weight.h5')
            statis.append(accuracy_mnist(model_temp,mnist))
            print('accuracy of origin:',accuracy_mnist(self.model,mnist))
            print('accuracy of change:',accuracy_mnist(model_temp,mnist))
            data.close()
            self.model.save_weights('my_model_weight.h5')
            data = h5py.File('my_model_weight.h5','r+')
        data.close()
        return statis


def accuracy_mnist(model, mnist, index):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    pred = model.predict(mnist.test.images)
    path = 'predict/cwv_' + str(index) + '.npy'
    np.save(path, pred)

    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), mnist.test.labels))
    return accuracy_score(test_label, pred)


def bp_kernel(model_path):
    model = load_model(model_path)
    extent_lst=[0.01,0.1,0.5,1.5,2,3,5,10]
    statistic={i:[] for i in extent_lst}
    ratio_lst=[0.01,0.03,0.05,0.1,0.2]

    for extent in extent_lst:
        for ratio in ratio_lst:
            lst=[]
            for i in range(10):
                _,_,_,_,model_change=model_mutation_single_neuron(model,cls='kernel',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist)
                lst.append(acc)
                print(i,acc)
        statistic[extent].append(lst)
    return statistic


if __name__=='__main__':

    mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
    model_path = 'basic_model.hdf5'
    model = load_model(model_path)

    # 参数准备
    num_list = [(i + 1)/100 for i in range(9)]
    temp_list = [(i+1)/10 for i in range(100)]
    extent_lst = num_list + temp_list
    ratio_lst = [(i + 1)/100 for i in range(20)]
    statistic_list = []

    for extent in extent_lst:
        for ratio in ratio_lst:
            name = 'extent_' + str(extent).replace('.', '') + '_ratio_' + str(ratio).replace('.', '')
            _, _, _, _, model_change = model_mutation_single_neuron(model, cls='kernel', random_ratio=ratio, extent=extent)
            acc = accuracy_mnist(model_change, mnist, name)
            statistic_list.append(acc)
            model_path = 'model/cwv_' + name + '.hdf5'
            save_model(model_change, model_path)
            print(name, acc)
    print(statistic_list)

