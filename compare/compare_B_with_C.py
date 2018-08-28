import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    model_path_1 = 'detail_data/change_label_01.npz'
    model_path_2 = 'detail_data/din_407.npz'

    model_data_1 = np.load(model_path_1)
    model_data_2 = np.load(model_path_2)

    print('B-accuracy: ' + str(model_data_1['accuracy']))
    print('C-accuracy: ' + str(model_data_2['accuracy']))

    # 统计
    both_right = []
    both_wrong = []
    only_B_right = []
    only_C_right = []

    predict_1 = model_data_1['predict']
    predict_2 = model_data_2['predict']
    standard_label = mnist.test.labels

    for i in range(len(standard_label)):
        label = np.argmax(standard_label[i])
        label1 = predict_1[i]
        label2 = predict_2[i]

        # print(label, label1, label2)
        if label1 == label:
            if label2 == label:
                both_right.append(i)
            else:
                only_B_right.append(i)
        else:
            if label2 == label:
                only_C_right.append(i)
            else:
                both_wrong.append(i)

    both_right = np.asarray(both_right)
    both_wrong = np.asarray(both_wrong)
    only_B_right = np.asarray(only_B_right)
    only_C_right = np.asarray(only_C_right)
    print('len-both_right:, len-only_B_right:, len-only_C_right:, len-both_wrong:')
    print(len(both_right), len(only_B_right), len(only_C_right), len(both_wrong))

    # 逐层比较
    level1_model1 = model_data_1['level1']
    level2_model1 = model_data_1['level2']
    level3_model1 = model_data_1['level3']
    level1_model2 = model_data_2['level1']
    level2_model2 = model_data_2['level2']
    level3_model2 = model_data_2['level3']

    # 随机选择一张图片
    br_index = np.random.randint(len(both_right))
    oB_index = np.random.randint(len(only_B_right))
    oC_index = np.random.randint(len(only_C_right))
    bw_index = np.random.randint(len(both_wrong))

    check_list = [both_right[br_index], only_B_right[oB_index], only_C_right[oC_index], both_wrong[bw_index]]
    print(check_list)

    # check_list = [2551, 4571, 4731, 2387]

    for index in check_list:
        print()
        print('第{}张图片'.format(index))
        print(np.argmax(standard_label[index]), predict_1[index], predict_2[index])
        level11 = level1_model1[index]
        level12 = level1_model2[index]
        level21 = level2_model1[index]
        level22 = level2_model2[index]
        level31 = level3_model1[index]
        level32 = level3_model2[index]
        label = np.argmax(standard_label[index])

        # 模型激活的神经元统计
        # 各自激活的神经元
        # print()
        print('第一层')
        rate11 = 1
        # print('各自激活的神经元,阈值为{}'.format(rate11))
        temp11 = np.where(level11 > rate11)
        # print(temp11)
        temp112 = np.where(level12 > rate11)
        # print(temp112)
        # 激活值差值
        rate12 = 1
        # print('激活值差的绝对值大于阈值的神经元如下，阈值为{}'.format(rate12))
        compare1 = level11 - level12
        compare1 = np.abs(compare1)
        minus1 = np.where(compare1 > rate12)
        # print(minus1)

        print('第二层')
        rate21 = 1
        # print('各自激活的神经元,阈值为{}'.format(rate21))
        temp21 = np.where(level21 > rate21)
        # print(temp21)
        temp212 = np.where(level22 > rate21)
        # print(temp212)
        # 激活值差值
        rate22 = 1
        # print('激活值差的绝对值大于阈值的神经元如下，阈值为{}'.format(rate22))
        compare2 = level21 - level22
        compare2 = np.abs(compare2)
        minus2 = np.where(compare2 > rate22)
        # print(minus2)

        print('第三层')
        rate31 = 0.1
        # print('各自激活的神经元,阈值为{}'.format(rate31))
        temp31 = np.where(level31 > rate31)
        # print(temp31)
        temp312 = np.where(level32 > rate31)
        # print(temp312)
        # 激活值差值
        rate32 = 0.05
        print('激活值差的绝对值大于阈值的神经元如下，阈值为{}'.format(rate32))
        compare3 = level31 - level32
        compare3 = np.abs(compare3)
        minus3 = np.where(compare3 > rate32)
        print(minus3)

        # # 第一层图片
        # # plt.hist(range(128), 128, weights=level11, color='blue', alpha=0.7, label='B model')
        # # plt.hist(range(128), 128, weights=level12, label='C model', color='red')
        # # plt.xlabel('neuron location in level 1 for label ' + str(label))
        # # plt.ylabel('activation value')
        # # plt.legend()
        # # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_1_label_' + str(label) + '.png')
        # # plt.show()
        #
        # plt.hist(range(128), 128, weights=level12, label='C model', color='red')
        # plt.hist(range(128), 128, weights=level11, color='blue', alpha=0.7, label='B model')
        # plt.xlabel('neuron location in level 1 for label ' + str(label))
        # plt.ylabel('activation value')
        # plt.legend()
        # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_1_label_' + str(label) + '.png')
        # plt.show()
        #
        # # 第二层图片
        # # plt.hist(range(64), 64, weights=level21, color='blue', alpha=0.7, label='B model')
        # # plt.hist(range(64), 64, weights=level22, label='C model', color='red')
        # # plt.xlabel('neuron location in level 2 for label ' + str(label))
        # # plt.ylabel('activation value')
        # # plt.legend()
        # # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_2_label_' + str(label) + '.png')
        # # plt.show()
        #
        # plt.hist(range(64), 64, weights=level22, label='C model', color='red')
        # plt.hist(range(64), 64, weights=level21, color='blue', alpha=0.7, label='B model')
        # plt.xlabel('neuron location in level 2 for label ' + str(label))
        # plt.ylabel('activation value')
        # plt.legend()
        # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_2_label_' + str(label) + '.png')
        # plt.show()
        #
        # # 第三层图片
        # # plt.hist(range(10), 10, weights=level31, color='blue', alpha=0.7, label='B model')
        # # plt.hist(range(10), 10, weights=level32, label='C model', color='red')
        # # plt.xlabel('neuron location in level 3 for label ' + str(label))
        # # plt.ylabel('activation value')
        # # plt.legend()
        # # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_3_label_' + str(label) + '.png')
        # # plt.show()
        #
        # plt.hist(range(10), 10, weights=level32, label='C model', color='red')
        # plt.hist(range(10), 10, weights=level31, color='blue', alpha=0.7, label='B model')
        # plt.xlabel('neuron location in level 3 for label ' + str(label))
        # plt.ylabel('activation value')
        # plt.legend()
        # plt.savefig("compare_pic/compare_index_" + str(index) + '_level_3_label_' + str(label) + '.png')
        # plt.show()

