from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':

    # B模型
    rate = 1
    rate_list = []
    for i in range(1000):
        rate_list.append(rate / 10000)
        rate += 1

    result = []

    for rate in rate_list:
        B_model = '../change_train_data/modify_label_data/predict/predict_cl_' + str(rate).replace('.', '') + '.npy'
        B_data = np.load(B_model)
        B_predict = list(map(lambda x: np.argmax(x), B_data))

        # DIN
        # for pixel in range(784):
        #     C_model = '../mutation_model/DIN/predict/din_' + str(pixel) + '.npy'
        #     C_data = np.load(C_model)
        #     C_predict = list(map(lambda x: np.argmax(x), C_data))
        #
        #     percent = accuracy_score(B_predict, C_predict)
        #     print('B: cl_'+str(rate)+', C: din_' + str(pixel) + ', similarity: ' + str(percent))
        #     result.append(percent)

        # DHN
        # num_list = [x+1 for x in range(20)]
        # for num in num_list:
        #     for k in range(20):
        #         location = str(num) + '_' + str(k)
        #         C_model = '../mutation_model/DHN/predict/dhn_' + location + '.npy'
        #         C_data = np.load(C_model)
        #         C_predict = list(map(lambda x: np.argmax(x), C_data))
        #
        #         percent = accuracy_score(B_predict, C_predict)
        #         print('B: cl_'+str(rate)+', C: dhn_' + location + ', similarity: ' + str(percent))
        #         result.append(percent)

        # CAF
        num_list = [x+1 for x in range(40)]
        for num in num_list:
            for k in range(20):
                location = str(num) + '_' + str(k)
                C_model = '../mutation_model/CAF/predict/caf_' + location + '.npy'
                C_data = np.load(C_model)
                C_predict = list(map(lambda x: np.argmax(x), C_data))

                percent = accuracy_score(B_predict, C_predict)
                print('B: cl_'+str(rate)+', C: caf_' + location + ', similarity: ' + str(percent))
                result.append(percent)

        # CWV
        # 参数准备
        # num_list = [(i + 1) / 100 for i in range(9)]
        # temp_list = [(i + 1) / 10 for i in range(100)]
        # extent_lst = num_list + temp_list
        # ratio_lst = [(i + 1) / 100 for i in range(20)]
        #
        # for extent in extent_lst:
        #     for ratio in ratio_lst:
        #         name = 'extent_' + str(extent).replace('.', '') + '_ratio_' + str(ratio).replace('.', '')
        #         C_model = '../mutation_model/CWV/predict/cwv_' + name + '.npy'
        #         C_data = np.load(C_model)
        #         C_predict = list(map(lambda x: np.argmax(x), C_data))
        #
        #         percent = accuracy_score(B_predict, C_predict)
        #         print('B: cl_' + str(rate) + ', C: cwv_' + name + ', similarity: ' + str(percent))
        #         result.append(percent)

        # CBV
        # 参数准备
        # num_list = [(i + 1) / 100 for i in range(9)]
        # temp_list = [(i + 1) / 10 for i in range(100)]
        # extent_lst = num_list + temp_list
        # ratio_lst = [(i + 1) / 100 for i in range(20)]
        #
        # for extent in extent_lst:
        #     for ratio in ratio_lst:
        #         name = 'extent_' + str(extent).replace('.', '') + '_ratio_' + str(ratio).replace('.', '')
        #         C_model = '../mutation_model/CBV/predict/cbv_' + name + '.npy'
        #         C_data = np.load(C_model)
        #         C_predict = list(map(lambda x: np.argmax(x), C_data))
        #
        #         percent = accuracy_score(B_predict, C_predict)
        #         print('B: cl_' + str(rate) + ', C: cbv_' + name + ', similarity: ' + str(percent))
        #         result.append(percent)
    print(result)
    result = np.asarray(result)
    print(np.max(result))
    print(np.min(result))
    np.save('compare_data_cl_caf_2.npy', result)
    origin = np.load('compare_data_cl_caf.npy')
    new = np.append(origin, result)
    print(new.shape)
    np.save('compare_data_cl_caf.npy', new)

