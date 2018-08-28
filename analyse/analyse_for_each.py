import numpy as np
import matplotlib.pyplot as plt


def get_count(data, min, max):
    pos = np.where((data >= min) & (data < max))
    return len(pos[0])


if __name__ == '__main__':
    # 数据处理
    cl_din_path = 'compare_data_din_cl.npy'
    cl_dhn_path = 'compare_data_dhn_cl.npy'
    cl_caf_path = 'compare_data_cl_caf.npy'
    cl_cwv_path = 'compare_data_cl_cwv.npy'
    cl_cbv_path = 'compare_data_cl_cbv.npy'

    cl_din = np.load(cl_din_path)
    cl_dhn = np.load(cl_dhn_path)
    cl_caf = np.load(cl_caf_path)
    cl_cwv = np.load(cl_cwv_path)
    cl_cbv = np.load(cl_cbv_path)

    cl_din = cl_din.reshape(1000, 784)
    cl_dhn = cl_dhn.reshape(1000, 400)
    cl_caf = cl_caf.reshape(1000, 800)
    cl_cwv = cl_cwv.reshape(1000, 2180)
    cl_cbv = cl_cbv.reshape(1000, 2180)

    print(cl_din.shape)
    print(cl_dhn.shape)
    print(cl_caf.shape)
    print(cl_cwv.shape)
    print(cl_cbv.shape)

    max_list = []
    for i in range(1000):
        index_list = np.append(cl_din[i], cl_dhn[i])
        index_list = np.append(index_list, cl_caf[i])
        index_list = np.append(index_list, cl_cwv[i])
        index_list = np.append(index_list, cl_cbv[i])
        max_list.append(np.max(index_list))

    max_list = np.asarray(max_list)
    count = 0
    start = 98
    x_axis = []
    y1_axis = []
    y2_axis = []
    length = len(max_list)
    while count < length:
        end = start + 1
        result = get_count(max_list, start/100, end/100)
        count += result
        print(start/100, result/length, count/length)
        x_axis.append(start/100)
        y1_axis.append(result)
        y2_axis.append(count/length)
        start -= 1

    plt.hist(max_list, len(x_axis), weights=np.zeros_like(max_list)+100.0/length*0.01, color='blue', alpha=0.7, range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    plt.xlabel('Segmentation value')
    plt.ylabel('Percent')
    plt.title('Compare for change_label and all mutation')
    plt.legend()
    plt.savefig("pic/max_list_hist.png")
    plt.show()

    plt.plot(x_axis, y2_axis, label='Cumulative percent')
    plt.xlabel('Segmentation value')
    plt.ylabel('Cumulative percent')
    plt.title('Compare for change_label and all mutation')
    plt.legend()
    plt.savefig("pic/max_list_plot.png")
    plt.show()
