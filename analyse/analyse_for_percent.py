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

    # cl_din = cl_din.reshape(1000, 784)
    # cl_dhn = cl_dhn.reshape(1000, 400)
    # cl_cwv = cl_cwv.reshape(1000, 2180)
    # cl_cbv = cl_cbv.reshape(1000, 2180)

    print(cl_din.shape)
    print(cl_dhn.shape)
    print(cl_caf.shape)
    print(cl_cwv.shape)
    print(cl_cbv.shape)

    # DIN
    cl_din = np.log(cl_din)
    print(cl_din.max())
    print(cl_din.min())
    # count = 0
    # start = 98
    # x_axis = []
    # y1_axis = []
    # y2_axis = []
    # length = len(cl_din)
    # while count < length:
    #     end = start + 1
    #     result = get_count(cl_din, start/100, end/100)
    #     count += result
    #     print(start/100, result/length, count/length)
    #     x_axis.append(start/100)
    #     y1_axis.append(result)
    #     y2_axis.append(count/length)
    #     start -= 1
    #
    # plt.hist(cl_din, len(x_axis), weights=np.zeros_like(cl_din)+100.0/length*0.01, color='blue', alpha=0.7, range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Percent')
    # plt.title('Compare for change_label and DIN')
    # plt.legend()
    # plt.savefig("pic/cl_din_hist.png")
    # plt.show()

    # plt.plot(x_axis, y2_axis, label='Cumulative percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Cumulative percent')
    # plt.title('Compare for change_label and DIN')
    # plt.legend()
    # plt.savefig("pic/cl_din_plot.png")
    # plt.show()

    # DHN
    # count = 0
    # start = 98
    # x_axis = []
    # y1_axis = []
    # y2_axis = []
    # length = len(cl_dhn)
    # while count < length:
    #     end = start + 1
    #     result = get_count(cl_dhn, start / 100, end / 100)
    #     count += result
    #     print(start / 100, result/length, count/length)
    #     x_axis.append(start / 100)
    #     y1_axis.append(result)
    #     y2_axis.append(count / length)
    #     start -= 1
    #
    # plt.hist(cl_dhn, len(x_axis), weights=np.zeros_like(cl_dhn) + 100.0 / length * 0.01, color='blue', alpha=0.7,
    #          range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Percent')
    # plt.title('Compare for change_label and DHN')
    # plt.legend()
    # plt.savefig("pic/cl_dhn_hist.png")
    # plt.show()
    #
    # plt.plot(x_axis, y2_axis, label='Cumulative percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Cumulative percent')
    # plt.title('Compare for change_label and DHN')
    # plt.legend()
    # plt.savefig("pic/cl_dhn_plot.png")
    # plt.show()

    # CAF
    # count = 0
    # start = 98
    # x_axis = []
    # y1_axis = []
    # y2_axis = []
    # length = len(cl_caf)
    # while count < length:
    #     end = start + 1
    #     result = get_count(cl_caf, start / 100, end / 100)
    #     count += result
    #     print(start / 100, result/length, count/length)
    #     x_axis.append(start / 100)
    #     y1_axis.append(result)
    #     y2_axis.append(count / length)
    #     start -= 1
    #
    # plt.hist(cl_caf, len(x_axis), weights=np.zeros_like(cl_caf) + 100.0 / length * 0.01, color='blue', alpha=0.7,
    #          range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Percent')
    # plt.title('Compare for change_label and CAF')
    # plt.legend()
    # plt.savefig("pic/cl_caf_hist.png")
    # plt.show()

    # plt.plot(x_axis, y2_axis, label='Cumulative percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Cumulative percent')
    # plt.title('Compare for change_label and CAF')
    # plt.legend()
    # plt.savefig("pic/cl_caf_plot.png")
    # plt.show()

    # CWV
    # count = 0
    # start = 98
    # x_axis = []
    # y1_axis = []
    # y2_axis = []
    # length = len(cl_cwv)
    # while count < length:
    #     end = start + 1
    #     result = get_count(cl_cwv, start / 100, end / 100)
    #     count += result
    #     print(start / 100, result/length, count/length)
    #     x_axis.append(start / 100)
    #     y1_axis.append(result)
    #     y2_axis.append(count / length)
    #     start -= 1
    #
    # plt.hist(cl_cwv, len(x_axis), weights=np.zeros_like(cl_cwv) + 100.0 / length * 0.01, color='blue', alpha=0.7,
    #          range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Percent')
    # plt.title('Compare for change_label and CWV')
    # plt.legend()
    # plt.savefig("pic/cl_cwv_hist.png")
    # plt.show()
    #
    # plt.plot(x_axis, y2_axis, label='Cumulative percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Cumulative percent')
    # plt.title('Compare for change_label and CWV')
    # plt.legend()
    # plt.savefig("pic/cl_cwv_plot.png")
    # plt.show()

    # CBV
    # count = 0
    # start = 98
    # x_axis = []
    # y1_axis = []
    # y2_axis = []
    # length = len(cl_cbv)
    # while count < length:
    #     end = start + 1
    #     result = get_count(cl_cbv, start / 100, end / 100)
    #     count += result
    #     print(start / 100, result/length, count/length)
    #     x_axis.append(start / 100)
    #     y1_axis.append(result)
    #     y2_axis.append(count / length)
    #     start -= 1
    #
    # plt.hist(cl_cbv, len(x_axis), weights=np.zeros_like(cl_cbv) + 100.0 / length * 0.01, color='blue', alpha=0.7,
    #          range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Percent')
    # plt.title('Compare for change_label and CBV')
    # plt.legend()
    # plt.savefig("pic/cl_cbv_hist.png")
    # plt.show()
    #
    # plt.plot(x_axis, y2_axis, label='Cumulative percent')
    # plt.xlabel('Segmentation value')
    # plt.ylabel('Cumulative percent')
    # plt.title('Compare for change_label and CBV')
    # plt.legend()
    # plt.savefig("pic/cl_cbv_plot.png")
    # plt.show()

    # WHOLE
    cl_whole = np.append(cl_din, cl_dhn)
    cl_whole = np.append(cl_whole, cl_caf)
    cl_whole = np.append(cl_whole, cl_cwv)
    cl_whole = np.append(cl_whole, cl_cbv)
    print('cl_whole', cl_whole.shape)
    count = 0
    start = 98
    x_axis = []
    y1_axis = []
    y2_axis = []
    length = len(cl_whole)
    while count < length:
        end = start + 1
        result = get_count(cl_whole, start / 100, end / 100)
        count += result
        # print(start / 100, result/length, count/length)
        x_axis.append(start / 100)
        y1_axis.append(result)
        y2_axis.append(count / length)
        start -= 1

    plt.hist(cl_whole, len(x_axis), weights=np.zeros_like(cl_whole) + 100.0 / length * 0.01, color='blue', alpha=0.7,
             range=(x_axis[-1], x_axis[0] + 0.01), label='Percent')
    plt.xlabel('Segmentation value')
    plt.ylabel('Percent')
    plt.title('Compare for change_label and all')
    plt.legend()
    plt.savefig("pic/cl_whole_hist.png")
    plt.show()

    plt.plot(x_axis, y2_axis, label='Cumulative percent')
    plt.xlabel('Segmentation value')
    plt.ylabel('Cumulative percent')
    plt.title('Compare for change_label and all')
    plt.legend()
    plt.savefig("pic/cl_whole_plot.png")
    plt.show()

    print("均值：din - dhn - caf - cwv -cbv -whole")
    print(np.mean(cl_din))
    print(np.mean(cl_dhn))
    print(np.mean(cl_caf))
    print(np.mean(cl_cwv))
    print(np.mean(cl_cbv))
    print(np.mean(cl_whole))

    print("方差：din - dhn - caf - cwv -cbv -whole")
    print(cl_din.var())
    print(cl_dhn.var())
    print(cl_caf.var())
    print(cl_cwv.var())
    print(cl_cbv.var())
    print(cl_whole.var())

    print("最大值：din - dhn - caf - cwv -cbv -whole")
    print(cl_din.max())
    print(cl_dhn.max())
    print(cl_caf.max())
    print(cl_cwv.max())
    print(cl_cbv.max())
    print(cl_whole.max())

    print("最小值：din - dhn - caf - cwv -cbv -whole")
    print(cl_din.min())
    print(cl_dhn.min())
    print(cl_caf.min())
    print(cl_cwv.min())
    print(cl_cbv.min())
    print(cl_whole.min())





