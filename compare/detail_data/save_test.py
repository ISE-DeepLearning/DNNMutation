import numpy as np

if __name__ == '__main__':
    data = np.load('change_label_1.npz')
    print(data['accuracy'], data['predict'].shape, data['level1'].shape, data['level2'].shape, data['level3'].shape)
