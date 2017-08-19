import numpy as np

def conver2onehot(annotation):
    shape = np.shape(annotation)
    res_shape = []
    res_shape.extend(shape)
    res_shape.append(151)
    res = np.zeros(res_shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for z in range(shape[2]):
                res[i, j, z, annotation[i, j, z]] = 1
    return res