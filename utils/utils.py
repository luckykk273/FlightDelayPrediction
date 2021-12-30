import numpy as np


def print_info(data):
    print('Data Information: ')
    print(data.info(), '\n\n')
    print('The first 10 data: ')
    print(data.head(10))


def predict_batch(data, model, batch_size):
    res = np.array([])
    for i in range(0, len(data), batch_size):
        pred = model.predict(data[i:i + batch_size])
        res = np.concatenate((res, pred), axis=0)

    res = np.asarray(res, dtype=np.int32)

    return res
