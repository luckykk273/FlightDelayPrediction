from keras.models import load_model
from utils.utils import predict_batch

import numpy as np
import pandas as pd
import pickle


MAX_DATA_SIZE_TO_PREDICT = 100000


def preprocess(filepath):
    # Read in raw data:
    data = pd.read_csv(filepath)

    # Drop missing value:
    data = data.dropna(axis=0, how='any')
    data = data.reset_index(drop=True)

    # Drop unnecessary columns:
    data = data.drop(columns=['flight_id', 'delay_time', 'Departure'])
    data = data.drop(columns='is_claim')

    # Transform time-related variables:
    data['flight_date'] = pd.to_datetime(data['flight_date'])
    data = data.rename(columns={'Week': 'week_of_year'})
    data['day_of_week'], data['day_of_month'], data['day_of_year'] = \
        data['flight_date'].dt.dayofweek, data['flight_date'].dt.day, data['flight_date'].dt.dayofyear
    data = data.drop(columns='flight_date')

    # Target encoding:
    cols = ['flight_no', 'Arrival', 'Airline']
    for col in cols:
        with open('./transform/' + 'target_' + col + '_encoder', 'rb') as f:
            te = pickle.load(f)

        data[col] = te.transform(data[col])

    # Z-score normalization:
    with open('./transform/z_score_scaler', 'rb') as f:
        z_score_scaler = pickle.load(f)

    data_normalized = z_score_scaler.transform(data)
    # data_normalized = pd.DataFrame(z_score_scaler.transform(data), columns=data.keys())

    return data_normalized


def test_ml(filepath, model_path):
    data_normalized = preprocess(filepath)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Because the prediction time will increase as data size increases,
    # we have to predict batch data and concatenate them instead of predicting them directly.
    res = predict_batch(data_normalized, model,  MAX_DATA_SIZE_TO_PREDICT) \
        if len(data_normalized) > MAX_DATA_SIZE_TO_PREDICT else model.predict(data_normalized)

    return res


def test_dl(filepath, model_path):
    data_normalized = preprocess(filepath)

    model = load_model(model_path, compile=False)

    res = model.predict_classes(data_normalized).ravel()

    return res


if __name__ == '__main__':
    filepath, model_path = './data/flight_delays_data.csv',  './model/logistic'
    res = test_ml(filepath, model_path)
    label = np.unique(res)
    print(label[0], '\t', label[1])
    print(len(res[res == label[0]]), len(res[res == label[1]]))

    model_path = './model/mlp.h5'
    res = test_dl(filepath, model_path)
    label = np.unique(res)
    print(label[0], '\t', label[1])
    print(len(res[res == label[0]]), len(res[res == label[1]]))
