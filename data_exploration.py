# Import necessary library
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

from utils.utils import print_info

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    # Read in raw data
    raw_data = pd.read_csv('./data/flight_delays_data.csv')

    # Copy the data for manipulating:
    data = raw_data.copy()

    # Catch a glimpse of data:
    print_info(data)

    ##############
    # Clean data #
    ##############
    # Check whether the dataset contain unknown/missing values or not:
    print('Missing values count: \n', data.isna().sum())

    # Drop the rows which contain unknown/missing values:
    # (because the unknown/missing values only occupy little percentage in the whole dataset(1714/899114))
    data = data.dropna(axis=0, how='any')

    # Reset the index:
    data = data.reset_index(drop=True)

    # First drop the variable we already know they are unnecessary: flight_id and delay_time
    #   1. Because we have row id in dataframe in pandas, we don't need another identity id `flight_id`.
    #   2. We cannot know how much time a flight will delay in advance so we can't see `delay_time` as a independent variable when testing.
    data = data.drop(columns=['flight_id', 'delay_time'])

    # Time-related variables should be transformed:
    # Transform data type from string to datetime:
    data['flight_date'] = pd.to_datetime(data['flight_date'])

    # Transform `flight_date` to `day_of_week`, `day_of_month`, `day_of_year`, and `Season` to capture the periodicity with different time spans.
    data = data.rename(columns={'Week': 'week_of_year'})
    data['Season'] = data['flight_date'].dt.quarter
    data['day_of_week'], data['day_of_month'], data['day_of_year'] = data['flight_date'].dt.dayofweek, data[
        'flight_date'].dt.day, data['flight_date'].dt.dayofyear

    # After transforming, `flight_date` can be dropped:
    data = data.drop(columns='flight_date')

    print_info(data)

    # Variable encoding:
    # There is a problem: if there are some categories in testing data not appear in training data, then encoding will be invalid.
    # Here I use TargetEncoder in scikit-learn contrib because we have to save the encoding model to use when testing.

    # Target encoding:
    cols = ['flight_no', 'Departure', 'Arrival', 'Airline']
    for col in cols:
        te = TargetEncoder()
        te.fit(data[col], data['is_claim'])
        with open('./transform/' + 'target_' + col + '_encoder', 'wb') as f:
            pickle.dump(te, f)

        data[col] = te.transform(data[col])

    # Transfer is_claim variable to 0, 1:
    data['is_claim'] = data['is_claim'].apply(lambda x: 0 if x == 0 else 1)

    # Check overall statistics: std of `Departure` is 0
    print(data.describe().transpose())

    # Check if all values in `Departure` column is the same: drop it because it doesn't contain useful information.
    print(raw_data['Departure'].unique())
    data = data.drop(columns='Departure')

    #################
    # Normalization #
    #################
    # Feature normalization is not necessary but it makes model training more stable.
    # Here we just observe the mean and standard deviation.
    # Remember to add Normalization layer into model later.
    print(data.describe().transpose()[['mean', 'std']])

    ####################
    # Inspect the data #
    ####################
    # Observe if independent variables are highly correlated:
    # `Season` and `week_of_year` are highly correlated, so we can use only `week_of_year` to capture the seasonality.
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    data = data.drop(columns='Season')

    ##################
    # Data imbalance #
    ##################
    # Here we just observe whether data imbalance exists or not.
    # Remember to deal with it when training.
    names = ['0', '1']
    freqs = [len(data.loc[data['is_claim'] == 0]), len(data.loc[data['is_claim'] == 1])]

    fig, ax = plt.subplots()
    ax.bar(names, freqs)
    for n, f in zip(names, freqs):
        ax.text(n, f + 1, f, ha='center', va='bottom')

    x = np.arange(len(names))
    plt.bar(x, freqs, tick_label=names)
    plt.title('Bar')
    plt.xlabel('is_claim')
    plt.ylabel('Frequency')
    plt.show()

    ###############################################
    # Split data to training set and testing set: #
    ###############################################
    # Split data:
    train_data = data.sample(frac=0.8, random_state=777)
    test_data = data.drop(train_data.index)

    # Split features from labels
    train_features, test_features = train_data.copy(), test_data.copy()
    train_labels, test_labels = train_features.pop('is_claim'), test_features.pop('is_claim')

    # Do the z-score normalization: Here we use training features to fit the scaler.
    z_score_scaler = StandardScaler()
    z_score_scaler.fit(train_features)
    with open('./transform/z_score_scaler', 'wb') as f:
        pickle.dump(z_score_scaler, f)

    train_features_normalized = pd.DataFrame(z_score_scaler.transform(train_features), columns=train_features.keys())
    test_features_normalized = pd.DataFrame(z_score_scaler.transform(test_features), columns=test_features.keys())

    # Check overall statistics again:
    print(train_features_normalized.describe().transpose())
    print(test_features_normalized.describe().transpose())

    # Save all split data for convenience:
    train_features_normalized.to_csv('./data/train_features_normalized.csv', index=False)
    test_features_normalized.to_csv('./data/test_features_normalized.csv', index=False)
    train_labels.to_csv('./data/train_labels.csv', index=False)
    test_labels.to_csv('./data/test_labels.csv', index=False)
