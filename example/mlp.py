from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from utils.losses import compromised_loss
from utils.metrics import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mlp():
    model = Sequential()
    model.add(Dense(units=32, input_dim=8, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=32, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=16, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=16, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=8, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=8, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=4, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=4, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=2, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=2, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='he_uniform', activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer=Adam(lr=3e-4), loss=compromised_loss, metrics=['accuracy', precision_m, recall_m, f1_score_m])

    return model


def train_mlp_class_weights(train_features, train_labels, test_features, test_labels):
    # Initialize model:
    model = mlp()

    class_weights = {0: len(train_labels[train_labels == 1]), 1: len(train_labels[train_labels == 0])}

    train_features, train_labels, test_features, test_labels = np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)

    # Train model:
    train_history = model.fit(train_features, train_labels, batch_size=16, epochs=100,
                              validation_data=(test_features, test_labels), class_weight=class_weights, verbose=2)

    # Save model:
    model.save('../model/mlp_class_weights.h5')

    # Confusion Matrix:
    train_pred = model.predict_classes(train_features).ravel()
    test_pred = model.predict_classes(test_features).ravel()

    train_cm = pd.crosstab(train_labels, train_pred, rownames=['label'], colnames=['predict'])
    test_cm = pd.crosstab(test_labels, test_pred, rownames=['label'], colnames=['predict'])

    print(train_cm)
    print(test_cm)

    return train_history


def train_mlp_downsampling(train_features, train_labels, test_features, test_labels):
    # Initialize model:
    model = mlp()

    train_features_ok = train_features.iloc[train_labels[train_labels == 0].index]
    train_features_ng = train_features.iloc[train_labels[train_labels == 1].index]
    train_labels_ok = train_labels[train_labels == 0]
    train_labels_ng = train_labels[train_labels == 1]

    train_features_ok, train_features_ng, train_labels_ok, train_labels_ng = \
        np.array(train_features_ok), np.array(train_features_ng), np.array(train_labels_ok), np.array(train_labels_ng)

    test_features, test_labels = np.array(test_features), np.array(test_labels)

    # Train model:
    for i in range(100):
        indices = np.random.choice(np.arange(len(train_features_ok)), size=len(train_features_ng), replace=False)

        composed_train_features = np.concatenate((train_features_ok[indices], train_features_ng), axis=0)
        composed_train_labels = np.concatenate((train_labels_ok[indices], train_labels_ng), axis=0)

        train_history = model.fit(composed_train_features, composed_train_labels, batch_size=32, epochs=20,
                                  validation_data=(test_features, test_labels), verbose=2)

    # Save model:
    model.save('../model/mlp_downsampling.h5')

    # Confusion Matrix:
    train_pred = model.predict_classes(train_features).ravel()
    test_pred = model.predict_classes(test_features).ravel()

    train_cm = pd.crosstab(train_labels, train_pred, rownames=['label'], colnames=['predict'])
    test_cm = pd.crosstab(test_labels, test_pred, rownames=['label'], colnames=['predict'])

    print(train_cm)
    print(test_cm)

    return train_history


def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


if __name__ == '__main__':
    train_features = pd.read_csv('../data/train_features_normalized.csv')
    test_features = pd.read_csv('../data/test_features_normalized.csv')
    train_labels = pd.read_csv('../data/train_labels.csv')
    test_labels = pd.read_csv('../data/test_labels.csv')

    train_history = train_mlp_class_weights(train_features, train_labels['is_claim'], test_features, test_labels['is_claim'])
    show_train_history(train_history)

    # train_history = train_mlp_downsampling(train_features, train_labels['is_claim'], test_features, test_labels['is_claim'])
    # show_train_history(train_history)644
