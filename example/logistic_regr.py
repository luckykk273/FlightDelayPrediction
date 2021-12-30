from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pandas as pd
import pickle


def train_logistic_regr(train_features, train_labels, test_features, test_labels):
    # Base model: Logistic regression
    logistic_regr = LogisticRegression()
    logistic_regr.fit(train_features, train_labels)
    print(logistic_regr.coef_)
    print(logistic_regr.intercept_)

    # Save logistic model:
    with open('../model/logistic', 'wb') as f:
        pickle.dump(logistic_regr, f)

    # Training accuracy:
    train_pred = logistic_regr.predict(train_features)
    train_acc = logistic_regr.score(train_features, train_labels)
    print(train_acc)

    # Testing accuracy:
    test_pred = logistic_regr.predict(test_features)
    test_acc = logistic_regr.score(test_features, test_labels)
    print(test_acc)

    # Confusion matrix:
    print(confusion_matrix(train_labels, train_pred))
    print(confusion_matrix(test_labels, test_pred))

    # Check the prediction is imbalanced or not: we can see the predictions are overfitted to label=0.
    print('Ground truth(Training):', len(train_labels[train_labels == 0]), len(train_labels[train_labels == 1]),
          len(train_labels[train_labels == 0]) / len(train_labels[train_labels == 1]))
    print('Ground truth(Testing):', len(test_labels[test_labels == 0]), len(test_labels[test_labels == 1]),
          len(test_labels[test_labels == 0]) / len(test_labels[test_labels == 1]))
    print('Prediction(Training):', len(train_pred[train_pred == 0]), len(train_pred[train_pred == 1]),
          len(train_pred[train_pred == 0]) / len(train_pred[train_pred == 1]))
    print('Prediction(Testing):', len(test_pred[test_pred == 0]), len(test_pred[test_pred == 1]),
          len(test_pred[test_pred == 0]) / len(test_pred[test_pred == 1]))


if __name__ == '__main__':
    train_features = pd.read_csv('../data/train_features_normalized.csv')
    test_features = pd.read_csv('../data/test_features_normalized.csv')
    train_labels = pd.read_csv('../data/train_labels.csv')
    test_labels = pd.read_csv('../data/test_labels.csv')

    train_logistic_regr(train_features, train_labels['is_claim'], test_features, test_labels['is_claim'])
