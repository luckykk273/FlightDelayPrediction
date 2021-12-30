from keras.losses import mean_absolute_error, mean_squared_error


def compromised_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    return alpha * mean_absolute_error(y_true, y_pred) + beta * mean_squared_error(y_true, y_pred)
