import tensorflow as tf
K = tf.keras.backend

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth