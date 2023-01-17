import tensorflow as tf
K = tf.keras.backend


def Iou_point_5(y_true, y_pred):
    return IoU(y_true, y_pred, 0.5)


def Iou_point_6(y_true, y_pred):
    return IoU(y_true, y_pred, 0.6)


def Iou_point_7(y_true, y_pred):
    return IoU(y_true, y_pred, 0.7)


def Iou_point_8(y_true, y_pred):
    return IoU(y_true, y_pred, 0.8)


def Iou_point_9(y_true, y_pred):
    return IoU(y_true, y_pred, 0.9)


def IoU(y_true, y_pred, threshold=None):
    """
    labels,prediction with shape of [batch,height,width,class_number=2]
    """
    labels = y_true
    predictions = y_pred
    # Perform threshold

    if threshold is None:
        labels_c = K.cast(labels, K.floatx())
        pred_c = K.cast(predictions, K.floatx())
    else:
        predictions_thresholded = tf.cast(predictions > threshold, tf.int32)

        labels_c = K.cast(K.equal(labels, 1), K.floatx())
        pred_c = K.cast(K.equal(predictions_thresholded, 1), K.floatx())

    labels_c_sum = K.sum(labels_c)
    pred_c_sum = K.sum(pred_c)

    intersect = K.sum(labels_c*pred_c)
    union = labels_c_sum + pred_c_sum - intersect
    iou = intersect / union

    return iou

def IoU_fz(y_true, y_pred):
    labels_c = K.cast(y_true, K.floatx())
    pred_c = K.cast(y_pred, K.floatx())

    # Fuzzy-set intersection / union
    iou = K.sum(K.minimum(labels_c, pred_c)) / K.sum(K.maximum(labels_c, pred_c))
    return iou
