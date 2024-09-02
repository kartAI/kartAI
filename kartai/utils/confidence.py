import tensorflow as tf

class Confidence(tf.keras.metrics.Metric):
    def __init__(self, confusion_weight=1.0, **kwargs):
        super(Confidence, self).__init__(**kwargs)
        self.confidence_sum = self.add_weight(
            name='confidence_sum', initializer='zeros')
        self.count = self.add_weight(name="count", initializer="zeros")
        self.confidence = []
        if confusion_weight < 0.:
            confusion_weight = 0.
        elif confusion_weight > 1.:
            confusion_weight = 1.
        self.confusion_weight = confusion_weight

    def update_state(self, y_true, y_pred, sample_weight=None):
        raw_confidence = tf.math.abs(y_pred - 0.5) * 2
        raw_confusion = tf.math.abs(y_pred - tf.cast(y_true, self._dtype))

        confidence = raw_confidence - raw_confusion * self.confusion_weight
        confidence = tf.math.maximum(confidence, 0.)
        confidence = tf.math.reduce_mean(confidence, axis=(1, 2, 3))
        self.confidence_sum.assign_add(tf.math.reduce_sum(confidence))
        self.count.assign_add(tf.cast(tf.size(confidence), self._dtype))

        # Return confidence
        self.confidence = confidence

    def result(self):
        return {"confidence": tf.math.divide_no_nan(self.confidence_sum, self.count),
                "sample_confidence": self.confidence}

    def reset_state(self):
        self.confidence_sum.assign(0.)
        self.count.assign(0.)
        self.confidence = None

