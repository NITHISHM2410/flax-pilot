from fpilot.trackers import *
from fpilot.common import jnp


class MSE(MeanMetric):
    def __init__(self):
        super().__init__()

    def update(self, true, pred):
        value = ((true - pred) ** 2).mean()
        self.update_value(value)


class MAE(MeanMetric):
    def __init__(self):
        super().__init__()

    def update(self, true, pred):
        value = jnp.abs(true - pred).mean()
        self.update_value(value)


class BinaryAccuracy(MeanMetric):
    def __init__(self, threshold):
        """
        Binary Accuracy/ MultiLabel accuracy.

        """
        self.threshold = threshold
        super().__init__()

    def update(self, true, pred):
        pred = pred > self.threshold
        acc = (true == pred).mean()
        self.update_value(acc)

    def flatten(self):
        children = (self.value, self.count)
        aux_data = {'threshold': self.threshold}
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls(**aux_data)
        obj.value, obj.count = children
        return obj


class Accuracy(MeanMetric):
    def __init__(self, ):
        """
        Multiclass Accuracy.
        Expects both true and pred to be one-hot encoded.

        """
        super().__init__()

    def update(self, true, pred):
        true, pred = true.argmax(axis=-1), pred.argmax(axis=-1)
        acc = (true == pred).mean()
        self.update_value(acc)


class Recall(ClassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        r = (self.tp / (self.tp + self.fn + 1e-7)).mean()
        return r


class Precision(ClassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        p = (self.tp / (self.tp + self.fp + 1e-7)).mean()
        return p


class F1Score(ClassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        r = self.tp / (self.tp + self.fn + 1e-7)
        p = self.tp / (self.tp + self.fp + 1e-7)
        f1 = (2 * r * p / (r + p + 1e-7)).mean()
        return f1


register_pytree([MAE, MSE, BinaryAccuracy, Accuracy, Precision, Recall, F1Score])
