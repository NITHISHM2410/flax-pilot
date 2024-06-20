import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node


class Mean:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, value):
        self.value += value
        self.count += 1

    def compute(self):
        return self.value / (self.count + 1e-7)

    def reset(self):
        self.value = 0
        self.count = 0

    def flatten(self):
        children = (self.value, self.count)
        aux_data = None
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls()
        obj.value, obj.count = children
        return obj


class MeanMetric:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update_value(self, value):
        self.value += value
        self.count += 1

    def update(self, true, pred):
        raise NotImplementedError

    def compute(self):
        return self.value / (self.count + 1e-7)

    def reset(self):
        self.value = 0
        self.count = 0

    def flatten(self):
        children = (self.value, self.count)
        aux_data = None
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls()
        obj.value, obj.count = children
        return obj


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


class ClassMetric:
    def __init__(self, num_classes, average, threshold):
        self.axis = None
        self.average = average
        self.threshold = threshold
        self.num_classes = num_classes

        assert self.average.lower() in ('micro', 'macro'), "Currently supported averages: micro, macro."
        if self.average.lower() != 'micro':
            self.axis = 0
            self.m_shape = (self.num_classes,)
        else:
            self.m_shape = ()

        self.tp = self.tn = self.fp = self.fn = 0
        self.count = 0

    def update(self, true: jax.Array, pred: jax.Array):
        pred = pred > self.threshold
        self.tp += (true * pred).sum(axis=self.axis)
        self.fp += ((1 - true) * pred).sum(axis=self.axis)
        self.fn += (true * (1 - pred)).sum(axis=self.axis)
        self.tn += ((1 - true) * (1 - pred)).sum(axis=self.axis)
        self.count += 1

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self.count = 0

    def flatten(self):
        children = (self.tp, self.tn, self.fp, self.fn, self.count)
        aux_data = {'threshold': self.threshold, 'num_classes': self.num_classes, 'average': self.average}
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls(**aux_data)
        obj.tp, obj.tn, obj.fp, obj.fn, obj.count = children
        return obj


class BinaryAccuracy(MeanMetric):
    def __init__(self, threshold):
        """
        Binary Accuracy/ MultiLabel accuracy.

        """
        self.threshold = threshold
        super().__init__()

    def update(self, true: jax.Array, pred: jax.Array):
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

    def update(self, true: jax.Array, pred: jax.Array):
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


register_pytree_node(Mean, Mean.flatten, Mean.unflatten)
register_pytree_node(MSE, MSE.flatten, MSE.unflatten)
register_pytree_node(MAE, MAE.flatten, MAE.unflatten)
register_pytree_node(ClassMetric, ClassMetric.flatten, ClassMetric.unflatten)
register_pytree_node(MeanMetric, MeanMetric.flatten, MeanMetric.unflatten)
register_pytree_node(Accuracy, Accuracy.flatten, Accuracy.unflatten)
register_pytree_node(BinaryAccuracy, BinaryAccuracy.flatten, BinaryAccuracy.unflatten)
register_pytree_node(Recall, Recall.flatten, Recall.unflatten)
register_pytree_node(Precision, Precision.flatten, Precision.unflatten)
register_pytree_node(F1Score, F1Score.flatten, F1Score.unflatten)
