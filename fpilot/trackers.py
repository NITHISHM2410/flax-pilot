import jax
from jax.tree_util import register_pytree_node


class Mean:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, value):
        self.value += value
        self.count += 1

    def compute(self):
        return self.value / self.count

    def reset(self):
        self.count = self.value = 0

    def flatten(self):
        children = (self.value, self.count)
        aux_data = None
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls()
        obj.value, obj.count = children
        return obj


class MSE(Mean):
    def __init__(self):
        super().__init__()

    def update(self, true, pred):
        value = ((true - pred) ** 2).mean()
        super().update(value)


class MAE(Mean):
    def __init__(self):
        super().__init__()

    def update(self, true, pred):
        value = jax.numpy.abs(true - pred).mean()
        super().update(value)


class CMetric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.tp = self.tn = self.fp = self.fn = 0
        self.count = 0

    def update(self, true: jax.Array, pred: jax.Array, avg=-2):
        pred = pred > self.threshold
        self.tp += (true * pred).sum(axis=avg)
        self.fp += ((1 - true) * pred).sum(axis=avg)
        self.fn += (true * (1 - pred)).sum(axis=avg)
        self.tn += ((1 - true) * (1 - pred)).sum(axis=avg)
        self.count += 1

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self.count = 0

    def flatten(self):
        children = (self.tp, self.tn, self.fp, self.fn, self.count)
        aux_data = {'threshold': self.threshold}
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        obj = cls(**aux_data)
        obj.tp, obj.tn, obj.fp, obj.fn, obj.count = children
        return obj


class Accuracy(CMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self):
        a = ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)).mean(axis=-1)
        return a


class Recall(CMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self):
        r = (self.tp / (self.tp + self.fn + 1e-7)).mean(axis=-1)
        return r


class Precision(CMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self):
        p = (self.tp / (self.tp + self.fp + 1e-7)).mean(axis=-1)
        return p


class F1Score(CMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self):
        r = self.tp / (self.tp + self.fn + 1e-7)
        p = self.tp / (self.tp + self.fp + 1e-7)
        f1 = (2 * r * p / (r + p)).mean(axis=-1)
        return f1


register_pytree_node(Mean, Mean.flatten, Mean.unflatten)
register_pytree_node(MSE, MSE.flatten, MSE.unflatten)
register_pytree_node(MAE, MAE.flatten, MAE.unflatten)
register_pytree_node(CMetric, CMetric.flatten, CMetric.unflatten)
register_pytree_node(Accuracy, Accuracy.flatten, Accuracy.unflatten)
register_pytree_node(Recall, Recall.flatten, Recall.unflatten)
register_pytree_node(Precision, Precision.flatten, Precision.unflatten)
register_pytree_node(F1Score, F1Score.flatten, F1Score.unflatten)
