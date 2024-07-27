from fpilot.common import np, jtu
import sys, inspect


def register_pytree(module):
    if isinstance(module, list):
        for mod in module:
            register_pytree(mod)
    else:
        jtu.register_pytree_node(module, module.flatten, module.unflatten)


class Mean:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    def update(self, value):
        self.value += value
        self.count += 1

    def compute(self):
        return self.value / (self.count + 1e-7)

    def reset(self):
        self.value = 0.0
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
        self.value = 0.0
        self.count = 0

    def update_value(self, value):
        self.value += value
        self.count += 1

    def update(self, true, pred):
        raise NotImplementedError

    def compute(self):
        return self.value / (self.count + 1e-7)

    def reset(self):
        self.value = 0.0
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

        self.tp = np.zeros(self.m_shape)
        self.fp = np.zeros(self.m_shape)
        self.fn = np.zeros(self.m_shape)
        self.tn = np.zeros(self.m_shape)
        self.count = 0

    def update(self, true, pred):
        pred = pred > self.threshold
        self.tp += (true * pred).sum(axis=self.axis)
        self.fp += ((1 - true) * pred).sum(axis=self.axis)
        self.fn += (true * (1 - pred)).sum(axis=self.axis)
        self.tn += ((1 - true) * (1 - pred)).sum(axis=self.axis)
        self.count += 1

    def reset(self):
        self.tp = np.zeros(self.m_shape)
        self.fp = np.zeros(self.m_shape)
        self.fn = np.zeros(self.m_shape)
        self.tn = np.zeros(self.m_shape)
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


register_pytree([Mean, MeanMetric, ClassMetric])
