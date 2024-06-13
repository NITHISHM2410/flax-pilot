# Flax-Pilot

Flax-Pilot aims to simplify the process of writing training loops for Google's Flax framework. As someone new to Flax, I started this project to deepen my understanding. This module represents a beginner's exploration into building
efficient training workflows, emphasizing the need for further expertise to refine and expand its capabilities. Future plans include integrating multiple optimizer training, diverse metric modules, callbacks, and advancing towards more complex training
loops, aiming to enhance its functionality and versatility. Flax-Pilot supports distributed training, ensuring scalability and efficiency across multiple devices.


## How to use?

### Write a flax.linen module

```python
import flax.linen as nn
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  
        x = nn.Dense(features=256)(x)
        x = nn.Dropout(rate=0.5, deterministic=deterministic)(x)
        x = nn.Dense(features=10)(x)
        return x
```

### Define optimizer, input_shapes and dict of loss & metric trackers
*Loss trackers (**lt**) takes in **scalar loss** value and averages it throughout training.*<br>
*Metric trackers (**mt**) takes in **y_true, y_pred** and computes metric score and averages throughout training.*<br>

```python
import optax as tx

opt = tx.adam(0.0001)
input_shape = {'x': (1, 28, 28, 1)}

import fpilot.trackers as tr

# Create tracker instances.
loss_metric_tracker_dict = {
    'lt': {'loss': tr.Mean()},
    'mt': {'F1': tr.F1Score(threshold=0.6)}
}
```

### Create loss_fn
A function that takes these certain params as written below in the code and returns scalar loss, dict of loss & metrics values.<br>

Key names **lt**, **mt** shouldn't be changed anywhere, as training loops depend on those keys. Subkey names, **loss**, **F1** are free to be changed
but must match across **loss_metric_tracker_dict** and **loss_metric_value_dict**.<br>
```python
import optax as tx

# This fn's 1st return value is differentiated wrt the fn's first param.
def loss_fn(params, apply, sample, deterministic, det_key):
    x, y = sample
    yp = apply(params, x, deterministic=deterministic, rngs={'dropout': det_key})
    loss = tx.softmax_cross_entropy(y, yp).mean()
    loss_metric_value_dict = {'lt': {'loss': loss}, 'mt': {'F1': (y, yp)}}
    return loss, loss_metric_value_dict
```

### Create Trainer instance
```python
from fpilot.trainer import Trainer
trainer = Trainer(CNN(), input_shape, optimizer, loss_fn, loss_metric_tracker_dict)
```

### Train the model & eval
```python
train_ds = ... # tf.data.Dataset as numpy iterator
val_ds = ... # tf.data.Dataset as numpy iterator
train_steps, val_steps = .... # steps per epoch
ckpt_path = "/saved/model/model_1"  # If set to None, no checkpoints will be saved during training.

trainer.train(epochs, train_ds, val_ds, train_steps, val_steps, ckpt_path)
```

## Compatibility
- Jax - 0.4.26
- Flax - 0.8.4
- Orbax-checkpoint - 0.5.15
- TensorFlow - 2.16.1

