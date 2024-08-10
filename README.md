# Flax-Pilot

Flax-Pilot aims to simplify the process of writing training loops for Google's Flax framework. As someone new to Flax, I started this project to deepen my understanding. This module represents a beginner's exploration into building
efficient training workflows, emphasizing the need for further expertise to refine and expand its capabilities. Future plans include integrating multiple optimizer training, diverse metric modules, callbacks, and advancing towards more complex training
loops, aiming to enhance its functionality and versatility. Flax-Pilot supports distributed training, ensuring scalability and efficiency across multiple devices.

**As of 27-7-2024, the trainer is available as package [![PyPI version](https://img.shields.io/pypi/v/flax-pilot.svg)](https://pypi.org/project/flax-pilot/) for GPU and [![PyPI version](https://img.shields.io/pypi/v/flax-pilot-cpu.svg)](https://pypi.org/project/flax-pilot-cpu/) for CPU.**

## How to Use?

### 🛠️ Write a flax.linen Module

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

### 🔧 Define Optimizer, Input Shapes, and Dict of Loss & Metric Trackers
*Loss trackers (**lt**) takes in **scalar loss** value and averages it throughout training.*<br>
*Metric trackers (**mt**) takes in **y_true, y_pred** and computes metric score and averages throughout training.*<br>

```python
import optax as tx

opt = tx.adam(0.0001)
input_shape = {'x': (1, 28, 28, 1)}

from fpilot import BasicTrackers as tr

# Create tracker instances.
loss_metric_tracker_dict = {
    'lt': {'loss': tr.Mean()},
    'mt': {'F1': tr.F1Score(threshold=0.6, num_classes=10, average='macro')}
}
```

### 🧮 Create loss_fn
A function that takes these certain params as written below in the code and returns scalar loss, dict of loss & metrics values.<br>

Key names **lt**, **mt** shouldn't be changed anywhere, as training loops depend on those keys. Subkey names, **loss**, **F1** are free to be changed
but must match across **loss_metric_tracker_dict** and **loss_metric_value_dict**.<br>
```python
import optax as tx

# This fn's 1st return value is differentiated wrt the fn's first param.
def loss_fn(params, apply, sample, deterministic, det_key, step):
    x, y = sample
    yp = apply(params, x, deterministic=deterministic, rngs={'dropout': det_key})
    loss = tx.softmax_cross_entropy(y, yp).mean()
    loss_metric_value_dict = {'lt': {'loss': loss}, 'mt': {'F1': (y, yp)}}
    return loss, loss_metric_value_dict
```

### 🏋️ Create Trainer Instance

```python
from fpilot import Trainer

trainer = Trainer(CNN(), input_shape, optimizer, loss_fn, loss_metric_tracker_dict)
```

### 📈 Train the Model & Evaluate
```python
train_ds = ... # tf.data.Dataset as numpy iterator
val_ds = ... # tf.data.Dataset as numpy iterator
train_steps, val_steps = 10000, 1000 # steps per epoch
ckpt_path = "/saved/model/model_1"  # If set to None, no checkpoints will be saved during training.

trainer.train(epochs, train_ds, val_ds, train_steps, val_steps, ckpt_path)
```

## What's next?
- Seperate package for TPU.
- Callbacks.
- TensorBoard logging.

## Demo
Review the 'examples' folder for training tutorials. The `vae-gan-cfg-using-pretrained` notebook demonstrates how to use 
the trainer as a Python package, while the other notebooks show how to use the trainer with git clone. 
Therefore, see the vae-gan-cfg-using-pretrained for a more simpler training.

