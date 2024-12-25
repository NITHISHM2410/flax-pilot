# General
import gc
import time
import typing
import random
import functools
from tqdm import tqdm
from copy import deepcopy as copy
from mergedeep import merge as dmerge

# Jax Computations
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from flax.jax_utils import replicate, unreplicate

# Flax
from flax import linen as nn
import orbax.checkpoint as ocp

# Optax
import optax as tx

