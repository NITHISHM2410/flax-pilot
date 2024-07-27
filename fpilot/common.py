# Utils
import gc
import time
import random
import functools
from typing import *
from tqdm import tqdm
from mergedeep import merge as dmerge

# Computations
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from flax.jax_utils import replicate, unreplicate