# Copyright 2023 (c) OpenAI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import haiku as hk
import functools
import jax.numpy as jnp

from .. import sde_lib
import jax
import numpy as np
from . import wideresnet_noise_conditional
from .. import checkpoints
from ..utils import T, batch_mul


# The dataclass that stores all training states
@flax.struct.dataclass
class State:
    step: int
    lr: float
    ema_rate: float
    params: Any
    params_ema: Any
    model_state: Any
    opt_state: Any
    rng_state: Any


@flax.struct.dataclass
class StateWithTarget:
    step: int
    lr: float
    ema_rate: float
    params: Any
    target_params: Any
    params_ema: Any
    model_state: Any
    opt_state: Any
    rng_state: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def init_model(rng, config):
    """Initialize a `flax.linen.Module` model."""
    rng = hk.PRNGSequence(rng)
    model_name = config.model.name
    model_def = functools.partial(get_model(model_name), config=config)
    input_shape = (
        jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    model = model_def()
    variables = model.init(
        {"params": next(rng), "dropout": next(rng)}, fake_input, fake_label
    )
    # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
    init_model_state, initial_params = variables.pop("params")
    return model, init_model_state, initial_params


def init_lpips(rng, config):
    assert config.training.loss_norm.lower() == "lpips", "LPIPS is not used in training"
    from .lpips import LPIPS

    model = LPIPS()
    params = model.init(rng, jnp.zeros((1, 256, 256, 3)), jnp.zeros((1, 256, 256, 3)))
    return model, params


def get_model_fn(model, params, states, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: A `flax.linen.Module` object the represent the architecture of score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all mutable states.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels, rng=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          rng: If present, it is the random state for dropout

        Returns:
          A tuple of (model output, new mutable states)
        """
        variables = {"params": params, **states}
        if not train:
            return model.apply(variables, x, labels, train=False, mutable=False), states
        else:
            rngs = {"dropout": rng}
            return model.apply(
                variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs
            )

    return model_fn


def get_denoiser_fn(sde, model, params, states, train=False, return_state=False):
    model_fn = get_model_fn(model, params, states, train=train)
    assert isinstance(
        sde, sde_lib.KVESDE
    ), "Only KVE SDE is supported for building the denoiser"

    def denoiser_fn(x, t, rng=None):
        in_x = batch_mul(x, 1 / jnp.sqrt(t**2 + sde.data_std**2))
        cond_t = 0.25 * jnp.log(t)
        denoiser, state = model_fn(in_x, cond_t, rng)
        denoiser = batch_mul(
            denoiser, t * sde.data_std / jnp.sqrt(t**2 + sde.data_std**2)
        )
        skip_x = batch_mul(x, sde.data_std**2 / (t**2 + sde.data_std**2))
        denoiser = skip_x + denoiser

        if return_state:
            return denoiser, state
        else:
            return denoiser

    return denoiser_fn


def get_distiller_fn(
    sde, model, params, states, train=False, return_state=False, pred_t=None
):
    assert isinstance(
        sde, sde_lib.KVESDE
    ), "Only KVE SDE is supported for building the denoiser"
    model_fn = get_model_fn(model, params, states, train=train)

    if pred_t is None:
        pred_t = sde.t_min

    def distiller_fn(x, t, rng=None):
        in_x = batch_mul(x, 1 / jnp.sqrt(t**2 + sde.data_std**2))
        cond_t = 0.25 * jnp.log(t)
        denoiser, state = model_fn(in_x, cond_t, rng)
        denoiser = batch_mul(
            denoiser,
            (t - pred_t) * sde.data_std / jnp.sqrt(t**2 + sde.data_std**2),
        )
        skip_x = batch_mul(
            x, sde.data_std**2 / ((t - pred_t) ** 2 + sde.data_std**2)
        )
        denoiser = skip_x + denoiser

        if return_state:
            return denoiser, state
        else:
            return denoiser

    return distiller_fn


def get_gaussianizer_fn(
    sde, model, params, states, train=False, return_state=False, pred_t=None
):
    assert isinstance(
        sde, sde_lib.KVESDE
    ), "Only KVE SDE is supported for building the denoiser"
    model_fn = get_model_fn(model, params, states, train=train)

    if pred_t is None:
        pred_t = sde.t_min

    def gaussianizer_fn(x, t, rng=None):
        in_x = x / sde.data_std
        cond_t = 0.25 * jnp.log(t)
        model_output, state = model_fn(in_x, cond_t, rng)
        model_output = x + batch_mul(model_output, t - pred_t)

        if return_state:
            return model_output, state
        else:
            return model_output

    return gaussianizer_fn


def get_score_fn(sde, model, params, states, train=False, return_state=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      return_state: If `True`, return the new mutable states alongside the model output.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, params, states, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t, rng=None):
            # Scale neural network output by standard deviation and flip sign
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            cond_t = t * 999
            model, state = model_fn(x, cond_t, rng)
            std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            score = batch_mul(-model, 1.0 / std)
            if return_state:
                return score, state
            else:
                return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t, rng=None):
            x = 2 * x - 1.0  # assuming x is in [0, 1]
            std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            score, state = model_fn(x, jnp.log(std), rng)
            score = batch_mul(score, 1.0 / std)
            if return_state:
                return score, state
            else:
                return score

    elif isinstance(sde, sde_lib.KVESDE):
        denoiser_fn = get_denoiser_fn(
            sde, model, params, states, train=train, return_state=True
        )

        def score_fn(x, t, rng=None):
            denoiser, state = denoiser_fn(x, t, rng)
            score = batch_mul(denoiser - x, 1 / t**2)
            if return_state:
                return score, state
            else:
                return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def get_denoiser_and_distiller_fn(
    sde, model, params, states, train=False, return_state=False, pred_t=None
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      return_state: If `True`, return the new mutable states alongside the model output.
      pred_t: The time at which the denoiser is identity.

    Returns:
      A score function.
    """
    assert isinstance(
        sde, sde_lib.KVESDE
    ), "Only KVE SDE is supported for joint training."

    model_fn = get_model_fn(model, params, states, train=train)

    if pred_t is None:
        pred_t = sde.t_min

    from .ncsnpp import NCSNpp, JointNCSNpp

    def denoiser_distiller_fn(x, t, rng=None):
        in_x = batch_mul(x, 1 / jnp.sqrt(t**2 + sde.data_std**2))
        cond_t = 0.25 * jnp.log(t)
        if isinstance(model, NCSNpp):
            model_output, state = model_fn(in_x, cond_t, rng)
            denoiser = model_output[..., :3]
            distiller = model_output[..., 3:]
        elif isinstance(model, JointNCSNpp):
            (denoiser, distiller), state = model_fn(in_x, cond_t, rng)

        denoiser = batch_mul(
            denoiser, t * sde.data_std / jnp.sqrt(t**2 + sde.data_std**2)
        )
        skip_x = batch_mul(x, sde.data_std**2 / (t**2 + sde.data_std**2))
        denoiser = skip_x + denoiser

        distiller = batch_mul(
            distiller,
            (t - pred_t) * sde.data_std / jnp.sqrt(t**2 + sde.data_std**2),
        )
        skip_x = batch_mul(
            x, sde.data_std**2 / ((t - pred_t) ** 2 + sde.data_std**2)
        )
        distiller = skip_x + distiller

        if return_state:
            return (denoiser, distiller), state
        else:
            return denoiser, distiller

    return denoiser_distiller_fn


def to_flattened_numpy(x):
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


def create_classifier(prng_key, batch_size, ckpt_path):
    """Create a noise-conditional image classifier.

    Args:
      prng_key: A JAX random state.
      batch_size: The batch size of input data.
      ckpt_path: The path to stored checkpoints for this classifier.

    Returns:
      classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
      classifier_params: A dictionary that contains trainable parameters of the classifier.
    """
    input_shape = (batch_size, 32, 32, 3)
    classifier = wideresnet_noise_conditional.WideResnet(
        blocks_per_group=4, channel_multiplier=10, num_outputs=10
    )
    initial_variables = classifier.init(
        {"params": prng_key, "dropout": jax.random.PRNGKey(0)},
        jnp.ones(input_shape, dtype=jnp.float32),
        jnp.ones((batch_size,), dtype=jnp.float32),
        train=False,
    )
    model_state, init_params = initial_variables.pop("params")
    classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
    return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
    """Create a logit function for the classifier."""

    def preprocess(data):
        image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
        image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
        return (data - image_mean[None, ...]) / image_std[None, ...]

    def logit_fn(data, ve_noise_scale):
        """Give the logits of the classifier.

        Args:
          data: A JAX array of the input.
          ve_noise_scale: time conditioning variables in the form of VE SDEs.

        Returns:
          logits: The logits given by the noise-conditional classifier.
        """
        data = preprocess(data)
        logits = classifier.apply(
            {"params": classifier_params},
            data,
            ve_noise_scale,
            train=False,
            mutable=False,
        )
        return logits

    return logit_fn


def get_classifier_grad_fn(logit_fn):
    """Create the gradient function for the classifier in use of class-conditional sampling."""

    def grad_fn(data, ve_noise_scale, labels):
        def prob_fn(data):
            logits = logit_fn(data, ve_noise_scale)
            prob = jax.nn.log_softmax(logits, axis=-1)[
                jnp.arange(labels.shape[0]), labels
            ].sum()
            return prob

        return jax.grad(prob_fn)(data)

    return grad_fn
