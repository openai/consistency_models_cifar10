# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import jax
import flax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
import haiku as hk

from jcm.utils import T
from .models import utils as mutils
import diffrax


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    ## Reverse-mode differentiation (slower)
    # def div_fn(x, t, eps):
    #     grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
    #     grad_fn_eps = jax.grad(grad_fn)(x)
    #     return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

    ## Forward-mode differentiation (faster)
    def div_fn(x, t, eps):
        jvp = jax.jvp(lambda x: fn(x, t), (x,), (eps,))[1]
        return jnp.sum(jvp * eps, axis=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(
    sde,
    model,
    hutchinson_type="Rademacher",
    rtol=1e-5,
    atol=1e-5,
    eps=1e-5,
    num_repeats=1,
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        model: A `flax.linen.Module` object that represents the architecture of the score-based model.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
        num_repeats: The number of times to repeat the black-box ODE solver for reduced variance.

    Returns:
        A function that takes random states, replicated training states, and a batch of data points
            and returns the log-likelihoods in bits/dim, the latent code, and the number of function
            evaluations cost by computation.
    """

    def drift_fn(state, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = mutils.get_score_fn(
            sde,
            model,
            state.params_ema,
            state.model_state,
            train=False,
        )
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def likelihood_fn(rng, state, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            rng: An array of random states.
            state: Replicated training state for running on multiple devices.
            data: A JAX array of shape [batch size, ...].

        Returns:
            bpd: A JAX array of shape [batch size]. The log-likelihoods on `data` in bits/dim.
            z: A JAX array of the same shape as `data`. The latent representation of `data` under the
                probability flow ODE.
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        div_fn = get_div_fn(lambda x, t: drift_fn(state, x, t))

        rng = hk.PRNGSequence(rng)
        shape = data.shape
        if hutchinson_type == "Gaussian":
            epsilon = jax.random.normal(next(rng), shape)
        elif hutchinson_type == "Rademacher":
            epsilon = jax.random.rademacher(next(rng), shape, dtype=data.dtype)
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        ## ODE function for diffrax ODE solver
        def ode_func(t, x, args):
            sample = x[..., :-1]
            vec_t = jnp.ones((sample.shape[0],)) * t
            drift = drift_fn(sample, vec_t)
            logp_grad = div_fn(sample, vec_t, epsilon)
            return jnp.stack([drift, logp_grad], axis=-1)

        term = diffrax.ODETerm(ode_func)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=sde.T,
            t1=eps,
            dt0=eps - sde.T,
            y0=jnp.stack([data, jnp.zeros_like((data.shape[0],))], axis=-1),
            stepsize_controller=stepsize_controller,
        )

        nfe = solution.stats["num_steps"]
        z = solution.ys[-1, ..., :-1]
        delta_logp = solution.ys[-1, ..., -1]
        prior_logp = sde.prior_logp(z)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N
        offset = 7.0
        bpd += offset
        return bpd, z, nfe

    def likelihood_fn_repeated(rng, state, data):
        def loop_fn(i, carry):
            bpd, nfe, rng = carry
            rng, step_rng = jax.random.split(rng)
            bpd_i, z_i, nfe_i = likelihood_fn(step_rng, state, data)
            bpd = bpd + bpd_i
            nfe = nfe + nfe_i
            return bpd, nfe, rng

        bpd, nfe, rng = jax.lax.fori_loop(
            0, num_repeats, loop_fn, (jnp.zeros(data.shape[0]), 0, rng)
        )
        bpd = bpd / num_repeats
        nfe = nfe / num_repeats
        return bpd, nfe

    return jax.pmap(likelihood_fn_repeated, axis_name="batch")
