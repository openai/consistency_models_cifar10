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

"""All functions related to loss computation and optimization.
"""

import optax
import jax
import jax.numpy as jnp
import haiku as hk
import jax.random as random

from . import checkpoints
from .models import utils as mutils
from .utils import batch_mul
from jcm import sde_lib
import numpy as np


def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer.lower() == "adam":
        if hasattr(config.optim, "linear_decay_steps"):  # for progressive distillation
            stable_training_schedule = optax.linear_schedule(
                init_value=config.optim.lr,
                end_value=0.0,
                transition_steps=config.optim.linear_decay_steps,
            )
        else:
            stable_training_schedule = optax.constant_schedule(config.optim.lr)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0,
                    end_value=config.optim.lr,
                    transition_steps=config.optim.warmup,
                ),
                stable_training_schedule,
            ],
            [config.optim.warmup],
        )

        if not np.isinf(config.optim.grad_clip):
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=config.optim.grad_clip),
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.optim.beta1,
                    eps=config.optim.eps,
                    weight_decay=config.optim.weight_decay,
                ),
            )
        else:
            optimizer = optax.adamw(
                learning_rate=schedule,
                b1=config.optim.beta1,
                eps=config.optim.eps,
                weight_decay=config.optim.weight_decay,
            )

    elif config.optim.optimizer.lower() == "radam":
        beta1 = config.optim.beta1
        beta2 = config.optim.beta2
        eps = config.optim.eps
        weight_decay = config.optim.weight_decay
        lr = config.optim.lr
        optimizer = optax.chain(
            optax.scale_by_radam(b1=beta1, b2=beta2, eps=eps),
            optax.add_decayed_weights(weight_decay, None),
            optax.scale(-lr),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    def optimize_fn(grads, opt_state, params):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return optimizer, optimize_fn


def get_loss_fn(config, sde, score_model, state, rng):
    likelihood_weighting = config.training.likelihood_weighting
    if config.training.loss.lower() in ["dsm", "ssm"]:
        ssm = config.training.loss.lower() == "ssm"
        train_loss_fn = get_score_matching_loss_fn(
            sde,
            score_model,
            train=True,
            likelihood_weighting=likelihood_weighting,
            ssm=ssm,
        )
        eval_loss_fn = get_score_matching_loss_fn(
            sde,
            score_model,
            train=False,
            likelihood_weighting=likelihood_weighting,
            ssm=ssm,
        )
    elif config.training.loss.lower().startswith(
        ("continuous", "consistency", "progressive_distillation")
    ):
        optimizer, optimize_fn = get_optimizer(config.training.ref_config)
        rng = hk.PRNGSequence(rng)
        ref_config = config.training.ref_config
        ref_model, init_ref_model_state, init_ref_params = mutils.init_model(
            next(rng), ref_config
        )
        ref_state = mutils.State(
            step=0,
            lr=ref_config.optim.lr,
            ema_rate=ref_config.model.ema_rate,
            params=init_ref_params,
            params_ema=init_ref_params,
            model_state=init_ref_model_state,
            opt_state=optimizer.init(init_ref_params),
            rng_state=rng.internal_state,
        )
        ref_state = checkpoints.restore_checkpoint(
            config.training.ref_model_path, ref_state
        )
        # Initialize the flow model from the denoiser model
        if config.training.finetune:
            state = state.replace(
                params=ref_state.params,
                params_ema=ref_state.params_ema,
                model_state=ref_state.model_state,
            )
        if config.training.loss_norm.lower() == "lpips":
            lpips_model, lpips_params = mutils.init_lpips(next(rng), config)
        else:
            lpips_model, lpips_params = None, None
        if config.training.loss.lower().startswith("continuous"):
            train_loss_fn = get_continuous_consistency_loss_fn(
                sde,
                ref_model,
                ref_state.params_ema,
                ref_state.model_state,
                score_model,
                train=True,
                loss_norm=config.training.loss_norm,
                stopgrad=config.training.stopgrad,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
                dsm_target=config.training.dsm_target,
            )
            eval_loss_fn = get_continuous_consistency_loss_fn(
                sde,
                ref_model,
                ref_state.params_ema,
                ref_state.model_state,
                score_model,
                train=False,
                loss_norm=config.training.loss_norm,
                stopgrad=config.training.stopgrad,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
                dsm_target=config.training.dsm_target,
            )
        elif config.training.loss.lower().startswith("consistency"):
            train_loss_fn = get_consistency_loss_fn(
                sde,
                ref_model,
                ref_state.params_ema,
                ref_state.model_state,
                score_model,
                train=True,
                loss_norm=config.training.loss_norm,
                weighting=config.training.weighting,
                stopgrad=config.training.stopgrad,
                dsm_target=config.training.dsm_target,
                solver=config.training.solver,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
            )
            eval_loss_fn = get_consistency_loss_fn(
                sde,
                ref_model,
                ref_state.params_ema,
                ref_state.model_state,
                score_model,
                train=False,
                loss_norm=config.training.loss_norm,
                weighting=config.training.weighting,
                stopgrad=config.training.stopgrad,
                dsm_target=config.training.dsm_target,
                solver=config.training.solver,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
            )

        elif config.training.loss.lower() == "progressive_distillation":
            train_loss_fn = get_progressive_distillation_loss_fn(
                sde,
                score_model,
                train=True,
                loss_norm=config.training.loss_norm,
                weighting=config.training.weighting,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
            )
            eval_loss_fn = get_progressive_distillation_loss_fn(
                sde,
                score_model,
                train=False,
                loss_norm=config.training.loss_norm,
                weighting=config.training.weighting,
                lpips_model=lpips_model,
                lpips_params=lpips_params,
            )
            assert (
                config.training.finetune
            ), "Finetuning is required for progressive distillation."
            state = state.replace(
                target_params=ref_state.params_ema,
            )

    else:
        raise ValueError(f"Unknown loss {config.training.loss}")

    return train_loss_fn, eval_loss_fn, state


def get_quarter_masks(t, ranges):
    return [(ranges[i] <= t) & (t < ranges[i + 1]) for i in range(len(ranges) - 1)]


def get_consistency_loss_fn(
    sde,
    ref_model,
    ref_params,
    ref_states,
    model,
    train,
    loss_norm="l1",
    weighting="uniform",
    stopgrad=True,
    dsm_target=False,
    solver="heun",
    lpips_model=None,
    lpips_params=None,
):
    assert isinstance(sde, sde_lib.KVESDE), "Only KVE SDEs are supported for now."
    denoiser_fn = mutils.get_denoiser_fn(
        sde,
        ref_model,
        ref_params,
        ref_states,
        train=False,
        return_state=False,
    )

    def heun_solver(samples, t, next_t, x0):
        x = samples
        if dsm_target:
            denoiser = x0
        else:
            denoiser = denoiser_fn(x, t)

        d = batch_mul(1 / t, x - denoiser)

        samples = x + batch_mul(next_t - t, d)
        if dsm_target:
            denoiser = x0
        else:
            denoiser = denoiser_fn(samples, next_t)
        next_d = batch_mul(1 / next_t, samples - denoiser)
        samples = x + batch_mul((next_t - t) / 2, d + next_d)

        return samples

    def euler_solver(samples, t, next_t, x0):
        x = samples
        if dsm_target:
            denoiser = x0
        else:
            denoiser = denoiser_fn(x, t)
        score = batch_mul(1 / t**2, denoiser - x)
        samples = x + batch_mul(next_t - t, -batch_mul(score, t))

        return samples

    if solver.lower() == "heun":
        ode_solver = heun_solver
    elif solver.lower() == "euler":
        ode_solver = euler_solver

    def loss_fn(rng, params, states, batch, target_params=None, num_scales=None):
        rng = hk.PRNGSequence(rng)
        x = batch["image"]
        if target_params is None:
            target_params = params

        if num_scales is None:
            num_scales = sde.N

        indices = jax.random.randint(next(rng), (x.shape[0],), 0, num_scales - 1)
        t = sde.t_max ** (1 / sde.rho) + indices / (num_scales - 1) * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t = t**sde.rho

        t2 = sde.t_max ** (1 / sde.rho) + (indices + 1) / (num_scales - 1) * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t2 = t2**sde.rho

        z = jax.random.normal(next(rng), x.shape)
        x_t = x + batch_mul(t, z)
        dropout_rng = next(rng)
        Ft, new_states = mutils.get_distiller_fn(
            sde, model, params, states, train=train, return_state=True
        )(x_t, t, rng=dropout_rng if train else None)

        x_t2 = ode_solver(x_t, t, t2, x)
        Ft2, new_states = mutils.get_distiller_fn(
            sde, model, target_params, new_states, train=train, return_state=True
        )(x_t2, t2, rng=dropout_rng if train else None)

        if stopgrad:
            Ft2 = jax.lax.stop_gradient(Ft2)

        diffs = Ft - Ft2

        if weighting.lower() == "uniform":
            weight = jnp.ones_like(t)
        elif weighting.lower() == "snrp1":
            weight = 1 / t**2 + 1.0
        elif weighting.lower() == "truncated_snr":
            weight = jnp.maximum(1 / t**2, jnp.ones_like(t))
        elif weighting.lower() == "snr":
            weight = 1 / t**2
        else:
            raise NotImplementedError(f"Weighting {weighting} not implemented")

        if loss_norm.lower() == "l1":
            losses = jnp.abs(diffs)
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "l2":
            losses = diffs**2
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "linf":
            losses = jnp.abs(diffs)
            losses = jnp.max(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "lpips":
            scaled_Ft = jax.image.resize(
                Ft, (Ft.shape[0], 224, 224, 3), method="bilinear"
            )
            scaled_Ft2 = jax.image.resize(
                Ft2, (Ft2.shape[0], 224, 224, 3), method="bilinear"
            )
            losses = jnp.squeeze(lpips_model.apply(lpips_params, scaled_Ft, scaled_Ft2))

        else:
            raise ValueError("Unknown loss norm: {}".format(loss_norm))

        loss = jnp.nansum(losses * batch["mask"] * weight / jnp.sum(batch["mask"]))
        log_stats = {}

        ## Uncomment to log loss per time step
        # for t_index in range(sde.N - 1):
        #     mask = (indices == t_index).astype(jnp.float32)
        #     log_stats["loss_t{}".format(t_index)] = jnp.nansum(
        #         losses * batch["mask"] * mask / jnp.sum(batch["mask"] * mask)
        #     )

        return loss, (new_states, log_stats)

    return loss_fn


def get_progressive_distillation_loss_fn(
    sde,
    model,
    train,
    loss_norm="l2",
    weighting="truncated_snr",
    lpips_model=None,
    lpips_params=None,
):
    assert isinstance(sde, sde_lib.KVESDE), "Only KVE SDEs are supported for now."

    def loss_fn(rng, params, states, batch, target_params, num_scales):
        rng = hk.PRNGSequence(rng)
        x = batch["image"]

        indices = jax.random.randint(next(rng), (x.shape[0],), 0, num_scales)
        t = sde.t_max ** (1 / sde.rho) + indices / num_scales * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t = t**sde.rho

        t2 = sde.t_max ** (1 / sde.rho) + (indices + 0.5) / num_scales * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t2 = t2**sde.rho

        t3 = sde.t_max ** (1 / sde.rho) + (indices + 1) / num_scales * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t3 = t3**sde.rho

        z = jax.random.normal(next(rng), x.shape)
        x_t = x + batch_mul(t, z)

        dropout_rng = next(rng)
        denoised_x, new_states = mutils.get_denoiser_fn(
            sde, model, params, states, train=train, return_state=True
        )(x_t, t, rng=dropout_rng if train else None)

        target_denoiser_fn = mutils.get_denoiser_fn(
            sde,
            model,
            target_params,
            states,
            train=False,
            return_state=False,
        )

        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = target_denoiser_fn(x, t, rng=None)
            score = batch_mul(1 / t**2, denoiser - x)
            samples = x + batch_mul(next_t - t, -batch_mul(score, t))

            return samples

        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - batch_mul(t, batch_mul(x_next_t - x_t, 1 / (next_t - t)))
            return denoiser

        x_t2 = euler_solver(x_t, t, t2)
        x_t3 = euler_solver(x_t2, t2, t3)

        target_x = jax.lax.stop_gradient(euler_to_denoiser(x_t, t, x_t3, t3))

        diffs = denoised_x - target_x

        if loss_norm.lower() == "l1":
            losses = jnp.abs(diffs)
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "l2":
            losses = diffs**2
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "linf":
            losses = jnp.abs(diffs)
            losses = jnp.max(losses.reshape(losses.shape[0], -1), axis=-1)
        elif loss_norm.lower() == "lpips":
            scaled_denoised_x = jax.image.resize(
                denoised_x, (denoised_x.shape[0], 224, 224, 3), method="bilinear"
            )
            scaled_target_x = jax.image.resize(
                target_x, (target_x.shape[0], 224, 224, 3), method="bilinear"
            )
            losses = jnp.squeeze(
                lpips_model.apply(lpips_params, scaled_denoised_x, scaled_target_x)
            )
        else:
            raise ValueError("Unknown loss norm: {}".format(loss_norm))

        if weighting.lower() == "snrp1":
            weight = 1 / t**2 + 1
        elif weighting.lower() == "truncated_snr":
            weight = jnp.maximum(1 / t**2, jnp.ones_like(t))
        elif weighting.lower() == "snr":
            weight = 1 / t**2

        loss = jnp.nansum(losses * batch["mask"] * weight / jnp.sum(batch["mask"]))
        log_stats = {}

        return loss, (new_states, log_stats)

    return loss_fn


def get_continuous_consistency_loss_fn(
    sde,
    ref_model,
    ref_params,
    ref_states,
    model,
    train,
    loss_norm="l1",
    stopgrad=False,
    lpips_model=None,
    lpips_params=None,
    dsm_target=False,
):
    assert isinstance(sde, sde_lib.KVESDE), "Only KVE SDEs are supported for now."
    score_fn = mutils.get_score_fn(
        sde,
        ref_model,
        ref_params,
        ref_states,
        train=False,
        return_state=False,
    )

    def loss_fn(rng, params, states, batch):
        rng = hk.PRNGSequence(rng)
        x = batch["image"]

        # sampling t according to the Heun sampler
        t = jax.random.uniform(
            next(rng),
            (x.shape[0],),
            minval=sde.t_min ** (1 / sde.rho),
            maxval=sde.t_max ** (1 / sde.rho),
        ) ** (sde.rho)

        weightings = jnp.ones_like(t)
        z = jax.random.normal(next(rng), x.shape)
        x_t = x + batch_mul(t, z)

        if dsm_target:
            score_t = batch_mul(x - x_t, 1 / t**2)
        else:
            score_t = score_fn(x_t, t)

        if train:
            step_rng = next(rng)
        else:
            step_rng = None

        def model_fn(data, time):
            return mutils.get_distiller_fn(
                sde, model, params, states, train=train, return_state=True
            )(data, time, rng=step_rng)

        Ft, diffs, new_states = jax.jvp(
            model_fn, (x_t, t), (batch_mul(t, score_t), -jnp.ones_like(t)), has_aux=True
        )

        if loss_norm.lower() == "l1":
            losses = jnp.abs(diffs)
            losses = jnp.mean(losses.reshape(losses.shape[0], -1), axis=1)
        elif loss_norm.lower() == "l2":
            losses = diffs**2
            losses = jnp.sqrt(jnp.sum(losses.reshape(losses.shape[0], -1), axis=1))
        elif loss_norm.lower() == "linf":
            losses = jnp.abs(diffs)
            losses = jnp.max(losses.reshape(losses.shape[0], -1), axis=1)
        elif loss_norm.lower() == "lpips":

            def metric(x):
                scaled_Ft = jax.image.resize(
                    Ft, (Ft.shape[0], 224, 224, 3), method="bilinear"
                )
                x = jax.image.resize(x, (x.shape[0], 224, 224, 3), method="bilinear")
                return jnp.sum(
                    jnp.squeeze(lpips_model.apply(lpips_params, scaled_Ft, x))
                )

            losses = (
                jax.grad(lambda x: jnp.sum(jax.grad(metric)(x) * diffs))(Ft) * diffs
            )
            losses = jnp.sum(losses.reshape(losses.shape[0], -1), axis=1)

        else:
            raise ValueError("Unknown loss norm: {}".format(loss_norm))

        if stopgrad:
            if loss_norm.lower() == "l2":
                pseudo_losses = -jax.lax.stop_gradient(diffs) * Ft
                pseudo_losses = jnp.sum(
                    pseudo_losses.reshape((pseudo_losses.shape[0], -1)), axis=-1
                )
                loss = jnp.nansum(
                    pseudo_losses * batch["mask"] * weightings / jnp.sum(batch["mask"])
                )
            elif loss_norm.lower() == "lpips":

                def metric_fn(x):
                    x = jax.image.resize(
                        x, (x.shape[0], 224, 224, 3), method="bilinear"
                    )
                    y = jax.image.resize(
                        jax.lax.stop_gradient(Ft),
                        (x.shape[0], 224, 224, 3),
                        method="bilinear",
                    )
                    return jnp.sum(jnp.squeeze(lpips_model.apply(lpips_params, x, y)))

                # forward-over-reverse
                def hvp(f, primals, tangents):
                    return jax.jvp(jax.grad(f), primals, tangents)[1]

                pseudo_losses = Ft * hvp(
                    metric_fn,
                    (jax.lax.stop_gradient(Ft),),
                    (-jax.lax.stop_gradient(diffs),),
                )
                pseudo_losses = jnp.sum(
                    pseudo_losses.reshape((pseudo_losses.shape[0], -1)), axis=-1
                )
                loss = jnp.nansum(
                    pseudo_losses * batch["mask"] * weightings / jnp.sum(batch["mask"])
                )
            else:
                raise NotImplementedError

        else:
            loss = jnp.nansum(
                losses * batch["mask"] * weightings / jnp.sum(batch["mask"])
            )

        quarter_masks = get_quarter_masks(
            t,
            np.linspace(sde.t_min ** (1 / sde.rho), sde.t_max ** (1 / sde.rho), 5)
            ** sde.rho,
        )
        loss_q1 = jnp.nansum(
            losses
            * quarter_masks[0]
            * batch["mask"]
            / jnp.sum(quarter_masks[0] * batch["mask"])
        )
        loss_q2 = jnp.nansum(
            losses
            * quarter_masks[1]
            * batch["mask"]
            / jnp.sum(quarter_masks[1] * batch["mask"])
        )
        loss_q3 = jnp.nansum(
            losses
            * quarter_masks[2]
            * batch["mask"]
            / jnp.sum(quarter_masks[2] * batch["mask"])
        )
        loss_q4 = jnp.nansum(
            losses
            * quarter_masks[3]
            * batch["mask"]
            / jnp.sum(quarter_masks[3] * batch["mask"])
        )

        log_stats = {
            "loss": loss,
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
            "loss_q3": loss_q3,
            "loss_q4": loss_q4,
        }

        return loss, (new_states, log_stats)

    return loss_fn


def get_score_matching_loss_fn(
    sde,
    model,
    train,
    likelihood_weighting=False,
    ssm=False,
    eps=1e-5,
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """

    def dsm_loss_fn(rng, params, states, batch):
        """Compute the loss function based on denoising score matching.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        data = batch["image"]
        rng = hk.PRNGSequence(rng)

        if isinstance(sde, sde_lib.KVESDE):
            t = random.normal(next(rng), (data.shape[0],)) * 1.2 - 1.2
            t = jnp.exp(t)
        else:
            t = random.uniform(next(rng), (data.shape[0],), minval=eps, maxval=sde.T)

        z = random.normal(next(rng), data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)

        if isinstance(sde, sde_lib.KVESDE):
            score_fn = mutils.get_score_fn(
                sde,
                model,
                params,
                states,
                train=train,
                return_state=True,
            )
            score, new_model_state = score_fn(perturbed_data, t, rng=next(rng))

            losses = jnp.square(batch_mul(score, std) + z)
            losses = batch_mul(
                losses, (std**2 + sde.data_std**2) / sde.data_std**2
            )
            losses = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1)

        else:
            score_fn = mutils.get_score_fn(
                sde,
                model,
                params,
                states,
                train=train,
                return_state=True,
            )

            score, new_model_state = score_fn(perturbed_data, t, rng=next(rng))

            if not likelihood_weighting:
                losses = jnp.square(batch_mul(score, std) + z)
                losses = jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1)

            else:
                g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
                losses = jnp.square(score + batch_mul(z, 1.0 / std))
                losses = jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.nansum(losses * batch["mask"] / jnp.sum(batch["mask"]))
        quarter_masks = get_quarter_masks(
            t,
            np.linspace(sde.t_min ** (1 / sde.rho), sde.t_max ** (1 / sde.rho), 5)
            ** sde.rho,
        )
        loss_q1 = jnp.nansum(
            losses
            * quarter_masks[0]
            * batch["mask"]
            / jnp.sum(quarter_masks[0] * batch["mask"])
        )
        loss_q2 = jnp.nansum(
            losses
            * quarter_masks[1]
            * batch["mask"]
            / jnp.sum(quarter_masks[1] * batch["mask"])
        )
        loss_q3 = jnp.nansum(
            losses
            * quarter_masks[2]
            * batch["mask"]
            / jnp.sum(quarter_masks[2] * batch["mask"])
        )
        loss_q4 = jnp.nansum(
            losses
            * quarter_masks[3]
            * batch["mask"]
            / jnp.sum(quarter_masks[3] * batch["mask"])
        )

        log_stats = {
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
            "loss_q3": loss_q3,
            "loss_q4": loss_q4,
        }

        return loss, (new_model_state, log_stats)

    def ssm_loss_fn(rng, params, states, batch):
        """Compute the loss function based on sliced score matching.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = mutils.get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            return_state=True,
        )
        data = batch["image"]
        rng = hk.PRNGSequence(rng)
        # DEBUG: beware of eps!
        if isinstance(sde, sde_lib.KVESDE):
            t = random.normal(next(rng), (data.shape[0],)) * 1.2 - 1.2
            t = jnp.exp(t)
        else:
            t = random.uniform(next(rng), (data.shape[0],), minval=eps, maxval=sde.T)
        # t = random.uniform(next(rng), (data.shape[0],), minval=eps, maxval=sde.T)
        z = random.normal(next(rng), data.shape)
        mean, std = sde.marginal_prob(data, t)
        perturbed_data = mean + batch_mul(std, z)

        def score_fn_for_jvp(x):
            return score_fn(x, t, rng=next(rng))

        epsilon = random.rademacher(next(rng), data.shape, dtype=data.dtype)
        score, score_trace, new_model_state = jax.jvp(
            score_fn_for_jvp, (perturbed_data,), (epsilon,), has_aux=True
        )
        score_norm = jnp.mean(jnp.square(score).reshape((score.shape[0], -1)), axis=-1)
        score_trace = jnp.mean(
            (2 * score_trace * epsilon).reshape((score.shape[0], -1)), axis=-1
        )

        if not likelihood_weighting:
            losses = (score_norm + score_trace) * std**2
        elif isinstance(sde, sde_lib.KVESDE):
            losses = score_norm + score_trace
            losses = (
                losses * std**2 * (std**2 + sde.data_std**2) / sde.data_std**2
            )
        else:
            g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
            losses = (score_norm + score_trace) * g2
        loss = jnp.nansum(losses * batch["mask"] / jnp.sum(batch["mask"]))
        quarter_masks = get_quarter_masks(
            t,
            np.linspace(sde.t_min ** (1 / sde.rho), sde.t_max ** (1 / sde.rho), 5)
            ** sde.rho,
        )
        loss_q1 = jnp.nansum(
            losses
            * quarter_masks[0]
            * batch["mask"]
            / jnp.sum(quarter_masks[0] * batch["mask"])
        )
        loss_q2 = jnp.nansum(
            losses
            * quarter_masks[1]
            * batch["mask"]
            / jnp.sum(quarter_masks[1] * batch["mask"])
        )
        loss_q3 = jnp.nansum(
            losses
            * quarter_masks[2]
            * batch["mask"]
            / jnp.sum(quarter_masks[2] * batch["mask"])
        )
        loss_q4 = jnp.nansum(
            losses
            * quarter_masks[3]
            * batch["mask"]
            / jnp.sum(quarter_masks[3] * batch["mask"])
        )

        log_stats = {
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
            "loss_q3": loss_q3,
            "loss_q4": loss_q4,
            "loss": loss,
        }

        return loss, (new_model_state, log_stats)

    return dsm_loss_fn if not ssm else ssm_loss_fn


def get_ema_scales_fn(config):
    if config.training.loss.lower() in ("dsm", "ssm", "continuous", "consistency"):

        def ema_and_scales_fn(step):
            return None, None

    else:

        def ema_and_scales_fn(step):
            if (
                config.training.target_ema_mode == "fixed"
                and config.training.scale_mode == "fixed"
            ):
                target_ema = float(config.training.target_ema)
                scales = int(config.model.num_scales)

            elif (
                config.training.target_ema_mode == "adaptive"
                and config.training.scale_mode == "progressive"
            ):
                start_ema = float(config.training.start_ema)
                start_scales = int(config.training.start_scales)
                end_scales = int(config.training.end_scales)
                total_steps = int(config.training.n_iters)
                scales = jnp.ceil(
                    jnp.sqrt(
                        (step / total_steps)
                        * ((end_scales + 1) ** 2 - start_scales**2)
                        + start_scales**2
                    )
                    - 1
                ).astype(jnp.int32)
                scales = jnp.maximum(scales, 1)
                c = -jnp.log(start_ema) * start_scales
                target_ema = jnp.exp(-c / scales)
                scales = scales + 1
            elif (
                config.training.target_ema_mode == "fixed"
                and config.training.scale_mode == "progdist"
            ):
                start_scales = int(config.training.start_scales)
                distill_steps_per_iter = int(config.training.distill_steps_per_iter)
                distill_stage = step // distill_steps_per_iter
                scales = start_scales // (2**distill_stage)
                scales = jnp.maximum(scales, 2)

                sub_stage = jnp.maximum(
                    step - distill_steps_per_iter * (jnp.log2(start_scales) - 1),
                    0,
                )
                sub_stage = sub_stage // (distill_steps_per_iter * 2)
                sub_scales = 2 // (2**sub_stage)
                sub_scales = jnp.maximum(sub_scales, 1)

                scales = jnp.where(scales == 2, sub_scales, scales)

                target_ema = 1.0
            else:
                raise NotImplementedError

            return target_ema, scales

    return ema_and_scales_fn


def get_step_fn(
    loss_fn,
    train,
    optimize_fn=None,
    ema_scales_fn=None,
):
    """Create a one-step training/evaluation function.

    Args:
        loss_fn: The loss function for training or evaluation. It should have the
            signature `loss_fn(rng, params, states, batch)`.
        train: `True` for training and `False` for evaluation.
        optimize_fn: An optimization function.
        ema_scales_fn: A function that returns the current EMA and number of scales. Useful for progressive training.

    Returns:
        A one-step function for training or evaluation.
    """

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            step = state.step
            params = state.params
            states = state.model_state
            opt_state = state.opt_state
            target_ema, num_scales = ema_scales_fn(step)
            if target_ema is None and num_scales is None:
                (
                    loss,
                    (new_model_state, log_stats),
                ), grad = grad_fn(step_rng, params, states, batch)

                grad = jax.lax.pmean(grad, axis_name="batch")
                new_params, new_opt_state = optimize_fn(grad, opt_state, params)
                new_params_ema = jax.tree_util.tree_map(
                    lambda p_ema, p: p_ema * state.ema_rate
                    + p * (1.0 - state.ema_rate),
                    state.params_ema,
                    new_params,
                )
                step = state.step + 1
                new_state = state.replace(
                    step=step,
                    params=new_params,
                    params_ema=new_params_ema,
                    model_state=new_model_state,
                    opt_state=new_opt_state,
                )
            else:
                target_params = state.target_params
                (loss, (new_model_state, log_stats)), grad = grad_fn(
                    step_rng, params, states, batch, target_params, num_scales
                )
                grad = jax.lax.pmean(grad, axis_name="batch")
                new_params, new_opt_state = optimize_fn(grad, opt_state, params)
                new_params_ema = jax.tree_util.tree_map(
                    lambda p_ema, p: p_ema * state.ema_rate
                    + p * (1.0 - state.ema_rate),
                    state.params_ema,
                    new_params,
                )
                new_target_params = jax.tree_util.tree_map(
                    lambda p_target, p: p_target * target_ema + p * (1.0 - target_ema),
                    target_params,
                    new_params,
                )
                step = state.step + 1
                new_state = state.replace(
                    step=step,
                    params=new_params,
                    params_ema=new_params_ema,
                    target_params=new_target_params,
                    model_state=new_model_state,
                    opt_state=new_opt_state,
                )
        else:
            target_ema, num_scales = ema_scales_fn(state.step)
            if target_ema is None and num_scales is None:
                loss, (_, log_stats) = loss_fn(
                    step_rng,
                    state.params_ema,
                    state.model_state,
                    batch,
                )
            else:
                loss, (_, log_stats) = loss_fn(
                    step_rng,
                    state.params_ema,
                    state.model_state,
                    batch,
                    state.target_params,
                    num_scales,
                )
            new_state = state

        loss = jax.lax.pmean(loss, axis_name="batch")

        mean_log_stats = jax.tree_map(
            lambda x: jax.lax.pmean(x, axis_name="batch"), log_stats
        )

        new_carry_state = (rng, new_state)
        return new_carry_state, (loss, mean_log_stats)

    return step_fn
