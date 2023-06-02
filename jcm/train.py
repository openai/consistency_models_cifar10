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

"""Training and evaluation for score-based generative models. """

import os
from typing import Any

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import logging
import functools
import haiku as hk

from . import checkpoints
import wandb

# Keep the import below for registering all model definitions
from .models import ddpm, ncsnv2, ncsnpp

from .models import utils as mutils
from . import losses
from . import sampling
from . import utils
from . import datasets
from . import sde_lib
import blobfile


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    blobfile.makedirs(sample_dir)

    rng = hk.PRNGSequence(config.seed)

    # Initialize model.
    score_model, init_model_state, initial_params = mutils.init_model(next(rng), config)
    optimizer, optimize_fn = losses.get_optimizer(config)

    if config.training.loss.lower().endswith(
        ("ema", "adaptive", "progressive_distillation")
    ):
        state = mutils.StateWithTarget(
            step=0,
            lr=config.optim.lr,
            ema_rate=config.model.ema_rate,
            params=initial_params,
            target_params=initial_params,
            params_ema=initial_params,
            model_state=init_model_state,
            opt_state=optimizer.init(initial_params),
            rng_state=rng.internal_state,
        )
    else:
        state = mutils.State(
            step=0,
            lr=config.optim.lr,
            ema_rate=config.model.ema_rate,
            params=initial_params,
            params_ema=initial_params,
            model_state=init_model_state,
            opt_state=optimizer.init(initial_params),
            rng_state=rng.internal_state,
        )

    # Setup SDEs
    sde = sde_lib.get_sde(config)

    # Build one-step training and evaluation functions
    train_loss_fn, eval_loss_fn, state = losses.get_loss_fn(
        config, sde, score_model, state, next(rng)
    )

    ema_scale_fn = losses.get_ema_scales_fn(config)

    train_step_fn = losses.get_step_fn(
        train_loss_fn,
        train=True,
        optimize_fn=optimize_fn,
        ema_scales_fn=ema_scale_fn,
    )
    # Pmap (and jit-compile) multiple training steps together for faster running
    p_train_step = jax.pmap(
        functools.partial(jax.lax.scan, train_step_fn),
        axis_name="batch",
    )
    eval_step_fn = losses.get_step_fn(
        eval_loss_fn,
        train=False,
        optimize_fn=optimize_fn,
        ema_scales_fn=ema_scale_fn,
    )
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(
        functools.partial(jax.lax.scan, eval_step_fn),
        axis_name="batch",
    )

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    blobfile.makedirs(checkpoint_dir)
    blobfile.makedirs(checkpoint_meta_dir)
    # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
    # `state.step` is JAX integer on the GPU/TPU devices
    initial_step = int(state.step)
    rng.replace_internal_state(state.rng_state)
    # Finished model initialization

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(
        config,
        additional_dim=config.training.n_jitted_steps,
        uniform_dequantization=config.data.uniform_dequantization,
    )
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size // jax.local_device_count(),
            config.data.image_size,
            config.data.image_size,
            config.data.num_channels,
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape)

    # Replicate the training state to run on multiple devices
    pstate = flax_utils.replicate(state)
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    if jax.process_index() == 0:
        logging.info("Starting training loop at step %d." % (initial_step,))

    rng = hk.PRNGSequence(jax.random.fold_in(next(rng), jax.process_index()))

    # JIT multiple training steps together for faster training
    n_jitted_steps = config.training.n_jitted_steps
    # Must be divisible by the number of steps jitted together
    assert (
        config.training.log_freq % n_jitted_steps == 0
        and config.training.snapshot_freq_for_preemption % n_jitted_steps == 0
        and config.training.eval_freq % n_jitted_steps == 0
        and config.training.snapshot_freq % n_jitted_steps == 0
    ), "The number of steps jitted together must be divisible by the logging frequency."

    for step in range(
        initial_step, num_train_steps + 1, config.training.n_jitted_steps
    ):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        try:
            data = next(train_iter)
        except StopIteration:
            # Restart the iterator when the dataset is exhausted.
            train_iter = iter(train_ds)
            data = next(train_iter)
        batch = jax.tree_util.tree_map(lambda x: x.detach().cpu().numpy(), data)
        next_rng = rng.take(jax.local_device_count())
        next_rng = jnp.asarray(next_rng)
        # Execute one training step
        (_, pstate), (ploss, p_log_stats) = p_train_step((next_rng, pstate), batch)
        loss = flax.jax_utils.unreplicate(ploss).mean()
        log_stats = jax.tree_map(
            lambda x: x.mean(), flax.jax_utils.unreplicate(p_log_stats)
        )
        # Log to console, file and tensorboard on host 0
        if jax.process_index() == 0 and step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss))
            if "dsm_loss" in log_stats and "distill_loss" in log_stats:
                logging.info(
                    "step: %d, dsm_loss: %.5e, distill_loss: %.5e"
                    % (step, log_stats["dsm_loss"], log_stats["distill_loss"])
                )
            wandb.log({"training_loss": float(loss)}, step=step)
            for key, value in log_stats.items():
                wandb.log({f"training_{key}": float(value)}, step=step)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                eval_data = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)
                eval_data = next(eval_iter)

            eval_batch = jax.tree_util.tree_map(
                lambda x: x.detach().cpu().numpy(), eval_data
            )
            next_rng = jnp.asarray(rng.take(jax.local_device_count()))
            (_, _), (peval_loss, peval_log_stats) = p_eval_step(
                (next_rng, pstate), eval_batch
            )
            eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
            eval_log_stats = jax.tree_map(
                lambda x: x.mean(), flax.jax_utils.unreplicate(peval_log_stats)
            )

            if jax.process_index() == 0:
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
                if "dsm_loss" in eval_log_stats and "distill_loss" in eval_log_stats:
                    logging.info(
                        "step: %d, dsm_loss: %.5e, distill_loss: %.5e"
                        % (
                            step,
                            eval_log_stats["dsm_loss"],
                            eval_log_stats["distill_loss"],
                        )
                    )
                wandb.log({"eval_loss": float(eval_loss)}, step=step)
                for key, value in eval_log_stats.items():
                    wandb.log({f"eval_{key}": float(value)}, step=step)

        if config.training.loss.lower() == "progressive_distillation":
            ema_scale_fn = losses.get_ema_scales_fn(config)

            if step > 0:
                scales = int(ema_scale_fn(step)[1])
                last_scales = int(ema_scale_fn(step - 1)[1])
                if scales != last_scales:
                    # Move to the next distillation iteration
                    if scales == 2 or scales == 1:
                        config.optim.linear_decay_steps = (
                            config.training.distill_steps_per_iter * 2
                        )
                    elif scales == 1:
                        config.optim.linear_decay_steps = config.training.n_iters - step
                    optimizer, optimize_fn = losses.get_optimizer(config)
                    state = flax.jax_utils.unreplicate(pstate)
                    state = state.replace(
                        target_params=state.params_ema,
                        params=state.params_ema,
                        opt_state=optimizer.init(state.params_ema),
                    )
                    pstate = flax.jax_utils.replicate(state)

                    train_step_fn = losses.get_step_fn(
                        train_loss_fn,
                        train=True,
                        optimize_fn=optimize_fn,
                        ema_scales_fn=ema_scale_fn,
                    )
                    # Pmap (and jit-compile) multiple training steps together for faster running
                    p_train_step = jax.pmap(
                        functools.partial(jax.lax.scan, train_step_fn),
                        axis_name="batch",
                    )
                    eval_step_fn = losses.get_step_fn(
                        eval_loss_fn,
                        train=False,
                        optimize_fn=optimize_fn,
                        ema_scales_fn=ema_scale_fn,
                    )
                    # Pmap (and jit-compile) multiple evaluation steps together for faster running
                    p_eval_step = jax.pmap(
                        functools.partial(jax.lax.scan, eval_step_fn),
                        axis_name="batch",
                    )

        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == num_train_steps
        ):
            # Save the checkpoint.
            if jax.process_index() == 0:
                saved_state = flax_utils.unreplicate(pstate)
                saved_state = saved_state.replace(rng_state=rng.internal_state)
                checkpoints.save_checkpoint(
                    checkpoint_dir,
                    saved_state,
                    step=step // config.training.snapshot_freq,
                    keep=np.inf,
                )

            # Generate and save samples
            if config.training.snapshot_sampling:
                # Use the same random seed for sampling to track progress
                sample_rng_seed = hk.PRNGSequence(42)
                sample_rng = jnp.asarray(sample_rng_seed.take(jax.local_device_count()))
                sample, n = sampling_fn(sample_rng, pstate)
                sample = (sample + 1.0) / 2.0
                this_sample_dir = os.path.join(
                    sample_dir, "iter_{}_host_{}".format(step, jax.process_index())
                )
                blobfile.makedirs(this_sample_dir)
                image_grid = sample.reshape((-1, *sample.shape[2:]))
                nrow = int(np.sqrt(image_grid.shape[0]))
                sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
                with blobfile.BlobFile(
                    os.path.join(this_sample_dir, "sample.np"),
                    "wb",
                ) as fout:
                    np.save(fout, sample)

                with blobfile.BlobFile(
                    os.path.join(this_sample_dir, "sample.png"),
                    "wb",
                ) as fout:
                    utils.save_image(image_grid, fout, nrow=nrow, padding=2)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        # Must execute at the last to avoid corner cases where the main checkpoint was not successfully saved
        if (
            step != 0
            and step % config.training.snapshot_freq_for_preemption == 0
            and jax.process_index() == 0
        ):
            saved_state = flax_utils.unreplicate(pstate)
            saved_state = saved_state.replace(rng_state=rng.internal_state)
            checkpoints.save_checkpoint(
                checkpoint_meta_dir,
                saved_state,
                step=step // config.training.snapshot_freq_for_preemption,
                keep=1,
            )
