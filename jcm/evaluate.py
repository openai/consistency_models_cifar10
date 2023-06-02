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

import io
import os
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import logging
import functools
import haiku as hk
import math
from collections import defaultdict

from . import checkpoints

# Keep the import below for registering all model definitions
from .models import ddpm, ncsnv2, ncsnpp

from .models import utils as mutils
from . import losses
from . import sampling
from . import datasets
from . import metrics
from . import likelihood
from . import sde_lib
from .metrics import get_samples_from_ckpt
import blobfile


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    blobfile.makedirs(eval_dir)

    rng = hk.PRNGSequence(config.seed + 1)
    # Initialize model
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

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    sde = sde_lib.get_sde(config)
    # Add one additional round to get the exact number of samples as required.

    # num_sampling_rounds and num_bpd_rounds must be computed in all cases.
    num_sampling_rounds = int(
        math.ceil(config.eval.num_samples / config.eval.batch_size)
    )
    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd = datasets.get_dataset(
        config,
        additional_dim=None,
        uniform_dequantization=True,
        evaluation=True,
        drop_last=False,
    )
    if config.eval.bpd_dataset.lower() == "train":
        ds_bpd = train_ds_bpd
    elif config.eval.bpd_dataset.lower() == "test":
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    num_bpd_rounds = len(ds_bpd)

    if config.eval.enable_loss:
        # Build datasets
        train_ds, eval_ds = datasets.get_dataset(
            config,
            additional_dim=1,
            uniform_dequantization=config.data.uniform_dequantization,
            evaluation=True,
            drop_last=False,
        )
        # Create the one-step evaluation function when loss computation is enabled
        train_loss_fn, eval_loss_fn, state = losses.get_loss_fn(
            config, sde, score_model, state, next(rng)
        )

        ema_scales_fn = losses.get_ema_scales_fn(config)

        eval_step = losses.get_step_fn(
            eval_loss_fn,
            train=False,
            optimize_fn=optimize_fn,
            ema_scales_fn=ema_scales_fn,
        )
        # Pmap (and jit-compile) multiple evaluation steps together for faster execution
        p_eval_step = jax.pmap(
            functools.partial(jax.lax.scan, eval_step),
            axis_name="batch",
        )

    if config.eval.enable_bpd:
        # Build the likelihood computation function when likelihood is enabled
        likelihood_fn = likelihood.get_likelihood_fn(
            sde,
            score_model,
            num_repeats=5 if config.eval.bpd_dataset.lower() == "test" else 1,
        )

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size // jax.local_device_count(),
            config.data.image_size,
            config.data.image_size,
            config.data.num_channels,
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape)

    # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
    rng = hk.PRNGSequence(jax.random.fold_in(next(rng), jax.process_index()))

    # A data class for storing intermediate results to resume evaluation after pre-emption
    @flax.struct.dataclass
    class EvalMeta:
        ckpt_id: int
        sampling_round_id: int
        bpd_round_id: int
        rng_state: Any

    # Restore evaluation after pre-emption
    eval_meta = EvalMeta(
        ckpt_id=config.eval.begin_ckpt,
        sampling_round_id=-1,
        bpd_round_id=-1,
        rng_state=rng.internal_state,
    )
    eval_meta = checkpoints.restore_checkpoint(
        eval_dir, eval_meta, step=None, prefix=f"meta_{jax.process_index()}_"
    )

    # avoid not starting from config.eval.begin_ckpt.
    if eval_meta.ckpt_id < config.eval.begin_ckpt:
        eval_meta = eval_meta.replace(
            ckpt_id=config.eval.begin_ckpt,
            sampling_round_id=-1,
            bpd_round_id=-1,
            rng_state=rng.internal_state,
        )

    # Evaluation order: first loss, then likelihood, then sampling
    if eval_meta.bpd_round_id < num_bpd_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = eval_meta.bpd_round_id + 1
        begin_sampling_round = 0

    elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
        begin_ckpt = eval_meta.ckpt_id
        begin_bpd_round = num_bpd_rounds
        begin_sampling_round = eval_meta.sampling_round_id + 1

    else:
        begin_ckpt = eval_meta.ckpt_id + 1
        begin_bpd_round = 0
        begin_sampling_round = 0

    rng.replace_internal_state(eval_meta.rng_state)
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        ## Part 1: Load checkpoint
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
        while not blobfile.exists(ckpt_filename):
            if not waiting_message_printed and jax.process_index() == 0:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        try:
            state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        except:
            time.sleep(60)
            try:
                state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
            except:
                raise OSError("checkpoint file is not ready for reading")

        # Replicate the training state to prepare for pmap
        pstate = flax.jax_utils.replicate(state)
        ## Part 2: Compute loss
        if config.eval.enable_loss:
            all_losses = []
            all_log_stats = defaultdict(list)
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = jax.tree_util.tree_map(
                    lambda x: x.detach().cpu().numpy(), batch
                )
                next_rng = jnp.asarray(rng.take(jax.local_device_count()))
                (_, _), (
                    p_eval_loss,
                    p_eval_log_stats,
                ) = p_eval_step((next_rng, pstate), eval_batch)
                eval_loss = flax.jax_utils.unreplicate(p_eval_loss)
                eval_log_stats = flax.jax_utils.unreplicate(p_eval_log_stats)

                all_losses.extend(eval_loss)
                for key, value in eval_log_stats.items():
                    all_log_stats[key].extend(value)

                if (i + 1) % 1000 == 0 and jax.process_index() == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = jnp.asarray(all_losses)
            all_log_stats = jax.tree_map(lambda x: jnp.asarray(x), all_log_stats)
            with blobfile.BlobFile(
                os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb"
            ) as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer,
                    all_losses=all_losses,
                    mean_loss=all_losses.mean(),
                    **all_log_stats,
                )
                fout.write(io_buffer.getvalue())

        ## Part 3: Compute likelihood (bits/dim)
        if config.eval.enable_bpd:
            bpds = []
            bpd_iter = iter(ds_bpd)
            for _ in range(begin_bpd_round):
                next(bpd_iter)
            for i, eval_batch in enumerate(bpd_iter):
                eval_batch = jax.tree_util.tree_map(
                    lambda x: x.detach().cpu().numpy(), eval_batch
                )
                step_rng = jnp.asarray(rng.take(jax.local_device_count()))
                bpd = likelihood_fn(step_rng, pstate, eval_batch["image"])[0]
                bpd = bpd.reshape(-1)
                bpds.extend(bpd)
                bpd_round_id = begin_bpd_round + i
                logging.info(
                    "ckpt: %d, round: %d, mean bpd: %6f"
                    % (ckpt, bpd_round_id, jnp.mean(jnp.asarray(bpds)))
                )
                # Save bits/dim to disk or Google Cloud Storage
                with blobfile.BlobFile(
                    os.path.join(
                        eval_dir,
                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz",
                    ),
                    "wb",
                ) as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, bpd)
                    fout.write(io_buffer.getvalue())

                eval_meta = eval_meta.replace(
                    ckpt_id=ckpt,
                    bpd_round_id=bpd_round_id,
                    rng_state=rng.internal_state,
                )
                # Save intermediate states to resume evaluation after pre-emption
                checkpoints.save_checkpoint(
                    eval_dir,
                    eval_meta,
                    step=ckpt * (num_bpd_rounds + num_sampling_rounds) + bpd_round_id,
                    keep=1,
                    prefix=f"meta_{jax.process_index()}_",
                )
        else:
            # Skip likelihood computation and save intermediate states for pre-emption
            eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
            checkpoints.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_bpd_rounds + num_sampling_rounds) + num_bpd_rounds - 1,
                keep=1,
                prefix=f"meta_{jax.process_index()}_",
            )

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:
            logging.info(f"Start sampling evaluation for ckpt {ckpt}")
            # Run sample generation for multiple rounds to create enough samples
            # Designed to be pre-emption safe. Automatically resumes when interrupted
            for r in range(begin_sampling_round, num_sampling_rounds):
                if jax.process_index() == 0:
                    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}_host_{jax.process_index()}"
                )
                blobfile.makedirs(this_sample_dir)
                sample_rng = jnp.asarray(rng.take(jax.local_device_count()))
                samples, n = sampling_fn(sample_rng, pstate)
                samples = (samples + 1.0) / 2.0
                samples = np.clip(samples * 255.0, 0, 255).astype(np.uint8)
                samples = samples.reshape(
                    (
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        config.data.num_channels,
                    )
                )
                # Write samples to disk or Google Cloud Storage
                with blobfile.BlobFile(
                    os.path.join(this_sample_dir, f"samples_{r}.npz"),
                    "wb",
                ) as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                # Save image samples and submit to the FID evaluation website
                if r == num_sampling_rounds - 1:
                    # Collect samples from all hosts and sampling rounds
                    if jax.process_index() == 0:
                        all_samples = get_samples_from_ckpt(eval_dir, ckpt)
                    all_samples = all_samples[: config.eval.num_samples]
                    sample_path = os.path.join(eval_dir, f"ckpt_{ckpt}_samples.npz")
                    with blobfile.BlobFile(sample_path, "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, all_samples)
                        fout.write(io_buffer.getvalue())

                # Update the intermediate evaluation state
                eval_meta = eval_meta.replace(
                    ckpt_id=ckpt, sampling_round_id=r, rng_state=rng.internal_state
                )
                # Save intermediate states to resume evaluation after pre-emption
                checkpoints.save_checkpoint(
                    eval_dir,
                    eval_meta,
                    step=ckpt * (num_sampling_rounds + num_bpd_rounds)
                    + r
                    + num_bpd_rounds,
                    keep=1,
                    prefix=f"meta_{jax.process_index()}_",
                )

        else:
            # Skip sampling and save intermediate evaluation states for pre-emption
            eval_meta = eval_meta.replace(
                ckpt_id=ckpt,
                sampling_round_id=num_sampling_rounds - 1,
                rng_state=rng.internal_state,
            )
            checkpoints.save_checkpoint(
                eval_dir,
                eval_meta,
                step=ckpt * (num_sampling_rounds + num_bpd_rounds)
                + num_sampling_rounds
                - 1
                + num_bpd_rounds,
                keep=1,
                prefix=f"meta_{jax.process_index()}_",
            )

        begin_bpd_round = 0
        begin_sampling_round = 0

    # Remove all meta files after finishing evaluation
    meta_files = blobfile.glob(os.path.join(eval_dir, f"meta_{jax.process_index()}_*"))
    for file in meta_files:
        blobfile.remove(file)
