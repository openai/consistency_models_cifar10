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


"""Return training and evaluation/test datasets from config files."""
import jax
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import numpy as np
import blobfile


def get_dataset(
    config,
    additional_dim=None,
    uniform_dequantization=False,
    evaluation=False,
    drop_last=True,
):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
            which equals the number of steps jitted together.
        uniform_dequantization: If `True`, add uniform dequantization to images.
        evaluation: If `True`, fix number of epochs to 1. Default is `False`.
        drop_last: If `True`, drop the last batch if it is smaller than the batch size. Default is `True`.
            if `False`, the last batch will be padded with zeros and a mask will be returned.

    Returns:
        train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    per_device_batch_size = batch_size // jax.device_count()
    # Create additional data dimension when jitting multiple steps together
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

    # Create dataset builders for each dataset.
    # assert config.data.dataset == "CIFAR10", "Only CIFAR10 is supported for now."

    if config.data.dataset.upper() == "CIFAR10":

        def uniform_deq(image_th):
            if uniform_dequantization:
                return (image_th * 255.0 + torch.rand_like(image_th)) / 256.0
            else:
                return image_th

        def data_augmentation(image):
            if config.data.random_flip is True and not evaluation:
                return RandomHorizontalFlip(p=0.5)(image)
            else:
                return image

        transforms = Compose(
            [
                data_augmentation,
                ToTensor(),
                uniform_deq,
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_ds = CIFAR10("./data", train=True, download=True, transform=transforms)
        eval_ds = CIFAR10("./data", train=False, download=True, transform=transforms)

        @torch.no_grad()
        def collate_fn(batch):
            image_tensor = torch.stack([x[0] for x in batch], dim=0)
            label_tensor = torch.tensor([x[1] for x in batch])
            if image_tensor.shape[0] == int(np.prod(batch_dims)):
                image_tensor = (
                    image_tensor.reshape(batch_dims + [3, 32, 32])
                    .transpose(-3, -2)
                    .transpose(-2, -1)
                )
                label_tensor = label_tensor.reshape(batch_dims)
                return {
                    "image": image_tensor,
                    "label": label_tensor,
                    "mask": torch.ones_like(label_tensor),
                }
            # If the batch size is not a multiple of the batch dimension, pad the batch with zeros.
            else:
                pad_size = int(np.prod(batch_dims)) - image_tensor.shape[0]
                padded_image = torch.concat(
                    [
                        image_tensor,
                        torch.zeros(pad_size, 3, 32, 32, dtype=image_tensor.dtype),
                    ],
                    axis=0,
                )
                padded_label = torch.concat(
                    [
                        label_tensor,
                        torch.zeros(pad_size, dtype=label_tensor.dtype),
                    ],
                    axis=0,
                )
                mask = torch.ones(int(np.prod(batch_dims)))
                mask[image_tensor.shape[0] :] = 0.0
                padded_image = (
                    padded_image.reshape(batch_dims + [3, 32, 32])
                    .transpose(-3, -2)
                    .transpose(-2, -1)
                )
                padded_label = padded_label.reshape(batch_dims)
                mask = mask.reshape(batch_dims)
                return {"image": padded_image, "label": padded_label, "mask": mask}

    train_ds_loader = DataLoader(
        train_ds,
        batch_size=int(np.prod(batch_dims)),
        shuffle=not evaluation,
        num_workers=16,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    eval_ds_loader = DataLoader(
        eval_ds,
        batch_size=int(np.prod(batch_dims)),
        shuffle=not evaluation,
        num_workers=16,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return train_ds_loader, eval_ds_loader
