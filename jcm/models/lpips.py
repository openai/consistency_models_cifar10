# Code from https://github.com/pcuenca/lpips-j/blob/main/src/lpips_j/lpips.py
#
# Original copyright statement:
# Copyright 2021 The DALL·E mini Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import flax.linen as nn
import jax.numpy as jnp

# from flaxmodels import VGG16
import flaxmodels.vgg as vgg
from huggingface_hub import hf_hub_download


class VGGExtractor(vgg.VGG):
    """
    VGG16 configured as a feature extractor for LPIPS, with weights
    downloaded from the huggingface hub.

    Note: subclasses `VGG` from `flaxmodels`, even though it was
    probably not meant to be subclassed (is not included in __all__).
    """

    def __init__(self):
        super().__init__(
            output="activations",
            pretrained="imagenet",
            architecture="vgg16",
            include_head=False,
        )

    def setup(self):
        weights_file = hf_hub_download(
            repo_id="pcuenq/lpips-jax", filename="vgg16_weights.h5"
        )
        self.param_dict = h5py.File(weights_file, "r")


class NetLinLayer(nn.Module):
    weights: jnp.array
    kernel_size = (1, 1)

    def setup(self):
        w = lambda *_: self.weights
        self.layer = nn.Conv(
            1, self.kernel_size, kernel_init=w, strides=None, padding=0, use_bias=False
        )

    def __call__(self, x):
        x = self.layer(x)
        return x


class LPIPS(nn.Module):
    def setup(self):
        # We don't add a scaling layer because `VGG16` already includes it
        self.feature_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        self.vgg = VGGExtractor()

        weights_file = hf_hub_download(
            repo_id="pcuenq/lpips-jax", filename="lpips_lin.h5"
        )
        lin_weights = h5py.File(weights_file)
        self.lins = [
            NetLinLayer(jnp.array(lin_weights[f"lin{i}"]))
            for i in range(len(self.feature_names))
        ]

    def __call__(self, x, t):
        x = self.vgg((x + 1) / 2)
        t = self.vgg((t + 1) / 2)

        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])
            diffs[i] = (feats_x[i] - feats_t[i]) ** 2

        # We should maybe vectorize this better
        res = [
            spatial_average(self.lins[i](diffs[i]), keepdims=True)
            for i in range(len(self.feature_names))
        ]

        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val


def normalize_tensor(x, eps=1e-10):
    # Use `-1` because we are channel-last
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdims=True):
    # Mean over W, H
    return jnp.mean(x, axis=[1, 2], keepdims=keepdims)
