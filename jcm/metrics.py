# Code modified from https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
# Original license below:
#                                MIT License
#
#                      Copyright (c) 2021 Gaurav Parmar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from cleanfid import fid
import torchvision
import numpy as np
import flax
import logging
from . import checkpoints

import tqdm
import time
import jax

import os
import io
import blobfile
import json
import uuid
import requests
import torch_fidelity


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fid.build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


class TorchDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fid.build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        assert img_resized.dtype == "uint8"
        img_t = (self.transforms(np.array(img_resized)) * 255).to(torch.uint8)

        return img_t


def compute_fid(
    samples,
    feat_model,
    dataset_name="cifar10",
    dataset_res=32,
    dataset_split="train",
    batch_size=1024,
    num_workers=12,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
):
    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    l_feats = []
    for batch in tqdm.tqdm(dataloader):
        l_feats.append(fid.get_batch_features(batch, feat_model, device))
    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    ref_mu, ref_sigma = fid.get_reference_statistics(
        dataset_name, dataset_res, mode=mode, seed=0, split=dataset_split
    )

    score = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)

    return score


def compute_all_metrics(
    samples,
    dataset_name="cifar10-train",
    mode="legacy_tensorflow",
    batch_size=1024,
    num_workers=12,
):
    dataset = TorchDataset(samples, mode=mode)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dataset,
        input2=dataset_name,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False,
    )
    return metrics_dict


def get_samples_from_ckpt(folder, ckpt):
    files = list(
        blobfile.glob(os.path.join(folder, f"ckpt_{ckpt}_host_*", "samples_*.npz"))
    )
    all_samples = []
    for file in files:
        with blobfile.BlobFile(file, "rb") as fin:
            all_samples.append(np.load(fin)["samples"])
    if len(all_samples) >= 1:
        all_samples = np.concatenate(all_samples)
    else:
        all_samples = np.zeros((0, 32, 32, 3), dtype=np.uint8)
    return all_samples


def get_fids(folder, ckpt_range, mode, device):
    ckpts = []
    fids = []
    feat_model = fid.build_feature_extractor(mode, device)
    for ckpt in ckpt_range:
        ckpts.append(ckpt)
        print("Loading samples from ckpt", ckpt)
        data = get_samples_from_ckpt(folder, ckpt)
        print(f"data.shape: {data.shape}")
        fids.append(
            compute_fid(
                data[:50000],
                mode="legacy_tensorflow",
                device=device,
                feat_model=feat_model,
            )
        )
        print("FID", fids[-1])
    return ckpts, fids


def compute_metrics(
    config,
    workdir,
    eval_folder,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
):
    """Compute the FID metrics from given samples.

    Args:
        config (dict): The config dict.
        workdir (str): The working directory.
        eval_folder (str): The folder to store the evaluation results.
    """
    eval_dir = os.path.join(workdir, eval_folder)
    blobfile.makedirs(eval_dir)

    @flax.struct.dataclass
    class MetricsMeta:
        ckpt_id: int

    metrics_meta = MetricsMeta(
        ckpt_id=config.eval.begin_ckpt,
    )
    metrics_meta = checkpoints.restore_checkpoint(
        eval_dir, metrics_meta, step=None, prefix="metrics_meta_"
    )
    feat_model = fid.build_feature_extractor(mode, device)

    begin_ckpt = max(metrics_meta.ckpt_id, config.eval.begin_ckpt)
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        print(f"Start metric evaluation for ckpt {ckpt}")

        all_samples = get_samples_from_ckpt(eval_dir, ckpt)
        waiting_message_printed = False
        while all_samples.shape[0] < config.eval.num_samples:
            if not waiting_message_printed and jax.process_index() == 0:
                logging.warning(f"Waiting for the arrival of samples for ckpt {ckpt}")
                waiting_message_printed = True
            time.sleep(100)
            all_samples = get_samples_from_ckpt(eval_dir, ckpt)

        fid_score = compute_fid(
            all_samples[: config.eval.num_samples],
            mode=mode,
            device=device,
            feat_model=feat_model,
        )

        with blobfile.BlobFile(
            os.path.join(eval_dir, f"metrics_{ckpt}.npz"),
            "wb",
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, fid=fid_score)
            fout.write(io_buffer.getvalue())

        metrics_meta = metrics_meta.replace(ckpt_id=ckpt + 1)
        checkpoints.save_checkpoint(
            eval_dir, metrics_meta, step=ckpt, keep=1, prefix="metrics_meta_"
        )

    meta_files = blobfile.glob(os.path.join(eval_dir, "metrics_meta_*.npz"))
    for file in meta_files:
        blobfile.remove(file)


def obtain_feature_extractor(mode="legacy_tensorflow", device=torch.device("cuda:0")):
    return fid.build_feature_extractor(mode, device)


def compute_fid_jupyter(
    all_samples,
    feature_extractor,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
):
    """Compute the FID metrics from given samples.

    Args:
        config (dict): The config dict.
        workdir (str): The working directory.
        eval_folder (str): The folder to store the evaluation results.
    """

    feat_model = feature_extractor

    fid_score = compute_fid(
        all_samples,
        mode=mode,
        device=device,
        feat_model=feat_model,
    )

    return fid_score
