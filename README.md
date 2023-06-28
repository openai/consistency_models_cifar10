# Consistency Models (CIFAR-10)

This is the codebase for [Consistency Models](https://arxiv.org/abs/2303.01469), implemented using JAX for our experiments on CIFAR-10. We have based our repository on [yang-song/score_sde](https://github.com/yang-song/score_sde), which was released under the Apache-2.0 license. We have modified the code to streamline diffusion model training, with additional implementations for consistency distillation, consistency training, and various sampling & editing algorithms included in the paper.

For code and checkpoints for experiments on ImageNet-64, LSUN Bedroom-256, and LSUN Cat-256, check [openai/consistency_models](https://github.com/openai/consistency_models).

# Pre-trained models

We have released checkpoints for the main models in the paper. Before using these models, please review the corresponding [model card](model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model checkpoint:

 * EDM on CIFAR-10: [edm_cifar10_ema](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/edm_cifar10_ema)
 * CD on CIFAR-10 with l1 metric: [cd-l1](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-l1/checkpoints/checkpoint_80)
 * CD on CIFAR-10 with l2 metric: [cd-l2](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-l2/checkpoints/checkpoint_80)
 * CD on CIFAR-10 with LPIPS metric: [cd-lpips](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-lpips/checkpoints/checkpoint_80)
 * CT on CIFAR-10 with adaptive schedules and LPIPS metric: [ct-lpips](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/ct-lpips/checkpoints/checkpoint_74)
 * Continuous-time CD on CIFAR-10 with l2 metric: [cifar10-continuous-cd-l2](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cifar10-continuous-cd-l2/checkpoints/checkpoint_40)
 * Continuous-time CD on CIFAR-10 with l2 metric and stopgrad: [cifar10-continuous-cd-l2-stopgrad](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cifar10-continuous-cd-l2-stopgrad/checkpoints/checkpoint_40)
 * Continuous-time CD on CIFAR-10 with LPIPS metric and stopgrad: [cifar10-continuous-cd-lpips-stopgrad](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cifar10-continuous-cd-lpips-stopgrad/checkpoints/checkpoint_40)
 * Continuous-time CT on CIFAR-10 with l2 metric: [continuous-ct-l2](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/continuous-ct-l2/checkpoints/checkpoint_80)
 * Continuous-time CT on CIFAR-10 with LPIPS metric: [continuous-ct-lpips](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/continuous-ct-lpips/checkpoints/checkpoint_40)

# Dependencies

To install all packages in this codebase along with their dependencies, run
```sh
pip install -e .
```
Then manually install jax by running
```sh
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
```
The code assumes Python 3.9, JAX 0.4.10, CUDA 11 and cuDNN 8.2. For different Python, CUDA and cuDNN versions, you need to modify the above command accordingly.

# Model training and sampling

We provide examples of EDM training, consistency distillation, consistency training, single-step generation, and model evaluation in [launch.sh](launch.sh).

# Zero-shot editing
We provide examples for multistep generation and zero-shot image editing in [editing_multistep_sampling.ipynb](editing_multistep_sampling.ipynb).

# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023},
}
```

This repo is built upon previous work [score_sde](https://github.com/yang-song/score_sde). Please consider citing

```bibtex
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
