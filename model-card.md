# Overview

These are diffusion models and consistency models described in the paper [Consistency Models](https://arxiv.org/abs/2303.01469). We include the following models in this release:

 * EDM models on CIFAR-10.
 * Consistency models trained by consistency distillation (CD) with both l2 and LPIPS metrics on CIFAR-10.
 * Consistency models trained by consistency training (CT) with adaptive schedule and LPIPS on CIFAR-10.
 * Consistency models trained by continuous-time CD (with and without stop gradient, l1, l2, and LPIPS metrics) on CIFAR-10.
 * Consistency models trained by continuous-time CT (with both l2 and LPIPS metrics) on CIFAR-10.

# Datasets

The models that we are making available have been trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). This dataset was created in 2009 and has 60000 pictures classified into ten classes. These classes are airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck. Pictures in CIFAR-10 are of relatively low resolution (32x32) and do not contain humans.


# Performance

These models are intended to generate samples consistent with their training distributions.
This has been measured in terms of FID, and Inception Score. These metrics all rely on the representations of a [pre-trained Inception-V3 model](https://arxiv.org/abs/1512.00567), which was trained on ImageNet, and so is likely to focus more on the ImageNet classes (such as animals) than on other visual features (such as human faces).

# Intended Use

These models are intended to be used for research purposes only.
In particular, they can be used as a baseline for generative modeling research, or as a starting point for such research. These models are not intended to be commercially deployed. Additionally, they are not intended to be used to create propaganda or offensive imagery.

# Limitations

Minimizing LPIPS results in better sample quality, as evidenced by improved FID and Inception scores. However, this approach carries the risk of overestimating model performance, because LPIPS uses a VGG network pre-trained on ImageNet, while FID and Inception scores also rely on convolutional neural networks (the Inception network in particular) pre-trained on ImageNet.
