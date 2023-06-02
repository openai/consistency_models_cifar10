from configs.default_cifar10_configs import get_default_configs
from configs.cifar10_k_ve import get_config as get_ref_config

import math


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "kvesde"
    training.loss = "consistency"
    training.ref_model_path = "/path/to/edm_cifar10_ema"
    training.ref_config = get_ref_config()
    training.n_iters = 800001
    training.n_jitted_steps = 1
    training.snapshot_freq_for_preemption = 5000
    training.snapshot_freq = 10000
    training.batch_size = 512
    training.loss_norm = "lpips"
    training.finetune = True
    training.stopgrad = True
    training.dsm_target = False
    training.solver = "heun"
    training.weighting = "uniform"

    # evaluation
    evaluate = config.eval
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 80
    evaluate.enable_loss = True

    # sampling
    sampling = config.sampling
    sampling.method = "onestep"
    sampling.std = config.model.t_max
    sampling.n_steps = 18

    # data
    data = config.data
    data.dataset = "CIFAR10"

    # model
    model = config.model
    model.name = "ncsnpp"
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3
    model.rho = 7.0
    model.data_std = 0.5
    model.num_scales = 18
    model.dropout = 0.0

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "radam"
    optim.lr = 4e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = int(1e7 / training.batch_size)  # warmup for 10M images
    optim.grad_clip = float("inf")  # no gradient clipping

    return config
