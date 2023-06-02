from configs.default_cifar10_configs import get_default_configs
from configs.cifar10_k_ve import get_config as get_ref_config

import math


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "kvesde"
    training.loss = "progressive_distillation"
    training.ref_model_path = "/path/to/edm_cifar10_ema"
    training.ref_config = get_ref_config()
    training.n_iters = 800001
    training.n_jitted_steps = 10
    training.snapshot_freq_for_preemption = 5000
    training.snapshot_freq = 10000
    training.batch_size = 512
    training.loss_norm = "l2"
    training.finetune = True
    training.target_ema_mode = "fixed"
    training.scale_mode = "progdist"
    training.start_scales = 4096
    training.distill_steps_per_iter = 50000
    training.weighting = "truncated_snr"

    # evaluation
    evaluate = config.eval
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 80
    evaluate.enable_loss = False

    # sampling
    sampling = config.sampling
    sampling.method = "progressive_distillation"
    sampling.denoise = False

    # data
    data = config.data
    data.dataset = "CIFAR10"

    # model
    model = config.model
    model.name = "ncsnpp"
    model.ema_rate = 0.0
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
    optim.optimizer = "adam"
    optim.lr = 5e-5
    optim.schedule = "linear"
    optim.linear_decay_steps = training.distill_steps_per_iter
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 0
    optim.grad_clip = 1.0

    return config
