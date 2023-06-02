from configs.default_cifar10_configs import get_default_configs
import math


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "kvesde"
    training.loss = "dsm"
    training.batch_size = 512
    training.n_iters = 400001
    training.n_jitted_steps = 2
    training.snapshot_freq = 10000
    training.snapshot_freq_for_preemption = 5000
    training.log_freq = 50
    training.eval_freq = 100

    # sampling
    sampling = config.sampling
    sampling.method = "heun"
    sampling.denoise = True

    # evaluation
    evaluate = config.eval
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 40

    # model
    model = config.model

    model.name = "ncsnpp"
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "none"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.embedding_type = "positional"
    model.fourier_scale = 16
    model.conv_size = 3
    model.ema_rate = math.exp(
        math.log(0.5) / (0.5e6 / training.batch_size)
    )  # half life of 0.5M images
    model.rho = 7.0
    model.data_std = 0.5
    model.num_scales = 18
    model.dropout = 0.13

    # optimization
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 1e-3
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = int(1e7 / training.batch_size)  # warmup for 10M images
    optim.grad_clip = float("inf")  # no gradient clipping

    return config
