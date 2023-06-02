#################
# EDM on CIFAR-10
#################

## Model training
python -m jcm.main --config configs/cifar10_k_ve.py --workdir cifar10-kve-score-loss --mode train
## Model evaluation (loss computation, sampling, and likelihood computation)
python -m jcm.main --config configs/cifar10_k_ve.py --workdir cifar10-kve-score-loss --mode eval --eval_folder eval
## Model metrics (FID & IS)
python -m jcm.main --config configs/cifar10_k_ve.py --workdir cifar10-kve-score-loss --mode metrics --eval_folder eval

######################################
# Consistency distillation on CIFAR-10
######################################

## CD with the l1 metric
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir /path/to/cd-l1 --mode train --config.optim.lr=0.0002 --config.training.loss_norm='l1'

## CD with the l2 metric
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir /path/to/cd-l2 --mode train --config.optim.lr=0.0004 --config.training.loss_norm='l2'

## CD with the LPIPS metric
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir /path/to/cd-lpips --mode train --config.optim.lr=0.0004 --config.training.loss_norm='lpips'

## Sampling for CD (LPIPS)
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir /path/to/cd-lpips --mode eval --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --eval_folder eval --config.eval.begin_ckpt=80

## Computing FID for CD (LPIPS)
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir /path/to/cd-lpips --mode metrics --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --eval_folder eval --config.eval.begin_ckpt=80

######################################################
# Continuous-time consistency distillation on CIFAR-10
######################################################

## Continuous-time CD with the l2 metric
python -m jcm.main --config configs/cifar10_ve_continuous.py --workdir /path/to/cifar10-continuous-cd-l2 --mode train --config.optim.lr=0.001 --config.model.dropout=0.3 --config.training.loss_norm='l2'

## Continuous-time CD with stopgrad and the l2 metric
python -m jcm.main --config configs/cifar10_ve_continuous.py --workdir /path/to/cifar10-continuous-cd-l2-stopgrad --mode train --config.optim.lr=0.000005 --config.training.stopgrad=True --config.training.loss_norm='l2'

## Continuous-time CD with stopgrad and the LPIPS metric
python -m jcm.main --config configs/cifar10_ve_continuous.py --workdir /path/to/cifar10-continuous-cd-lpips-stopgrad --mode train --config.optim.lr=0.000005 --config.training.stopgrad=True --config.training.loss_norm='lpips'

##################################
# Consistency training on CIFAR-10
##################################

## CT with fixed N and mu, and the LPIPS metric
python -m jcm.main --config configs/cifar10_ve_ct_ema.py --mode train --workdir /path/to/ct-n-120-mu-0.99 --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --config.model.num_scales=120 --config.training.dsm_target --config.training.target_ema=0.99

## CT with adaptive schedules and the LPIPS metric
python -m jcm.main --config configs/cifar10_ve_ct_adaptive.py --workdir /path/to/ct --mode train --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --config.training.dsm_target --config.training.start_scales=2 --config.training.end_scales=150 --config.training.start_ema=0.9

## Sampling for CT
python -m jcm.main --config configs/cifar10_ve_ct_adaptive.py --workdir /path/to/ct --mode eval --eval_folder eval --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --config.training.dsm_target --config.training.start_scales=2 --config.training.end_scales=150 --config.training.start_ema=0.9 --config.eval.begin_ckpt=74 --config.eval.end_ckpt=74

## FID computation for CT
python -m jcm.main --config configs/cifar10_ve_ct_adaptive.py --workdir /path/to/ct --mode metrics --eval_folder eval --config.optim.lr=0.0004 --config.training.loss_norm='lpips' --config.training.dsm_target --config.training.start_scales=2 --config.training.end_scales=150 --config.training.start_ema=0.9 --config.eval.begin_ckpt=74 --config.eval.end_ckpt=74

##################################################
# Continuous-time consistency training on CIFAR-10
##################################################

## Continuous-time CT with the l2 metric
python -m jcm.main --config configs/cifar10_ve_continuous.py --workdir /path/to/continuous-ct-l2 --mode train --config.optim.lr=0.000005 --config.training.loss_norm='l2' --config.training.dsm_target=True --config.training.stopgrad=True --config.training.finetune=True

## Continuous-time CT with the LPIPS metric
python -m jcm.main --config configs/cifar10_ve_continuous.py --workdir /path/to/continuous-ct-lpips --mode train --config.optim.lr=0.000005 --config.training.loss_norm='lpips' --config.training.dsm_target=True --config.training.stopgrad=True --config.training.finetune=True
