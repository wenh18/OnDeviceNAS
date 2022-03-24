#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import math
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mytools import init_resnet_multiblocks
from mytimm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from mytimm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from mytimm.utils import *
from mytimm.loss import *
from mytimm.optim import create_optimizer_v2, optimizer_kwargs
from mytimm.scheduler import create_scheduler
from mytimm.utils import ApexScaler, NativeScaler
from mytimm.models.layers import Linear
import timm

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--pretrain_epochs', type=int, default=50, metavar='N',
                    help='number of epochs before training the final network (default: 10)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=int, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
# TODO:build teahcer model, train student in stages
parser.add_argument('--temp', type=float, default=5.0,
                    help='temperature of distillation')
parser.add_argument('--alpha', type=float, default=0.7,
                    help='percentage of distillation loss')
parser.add_argument('--save_freq', type=int, default=5,
                    help='save frequency')
parser.add_argument('--ALPHALOSS', type=float, default=15,
                    help='save frequency')
parser.add_argument('--savemodelfreq', type=int, default=5,
                    help='save frequency')

parser.add_argument('--checkpointdir', metavar='DIR',
                    help='path to checkpoint')
parser.add_argument('--withcheckpoint', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--stage', type=int, default=0,
                    help='0 is distilling and 1 is training')
parser.add_argument('--original_model', metavar='DIR',
                    help='path to original model')
parser.add_argument('--distilled_model', metavar='DIR',
                    help='path to distilled model')
parser.add_argument('--start-epoch1', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained_epochs', type=int, default=0,
                    help='0 is distilling and 1 is training')
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def no_grad_fn(subnetchoice, resnet=False):
    if resnet:
        no_grad = True
        for layer in subnetchoice:
            for i in layer:
                if i != 3:
                    no_grad = False
                    break
        return no_grad
    else:
        no_grad = True
        for i in subnetchoice:
            if i > 1:
                no_grad = False
                break
        return no_grad


def get_mbv3teacher_output(x, teacher_model, stage):
    if stage < 3:
        x = teacher_model.conv_stem(x)
        x = teacher_model.bn1(x)
        x = teacher_model.act1(x)
        if stage == 0:
            x = teacher_model.blocks[0:3](x)
            return x
        elif stage == 1:
            x = teacher_model.blocks[0:5](x)
            return x
        elif stage == 2:
            x = teacher_model.blocks(x)
            return x
    else:
        x = teacher_model(x)
        return x


def get_teacher_outputv2(x, teacher_model, idx1, idx2):
    x = teacher_model.conv_stem(x)
    x = teacher_model.bn1(x)
    x = teacher_model.act1(x)
    # for i in range(0, idx1):
    x = teacher_model.blocks[0:idx1](x)
    x = teacher_model.blocks[idx1][0:idx2](x)
    return x


def get_resnet_output(x, teacher_model, idxs):
    teacher_features = []
    x = teacher_model.conv1(x)
    x = teacher_model.bn1(x)
    x = teacher_model.act1(x)
    x = teacher_model.maxpool(x)
    for idx1 in range(len(teacher_model.layer1)):
        # print(";;;;;", idx1)
        x = teacher_model.layer1[idx1](x)
        if [1, idx1] in idxs:
            # print(";;;", idx1)
            teacher_features.append(x)
            # print(x.shape)
    for idx2 in range(len(teacher_model.layer2)):
        x = teacher_model.layer2[idx2](x)
        if [2, idx2] in idxs:
            teacher_features.append(x)
    for idx3 in range(len(teacher_model.layer3)):
        x = teacher_model.layer3[idx3](x)
        if [3, idx3] in idxs:
            teacher_features.append(x)
    for idx4 in range(len(teacher_model.layer4)):
        x = teacher_model.layer4[idx4](x)
        if [4, idx4] in idxs:
            teacher_features.append(x)
    return teacher_features


# import inspect
# def get_variable_name(variable):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is variable]

def load_to_MultiModel(MultiModel, path):
    pretrained_model = torch.load(path)
    model_dict = MultiModel.state_dict()
    # for k, v in model_dict.items():
    #     print(k)
    state = {}
    multimodel_idx = 0
    idx1 = idx2 = 0
    teacher_model_dict = []
    for k, v in pretrained_model.items():
        # print(k)
        if k in model_dict.keys():
            state[k] = v
        else:
            new_idx1 = int(k[7])
            new_idx2 = int(k[9])
            if idx1 != new_idx1 or idx2 != new_idx2:
                idx1 = new_idx1
                idx2 = new_idx2
                multimodel_idx += 1
                teacher_model_dict.append([new_idx1, new_idx2])
            key = "multiblocks." + str(multimodel_idx) + ".0." + k[11:]
            state[key] = v
            # key = k[7:]
            # state[key] = v
    model_dict.update(state)
    MultiModel.load_state_dict(model_dict)
    for (name, parameter) in MultiModel.named_parameters():
        if name in state:
            # if ("multiblocks" in name and (name[14] == "0" or name[15] == "0")) or "classifier" in name or "conv_head" in name or "conv_stem" in name:
            #     print(";;;", name)
            # print(parameter.requires_grad)
            parameter.requires_grad = False
    print("loaded")
    return MultiModel, teacher_model_dict


def load_to_MultiModelv2(MultiModel, path):  # layer1.0.conv1.weight -> multiblocks.0.0.conv1.weight
    # print("started")
    pretrained_model = torch.load(path)
    print("loaded pretrained")
    multimodel_dict = MultiModel.state_dict()
    # import pdb;pdb.set_trace()
    state = {}
    multimodel_idx = 0
    idx1 = 1
    idx2 = 0
    teacher_model_dict = [[1, 0]]
    for k, v in pretrained_model.items():
        # print(k)
        if k in multimodel_dict.keys():
            state[k] = v
        else:
            new_idx1 = int(k[5])
            new_idx2 = int(k[7])
            if idx1 != new_idx1 or idx2 != new_idx2:
                idx1 = new_idx1
                idx2 = new_idx2
                multimodel_idx += 1
                teacher_model_dict.append([new_idx1, new_idx2])
            key = "multiblocks." + str(multimodel_idx) + ".0." + k[9:]
            state[key] = v
    multimodel_dict.update(state)
    MultiModel.load_state_dict(multimodel_dict)
    for (name, parameter) in MultiModel.named_parameters():
        if name in state:
            # print(";;;", name)
            parameter.requires_grad = False
    print("loaded")
    return MultiModel, teacher_model_dict

    #     # print(k)
    # print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    # for k, v in multimodel_dict.items():
    #     print(k)
def load_resnet_checkpoint(multimodel, path):
    state = {}
    model_dict = multimodel.state_dict()
    # print_id = 0
    # for k, v in model_dict.items():
    #     if print_id < 100:
    #         print(k)
    #     print_id += 1
    pretrained_model = torch.load(path, map_location=torch.device('cpu'))
    for k, v in pretrained_model.items():
        # print(k)
        # key = k[7:]
        state[k] = v
    # import pdb;pdb.set_trace()
    model_dict.update(state)
    multimodel.load_state_dict(model_dict)
    for (name, parameter) in multimodel.named_parameters():
        # print(name)
        name_list = name.split('.')
        if name_list[1] == "multiblocks" and name_list[3] == "0":
            print(";;;", name)
            parameter.requires_grad = False
        if "conv1" in name_list[1] or name_list[1] == "bn1" or name_list[1] == "bn2" or name_list[1] == "fc":
            print(";;;", name)
            parameter.requires_grad = False
        # if "classifier" in name:
        #     parameter.requires_grad = False
        #     print(";;;", name)
        # if ("multiblocks" in name and (name[14] == "0" or name[15] == "0")) or "classifier" in name or "conv_head" in name \
        #         or "conv_stem" in name or name == "bn1.weight" or name == "bn1.bias":
            # print(";;;;", name)
            # parameter.requires_grad = False
    return multimodel

def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)

    # subnets = []
    # # subnetchoices = []
    # for i in range(5000000):#(2 * (int(512 / args.batch_size * 2503) + 1) * args.epochs):
    #     subnet = model.generate_random_subnet()
    #     subnets.append(subnet)
    #     # subnetchoices.append(subnetchoice)
    #     if i % 1000000 == 0:
    #         print(i)
    # print("created")
    # subnets = np.array(subnets)
    # np.save('resnet_subnets.npy', subnets)
    # print("saved")
    subnets = np.load('resnet_subnets.npy')
    print(subnets.shape[0])
    print('loaded')
    # import pdb;pdb.set_trace()

    teachermodel = timm.create_model('resnet50', pretrained=True)
    teachermodel.eval()
    teachermodel.cuda()
    model, teacher_map = load_to_MultiModelv2(model, args.original_model)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    # print(model.affine, "--------------------------------------------")
    # import pdb;pdb.set_trace()
    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    init_resnet_multiblocks(model, args)

    # import pdb;pdb.set_trace()

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)


    freeze_dict = ["multiblocks." + str(i) + ".0." for i in range(16)]
    freeze_dict += ["conv1", "module.bn1", "module.fc"]
    for (name, parameter) in model.named_parameters():
        # print(name)
        for freeze_name in freeze_dict:
            if freeze_name in name:
                print(";;;", name)
                parameter.requires_grad = False
                # if "conv1.1.running_mean" in name:
                    # print(name, parameter)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb,
                              find_unused_parameters=True)
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)
    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    writer = SummaryWriter('./image/multimbv3_imagenet')
    # model.load_state_dict(torch.load('nostagestage_3epoch9.pth'), strict=True)

    if args.withcheckpoint:
        model = load_resnet_checkpoint(model, path=args.distilled_model)

    try:
        for stage in range(args.stage, 2):
            # setup learning rate schedule and starting epoch
            lr_scheduler, num_epochs = create_scheduler(args, optimizer, stage)
            start_epoch = 0
            if args.start_epoch is not None:
                # a specified start_epoch will always override the resume epoch
                start_epoch = args.start_epoch
                if stage == 1:
                    start_epoch = args.start_epoch1
            elif resume_epoch is not None:
                start_epoch = resume_epoch
            if lr_scheduler is not None and start_epoch > 0:
                lr_scheduler.step(start_epoch)


            if args.local_rank == 0:
                _logger.info('Scheduled epochs: {}'.format(num_epochs))

            for epoch in range(start_epoch, num_epochs):
                if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                    loader_train.sampler.set_epoch(epoch)
                # for batch_idx, (input, target) in enumerate(loader_train):
                #     module_input, module_output = get_teacher_output(input, 1)

                train_metrics = train_one_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,
                    subnets[epoch * int(1281167 / args.batch_size):(epoch + 1) * int(1281167 / args.batch_size), :],
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn,
                    stage=stage, teacher_model=teachermodel,
                    writer=writer, teachernet_map=teacher_map)  # , subnet=subnet, subnetchoice=subnetchoice)

                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

                eval_metrics, originacc = validate(model, loader_eval, validate_loss_fn, args,
                                        subnets[subnets.shape[0]-1-int(50000 / args.batch_size) * (epoch + 1):len(subnets)-1-int(50000 / args.batch_size) * (epoch), :],
                                        amp_autocast=amp_autocast,
                                        teacher_model=teachermodel, stage=stage, is_distributed=False, writer=writer,
                                        epoch=epoch)
                # ,subnet = subnet, subnetchoice = subnetchoice)

                # if model_ema is not None and not args.model_ema_force_cpu:
                #     if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                #         distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                #     ema_eval_metrics = validate(
                #         model_ema.module, loader_eval, validate_loss_fn, args,
                #         # subnets[subnets.shape[0]-1-int(50000 / args.batch_size) * (epoch + 1):len(subnets)-1-int(50000 / args.batch_size) * (epoch), :],
                #         #                 subnetchoices[subnets.shape[0] - 1 - int(50000 / args.batch_size) * (epoch + 1):len(
                #         #                     subnets) - 1 - int(50000 / args.batch_size) * (epoch), :],
                #         amp_autocast=amp_autocast,
                #         log_suffix=' (EMA)', teacher_model=teachermodel, stage=stage,
                #         is_distributed=True)  # , subnet = subnet, subnetchoice = subnetchoice)
                #     eval_metrics = ema_eval_metrics
                #
                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
                #
                # if output_dir is not None:
                #     update_summary(
                #         epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                #         write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)
                #
                # if saver is not None:
                #     # save proper checkpoint with eval metric
                #     if (epoch + 1) % args.save_freq == 0:
                #         save_metric = eval_metrics[eval_metric]
                #         best_metric, best_epoch = saver.save_checkpoint(epoch, stage, metric=save_metric)

                if (epoch + 1) % args.savemodelfreq == 0:
                    name = "output/resnet_weights/resnet_128_" + str(stage) + "epoch" +  str(epoch) + "acc" + str(int(round(originacc))) + ".pth"
                    # emaname = "output/resnet_weights/ema_resnet_" + str(stage) + "epoch" + str(epoch) + ".pth"
                    torch.save(model.state_dict(), name)
                    # torch.save(model_ema.state_dict(), emaname)
        # print(lrs)
    #
    except KeyboardInterrupt:
        pass
    #
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args, subnets=None, subnetchoices=None,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, stage=None, teacher_model=None, writer=None, subnet=None,
        subnetchoice=None, teachernet_map=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    # freeze the original part
    if args.distributed:
        model.module.conv1.eval()
        model.module.bn1.eval()
        model.module.fc.eval()
        for blockchoice in model.module.multiblocks:
            blockchoice[0].bn1.eval()
            blockchoice[0].bn2.eval()
            blockchoice[0].bn3.eval()
            if hasattr(blockchoice[0], 'downsample'):
                if blockchoice[0].downsample is not None:
                    blockchoice[0].downsample[1].eval()
    else:
        model.conv1.eval()
        model.bn1.eval()
        model.fc.eval()
        for blockchoice in model.multiblocks:
            blockchoice[0].bn1.eval()
            blockchoice[0].bn2.eval()
            blockchoice[0].bn3.eval()
            if hasattr(blockchoice[0], 'downsample'):
                if blockchoice[0].downsample is not None:
                    blockchoice[0].downsample[1].eval()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    # subnet, subnetchoice = model.generate_random_subnet(stage)
    # print(subnet, subnetchoice)
    # while no_grad_fn(subnetchoice):
    #     subnet, subnetchoice = model.generate_random_subnet(stage)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # teacher_output = get_mbv3teacher_output(input, teacher_model, stage).detach()

        with amp_autocast():
            # subnet, subnetchoice = model.generate_random_subnet(stage)
            # subnet = model.module.generate_random_subnet()
            # while no_grad_fn(subnet):
            #     # subnet, subnetchoice = model.generate_random_subnet(stage)
            #     subnet = model.module.generate_random_subnet()
            # output, features, feature_idx_outs, block_cls = model(input, subnet, subnetchoice, stage)
            if args.distributed:
                subnet = subnets[batch_idx]
            else:
                subnet = model.generate_random_subnet()
                while no_grad_fn(subnet):
                    subnet = model.generate_random_subnet()
            output, features, feature_idxs = model(input, subnet)
            # print(subnet, feature_idxs)
            teacher_feature_idxs = []
            for i in range(len(feature_idxs)):
                teacher_feature_idxs.append(teachernet_map[feature_idxs[i]])
            # print(output.shape, teacher_output.shape)
            if stage == 1:
                loss = loss_fn(output, target)
                # loss2 = torch.nn.KLDivLoss()(F.log_softmax(output / args.temp, dim=1),
                #                              F.softmax(teacher_output / args.temp,
                #                                        dim=1)) * args.temp * args.temp * 2.0 * args.alpha
            else:
                loss = 0
                # print(feature_idxs)
                teacher_features = get_resnet_output(input, teacher_model, teacher_feature_idxs)
                # import pdb;pdb.set_trace()
                for i in range(len(teacher_features)):
                    # print(i, teacher_features[i].shape, features[i].shape)
                    loss += torch.nn.MSELoss()(teacher_features[i], features[i])
                    # loss += torch.norm((teacher_features[i] - features[i]), p=1) / args.batch_size
                for i in range(len(teacher_features)):
                    del teacher_features[len(teacher_features) - 1]
                loss /= len(features)
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        # if lr_scheduler is not None:
        #     lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

                writer.add_scalar('decay3SINGLEbranch_train_loss', losses_m.avg, epoch)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, subnets=None, subnetchoices=None, amp_autocast=suppress, log_suffix='',
             teacher_model=None, stage=None,
             is_distributed=False, writer=None, epoch=None, subnet=None, subnetchoice=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    # subnet, subnetchoice = model.generate_random_subnet(stage)
    # while no_grad_fn(subnetchoice):
    #     subnet, subnetchoice = model.generate_random_subnet(stage)
    end = time.time()

    last_idx = len(loader) - 1
    # if stage == 3:
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            # else:
            #     teacher_output = get_mbv3teacher_output(input, teacher_model, stage).detach()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            if args.distributed:
                if batch_idx < last_idx:
                    subnet = subnets[batch_idx]
                else:
                    subnet = subnets[0]
            else:
                subnet = model.generate_random_subnet()
            # if batch_idx >= last_idx:
            #     subnet = subnets[0]
            with amp_autocast():
                output = model(input, subnet)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                # if stage == 3:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            # if stage == 3:
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                # print(subnet, subnetchoice)
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                if not is_distributed:
                    writer.add_scalar('decay3SINGLE_val_acc', top1_m.avg, epoch)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics, top1_m.avg


if __name__ == '__main__':
    main()
