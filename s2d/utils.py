'''
Description        : The utility functions for S2DNET.
'''

import os
import torch
import torchvision
import shutil
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import logging
import math
from fvcore.nn.flop_count import flop_count
from torch.utils.data._utils.collate import default_collate


def parse_store_name(cfg):
    model_name = '_'.join([
        cfg.TRAIN.DATASET.upper(),
        '%dx%d' % (cfg.S2D.SNET_FRAME_NUM, cfg.S2D.DNET_FRAME_NUM),
        '%sx%s' % (cfg.S2D.SNET_MODEL, cfg.S2D.DNET_MODEL)
    ])
    if len(cfg.MODEL_SUFFIX) != 0:
        model_name = cfg.MODEL_SUFFIX + '_' + model_name
    return model_name


def check_rootfolders(cfg):
    """Create log and model folder"""
    folders_util = [
        cfg.ROOT_LOG, cfg.ROOT_MODEL,
        os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME),
        os.path.join(cfg.ROOT_MODEL, cfg.STORE_NAME)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


def is_master_rank(cfg):
    """The global rank is 0 or not."""
    if cfg.DIST:
        try:
            return dist.get_rank() == 0
        except:
            print('Not initialized distirbuted')
            return True
    else:
        return True


def save_cfg(cfg):
    """save config information to the experiment dir."""
    model_dir = '%s/%s' % (cfg.ROOT_MODEL, cfg.STORE_NAME)
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    filename = '%s/config.yaml' % (model_dir)
    with open(filename, 'w') as f:
        f.write(cfg.dump())


def save_checkpoint(state_dict, is_best, cfg, epoch, spec_name=None):
    """save checkpoint to the experiment dir

    Parameters
    ----------
    state_dict : dict
        the state dictionary of the checkpoint
    is_best : bool
        best checkpoint or newest checkpoint
    cfg : CfgNode
        the configuration
    epoch : int
        the epoch id
    spec_name : str
        when given, save as '<spec_name>.pth.tar'
    """

    if spec_name is None:
        filename = '%s/%s/newest.pth.tar' % (cfg.ROOT_MODEL, cfg.STORE_NAME)
        torch.save(state_dict, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('newest', 'best'))
        if epoch % cfg.DEBUG.SAVE_CKPT_FREQ == 0:
            shutil.copyfile(
                filename, filename.replace('newest', 'ep{:03d}'.format(epoch)))
    else:
        filename = '%s/%s/%s.pth.tar' % (cfg.ROOT_MODEL, cfg.STORE_NAME,
                                         spec_name)
        torch.save(state_dict, filename)


def adjust_learning_rate(optimizer, epoch, cfg, lr_min_factor=0.01):
    """Adjust the learning rate according to the selected policy and the epoch
    
    Parameters
    ----------
    optimizer : Optimizer
        an instance of PyTorch Optimizer 
    epoch : int
        the epoch id
    cfg : CfgNode
        the configuration
    lr_min_factor : float
        min_learning_rate = base_learning_rate * lr_min_factor

    """
    if isinstance(optimizer, list) is False:
        optimizers = [optimizer]
    else:
        optimizers = optimizer

    if epoch <= cfg.TRAIN.WARMUP_EPOCHS:
        lr_step = (cfg.TRAIN.BASE_LR -
                   cfg.TRAIN.WARMUP_LR) / cfg.TRAIN.WARMUP_EPOCHS
        lr = cfg.TRAIN.WARMUP_LR + epoch * lr_step
    else:
        if cfg.TRAIN.LR_DECAY_TYPES == 'step':
            decay = 0.1**(sum(epoch >= np.array(cfg.TRAIN.LR_DECAY_STEPS)))
            lr = cfg.TRAIN.BASE_LR * decay
        elif cfg.TRAIN.LR_DECAY_TYPES == 'cos':
            import math
            lr_min = cfg.TRAIN.BASE_LR * lr_min_factor
            lr_max = cfg.TRAIN.BASE_LR
            lr = lr_min + 0.5 * (lr_max - lr_min) * (
                1 + math.cos(math.pi * epoch / cfg.TRAIN.EPOCHS))
        else:
            raise NotImplementedError

    cnet_lr = lr
    if cfg.S2D.SNET_EARLY_DECAY_EPOCH > 0:
        """ To allow SNet decay earlier """
        assert (
            cfg.S2D.SNET_EARLY_DECAY_EPOCH >= cfg.TRAIN.WARMUP_EPOCHS
        ), 'invalid cfg.S2D.SNET_EARLY_DECAY_EPOCH={}, should be greater than cfg.TRAIN.WARMUP_EPOCHS={}'.format(
            cfg.S2D.SNET_EARLY_DECAY_EPOCH, cfg.TRAIN.WARMUP_EPOCHS)

        if epoch > cfg.TRAIN.WARMUP_EPOCHS:
            if epoch >= cfg.S2D.SNET_EARLY_DECAY_EPOCH:
                cnet_lr = 0.
            else:
                if cfg.TRAIN.LR_DECAY_TYPES == 'step':
                    cnet_lr = lr  # not scale automatically
                elif cfg.TRAIN.LR_DECAY_TYPES == 'cos':
                    cnet_lr_min = 0.
                    cnet_lr_max = cfg.TRAIN.BASE_LR
                    cnet_lr = cnet_lr_min + 0.5 * (
                        cnet_lr_max - cnet_lr_min) * (1 + math.cos(
                            math.pi * epoch / cfg.S2D.SNET_EARLY_DECAY_EPOCH))
                else:
                    raise NotImplementedError
            print('epoch={}, SNET_EARLY_DECAY_EPOCH={}, cnet_lr:{}'.format(
                epoch, cfg.S2D.SNET_EARLY_DECAY_EPOCH, cnet_lr))

    for optim in optimizers:
        for param_group in optim.param_groups:
            if 'cnet_' in param_group['name']:
                param_group['lr'] = cnet_lr * param_group['lr_mult']
            else:
                param_group['lr'] = lr * param_group['lr_mult']


def get_criterion(smooth_alpha, require_logsoftmax=False):
    """Get the loss function.
    
    Parameters
    ----------
    smooth_alpha : float
        alpha for label smoothing, \in [0,1] 
    require_logsoftmax : bool
        apply CrossEntropy or NLL (Negative Log Likelihood).
    """
    if require_logsoftmax:
        return torch.nn.CrossEntropyLoss(reduction='none',
                                         label_smoothing=smooth_alpha)
    else:
        print('label smoothing is disabled for NLL loss (w/o logsoftmax)')
        return torch.nn.NLLLoss(reduction='none')


def load_state_dict(net, state_dict, ignores='', strict=True):
    """Load state_dict for the given model.
    
    Parameters
    ----------
    net : 
        the model
    state_dict : dict
        the state_dict to load
    ignores : list
        the keywords that help filter out unwanted weights
    """
    init_state_dict = net.state_dict()
    has_prefix = list(init_state_dict.keys())[0].startswith('module.')
    new_state_dict = {}
    ignores = [
        ign for ign in ignores.replace(' ', '').split(',') if len(ign) > 0
    ]
    print('Ignored Weights: {}'.format(ignores))
    for k, v in state_dict.items():
        if not has_prefix:
            k = k.replace('module.', '')
        if any(map(k.__contains__, ignores)):
            print('=> ignore {}'.format(k))
            if k in init_state_dict.keys():
                new_state_dict[k] = init_state_dict[k]
        else:
            new_state_dict[k] = v
    adds = []
    mismatches = []
    for k, v in init_state_dict.items():
        if not has_prefix:
            k = k.replace('module.', '')
        if k not in new_state_dict.keys():
            adds.append(k)
            new_state_dict[k] = init_state_dict[k]
        else:
            if not strict:
                if new_state_dict[k].shape != init_state_dict[k].shape:
                    mismatches.append(k)
                    new_state_dict[k] = init_state_dict[k]

    print('Newly Added Weights: {}'.format(adds))
    print('Mismatched Weights (Used only when not strict): {}'.format(
        mismatches))
    net.load_state_dict(new_state_dict)


def create_sampler(dataset, cfg):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset) if cfg.NUM_GPUS > 1 else None
    return sampler


def reset_parameters(layer, factor=0.1):
    """reinitialize the parameters

    Parameters
    ----------
    layer : nn.Module
        the layer that needs parameter re-initialization.
    factor : float, optional
        the factor that control std of initialized parameters, by default 0.1
    """
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109; bound = sqrt(6/(1+a^2)*fan_in)

    # weight~uniform(-0.1/sqrt(in_features), 0.1/sqrt(in_features))
    # bias~uniform(-0.1/sqrt(in_features), 0.1/sqrt(in_features))
    a = math.sqrt(6 / factor - 1)
    torch.nn.init.kaiming_uniform_(layer.weight, a=a)
    if layer.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = factor / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(layer.bias, -bound, bound)


def _get_model_analysis_input(cfg, is_train, device):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, return the input for training. Otherwise,
            return the input for testing.
    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if cfg.S2D.DNET_MODEL == 'TSM':
        input_tensors = torch.rand(
            cfg.DATA.MAX_FRAME_NUM,
            rgb_dimension,
            cfg.S2D.DNET_INPUT_SIZE,
            cfg.S2D.DNET_INPUT_SIZE,
        )
    else:
        raise NotImplementedError

    meta = {'in_frame_num': torch.Tensor([300.]).long().to(device)}
    input_tensors = input_tensors.unsqueeze(0).to(device)
    logging.info('analysis input:{}'.format(input_tensors.shape))
    logging.info('analysis meta:{}'.format(meta))
    return (input_tensors, meta)


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(cfg, is_train, cfg.DEVICE)
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def s2d_collate_func(batch):
    """collect all data appropriately"""
    inputs, labels, extra_data = zip(*batch)
    inputs = [i for i in inputs]  # saves as list of Tensors
    labels = default_collate(labels)
    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        collated_extra_data[key] = default_collate(data)
    return inputs, labels, collated_extra_data


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    cfg,
    output_queue=None,
):
    """
    for DistributedDataParallel (DDP)
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        output_queue (queue): can optionally be used to return values from the
            master process.
    """

    # the correct local_rank can be obtained in this function
    cfg.LOCAL_RANK = local_rank
    cfg.DEVICE = 'cuda:{}'.format(cfg.LOCAL_RANK)

    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    ret = func(cfg)
    if output_queue is not None and local_rank == 0:
        output_queue.put(ret)
