'''
Description        : The main function for S2DNet.
'''

import os
import json
import time
import random
import numpy as np
import shutil
import logging
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import s2d.utils as utils
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy

from tensorboardX import SummaryWriter
from s2d.dataset import S2DDataSet
from s2d.parser import parser
from s2d.models import S2DNetwork
from configs.default_config import get_cfg as get_tsm_cfg
from s2d.default_config import get_cfg as get_s2d_cfg

logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=logformat)

best_prec1 = 0

# torch.autograd.set_detect_anomaly(True)


def set_seed(seed):
    """set seed for the experiemnt.

    Parameters
    ----------
    seed : int
        the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def group_lr_warmup(opts, opts_base_lrs, step, warmup_steps):
    """Perform learning rate warm up for all given optimizers (inplace operation).

    Parameters
    ----------
    opts : list
        list of the optimizers that require lr warmup.
    opts_base_lrs : list
        list of the base learning rates for the given optimizers.
    step : int
        the current optimizing step (global steps across all epoches).
    warmup_steps : int
        the total num of the warmup steps.
    """
    ratio = (step + 1) / warmup_steps
    logging.info('Performing warmup %.10f' % (ratio))
    for opt, lr in zip(opts, opts_base_lrs):
        if opt is None:
            continue
        for g in opt.param_groups:
            g['lr'] = lr * ratio


def get_transform(augmentation, cfg):
    return torchvision.transforms.Compose([
        augmentation,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(cfg.DATA.MEAN, cfg.DATA.STD)
    ])


def get_datetime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def update_to_log_pth(log_pth, msg):
    with open(log_pth, 'a+') as w:
        w.write(msg)


def has_module_prefix(model):
    if isinstance(model, (torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        return True
    else:
        return False


def main(cfg):
    """Get default settings"""
    cudnn.benchmark = True
    cudnn.deterministic = False
    global best_prec1
    cfg.DATA.NUM_CLASS, cfg.DATA.TRAIN_ANN_FILE, cfg.DATA.VAL_ANN_FILE, cfg.DATA.ROOT_PATH, _, _ = dataset_config.return_dataset(
        cfg.TRAIN.DATASET, cfg.DATA.MODALITY, cfg.DATA.RAW_INPUT_SAMPLING_RATE)
    if cfg.TRAIN.DATASET == 'kinetics' or 'mini-kinetics' in cfg.TRAIN.DATASET:
        cfg.DATA.MAX_FRAME_NUM = 300 // (cfg.S2D.SAMPLING_RATE *
                                         cfg.DATA.RAW_INPUT_SAMPLING_RATE)
        cfg.DATA.DENSE_SAMPLE = False
        cfg.DATA.TWICE_SAMPLE = False
        train_augmentation = torchvision.transforms.Compose([
            GroupMultiScaleCrop(  # 
                cfg.S2D.DNET_INPUT_SIZE, cfg.DATA.AUGMENTATION_SCALES),
            GroupRandomHorizontalFlip(is_flow=False)
        ])
    elif cfg.TRAIN.DATASET == 'somethingv2':
        cfg.DATA.MAX_FRAME_NUM = 170 // (cfg.S2D.SAMPLING_RATE *
                                         cfg.DATA.RAW_INPUT_SAMPLING_RATE)
        cfg.DATA.DENSE_SAMPLE = False
        cfg.DATA.TWICE_SAMPLE = False
        train_augmentation = torchvision.transforms.Compose([
            GroupMultiScaleCrop(cfg.S2D.DNET_INPUT_SIZE,
                                cfg.DATA.AUGMENTATION_SCALES)
        ])
    val_augmentation = torchvision.transforms.Compose([
        GroupScale(int(cfg.DATA.IN_S_SCALE)),
        GroupCenterCrop(int(cfg.S2D.DNET_INPUT_SIZE)),
    ])
    cfg.STORE_NAME = utils.parse_store_name(cfg)

    if utils.is_master_rank(cfg):
        logging.info(cfg)
        utils.check_rootfolders(cfg)
        utils.save_cfg(cfg)
    logging.info('is_master_rank(cfg): {}'.format(utils.is_master_rank(cfg)))
    logging.info('Exp Dir is {}/{}'.format(cfg.ROOT_LOG, cfg.STORE_NAME))
    """Build Models"""
    model = S2DNetwork(cfg).to(cfg.DEVICE)
    if cfg.DIST:
        if cfg.TRAIN.SYNC_BN:
            logging.info('Apply the Sync BatchNorm')
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(cfg.DEVICE)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.LOCAL_RANK],
            output_device=cfg.LOCAL_RANK,
            find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=[cfg.DEVICE])
    """Build Optimizers"""
    policies = model.module.get_optim_policies()
    if utils.is_master_rank(cfg):
        for group in policies:
            logging.info(
                ('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                    group['name'], len(group['params']), group['lr_mult'],
                    group['decay_mult'])))

    main_params = [
        group for group in policies if 'sampler' not in group['name']
    ]
    sampler_params = [
        group for group in policies if 'sampler' in group['name']
    ]

    optimizer = []
    train_only = cfg.S2D.TRAIN_ONLY_PARAMS.replace(' ', '').replace(',', '')
    if 'sampler' not in train_only and 'spatial' not in train_only and 'temporal' not in train_only:
        if 'sgd' in cfg.TRAIN.OPTIM:
            main_optimizer = torch.optim.SGD(
                main_params,
                cfg.TRAIN.BASE_LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        elif 'adamw' in cfg.TRAIN.OPTIM:
            main_optimizer = torch.optim.AdamW(
                main_params,
                cfg.TRAIN.BASE_LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        elif 'lamb' in cfg.TRAIN.OPTIM:
            main_optimizer = utils.Lamb(main_params,
                                        lr=cfg.TRAIN.BASE_LR,
                                        weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            main_optimizer = torch.optim.Adam(
                main_params,
                cfg.TRAIN.BASE_LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        optimizer.append(main_optimizer)
    else:
        logging.info(
            'Train the sampler only so that main optimizer is not build!')

    if cfg.S2D.TWOSTAGE_TRAINING_STAGE != 'WARMUP' or cfg.S2D.TWOSTAGE_TRAINING is False:
        if cfg.S2D.SAMPLER_OPTIM == 'sgd':
            sampler_optimizer = torch.optim.SGD(
                sampler_params,
                lr=cfg.TRAIN.BASE_LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        elif cfg.S2D.SAMPLER_OPTIM == 'lamb':
            sampler_optimizer = utils.Lamb(sampler_params,
                                           lr=cfg.TRAIN.BASE_LR,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        elif cfg.S2D.SAMPLER_OPTIM == 'adamw':
            sampler_optimizer = torch.optim.AdamW(
                sampler_params,
                lr=cfg.TRAIN.BASE_LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            sampler_optimizer = torch.optim.Adam(
                sampler_params,
                lr=cfg.TRAIN.BASE_LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        optimizer += [sampler_optimizer]
    else:
        logging.info('Warmup S2D, so that sampler optimizer is not build!')

    assert (len(optimizer) >= 1)

    schedulers = []
    """Resume Models"""
    if cfg.TRAIN.RESUME:
        if os.path.isfile(cfg.TRAIN.RESUME):
            logging.info(
                ("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME)))
            checkpoint = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
            utils.load_state_dict(model, checkpoint['state_dict'],
                                  cfg.TRAIN.RESUME_IGNORE_FLAGS)

            if cfg.TRAIN.RESUME_TRAINING:
                logging.info('=> resuming optimizer !')
                for i, optim in enumerate(optimizer):
                    optim.load_state_dict(checkpoint['optimizer'][i])
        else:
            logging.info(
                ("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME)))
    else:
        if cfg.S2D.DNET_RESUME:
            if os.path.isfile(cfg.S2D.DNET_RESUME):
                logging.info(("=> DETAIL NET loading checkpoint '{}'".format(
                    cfg.S2D.DNET_RESUME)))
                checkpoint = torch.load(cfg.S2D.DNET_RESUME,
                                        map_location='cpu')
                utils.load_state_dict(model.module.DNet.model,
                                      checkpoint['state_dict'],
                                      cfg.TRAIN.RESUME_IGNORE_FLAGS,
                                      strict=False)
            else:
                logging.info(("=> no checkpoint found at '{}'".format(
                    cfg.S2D.DNET_RESUME)))
        if cfg.S2D.SNET_RESUME:
            if os.path.isfile(cfg.S2D.SNET_RESUME):
                logging.info(("=> SYNOPSIS NET loading checkpoint '{}'".format(
                    cfg.S2D.SNET_RESUME)))
                checkpoint = torch.load(cfg.S2D.SNET_RESUME,
                                        map_location='cpu')
                utils.load_state_dict(model.module.SNet.model,
                                      checkpoint['state_dict'],
                                      cfg.TRAIN.RESUME_IGNORE_FLAGS,
                                      strict=False)
            else:
                logging.info(("=> no checkpoint found at '{}'".format(
                    cfg.S2D.SNET_RESUME)))
    """Build data loader"""

    train_dataset = S2DDataSet(cfg,
                               cfg.DATA.TRAIN_ANN_FILE,
                               random_shift=True,
                               transform=get_transform(train_augmentation,
                                                       cfg))
    val_dataset = S2DDataSet(cfg,
                             cfg.DATA.VAL_ANN_FILE,
                             test_mode=True,
                             random_shift=False,
                             transform=get_transform(val_augmentation, cfg))
    if cfg.DIST:
        train_sampler = utils.create_sampler(train_dataset, cfg=cfg)
        val_sampler = utils.create_sampler(val_dataset, cfg=cfg)
    else:
        train_sampler = None
        val_sampler = None

    logging.info('cfg.TRAIN.BATCH_SIZE : {}'.format(cfg.TRAIN.BATCH_SIZE))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.WORKERS,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        collate_fn=utils.s2d_collate_func)  # prevent something not % n_GPU
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.TRAIN.BATCH_SIZE,
                                             num_workers=cfg.TRAIN.WORKERS,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             collate_fn=utils.s2d_collate_func)
    """Define loss function (criterion) and optimizer"""
    if cfg.S2D.GATHER_DNET_LOSS:
        logging.info('using GATHER_DNET_LOSS')
        dnet_criterion = utils.get_criterion(cfg.S2D.DNET_SMOOTH_ALPHA,
                                             require_logsoftmax=True).cuda()
        val_dnet_criterion = utils.get_criterion(
            cfg.S2D.DNET_SMOOTH_ALPHA, require_logsoftmax=True).cuda()
    elif cfg.S2D.NAIVE_DNET_LOSS:
        logging.info('using NAIVE_DNET_LOSS')
        dnet_criterion = utils.get_criterion(cfg.S2D.DNET_SMOOTH_ALPHA,
                                             require_logsoftmax=True).cuda()
        val_dnet_criterion = utils.get_criterion(
            cfg.S2D.DNET_SMOOTH_ALPHA, require_logsoftmax=True).cuda()
    else:
        logging.info('using MASKED_DNET_LOSS')
        dnet_criterion = utils.get_criterion(cfg.S2D.DNET_SMOOTH_ALPHA,
                                             require_logsoftmax=False).cuda()
        val_dnet_criterion = utils.get_criterion(
            cfg.S2D.DNET_SMOOTH_ALPHA, require_logsoftmax=False).cuda()
    snet_criterion = utils.get_criterion(cfg.S2D.SNET_SMOOTH_ALPHA,
                                         require_logsoftmax=True).cuda()
    val_snet_criterion = utils.get_criterion(cfg.S2D.SNET_SMOOTH_ALPHA,
                                             require_logsoftmax=True).cuda()

    if cfg.EVALUATE:
        validate(cfg, val_loader, model,
                 (val_dnet_criterion, val_snet_criterion), 0)
        return
    """Set up logger"""
    log_losses_pth = None
    log_metrics_pth = None
    log_print = None
    tf_writer = None
    if utils.is_master_rank(cfg):  # local master
        log_print = open(os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME, 'log.csv'),
                         'a+')
    if utils.is_master_rank(cfg):  # global master
        log_losses_pth = os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME,
                                      'losses.csv')  # open(, 'a+')
        log_metrics_pth = os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME,
                                       'metrics.csv')  # open(, 'a+')
        tf_writer = SummaryWriter(
            log_dir=os.path.join(cfg.ROOT_LOG, cfg.STORE_NAME))
    """Training Loops"""
    scaler = GradScaler() if cfg.AMP else None
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        epoch_start = time.time()

        if cfg.DIST:
            train_sampler.set_epoch(epoch)

        # switch to train mode
        model.train()
        utils.adjust_learning_rate(optimizer, epoch, cfg)
        for optim in optimizer:
            for param_group in optim.param_groups:
                if utils.is_master_rank(cfg):
                    logging.info('Adjust, ep:{}, name:{}, lr:{}'.format(
                        epoch, param_group['name'], param_group['lr']))

        # train for one epoch
        msg = train(cfg, train_loader, model, (dnet_criterion, snet_criterion),
                    optimizer, epoch, log_print, tf_writer, scaler)

        if utils.is_master_rank(cfg):
            if epoch == cfg.TRAIN.START_EPOCH:
                title = ' '.join(['ep'] +
                                 ['{}'.format(k) for k, v in msg.items()])
                update_to_log_pth(log_losses_pth, title + '\n')
            msg = ' '.join([str(epoch)] +
                           ['{:.4f},'.format(v) for k, v in msg.items()])
            update_to_log_pth(log_losses_pth, msg + '\n')
        """evaluate on validation set"""
        if (epoch + 1) % cfg.EVAL_FREQ == 0 or epoch == cfg.TRAIN.EPOCHS - 1:
            val_loss, val_dnet_loss, val_snet_loss, prec1, prec5, snet_top1, snet_top5 = validate(
                cfg, val_loader, model,
                (val_dnet_criterion, val_snet_criterion), epoch, log_print,
                tf_writer)

            # record best prec@1 and save checkpoint
            if utils.is_master_rank(cfg):
                is_best = prec1 >= best_prec1  # >= to keep the newest best model
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)
                output_best = '--- Best Prec@1: %.3f' % (best_prec1)
                logging.info(output_best)
                log_print.write(output_best + '\n')
                log_print.flush()

                utils.save_checkpoint(
                    {
                        'epoch': epoch,
                        'cfg': cfg.dump(),
                        'state_dict': model.state_dict(),
                        'optimizer':
                        [optim.state_dict() for optim in optimizer],
                        'best_prec1': best_prec1,
                    }, is_best, cfg, epoch)

                if epoch == cfg.TRAIN.START_EPOCH:
                    title = ' '.join([
                        'ep', 'prec1', 'prec5', 'coarse1', 'coarse5', 'loss',
                        'fine_loss', 'coarse_loss', 'reg_loss'
                    ])
                    update_to_log_pth(log_metrics_pth, title + '\n')
                msg = ' '.join([str(epoch)] + [
                    '{:.4f},'.format(v) for v in [
                        prec1, prec5, snet_top1, snet_top5, val_loss,
                        val_dnet_loss, val_snet_loss, 0.
                    ]
                ])
                update_to_log_pth(log_metrics_pth, msg + '\n')

        if utils.is_master_rank(cfg):
            logging.info('--- epoch timecost: {}'.format(time.time() -
                                                         epoch_start))

    if utils.is_master_rank(cfg):
        if cfg.S2D.TWOSTAGE_TRAINING and cfg.S2D.TWOSTAGE_TRAINING_STAGE == 'WARMUP':
            utils.save_checkpoint(
                {
                    'epoch': epoch,
                    'cfg': cfg.dump(),
                    'state_dict': model.state_dict(),
                    'optimizer': [optim.state_dict() for optim in optimizer],
                    'best_prec1': best_prec1,
                },
                is_best,
                cfg,
                epoch,
                spec_name='warmup')
    if utils.is_master_rank(cfg):
        log_print.close()


def train(cfg,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          log,
          tf_writer,
          scaler=None):
    if isinstance(criterion, tuple):
        dnet_criterion, snet_criterion = criterion
    else:
        dnet_criterion = snet_criterion = criterion

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dnet_losses = AverageMeter()
    snet_losses = AverageMeter()
    dnet_top1 = AverageMeter()
    dnet_top5 = AverageMeter()
    snet_top1 = AverageMeter()
    snet_top5 = AverageMeter()

    if cfg.AMP:
        assert (scaler is not None)

    end = time.time()

    for optim in optimizer:
        optim.zero_grad()

    for i, (input, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        meta = {
            k: v.cuda()
            for k, v in meta.items() if isinstance(v, torch.Tensor)
        }
        meta['label'] = target

        # compute output
        with autocast(enabled=cfg.AMP):
            dnet_output, snet_output, _, raw_dnet_output = model(input, meta)
            snet_output = snet_output.float()
            dnet_output = dnet_output.float()
            if cfg.S2D.GATHER_DNET_LOSS:
                topk_indices = torch.topk(snet_output, k=5, dim=-1).indices
                new_t = []
                for (_t, _i) in zip(target, topk_indices):
                    if _t in _i:
                        this_ranking = (_i == _t).nonzero(
                            as_tuple=True)[0].squeeze()
                        new_t.append(this_ranking)
                    else:
                        new_t.append(torch.ones_like(_t) *
                                     -100)  # ignore index
                topk_target = torch.stack(new_t, dim=0)
                topk_values = dnet_output.gather(dim=-1, index=topk_indices)
                dnet_loss = dnet_criterion(topk_values, topk_target).mean()
            else:
                dnet_loss = dnet_criterion(dnet_output.float(), target).mean()

            snet_loss = snet_criterion(snet_output, target)
            if cfg.S2D.TOPK_SNET_LOSS:
                topk_ = snet_output.topk(dim=-1, k=5)[1]
                miss = 1 - (topk_ == target.unsqueeze(-1)).sum(-1)
                snet_loss = (snet_loss * miss)
            snet_loss = snet_loss.mean()

            # FIXME: use one option for the two
            if cfg.S2D.SNET_LR_RATIO <= 0.:
                cfg.S2D.SNET_LOSS_WEIGHT = 0.

            loss = cfg.S2D.DNET_LOSS_WEIGHT * dnet_loss + cfg.S2D.SNET_LOSS_WEIGHT * snet_loss

            loss = loss / cfg.S2D.ITERS_TO_ACCUMULATE

        if cfg.AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % (cfg.S2D.ITERS_TO_ACCUMULATE) == 0:
            # clip before backward
            if cfg.TRAIN.CLIP_GRADIENT is not None and cfg.TRAIN.CLIP_GRADIENT != 0.:
                if cfg.AMP:
                    for optim in optimizer:
                        scaler.unscale_(optim)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_nan_gradients = torch.isnan(param.grad).any()
                        has_inf_gradients = torch.isinf(param.grad).any()
                        if has_nan_gradients or has_inf_gradients:
                            logging.info(
                                'weight: {}, isnan: {}, isinf: {}.'.format(
                                    name, has_nan_gradients,
                                    has_inf_gradients))
                            # logging.info('weight: {}, isnan: {}, isinf: {}. Perform nan_to_num.'.format(name, has_nan_gradients, has_inf_gradients))
                            # param.grad = param.grad.nan_to_num(nan=0., posinf=1*cfg.TRAIN.CLIP_GRADIENT, neginf=-1*cfg.TRAIN.CLIP_GRADIENT)

            clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRADIENT)

            if cfg.S2D.ITERS_TO_ACCUMULATE != 1:
                logging.info('Update using accumulated gradients.')
            if cfg.AMP:
                for optim in optimizer:
                    scaler.step(optim)
                scaler.update()
                for optim in optimizer:
                    optim.zero_grad()
            else:
                for optim in optimizer:
                    optim.step()
                    optim.zero_grad()

        # get Metrics
        sprec1, sprec5 = accuracy(snet_output.data, target, topk=(1, 5))
        if cfg.S2D.GATHER_DNET_LOSS:
            dprec1, _ = accuracy(topk_values.data, topk_target, topk=(1, 5))
            dprec5 = sprec5
        else:
            dprec1, dprec5 = accuracy(dnet_output.data, target, topk=(1, 5))

        # reduce
        if cfg.DIST:
            loss, dnet_loss, snet_loss, dprec1, dprec5, sprec1, sprec5 = all_reduce(
                [loss, dnet_loss, snet_loss, dprec1, dprec5, sprec1, sprec5])

        # logging
        losses.update(loss.item(), len(input))
        dnet_losses.update(dnet_loss.item(), len(input))
        snet_losses.update(snet_loss.item(), len(input))
        dnet_top1.update(dprec1.item(), len(input))
        dnet_top5.update(dprec5.item(), len(input))
        snet_top1.update(sprec1.item(), len(input))
        snet_top5.update(sprec5.item(), len(input))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and utils.is_master_rank(
                cfg):  # local rank 0
            rest_time = time.strftime(
                '%H:%M:%S',
                time.gmtime(int((len(train_loader) - i) * batch_time.avg)))
            if utils.is_master_rank(cfg):  # global rank 0
                output = (
                    '[{0}] Epoch: ({1})({2}/{3}) (rt: {4}), lr: {lr:.7f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Fine_Loss {dnet_loss.val:.4f} ({dnet_loss.avg:.4f})\t'
                    'Coarse_Loss {snet_loss.val:.4f} ({snet_loss.avg:.4f})\t'
                    'Fine_Prec@1 {dnet_top1.val:.3f} ({dnet_top1.avg:.3f})\t'
                    'Fine_Prec@5 {dnet_top5.val:.3f} ({dnet_top5.avg:.3f})\t'
                    'Coarse_Prec@1 {snet_top1.val:.3f} ({snet_top1.avg:.3f})\t'
                    'Coarse_Prec@5 {snet_top5.val:.3f} ({snet_top5.avg:.3f})'.
                    format(get_datetime(),
                           epoch,
                           i,
                           len(train_loader),
                           rest_time,
                           batch_time=batch_time,
                           data_time=data_time,
                           loss=losses,
                           dnet_loss=dnet_losses,
                           snet_loss=snet_losses,
                           dnet_top1=dnet_top1,
                           dnet_top5=dnet_top5,
                           snet_top1=snet_top1,
                           snet_top5=snet_top5,
                           lr=optimizer[0].param_groups[0]['lr']))
            else:
                output = (
                    '[{0}] Epoch: ({1})({2}/{3}) (rt: {4}), lr: {lr:.7f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        get_datetime(),
                        epoch,
                        i,
                        len(train_loader),
                        rest_time,
                        batch_time=batch_time,
                        data_time=data_time,
                        lr=optimizer[0].param_groups[0]['lr']))
            logging.info(output)
            log.write(output + '\n')
            log.flush()

    msg = {
        'loss/train': losses.avg,
        'loss/train_coarse': snet_losses.avg,
        'loss/train_fine': dnet_losses.avg,
        'loss/reg_terms': 0.,
        'acc/train_fine_top1': dnet_top1.avg,
        'acc/train_fine_top5': dnet_top5.avg,
        'acc/train_coarse_top1': snet_top1.avg,
        'acc/train_coarse_top5': snet_top5.avg,
        'lr': optimizer[0].param_groups[-1]['lr']
    }

    if utils.is_master_rank(cfg):
        for k, v in msg.items():
            tf_writer.add_scalar(k, v, epoch)

    return msg


def validate(cfg,
             val_loader,
             model,
             criterion,
             epoch,
             log=None,
             tf_writer=None):
    if isinstance(criterion, tuple):
        dnet_criterion, snet_criterion = criterion
    else:
        dnet_criterion = snet_criterion = criterion

    batch_time = AverageMeter()
    losses = AverageMeter()
    dnet_losses = AverageMeter()
    snet_losses = AverageMeter()

    snet_top1 = AverageMeter()
    snet_top5 = AverageMeter()
    dnet_top1 = AverageMeter()
    dnet_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if cfg.EVALUATE:
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        starter, ender = None, None

    end = time.time()
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            input = [i.cuda() for i in input]
            target = target.cuda()
            meta = {
                k: v.cuda()
                for k, v in meta.items() if isinstance(v, torch.Tensor)
            }
            meta['label'] = target

            if cfg.EVALUATE and i == 0:
                from fvcore.nn import FlopCountAnalysis, parameter_count
                model.module.DNet.sampler.set_jit(True)
                _input = (input[:], {k: v[:] for k, v in meta.items()})
                flops = FlopCountAnalysis(model,
                                          _input).total() / 1e9 / len(input)
                param_num = parameter_count(model)
                print('Param DNet',
                      parameter_count(model)['module.DNet'] / 1e6)
                print('Param SNet',
                      parameter_count(model)['module.SNet'] / 1e6)
                param_num = parameter_count(model)[
                    'module.DNet'] + parameter_count(model)['module.SNet']
                param_num = param_num / 1e6
                logging.info('GFLOPs: {:.1f}, Num Of Params:{}'.format(
                    flops, param_num))
                model.module.DNet.sampler.set_jit(False)

                # GPU WARMUP
                for i in range(10):
                    with autocast(enabled=cfg.AMP):
                        _, _, _, _ = model(input, meta)
                repetitions = 100
                timings = np.zeros((repetitions, 1))
                model.module.DNet.runtime = True
                model.module.SNet.runtime = True
                for rep in range(repetitions):
                    starter.record()
                    with autocast(enabled=cfg.AMP):
                        _, _, _, _ = model(input, meta)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
                mean_syn = np.sum(timings) / repetitions
                logging.info(
                    'inference time (w/o divided by BS): {:.5f}'.format(
                        mean_syn))
                logging.info('SNet inference time: {:.5f}'.format(
                    model.module.SNet.runtime_stats / repetitions))
                logging.info('DNet sampler inference time: {:.5f}'.format(
                    model.module.DNet.runtime_splr_stats / repetitions))
                logging.info('DNet backbone inference time: {:.5f}'.format(
                    model.module.DNet.runtime_bone_stats / repetitions))
                model.module.DNet.runtime = False
                model.module.SNet.runtime = False
                assert (), 'check time'

            with autocast(enabled=cfg.AMP):
                dnet_output, snet_output, _, raw_dnet_output = model(
                    input, meta)
            snet_output = snet_output.float()
            dnet_output = dnet_output.float()
            if cfg.S2D.GATHER_DNET_LOSS:
                topk_indices = torch.topk(snet_output, k=5, dim=-1).indices
                new_t = []
                for (_t, _i) in zip(target, topk_indices):
                    if _t in _i:
                        this_ranking = (_i == _t).nonzero(
                            as_tuple=True)[0].squeeze()
                        new_t.append(this_ranking)
                    else:
                        new_t.append(torch.ones_like(_t) *
                                     -100)  # ignore index
                topk_target = torch.stack(new_t, dim=0)
                topk_values = dnet_output.gather(dim=-1, index=topk_indices)
                dnet_loss = dnet_criterion(topk_values, topk_target).mean()
            else:
                # NAIVE DNET LOSS or Mask Fine Loss
                dnet_loss = dnet_criterion(dnet_output, target).mean()

            snet_loss = snet_criterion(snet_output, target)
            if cfg.S2D.TOPK_SNET_LOSS:
                topk_ = snet_output.topk(dim=-1, k=5)[1]
                miss = 1 - (topk_ == target.unsqueeze(-1)).sum(-1)
                snet_loss = snet_loss * miss
            snet_loss = snet_loss.mean()

            loss = cfg.S2D.DNET_LOSS_WEIGHT * dnet_loss + cfg.S2D.SNET_LOSS_WEIGHT * snet_loss

            # get Metrics
            sprec1, sprec5 = accuracy(snet_output.detach(),
                                      target,
                                      topk=(1, 5))
            if cfg.S2D.GATHER_DNET_LOSS:
                dprec1, _ = accuracy(topk_values.detach(),
                                     topk_target,
                                     topk=(1, 5))
                dprec5 = sprec5
            else:
                dprec1, dprec5 = accuracy(dnet_output.detach(),
                                          target,
                                          topk=(1, 5))

            # reduce
            if cfg.DIST:
                loss, dnet_loss, snet_loss, dprec1, dprec5, sprec1, sprec5 = all_reduce(
                    [
                        loss, dnet_loss, snet_loss, dprec1, dprec5, sprec1,
                        sprec5
                    ])

            # logging
            losses.update(loss.item(), len(input))
            dnet_losses.update(dnet_loss.item(), len(input))
            snet_losses.update(snet_loss.item(), len(input))
            snet_top1.update(sprec1.item(), len(input))
            snet_top5.update(sprec5.item(), len(input))
            dnet_top1.update(dprec1.item(), len(input))
            dnet_top5.update(dprec5.item(), len(input))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0 and utils.is_master_rank(
                    cfg):  # local rank 0
                rest_time = time.strftime(
                    '%H:%M:%S',
                    time.gmtime(int((len(val_loader) - i) * batch_time.avg)))
                if utils.is_master_rank(cfg):  # global rank 0
                    output = (
                        '[{0}] Test: ({1}/{2}) (rt: {3})\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Fine_Loss {dnet_loss.val:.4f} ({dnet_loss.avg:.4f})\t'
                        'Coarse_Loss {snet_loss.val:.4f} ({snet_loss.avg:.4f})\t'
                        'Fine_Prec@1 {dnet_top1.val:.3f} ({dnet_top1.avg:.3f})\t'
                        'Fine_Prec@5 {dnet_top5.val:.3f} ({dnet_top5.avg:.3f})\t'
                        'Coarse_Prec@1 {snet_top1.val:.3f} ({snet_top1.avg:.3f})\t'
                        'Coarse_Prec@5 {snet_top5.val:.3f} ({snet_top5.avg:.3f})'
                        .format(get_datetime(),
                                i,
                                len(val_loader),
                                rest_time,
                                batch_time=batch_time,
                                loss=losses,
                                dnet_loss=dnet_losses,
                                snet_loss=snet_losses,
                                dnet_top1=dnet_top1,
                                dnet_top5=dnet_top5,
                                snet_top1=snet_top1,
                                snet_top5=snet_top5))
                else:
                    output = (
                        '[{0}] Test: ({1}/{2}) (rt: {3})\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.
                        format(get_datetime(),
                               i,
                               len(val_loader),
                               rest_time,
                               batch_time=batch_time))
                logging.info(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    if utils.is_master_rank(cfg):
        output = ('Testing Results: Fine_Prec@1 {dnet_top1.avg:.3f}\t'
                  'Fine_Prec@5 {dnet_top5.avg:.3f}\t'
                  'Coarse_Prec@1 {snet_top1.avg:.3f}\t'
                  'Coarse_Prec@5 {snet_top5.avg:.3f}\t'
                  'Loss {loss.avg:.5f}'.format(dnet_top1=dnet_top1,
                                               dnet_top5=dnet_top5,
                                               snet_top1=snet_top1,
                                               snet_top5=snet_top5,
                                               loss=losses))
        logging.info(output)
        if log is not None:
            log.write(output + '\n')
            log.flush()

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('loss/test_snet', snet_losses.avg, epoch)
            tf_writer.add_scalar('loss/test_dnet', dnet_losses.avg, epoch)
            tf_writer.add_scalar('acc/test_dnet_top1', dnet_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_dnet_top5', dnet_top5.avg, epoch)
            tf_writer.add_scalar('acc/test_snet_top1', snet_top1.avg, epoch)
            tf_writer.add_scalar('acc/test_snet_top5', snet_top5.avg, epoch)

    logging.info('avg dnet_top1: {}'.format(dnet_top1.avg))
    return losses.avg, dnet_losses.avg, snet_losses.avg, dnet_top1.avg, dnet_top5.avg, snet_top1.avg, snet_top5.avg


def parse_opts(opts):
    """Parse the opts from command line. 

    Parameters
    ----------
    opts : list
        [k1,v1,k2,v2,....]

    Returns
    -------
    tuple
        codebase_opts: will be loaded immediately after parsing
        snet_opts: will be loaded when building SNet (so as to overwrite the default configs of the model)
        dnet_opts: will be loaded when building DNet (so as to overwrite the default configs of the model)

        Note, the `snet_opts`/`dnet_opts` will be loaded when building the models of  SNet/DNet so as to overwrite
        the default configs of the model. The process is especially important when SNet and DNet are using the same
        model (e.g., TSM) while we would like to use them differently in the two models. For example, let say 
        we use mobilenetv2 in SNet, use R50 in DNet and keep the other settings the same. 

    """
    codebase_opts = []
    snet_opts = []
    dnet_opts = []
    for i in range(1, len(opts), 2):
        if opts[i] == 'true' or opts[i] == 'True':
            opts[i] = True
        if opts[i] == 'false' or opts[i] == 'False':
            opts[i] = False
    for i in range(0, len(opts), 2):
        if 'SNET.' in opts[i]:
            opts[i] = opts[i].replace('SNET.', '')
            snet_opts += opts[i:i + 2]
        elif 'DNET.' in opts[i]:
            opts[i] = opts[i].replace('DNET.', '')
            dnet_opts += opts[i:i + 2]
        else:
            codebase_opts += opts[i:i + 2]

    return codebase_opts, snet_opts, dnet_opts


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup codebase default cfg.
    cfg = get_tsm_cfg()

    # Setup s2d default cfg.
    _c = get_s2d_cfg()
    for k in _c.keys():
        cfg[k] = _c[k]

    # Load config from cfg_file (load the configs that vary accross datasets).
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Load config from command line, overwrite config from opts (for the convenience of experiemnts).
    if args.opts is not None:
        codebase_opts, snet_opts, dnet_opts = parse_opts(args.opts)
        cfg.merge_from_list(codebase_opts)
        cfg.S2D.SNET_OPTS = snet_opts
        cfg.S2D.DNET_OPTS = dnet_opts

    cfg.TRAIN.DATASET = args.dataset
    if cfg.DIST and cfg.NUM_GPUS > 1:
        # per-gpu lr -> global lr (for optimizer)
        world_size = cfg.NUM_GPUS * cfg.NUM_SHARDS
        cfg.TRAIN.BASE_LR = cfg.TRAIN.BASE_LR * world_size * cfg.TRAIN.BATCH_SIZE * cfg.S2D.ITERS_TO_ACCUMULATE
        cfg.TRAIN.WARMUP_LR = cfg.TRAIN.WARMUP_LR * world_size * cfg.TRAIN.BATCH_SIZE * cfg.S2D.ITERS_TO_ACCUMULATE
        logging.info('World Size: {}, Global Lr: {}, Warmup Lr: {}'.format(
            world_size, cfg.TRAIN.BASE_LR, cfg.TRAIN.WARMUP_LR))

    return cfg


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    cfg = load_config(args)
    set_seed(8999)

    if cfg.DIST and cfg.NUM_GPUS >= 1:
        # launch_job
        torch.multiprocessing.spawn(
            utils.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                main,
                cfg.INIT_METHOD,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                'nccl',
                cfg,
            ),
        )
    else:
        cfg.LOCAL_RANK = args.local_rank
        if cfg.NUM_GPUS >= 1:
            cfg.DEVICE = 'cuda:{}'.format(cfg.LOCAL_RANK)
        else:
            cfg.DEVICE = 'cpu'
        cfg.DIST = False
        main(cfg)
