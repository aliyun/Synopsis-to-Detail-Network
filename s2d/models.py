'''
Description        : Models of the S2DNet.
'''

from .sampler import UniformSampler, AdaptiveSampler, get_feature_shape, FEATURE_DIMS
import torch.nn as nn
import torch
import os
import sys
import random
import numpy as np
from fvcore.common.config import CfgNode
from .utils import is_master_rank

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))  # parent dir
if 'slowfast' in os.getcwd() or 'SlowFast' in os.getcwd():
    from slowfast.models.build import MODEL_REGISTRY
else:
    from fvcore.common.registry import Registry
    MODEL_REGISTRY = Registry("MODEL")

import logging

logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=logformat)


def set_bn_affine_eval(m, name=None):
    classname = m.__class__.__name__
    if 'base_model.bn1' not in classname:
        if classname.find('BatchNorm') != -1:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def get_optim_policies(
        model, cfg, model_cfg,
        train_only_params):  # sep_low_level seperate high-low level
    """A general verion of `get_optim_policies` from TSM"""
    _enable_pbn = cfg.TRAIN.FREEZE_BN
    tuning = (model_cfg.TUNE_FROM
              and model_cfg.TRAIN.DATASET in model_cfg.TUNE_FROM)
    if tuning:
        linear_mul = 1
    else:
        linear_mul = 5
    sampler_mul = cfg.S2D.SAMPLER_LR_RATIO
    spa_mul = cfg.S2D.SPATIAL_LR_RATIO
    tem_mul = cfg.S2D.TEMPORAL_LR_RATIO

    individual_params = []
    conv_weight, conv_bias, lr_weight, lr_bias, bn = [], [], [], [], []
    sampler_spatial_conv_weight, sampler_spatial_conv_bias, sampler_temporal_conv_weight, sampler_temporal_conv_bias, sampler_spatial_bn, sampler_temporal_bn = [], [], [], [], [], []
    sampler_fc_spatial_lr_weight, sampler_fc_spatial_lr_bias, sampler_fc_temporal_lr_weight, sampler_fc_temporal_lr_bias = [],[], [], []
    conv_cnt, lr_cnt, bn_cnt, sampler_conv_cnt, sampler_lr_cnt, sampler_bn_cnt = 0, 0, 0, 0, 0, 0

    logging.info('`train_only_params` params: {}'.format(train_only_params))
    for name, m in model.named_modules():
        if len(train_only_params) > 0:
            train_flag = False
            for top in train_only_params:
                if top in name:
                    train_flag = True
                    break
            if not train_flag:
                continue

        if 'sampler' in name:
            if isinstance(m,
                          (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                """CONV"""
                sampler_conv_cnt += 1
                ps = list(m.parameters())
                if 'spatial' in name:
                    sampler_spatial_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        sampler_spatial_conv_bias.append(ps[1])
                elif 'temporal' in name or 'space_time' in name or 'target' in name:
                    sampler_temporal_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        sampler_temporal_conv_bias.append(ps[1])
                else:
                    raise NotImplementedError
            elif isinstance(m, (torch.nn.Linear)):
                """LINEAR"""
                sampler_lr_cnt += 1
                ps = list(m.parameters())
                if 'spatial' in name:
                    sampler_fc_spatial_lr_weight.append(ps[0])
                    if len(ps) == 2:
                        sampler_fc_spatial_lr_bias.append(ps[1])
                elif 'temporal' in name or 'space_time' in name or 'target' in name:
                    sampler_fc_temporal_lr_weight.append(ps[0])
                    if len(ps) == 2:
                        sampler_fc_temporal_lr_bias.append(ps[1])
                else:
                    raise NotImplementedError
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d,
                                torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                """BN"""
                sampler_bn_cnt += 1
                if 'spatial' in name:
                    sampler_spatial_bn.extend(list(m.parameters()))
                elif 'temporal' in name or 'space_time' in name or 'target' in name:
                    sampler_temporal_bn.extend(list(m.parameters()))
                else:
                    raise NotImplementedError
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy"
                        .format(type(m)))
            continue

        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
            """CONV"""
            ps = list(m.parameters())
            conv_cnt += 1
            conv_weight.append(ps[0])
            if len(ps) == 2:
                conv_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            """LINEAR"""
            lr_cnt += 1
            ps = list(m.parameters())
            lr_weight.append(ps[0])
            if len(ps) == 2:
                lr_bias.append(ps[1])
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            """BN"""
            bn_cnt += 1
            # later BN's are frozen
            if not _enable_pbn or ('base_model.bn1' in name) or (
                    'base_model.features.0'
                    in name) or ('fusion' in name) or ('ctx_feature_layer'
                                                       in name):
                bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError(
                    "New atomic module type: {}. Need to give it a learning policy"
                    .format(type(m)))

    for i, (name, m) in enumerate(model.named_parameters()):
        if 'logit_scale' in name:
            individual_params.append(m)

    # follow TSM settings
    return [
        {
            'params': individual_params,
            'lr_mult': 1,
            'decay_mult': 1,
            'name': "individual_params"
        },
        {
            'params': conv_weight,
            'lr_mult': 1,
            'decay_mult': 1,
            'name': "conv_weight"
        },
        {
            'params': conv_bias,
            'lr_mult': 2,
            'decay_mult': 0,
            'name': "conv_bias"
        },
        {
            'params': lr_weight,
            'lr_mult': 1 * linear_mul,
            'decay_mult': 1,
            'name': "lr_weight_{}x".format(linear_mul)
        },
        {
            'params': lr_bias,
            'lr_mult': 2 * linear_mul,
            'decay_mult': 0,
            'name': "lr_bias_{}x".format(linear_mul)
        },
        {
            'params': bn,
            'lr_mult': 1,
            'decay_mult': 0,
            'name': "BN scale/shift"
        },
        {
            'params': sampler_spatial_conv_weight,
            'lr_mult': 1 * sampler_mul * spa_mul,
            'decay_mult': 1,
            'name': "sampler_spatial_weight"
        },
        {
            'params': sampler_spatial_conv_bias,
            'lr_mult': 1 * sampler_mul * spa_mul,
            'decay_mult': 0,
            'name': "sampler_spatial_bias"
        },
        {
            'params': sampler_temporal_conv_weight,
            'lr_mult': 1 * sampler_mul * tem_mul,
            'decay_mult': 1,
            'name': "sampler_temporal_weight"
        },
        {
            'params': sampler_temporal_conv_bias,
            'lr_mult': 1 * sampler_mul * tem_mul,
            'decay_mult': 0,
            'name': "sampler_temporal_bias"
        },
        {
            'params': sampler_fc_spatial_lr_weight,
            'lr_mult': 1 * sampler_mul * spa_mul,
            'decay_mult': 1,
            'name': "sampler_fc_spatial_lr_weight"
        },
        {
            'params': sampler_fc_spatial_lr_bias,
            'lr_mult': 1 * sampler_mul * spa_mul,
            'decay_mult': 0,
            'name': "sampler_fc_spatial_lr_bias"
        },
        {
            'params': sampler_fc_temporal_lr_weight,
            'lr_mult': 1 * sampler_mul * tem_mul,
            'decay_mult': 1,
            'name': "sampler_fc_temporal_lr_weight"
        },
        {
            'params': sampler_fc_temporal_lr_bias,
            'lr_mult': 1 * sampler_mul * tem_mul,
            'decay_mult': 0,
            'name': "sampler_fc_temporal_lr_bias"
        },
        {
            'params': sampler_spatial_bn,
            'lr_mult': 1 * sampler_mul * spa_mul,
            'decay_mult': 0,
            'name': "sampler_spatial_bn"
        },
        {
            'params': sampler_temporal_bn,
            'lr_mult': 1 * sampler_mul * tem_mul,
            'decay_mult': 0,
            'name': "sampler_temporal_bn"
        },
    ]


@MODEL_REGISTRY.register()
class S2DNetwork(nn.Module):

    def __init__(self, cfg):
        super(S2DNetwork, self).__init__()
        self.cfg = cfg
        self.SNet = SynopsisNetwork(cfg)
        self.DNet = DetailNetwork(cfg)
        if is_master_rank(cfg):
            logging.info('SNet: {}'.format(self.SNet))
            logging.info('DNET: {}'.format(self.DNet))

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters

        Returns
        -------
        mode: bool
            If True, training mode, else evaluating mode.

        """
        super(S2DNetwork, self).train(mode)
        if not mode:
            return self

        cfg = self.cfg
        sampler = self.DNet.sampler
        if cfg.S2D.TWOSTAGE_TRAINING:
            sampler.fixed_spatial_sampling = False
            sampler.fixed_temporal_sampling = False
            if cfg.S2D.TWOSTAGE_TRAINING_STAGE == 'WARMUP':
                sampler.eval()
                sampler.fixed_spatial_sampling = sampler.fixed_temporal_sampling = True
                for param in sampler.parameters():
                    param.requires_grad = False
            elif cfg.S2D.TWOSTAGE_TRAINING_STAGE == 'SAMPLING':
                if cfg.S2D.TRAIN_ONLY_PARAMS == '':
                    logging.info('End-to-end training')
                    self.DNet.train()
                    self.SNet.train()
                else:
                    super(S2DNetwork, self).train(mode=False)
                    if 'sampler' in cfg.S2D.TRAIN_ONLY_PARAMS:
                        logging.info('Training sampler only')
                        sampler.train()
                    elif 'spatial' in cfg.S2D.TRAIN_ONLY_PARAMS:
                        assert (cfg.S2D.SPACE_TIME_SHARE_CONVS is False)
                        logging.info(
                            'Training spatial sampling only (freezing temporal sampling)...'
                        )
                        sampler.fixed_temporal_sampling = True
                        sampler.spatial_convs.train()
                        sampler.fc_spatial_offset.train()
                        sampler.fc_spatial_stride.train()
                        if cfg.S2D.PREDICT_SIGMA:
                            sampler.fc_spatial_sigma.train()
                    elif 'temporal' in cfg.S2D.TRAIN_ONLY_PARAMS:
                        assert (cfg.S2D.SPACE_TIME_SHARE_CONVS is False)
                        logging.info(
                            'Training temporal sampling only (freezing spatial sampling)...'
                        )
                        sampler.fixed_spatial_sampling = True
                        sampler.temporal_convs.train()
                        sampler.fc_temporal_offset.train()
                        sampler.fc_temporal_stride.train()
                        if cfg.S2D.PREDICT_SIGMA:
                            sampler.fc_temporal_sigma.train()
            else:
                raise NotImplementedError

        if cfg.S2D.SNET_LR_RATIO == 0.:
            self.SNet.eval()

        if cfg.S2D.USE_DNET_DROPOUT:
            getattr(self.DNet.model.base_model,
                    self.DNet.model.base_model.last_layer_name).train()
        else:
            getattr(self.DNet.model.base_model,
                    self.DNet.model.base_model.last_layer_name).eval()

        if cfg.S2D.USE_SNET_DROPOUT:
            getattr(self.SNet.model.base_model,
                    self.SNet.model.base_model.last_layer_name).train()
            self.SNet.model.ctx_dropout.train()
        else:
            getattr(self.SNet.model.base_model,
                    self.SNet.model.base_model.last_layer_name).eval()
            self.SNet.model.ctx_dropout.eval()

        if cfg.TRAIN.FREEZE_BN:
            logging.info('Freezing All BatchNorm Except The First One.')
            for name, m in self.SNet.model.named_modules():  # ignore sampler
                # for resnet, `bn1``; for mn2, `features.0`
                if name == 'bn1' or 'features.0' in name:
                    print('FBN filtered name for SNet:', name)
                    continue
                set_bn_affine_eval(m, name)
            for name, m in self.DNet.model.named_modules():  # ignore sampler
                # for resnet, `bn1``; for mn2, `features.0`
                if name == 'bn1' or 'features.0' in name:
                    print('FBN filtered name for DNet:', name)
                    continue
                if 'sampler' in name or 'ctx_feature_layer' in name or 'fusion' in name:
                    print('FBN filtered name for DNet:', name)
                    continue
                set_bn_affine_eval(m, name)

        return self

    def get_optim_policies(self):
        """
            returns the optimizable parameters (for TSM)
        """
        train_only_params = self.cfg.S2D.TRAIN_ONLY_PARAMS
        train_only_params = [
            p.lower() for p in train_only_params.replace(' ', '').split(',')
            if len(p) > 0
        ]
        snet_train_only_params = [
            p.replace('snet.', '') for p in train_only_params
            if 'dnet' not in p
        ]  # ignore dnet params, and remove snet. prefix
        dnet_train_only_params = [
            p.replace('dnet.', '') for p in train_only_params
            if 'snet' not in p
        ]  # ignore snet params, and remove dnet. prefix

        c_policies = get_optim_policies(self.SNet, self.cfg,
                                        self.SNet.model_cfg,
                                        snet_train_only_params)
        f_policies = get_optim_policies(self.DNet, self.cfg,
                                        self.DNet.model_cfg,
                                        dnet_train_only_params)
        s2d_policies = []

        # SNET
        for row in c_policies:
            row['name'] = 'snet_' + row['name']
            if 'sampler' in row['name']:
                """i.e., snet_features_convs"""
                row['lr_mult'] = row['lr_mult']
                s2d_policies.append(row)
            else:
                """SNet backbone """
                if self.cfg.S2D.SNET_LR_RATIO > 0.:
                    row['lr_mult'] = row['lr_mult'] * self.cfg.S2D.SNET_LR_RATIO
                    s2d_policies.append(row)

        # DNET
        for row in f_policies:
            row['name'] = 'dnet_' + row['name']
            s2d_policies.append(row)
        return s2d_policies

    def make_somehot_vectors(self, indics, num_class):
        """Get some-hot vectors according to the given indics. 
        Example: when indics=[[1,4,6]] and num_class=10, we have somehot=[[0,1,0,0,1,0,1,0,0,0]]

        Parameters
        ----------
        indics : (torch.Tensor)
            soft target (B,a)
        num_class : int
            number of classes
        """
        onehot = torch.nn.functional.one_hot(indics, num_class).float()
        somehot = onehot.sum(dim=1)
        return somehot

    def masked_softmax(self, x, mask, dim=-1):
        """Returns the masked softmax
        
        Parameters
        ----------
        x : torch.Tensor
            (B,A), output actions
        mask : torch.Tensor
            (B,A), some-hot vectors, e.g., [1,1,1,0,0,0,1]
        """
        max_x = torch.max(x, dim=dim,
                          keepdim=True)[0]  # -max_x is used to avoid nan
        x = torch.exp(x - max_x) * mask
        x = x / (torch.sum(x, dim=dim, keepdim=True) + 1e-6)
        return x

    def get_synopsis(self, targets, cdict):
        """Extract the synopsis information from the output of SNet

        Parameters
        ----------
        targets : torch.Tensor
            top-k vector
        cdict : dict
            intermediate output from SNet, `backbone` is the backbone feature maps,
            `gap_backbone` is the backbone feature vector.
        """
        targets = targets
        snet_features = cdict['backbone']  # (B,T,C,H,W)
        gap_snet_features = cdict['gap_backbone']

        return (targets, snet_features, gap_snet_features)

    def forward(self, x, meta):
        if self.cfg.S2D.CODEBASE == 'slowfast':
            dtype = x[0][0].dtype
        else:
            dtype = x[0].dtype

        # Get snet output
        snet_actions, snet_mid_features = self.SNet(x, meta)

        # Process snet output
        _, indices = torch.sort(snet_actions, dim=-1, descending=True)
        snet_indices = indices[:, :5]
        somehot = self.make_somehot_vectors(
            snet_indices, self.cfg.DATA.NUM_CLASS).type(dtype)
        if self.cfg.ABLATION.SNET_ACTION_TYPE == 'PROB':
            snet_targets = self.masked_softmax(snet_actions, somehot)
        else:
            snet_targets = somehot

        # Get S2D Cues
        synopsis = self.get_synopsis(snet_targets, snet_mid_features)

        # Pre-Softmax Output (B,K)
        dnet_actions, reg_term = self.DNet(x, synopsis, meta=meta)

        if self.cfg.S2D.GATHER_DNET_LOSS or self.cfg.S2D.NAIVE_DNET_LOSS:
            dnet_prob = dnet_actions

        else:
            # From a probabilistic view: P(a) = 1/Z * P(a|T) * P(T) * I(a in T)
            dnet_prob = torch.nn.functional.softmax(dnet_actions,
                                                    dim=-1)  # = P(a|T)

            dnet_prob = dnet_prob * somehot  # P(a|T) * P(T) * I(a in T),
            dnet_prob = dnet_prob / (dnet_prob.sum(dim=-1, keepdim=True) + 1e-6
                                     )  # norm,

            dnet_prob = torch.log(
                dnet_prob + 1e-6)  # get log_softmax, +1e-6 to avoid nan (log0)

        return dnet_prob, snet_actions, reg_term, dnet_actions


class SynopsisNetwork(nn.Module):

    def __init__(self, cfg):
        '''
        SNet consists of a uniform sampler and a light-weighted model. 
        '''
        super(SynopsisNetwork, self).__init__()
        self.cfg = cfg
        self.model = self.build_from_cfg(cfg)
        self.sampler = UniformSampler(cfg)
        self.runtime = False
        self.runtime_stats = 0.
        if self.runtime:
            self._starter, self._ender = torch.cuda.Event(
                enable_timing=True), torch.cuda.Event(enable_timing=True)

    def build_from_cfg(self, cfg):
        """Build Synopsis Network using the cfg

        Parameters
        ----------
        cfg : CfgNode
            cfg of S2DNet
        """
        print('Building Synopsis Network ...')

        if cfg.S2D.SNET_MODEL == 'TSM':
            from ops.models import TSN
            model_cfg = cfg.TSM.clone()
            model_cfg.merge_from_list(cfg.S2D.SNET_OPTS)
            model_cfg.IMG_FEATURE_DIM = cfg.S2D.SNET_FEATURE_DIM
            model = TSN(
                cfg.DATA.NUM_CLASS,
                cfg.S2D.SNET_FRAME_NUM,
                cfg.DATA.MODALITY,
                base_model=model_cfg.ARCH,
                consensus_type=model_cfg.CONSENSUS_TYPE,
                dropout=cfg.S2D.SNET_DROPOUT,
                img_feature_dim=model_cfg.IMG_FEATURE_DIM,
                partial_bn=model_cfg.PARTIAL_BN,
                pretrain=model_cfg.PRETRAIN,
                is_shift=model_cfg.SHIFT,
                shift_div=model_cfg.SHIFT_DIV,
                shift_place=model_cfg.SHIFT_PLACE,
                fc_lr5=not (model_cfg.TUNE_FROM and model_cfg.TRAIN.DATASET
                            in model_cfg.TUNE_FROM),
                temporal_pool=model_cfg.TEMPORAL_POOL,
                non_local=model_cfg.NON_LOCAL,
                cfg=cfg)
        # elif 'X3D' in cfg.S2D.SNET_MODEL:
        #     from slowfast.models import build_model
        #     model_cfg = cfg.SNET.clone()
        #     model_cfg.merge_from_list(cfg.S2D.SNET_OPTS)
        #     model_cfg.S2D = CfgNode()
        #     for k,v in cfg.S2D.clone().items():
        #         model_cfg.S2D[k] = v
        #     model = build_model(model_cfg, submodel=True)
        else:
            raise NotImplementedError
        model.model_name = 'snet'
        self.mount_snet_feature_convs(model)
        self.model_cfg = model_cfg
        return model

    def mount_snet_feature_convs(self, model):
        """Mount convolutions that process feature maps/vectors to SNet.

        Parameters
        ----------
        model : 
            SNet model
        """
        snet_feature_layer = sorted(
            self.cfg.S2D.SNET_FEATURE_LAYER.replace(' ', '').split(','))
        snet_feature_layer = [n for n in snet_feature_layer if n != '']
        model.snet_feature_layer = snet_feature_layer

        base_model_name = model.base_model_name
        if len(snet_feature_layer) > 1:
            snet_features_convs = nn.ModuleList()
            for i in snet_feature_layer:
                assert (i in ('layer1', 'layer2', 'layer3', 'layer4', 's2',
                              's3', 's4', 's5', 'conv_5'))
                in_dim = FEATURE_DIMS[base_model_name][i]
                out_dim = self.cfg.S2D.HIDDEN_DIM
                snet_features_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True),
                        # nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                    ))
        else:
            snet_features_convs = nn.ModuleList()
            snet_features_convs.append(nn.Identity())

        model.snet_features_convs = snet_features_convs

    def forward(self, x, meta):
        if self.runtime:
            self._starter.record()

        # uniform sampler does not require gradients.
        with torch.no_grad():
            out, _ = self.sampler(x, meta)

        out = self.model(out,
                         return_features=True,
                         backbone_no_grad=(self.cfg.S2D.SNET_LR_RATIO == 0))

        if self.runtime:
            self._ender.record()
            torch.cuda.synchronize()
            self.runtime_stats += self._starter.elapsed_time(self._ender)
        return out


class DetailNetwork(nn.Module):

    def __init__(self, cfg):
        '''
        DNet consists of an adaptive sampler and a model.
        '''
        super(DetailNetwork, self).__init__()
        self.cfg = cfg
        if cfg.S2D.DETAIL_SAMPLING:
            self.sampler = AdaptiveSampler(cfg)
        else:
            self.sampler = UniformSampler(cfg,
                                          size=cfg.S2D.DNET_INPUT_SIZE,
                                          num_frame=cfg.S2D.DNET_FRAME_NUM,
                                          use_posenc=False)
        self.model = self.build_from_cfg(cfg)

        if cfg.S2D.LATERAL_FUSION:
            in_dim = cfg.S2D.SNET_FEATURE_DIM + cfg.DATA.NUM_CLASS
            out_dim = cfg.S2D.CTX_FEATURE_DIM
            self.ctx_feature_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )

        else:
            self.ctx_feature_layer = None

        self.runtime = False
        self.runtime_splr_stats = 0.
        self.runtime_bone_stats = 0.
        self._starter = torch.cuda.Event(enable_timing=True)
        self._ender = torch.cuda.Event(enable_timing=True)

    def mount_lateral_fusion(self, model):
        fusion_layer_names = sorted(
            self.cfg.S2D.FUSION_LAYERS.replace(' ', '').split(','))
        fusion_layer_names = [n for n in fusion_layer_names if n != '']
        model.fusion_layer_names = fusion_layer_names
        base_model_name = model.base_model_name

        fusion_layers = nn.ModuleList()
        ctx_dim = self.cfg.S2D.CTX_FEATURE_DIM
        if self.cfg.S2D.FUSION_TYPE == 'concat-multilevel':
            for layer_name in fusion_layer_names:
                in_dim = FEATURE_DIMS[base_model_name][layer_name]
                fusion_layers.append(
                    nn.Sequential(nn.Linear(in_dim + ctx_dim, in_dim), ))
        elif self.cfg.S2D.FUSION_TYPE == 'concat-multilevel-3d':
            for layer_name in fusion_layer_names:
                in_dim = FEATURE_DIMS[base_model_name][layer_name]
                fusion_layers.append(
                    nn.Sequential(
                        nn.Conv3d(in_dim + ctx_dim,
                                  in_dim,
                                  kernel_size=1,
                                  padding=0,
                                  bias=False),
                        # nn.BatchNorm3d(in_dim),
                        nn.ReLU(inplace=True),
                    ))

        model.fusion_layers = fusion_layers

        if 'X3D' in self.cfg.S2D.DNET_MODEL:
            model.head.projection = nn.Linear(
                model.head.projection.in_features + ctx_dim + model.num_class,
                model.head.projection.out_features)
        else:
            model.new_fc = nn.Linear(
                model.new_fc.in_features + ctx_dim + model.num_class,
                model.num_class)

    def build_from_cfg(self, cfg):
        """Build Detail Network using the cfg

        Parameters
        ----------
        cfg : CfgNode
            cfg of S2DNet
        """
        logging.info('Building Detail Network ...')
        if cfg.S2D.DNET_MODEL == 'TSM':
            from ops.models import TSN
            model_cfg = cfg.TSM.clone()
            model_cfg.merge_from_list(cfg.S2D.DNET_OPTS)
            model_cfg.IMG_FEATURE_DIM = cfg.S2D.DNET_FEATURE_DIM
            model = TSN(
                cfg.DATA.NUM_CLASS,
                cfg.S2D.DNET_FRAME_NUM,
                cfg.DATA.MODALITY,
                base_model=model_cfg.ARCH,
                consensus_type=model_cfg.CONSENSUS_TYPE,
                dropout=cfg.S2D.DNET_DROPOUT,
                img_feature_dim=model_cfg.IMG_FEATURE_DIM,
                partial_bn=model_cfg.PARTIAL_BN,
                pretrain=model_cfg.PRETRAIN,
                is_shift=model_cfg.SHIFT,
                shift_div=model_cfg.SHIFT_DIV,
                shift_place=model_cfg.SHIFT_PLACE,
                fc_lr5=not (model_cfg.TUNE_FROM and model_cfg.TRAIN.DATASET
                            in model_cfg.TUNE_FROM),
                temporal_pool=model_cfg.TEMPORAL_POOL,
                non_local=model_cfg.NON_LOCAL,
                cfg=cfg)
        # elif 'X3D' in cfg.S2D.DNET_MODEL:
        #     from slowfast.models import build_model
        #     model_cfg = cfg.DNET.clone()
        #     model_cfg.merge_from_list(cfg.S2D.DNET_OPTS)
        #     model_cfg.S2D = CfgNode()
        #     for k,v in cfg.S2D.clone().items():
        #         model_cfg.S2D[k] = v
        #     model = build_model(model_cfg, submodel=True)
        else:
            raise NotImplementedError
        model.model_name = 'dnet'
        if cfg.S2D.LATERAL_FUSION:
            self.mount_lateral_fusion(model)
        self.model_cfg = model_cfg
        return model

    def get_ctx_feature(self, cues):
        ctx_feature = None
        if self.ctx_feature_layer is not None:
            T = self.cfg.S2D.SNET_FRAME_NUM
            targets, _, gap_snet_features = cues
            B = targets.size(0)
            ctx_feature = torch.cat([targets, gap_snet_features],
                                    dim=1)  # (B,C+A)
            ctx_feature = self.ctx_feature_layer(ctx_feature.view(B, -1))
        return ctx_feature

    def get_sample_frames(self, x, cues, meta):
        if self.cfg.S2D.DETAIL_SAMPLING:
            frames, reg_term = self.sampler(x, cues=cues, meta=meta)
        else:
            frames, reg_term = self.sampler(x, meta=meta)
        return frames, reg_term

    def forward(self, x, synopsis, meta):
        if self.runtime:
            self._starter.record()
        out, reg_term = self.get_sample_frames(x, cues=synopsis, meta=meta)
        if self.runtime:
            self._ender.record()
            torch.cuda.synchronize()
            self.runtime_splr_stats += self._starter.elapsed_time(self._ender)

        if self.runtime:
            self._starter.record()
        ctx_feature = self.get_ctx_feature(synopsis)
        topk_vector = synopsis[0]
        out = self.model(out, ctx_feature=(ctx_feature, topk_vector))
        if self.runtime:
            self._ender.record()
            torch.cuda.synchronize()
            self.runtime_bone_stats += self._starter.elapsed_time(self._ender)
        return out, reg_term
