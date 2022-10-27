"""
The TSM model for S2DNet, which is modified using TSM/ops/models.py. 
Please prepare the necessary auxiliary codes for TSM (e.g., transform.py) according to  the original repo.
"""

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from torch.nn import functional as F

import logging

logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=logformat)


class TSN(nn.Module):

    def __init__(self,
                 num_class,
                 num_segments,
                 modality,
                 base_model='resnet50',
                 new_length=None,
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.8,
                 img_feature_dim=256,
                 crop_num=1,
                 partial_bn=True,
                 print_spec=True,
                 pretrain='imagenet',
                 is_shift=False,
                 shift_div=8,
                 shift_place='blockres',
                 fc_lr5=False,
                 temporal_pool=False,
                 non_local=False,
                 mean_multi_views=False,
                 num_views=6,
                 experimental_features=dict(),
                 cfg=None):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.mean_multi_views = mean_multi_views  ## for bs=1 testing
        self.num_views = num_views
        self.experimental_features = experimental_features
        self.cfg = cfg
        self.num_class = num_class
        self.model_name = 'tsm'

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if print_spec:
            logging.info(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments,
                       consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        self.snet_feature_layer = ''
        if cfg is not None:
            self.ctx_dropout = nn.Dropout(cfg.S2D.CVECTOR_DROPOUT)
        else:
            self.ctx_dropout = nn.Identity()

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model,
                        self.base_model.last_layer_name).weight, 0, std)
            constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias,
                0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        logging.info('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(
                torchvision.models,
                base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                logging.info('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model,
                                    self.num_segments,
                                    n_div=self.shift_div,
                                    place=self.shift_place,
                                    temporal_pool=self.temporal_pool)

            if self.non_local:
                logging.info('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(
                1
            )  # replace nn.AdaptiveAvgPool2d(1), to retain the spatial dim.

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain ==
                                           'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(
                            m.conv) == 8 and m.use_res_connect:
                        m.conv[0] = TemporalShift(m.conv[0],
                                                  n_segment=self.num_segments,
                                                  n_div=self.shift_div)

            if self.non_local:
                raise NotImplementedError

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            logging.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self  # consisent with the original `train` function.

    def partialBN(self, enable):
        self._enable_pbn = enable

    def lateral_fuse(self, x, ctx_feature, fusion_idx=0, mode='1D'):
        """Get the contextual features from snet_targest and snet_features

        Parameters
        ----------
        cues : list
            [0] snet_targets: torch.Tensor
                    top-5 softmax probs, (B, C_snet)
            [1] snet_features: torch.Tensor
                    features of SNet, (B*T, C)
        Returns
        -------
        torch.Tensor
            the features used for the final fc, (B*T, C)
        """
        _, C, H, W = x.size()
        B = ctx_feature.size(0)
        T = self.num_segments
        x = x.view((-1, T) + (C, H, W))  # (B*Tf,C,H,W) -> (B,Tf,C,H,W)
        ctx_dim = ctx_feature.size(1)
        if mode == '1D':
            ctx_feature = ctx_feature.view(B, ctx_dim).contiguous()
            new_x = torch.cat([x.mean([1, 3, 4]), ctx_feature], dim=-1)
            new_x = self.fusion_layers[fusion_idx](new_x).view(B, 1, -1, 1, 1)
        elif mode == '3D':
            ctx_feature = ctx_feature.view(B, 1, ctx_dim, 1,
                                           1).expand(-1, T, -1, H,
                                                     W).contiguous()
            new_x = torch.cat([x, ctx_feature],
                              dim=2).transpose(1, 2)  # -> (B,C+ctx_dim,T,H,W)
            new_x = self.fusion_layers[fusion_idx](new_x).transpose(
                1, 2).contiguous()  # -> (B,T,C,H,W)

        x = new_x
        # Follows ResNet implementation
        # x = x + torch.nn.functional.relu(new_x)

        x = x.view((-1, ) + (C, H, W))  # (B*Tf,C,H,W)
        return x

    def forward_module(self, x, module, no_grad=False, flatten=False):
        if flatten:
            x = torch.flatten(x, 1)
        if no_grad:
            with torch.no_grad():
                x = module(x)
        else:
            x = module(x)
        return x

    def get_middle_features(self, feature_dict):
        stages = self.snet_feature_layer
        assert (len(stages) > 0)

        coarsest_lv = stages[0]
        HW = feature_dict[coarsest_lv].size()[-2:]
        convs = self.snet_features_convs
        detach_splr = self.cfg.S2D.SAMPLER_STOP_GRADIENTS
        detach_feat = self.cfg.S2D.CTX_STOP_GRADIENTS

        mid_features = {}
        # get backbone features
        x = []
        for i, stage in enumerate(stages):
            _x = feature_dict[stage]
            _x = convs[i](_x)
            if i > 0:
                _x = F.interpolate(_x,
                                   size=HW,
                                   mode='bilinear',
                                   align_corners=True)
            x.append(_x)
        x = torch.cat(x, dim=1)
        x = x.view((-1, self.num_segments) + x.size()[-3:])  # (B,T,C,H,W)
        mid_features['backbone'] = x.detach() if detach_splr else x

        # get contextual feature vector
        if 'gap_backbone' in feature_dict.keys():
            mid_features['gap_backbone'] = feature_dict['gap_backbone'].detach(
            ) if detach_feat else feature_dict['gap_backbone']

        return mid_features

    def forward_base_model(self,
                           x,
                           return_features=False,
                           ctx_feature=None,
                           backbone_no_grad=False,
                           topk_vector=None):
        running_snet = return_features
        running_dnet = ctx_feature is not None

        if running_snet or running_dnet:
            assert (running_snet != running_dnet)

        mid_features = None
        if running_snet:
            """ SNet """
            feature_dict = dict()
            for name, module in self.base_model._modules.items():
                is_flatten = False
                if name == self.base_model.last_layer_name:  # dropout
                    # save the before-dropout feature vector
                    _x = torch.flatten(x, 1)
                    _x = _x.view((-1, self.num_segments) +
                                 _x.size()[-1:]).mean(dim=1)  # (B,T,C)
                    _x = self.ctx_dropout(_x)
                    feature_dict['gap_backbone'] = _x
                    is_flatten = True

                x = self.forward_module(x,
                                        module,
                                        backbone_no_grad,
                                        flatten=is_flatten)

                if name in self.snet_feature_layer:
                    feature_dict[name] = x

            mid_features = self.get_middle_features(feature_dict)

        elif running_dnet:
            """ DNet """
            if 'resnet' in self.base_model_name:
                for name, module in self.base_model._modules.items():
                    x = self.forward_module(
                        x,
                        module,
                        backbone_no_grad,
                        flatten=name is self.base_model.last_layer_name)

                    if name in self.fusion_layer_names:
                        fusion_idx = self.fusion_layer_names.index(name)
                        if self.cfg.S2D.FUSION_TYPE == 'concat-multilevel':
                            x = self.lateral_fuse(x,
                                                  ctx_feature,
                                                  fusion_idx,
                                                  mode='1D')
                        elif self.cfg.S2D.FUSION_TYPE == 'concat-multilevel-3d':
                            x = self.lateral_fuse(x,
                                                  ctx_feature,
                                                  fusion_idx,
                                                  mode='3D')
                        else:
                            raise NotImplementedError

                    # Final, always concat out features of DNet with ctx_features
                    if name == self.base_model.last_layer_name:  # after dropout
                        B, C = ctx_feature.size()
                        T = self.num_segments
                        ctx_feature = ctx_feature.view(B, 1, C).expand(
                            -1, T, -1).contiguous().view(-1,
                                                         C)  # (B,C)->(B*T,C)
                        topk_vector = topk_vector.repeat(T, 1)
                        x = torch.cat([x, ctx_feature, topk_vector], dim=-1)
            else:
                raise NotImplementedError

        else:
            """ Naive TSM """
            x = self.base_model(x)

        return x, mid_features

    def forward(self,
                input,
                ctx_feature=None,
                no_reshape=False,
                return_features=False,
                backbone_no_grad=False):
        if isinstance(ctx_feature, tuple):
            ctx_feature, topk_vector = ctx_feature
        else:
            topk_vector = None

        base_out, base_features = self.forward_base_model(
            input.view((-1, 3) + input.size()[-2:]), return_features,
            ctx_feature, backbone_no_grad, topk_vector)

        base_out = self.new_fc(base_out)
        if self.is_shift and self.temporal_pool:
            base_out = base_out.view((-1, self.num_segments // 2) +
                                     base_out.size()[1:])
        else:
            base_out = base_out.view((-1, self.num_segments) +
                                     base_out.size()[1:])
        output = self.consensus(base_out)
        if self.mean_multi_views:  # for visualization (e.g., Grad-CAM)
            output = output.view((-1, self.num_views) +
                                 output.size()[1:]).mean(
                                     1)  # require the batch size to be 1.
        if return_features:
            return [output.squeeze(1), base_features]
        else:
            return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True, experimental_features=dict()):
        if self.modality == 'RGB':
            ## exp
            if 'full_resize' in experimental_features.keys():
                if experimental_features['full_resize']:
                    if flip:
                        return torchvision.transforms.Compose([
                            GroupScale((self.input_size, self.input_size)),
                            GroupCenterCrop(self.input_size),
                            GroupRandomHorizontalFlip(is_flow=False)
                        ])
                    else:
                        return torchvision.transforms.Compose([
                            GroupScale((self.input_size, self.input_size)),
                            GroupCenterCrop(self.input_size)
                        ])

            if flip:
                return torchvision.transforms.Compose([
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False)
                ])
            else:
                logging.info('NO FLIP!!!')
                return torchvision.transforms.Compose([
                    GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])
                ])  # input_size=224
