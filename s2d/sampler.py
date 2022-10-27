'''
Description        : The sampling functions that selects input frames for the Synopsis/Detail networks.
'''
import os
import sys

cwd = os.getcwd()
if 'slowfast' in cwd or 'SlowFast' in cwd:
    cwd = os.path.join(cwd, 'slowfast')

sys.path.append(cwd)  # parent dir

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=logformat)
from torch.nn import init
from s2d.utils import is_master_rank, reset_parameters

FEATURE_DIMS = {
    'resnet50': {
        'layer4': 2048,
        'layer3': 1024,
        'layer2': 512,
        'layer1': 256,
    },
    'mobilenetv2': {
        'features': 1280
    },
    'X3D': {
        's2': 24,
        's3': 48,
        's4': 96,
        's5': 192,
        'conv_5': 432,
    },
}


def get_feature_shape(cfg):
    snet_feature_layer = [
        i for i in cfg.S2D.SNET_FEATURE_LAYER.replace(' ', '').split(',')
        if i != ''
    ]
    snet_feature_layer = sorted(
        snet_feature_layer)  # layer1 -> layer4 -> layer_fc
    if 'ARCH' in cfg.S2D.SNET_OPTS:
        arch = cfg.S2D.SNET_OPTS[cfg.S2D.SNET_OPTS.index('ARCH') + 1]

    for i in range(0, len(cfg.S2D.SNET_OPTS), 2):
        if cfg.S2D.SNET_OPTS[i] == 'CNET.ARCH':
            arch = cfg.S2D.SNET_OPTS[i + 1]

    if 'X3D' in cfg.S2D.SNET_MODEL:
        arch = cfg.S2D.SNET_MODEL

    if len(snet_feature_layer) == 1:
        dim = FEATURE_DIMS[arch][snet_feature_layer[0]]
    else:
        dim = cfg.S2D.HIDDEN_DIM * len(snet_feature_layer)  # 0 or L*HID_DIM
    assert (
        dim > 0
    ), 'check `S2D.SNET_FEATURE_LAYER`, `S2D.USE_FC_MAP` and `S2D.USE_FC_CORRELATION`'

    if arch == 'resnet50':
        if 'layer1' in snet_feature_layer:
            space_sz = cfg.S2D.SNET_INPUT_SIZE / 4
        elif 'layer2' in snet_feature_layer:
            space_sz = cfg.S2D.SNET_INPUT_SIZE / 8
        elif 'layer3' in snet_feature_layer:
            space_sz = cfg.S2D.SNET_INPUT_SIZE / 16
        elif 'layer4' in snet_feature_layer:
            space_sz = cfg.S2D.SNET_INPUT_SIZE / 32
        elif dim == cfg.DATA.NUM_CLASS:
            space_sz = cfg.S2D.SNET_INPUT_SIZE / 32
    elif arch == 'mobilenetv2':
        # assert('features' in snet_feature_layer)
        space_sz = cfg.S2D.SNET_INPUT_SIZE / 32
    elif arch == 'X3D':
        space_sz = cfg.S2D.SNET_INPUT_SIZE / 32
    space_sz = int(math.ceil(space_sz))
    return dim, space_sz


def _make_fc(in_channels, out_channels, use_bn=False, bias=True, factor=1):
    """
        smaller the factor, smaller the init std.
    """
    fc = nn.Linear(in_channels, out_channels, bias=bias)
    if factor == 0.:
        torch.nn.init.constant_(fc.weight, 0)
    else:
        torch.nn.init.kaiming_uniform_(
            fc.weight, a=math.sqrt(6 / factor - 1))  # when 1, default
    if fc.bias is not None:
        init.constant_(fc.bias, 0)
    bn = nn.Identity()

    return nn.Sequential(fc, bn)


def _make_conv_3d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False,
                  factor=1):
    conv = nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias)
    # bn = nn.BatchNorm3d(out_planes)
    relu = nn.ReLU(inplace=True)
    torch.nn.init.kaiming_uniform_(
        conv.weight, a=math.sqrt(6 / factor - 1))  # when 1, default
    if conv.bias is not None:
        init.constant_(conv.bias, 0)
    # init.constant_(bn.weight, 1)
    # init.constant_(bn.bias, 0)
    # return nn.Sequential(conv, bn, relu)
    return nn.Sequential(conv, relu)


def ProgressiveOffset(
    x: torch.Tensor,
    k: int = 2,
) -> torch.Tensor:
    N, T, _ = x.shape
    x = x.view(N, T)
    device = x.device
    dtype = x.dtype
    offset = torch.nn.functional.pad(x.unsqueeze(1), (1, 1),
                                     mode='replicate').squeeze(1)  # (N,T+2)
    x = x + offset[:, range(0, T)]
    x = x + offset[:, range(2, T + 2)]
    x = x / 3
    return x


def clip_variable_grad_norm(x, max_norm, batchwise=False, norm_type=2.0):
    """
    Follows the pytorch implementation of `clip_grad_norm_` 
    Note, however, since the input x here has a dimension of batch size.
    Optionally, the normalization can be individual per sample by setting batchwise=True
    """
    device = x.device
    size = x.size()
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    x = x.detach()
    if batchwise:
        x = x.view(size[0], -1)
        total_norm = torch.norm(x, norm_type, dim=-1).to(device)
    else:
        total_norm = torch.norm(x, norm_type).to(device)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(
        clip_coef, max=1.0)  # if total_norm < max_norm, unchanged

    x = x * clip_coef_clamped.to(device)
    x = x.view(size)

    return x


def safe_exp_sum(x, dim, keepdim):
    res = torch.logsumexp(x, dim, keepdim=keepdim).exp()
    res2 = torch.sum(x.exp(), dim, keepdim=keepdim)
    assert (torch.allclose(res, res2)), 'res:{}, res2:{}'.format(res, res2)
    return res


class VideoSampler(nn.Module):

    def __init__(self, cfg):
        super(VideoSampler, self).__init__()
        self.cfg = cfg

    def positional_encoding(self, meta, device):
        """pos = temporal_index (from 0 to val_num)
        PE(pos, 2i) = sin(pos/10000^(2i/posdim))
        PE(pos, 2i+1) = cos(pos/10000^(2i/posdim))

        Parameters
        ----------
        meta : dict
            the meta information of the video, 
                - in_frame_num: num of valid frames
        device : str
            the required device of the output

        """
        posdim, space_sz = get_feature_shape(self.cfg)
        maxnum = self.cfg.DATA.MAX_FRAME_NUM
        embs = []
        for val_num in meta['in_frame_num']:
            pos = torch.arange(val_num).to(device).view(val_num, 1).expand(
                val_num, posdim).float()  # (val_num, 1)
            dim_indexes = 2 * (torch.arange(posdim).to(device) // 2).view(
                1, posdim).float()  # 2i+1 -> 2i, (1,posdim)
            dim_indexes = 1e4**(dim_indexes / posdim
                                )  # 10000^(2i/posdim), (1,posdim)
            dim_indexes = dim_indexes.expand(val_num,
                                             posdim)  # (val_num, posdim)
            emb = torch.zeros_like(pos)  # (val_num, posdim)
            emb[:, ::2] = torch.sin(pos[:, ::2]) / dim_indexes[:, ::2]
            emb[:, 1::2] = torch.cos(pos[:, 1::2]) / dim_indexes[:, 1::2]
            emb = F.pad(
                emb,
                (0, 0, 0, maxnum - val_num))  # (val_num -> maxnum, posdim)
            embs.append(emb)
        return torch.stack(embs)  # (N, MaxNum, posdim)


class UniformSampler(VideoSampler):

    def __init__(self, cfg, size=None, num_frame=None, use_posenc=True):
        super(UniformSampler, self).__init__(cfg)
        self.cfg = cfg
        self.size = size
        self.num_frame = num_frame
        self.use_posenc = use_posenc

    def spatial_sampling(self, x):
        """For uniform sampler, downsample the input frames to the required scale (used after temporal sampling).

        Parameters
        ----------
        x : torch.Tensor
            the input frames (after temporal sampling)
        """

        if self.size is None:
            snet_size = self.cfg.S2D.SNET_INPUT_SIZE
        else:
            snet_size = self.size

        B, D1, D2, _, _ = x.size()
        x = x.view((B, -1) + x.size()[-2:])
        x = F.interpolate(x, size=snet_size)
        x = x.view((B, D1, D2) + x.size()[-2:])

        return x

    def temporal_sampling(self, x, meta, dim=1):
        """For uniform sampler, sample uniformly across all frames.

        Parameters
        ----------
        x : torch.Tensor
            raw input frames
        meta : dict
            the meta information of the video, 
                - in_frame_num: num of valid frames
        dim : int, optional
            the dim to sample (starting from the first non-Batch dim), by default 0
        """
        if self.num_frame is None:
            fnum = self.cfg.S2D.SNET_FRAME_NUM
        else:
            fnum = self.num_frame
        tmp_x = []
        for _x, val_num in zip(x, meta['in_frame_num']):
            fid = torch.linspace(0, val_num - 1, steps=fnum,
                                 device=_x.device).long()
            tmp_x += [torch.index_select(_x, dim - 1, fid)]
        x = torch.stack(tmp_x)
        return x

    def forward(self, x, meta):
        posemb = None  # posemb is currently deprecated
        if self.cfg.S2D.CODEBASE == 'slowfast':  # SlowFast
            x = [self.temporal_sampling(xx, meta, dim=2)
                 for xx in x]  # (B,C,*T*,H,W)
            x = [self.spatial_sampling(xx) for xx in x]
        else:  # TSM
            x = self.temporal_sampling(x, meta, dim=1)  # (B,*T*,C,H,W)
            x = self.spatial_sampling(x)
        return x, posemb


class AdaptiveSampler(VideoSampler):

    def __init__(self, cfg):
        super(AdaptiveSampler, self).__init__(cfg)
        self.cfg = cfg
        Tc = cfg.S2D.SNET_FRAME_NUM
        in_channels, space_sz = get_feature_shape(cfg)
        hid_channels = cfg.S2D.HIDDEN_DIM

        self.fixed_spatial_sampling = False
        self.fixed_temporal_sampling = False
        corr_only = in_channels == cfg.DATA.NUM_CLASS
        self.jit = False
        self.t_sigma = math.log(cfg.S2D.MANUAL_T_SIGMA)
        self.s_sigma = math.log(cfg.S2D.MANUAL_S_SIGMA)

        # Encode the synopsis targets for detail sampler
        if cfg.S2D.DETAIL_SAMPLING:
            A = 0 if corr_only else cfg.DATA.NUM_CLASS
            if cfg.S2D.TARGET_ENCODE:
                self.target_encoder = nn.Sequential(
                    nn.Linear(A, cfg.S2D.TARGET_EMBEDDING_DIM),
                    nn.BatchNorm1d(cfg.S2D.TARGET_EMBEDDING_DIM),
                    nn.ReLU(inplace=True),
                )
                in_channels += cfg.S2D.TARGET_EMBEDDING_DIM
            else:
                self.target_encoder = nn.Identity()
                in_channels += A

        rate = cfg.S2D.DETAIL_SAMPLING_DIM_RATE
        pool = cfg.S2D.CONTEXT_POOL
        space_sz = math.ceil(cfg.S2D.SNET_INPUT_SIZE / (2**5))

        fc_hidden_dim = (hid_channels // rate) * Tc * space_sz * space_sz

        if cfg.S2D.SPACE_SAMPLING:
            if cfg.S2D.SPACE_TIME_SHARE_CONVS:
                self.space_time_convs = nn.Sequential(
                    _make_conv_3d(in_channels,
                                  hid_channels,
                                  kernel_size=1,
                                  padding=0,
                                  bias=True), nn.BatchNorm3d(hid_channels),
                    nn.Flatten(), _make_fc(fc_hidden_dim, 1024, bias=True),
                    nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
                    _make_fc(1024, 512, bias=True), nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True))

            else:
                self.spatial_convs = nn.Sequential(
                    _make_conv_3d(in_channels,
                                  hid_channels,
                                  kernel_size=1,
                                  padding=0),  # , bias=False
                    _make_conv_3d(hid_channels,
                                  hid_channels // rate,
                                  kernel_size=3,
                                  padding=1),  # , bias=False
                    nn.AdaptiveAvgPool3d((1, 1, 1)) if pool else nn.Identity())
                self.temporal_convs = nn.Sequential(
                    _make_conv_3d(in_channels,
                                  hid_channels,
                                  kernel_size=1,
                                  padding=0),  # , bias=False
                    _make_conv_3d(hid_channels,
                                  hid_channels // rate,
                                  kernel_size=3,
                                  padding=1),  # , bias=False
                    nn.AdaptiveAvgPool3d((1, 1, 1)) if pool else nn.Identity())

            use_bn = cfg.S2D.BN_SAMPLE_PARAM
            time_sz = cfg.S2D.DNET_FRAME_NUM
            if pool:
                space_sz = 1  # new pool seetings

            space_extra = time_extra = 0
            if cfg.S2D.INPUT_VIDEO_LEN:
                space_extra += 1
                time_extra += 1
            if cfg.S2D.SPACE_FOLLOW_TIME:
                space_extra += 2

            Tspace = cfg.S2D.DNET_FRAME_NUM

            time_fc_in_channels = 512 + time_extra
            space_fc_in_channels = 512 + space_extra

            self.fc_temporal_offset = nn.Sequential(
                _make_fc(
                    # (hid_channels//rate) * time_sz * space_sz * space_sz,
                    time_fc_in_channels,
                    1,
                    use_bn,
                    bias=cfg.S2D.SAMPLER_FC_BIAS,
                    factor=cfg.S2D.TIME_PARAM_INIT_FACTOR))
            self.fc_temporal_stride = nn.Sequential(
                _make_fc(
                    # (hid_channels//rate) * time_sz * space_sz * space_sz,
                    time_fc_in_channels,
                    1,
                    use_bn,
                    bias=cfg.S2D.SAMPLER_FC_BIAS,
                    factor=cfg.S2D.TIME_PARAM_INIT_FACTOR))
            self.fc_spatial_offset = nn.Sequential(
                _make_fc(
                    # (hid_channels//rate) * time_sz * space_sz * space_sz + extra,
                    space_fc_in_channels,
                    Tspace * 2,
                    use_bn,
                    bias=cfg.S2D.SAMPLER_FC_BIAS,
                    factor=cfg.S2D.SPACE_PARAM_INIT_FACTOR))
            self.fc_spatial_stride = nn.Sequential(
                _make_fc(
                    # (hid_channels//rate) * time_sz * space_sz * space_sz + extra,
                    space_fc_in_channels,
                    Tspace * 2,
                    use_bn,
                    bias=cfg.S2D.SAMPLER_FC_BIAS,
                    factor=cfg.S2D.SPACE_PARAM_INIT_FACTOR))

            self.convs_temporal = nn.Identity()
            self.pool_temporal = nn.Identity()

            if cfg.S2D.PREDICT_SIGMA:
                self.fc_temporal_sigma = nn.Sequential(
                    _make_fc(
                        # (hid_channels//rate) * time_sz * space_sz * space_sz,
                        space_fc_in_channels,
                        1,
                        use_bn,
                        bias=cfg.S2D.SAMPLER_FC_BIAS,
                        factor=cfg.S2D.TIME_PARAM_INIT_FACTOR))
                self.fc_spatial_sigma = nn.Sequential(
                    _make_fc(
                        # (hid_channels//rate) * time_sz * space_sz * space_sz + extra,
                        time_fc_in_channels,
                        Tspace * 2,
                        use_bn,
                        bias=cfg.S2D.SAMPLER_FC_BIAS,
                        factor=cfg.S2D.SPACE_PARAM_INIT_FACTOR))

        else:
            assert (cfg.S2D.TIME_SAMPLING is True)
            """ temporal only """
            self.convs = nn.AdaptiveAvgPool2d(1)
            self.convs_temporal = nn.Sequential(
                nn.Conv1d(in_channels, hid_channels, kernel_size=1,
                          bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(hid_channels,
                          hid_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(hid_channels,
                          hid_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.ReLU(inplace=True),
            )
            self.pool_temporal = lambda x: torch.mean(x, dim=1)
            self.fc_temporal = nn.Sequential(nn.Linear(Tc, 2), )

    def set_jit(self, mode=False):
        self.jit = mode

    def bounded_range(self, x, low_bound, up_bound):
        """Project the initial delta_t into the bounded range.

        Parameters
        ----------
        x : torch.Tensor
            the iniitial delta_t
        low_bound : float
            the lower bound of the range.
        up_bound : float
            the upper bound of the range
        """
        assert (up_bound >
                low_bound), 'Check the upper bound: {}'.format(up_bound)
        return torch.sigmoid(x) * (up_bound - low_bound) + low_bound

    def sample(self, x, params, meta, mode=None):
        """sample the frames according to the constant time prior (center / stride) from 3D Local CNNs.

        Note, (H,W) denotes the size of input images and (h,w) denotes the size of output images.

        Parameters
        ----------
        x : torch.Tensor
            all input frames of the video, (B,C,T,H,W) or (B,T,C,H,W)
        params : list of torch.Tensor
            spatial params (B,T,4), temporal params (B,2)
        meta : dict
            the meta information of the video, 
                - in_frame_num: num of valid frames
        """
        B = len(x)
        T = self.cfg.DATA.MAX_FRAME_NUM
        # Get (B,T,C,H,W) input
        if x[0].size()[1] == 3:
            x = x
            tranpose_flag = False
        elif x[0].size()[0] == 3:  # (B,C,T,H,W) -> (B,T,C,H,W)
            x = [_x.transpose(0, 1) for _x in x]
            tranpose_flag = True
        else:
            raise NotImplementedError

        cfg = self.cfg
        device = x[0].device
        dtype = x[0].dtype
        tmp_x = []
        mean_dt = []
        mean_stride_t = []
        mean_stride_x = []
        mean_stride_y = []
        mean_dx = []
        mean_dy = []

        in_h, in_w = x[0].size(2), x[0].size(3)
        if self.jit:
            in_h, in_w = in_h.item(), in_w.item()
        anchor_x = (in_w - 1) / 2  # int(round(in_w / 2.0))
        anchor_y = (in_h - 1) / 2  # int(round(in_h / 2.0))

        atten_out_t = cfg.S2D.DNET_FRAME_NUM
        spatial_sample = True

        notanh_centers = self.cfg.S2D.DRAW_CENTERS
        if len(params) == 6 or len(params) == 9:  # gamma -> confidence * input
            """ parse space-time parameters """
            batch_dx, batch_dy, batch_delta_x, batch_delta_y, batch_dt, batch_delta_t = params[:
                                                                                               6]
            if cfg.S2D.PREDICT_SIGMA:
                assert (len(params) == 9)
                batch_sigma_x, batch_sigma_y, batch_sigma_t = params[6:]
            else:
                batch_sigma_x = batch_dx.new_zeros(
                    batch_dx.size()) + self.s_sigma
                batch_sigma_y = batch_dx.new_zeros(
                    batch_dy.size()) + self.s_sigma
                batch_sigma_t = batch_dt.new_zeros(
                    batch_dt.size()) + self.t_sigma

            atten_out_w = cfg.S2D.SPACE_SAMPLING_SIZE
            atten_out_h = cfg.S2D.SPACE_SAMPLING_SIZE

            spatial_sample = True
            interpolated_dx = []
            interpolated_dy = []
            interpolated_delta_x = []
            interpolated_delta_y = []
            interpolated_sigma_x = []
            interpolated_sigma_y = []

            for i, _x in enumerate(x):
                val_num = meta['in_frame_num'][i]

                _x = _x[:val_num]  # (val_num,3,H,W)
                # centers
                if notanh_centers:
                    dx = batch_dx[i:i + 1]
                    dy = batch_dy[i:i + 1]
                else:
                    dx = torch.tanh(batch_dx[i:i + 1])
                    dy = torch.tanh(batch_dy[i:i + 1])

                # strides
                if self.cfg.ABLATION.SPACE_STRIDE == 'exp':
                    # self.cfg.S2D.DEFAULT_SPACE_RANGE controls the init (relative control, 1 cover the inpu spatially, 0.5 the half)
                    delta_x = self.cfg.S2D.DEFAULT_SPACE_RANGE * (
                        in_w / (atten_out_w - 1)) * torch.exp(
                            batch_delta_x[i:i + 1]).view(1, -1, 1)
                    delta_y = self.cfg.S2D.DEFAULT_SPACE_RANGE * (
                        in_h / (atten_out_h - 1)) * torch.exp(
                            batch_delta_y[i:i + 1]).view(1, -1, 1)

                    # self.cfg.S2D.DEFAULT_SPACE_RANGE is the init stride (absolute control)
                    # delta_x = self.cfg.S2D.DEFAULT_SPACE_RANGE * torch.exp(batch_delta_x[i:i+1]).view(1, -1 ,1)
                    # delta_y = self.cfg.S2D.DEFAULT_SPACE_RANGE * torch.exp(batch_delta_y[i:i+1]).view(1, -1 ,1)
                else:
                    assert (2 - in_w / atten_out_w > 0)
                    delta_x = self.bounded_range(batch_delta_x[i:i + 1],
                                                 low_bound=0.5,
                                                 up_bound=1.5)
                    delta_y = self.bounded_range(batch_delta_y[i:i + 1],
                                                 low_bound=0.5,
                                                 up_bound=1.5)

                # sigma
                sigma_x = torch.exp(batch_sigma_x[i:i + 1]).view(-1, 1, 1)
                sigma_y = torch.exp(batch_sigma_y[i:i + 1]).view(-1, 1, 1)

                interpolated_dx.append(dx)
                interpolated_dy.append(dy)
                interpolated_delta_x.append(delta_x)
                interpolated_delta_y.append(delta_y)
                interpolated_sigma_x.append(sigma_x)
                interpolated_sigma_y.append(sigma_y)

        elif len(params) == 2:
            """ parse temporal parameters """
            x = x.permute(0, 2, 3, 4,
                          1).contiguous()  # (B,T,C,H,W) -> (B,C,H,W,T)
            batch_dt, batch_delta_t = params
            atten_out_w = in_h  # h=H
            atten_out_h = in_w  # w=W

        else:
            raise NotImplementedError

        # if not cfg.EVALUATE:
        if self.fixed_spatial_sampling:
            interpolated_dx = [p.new_zeros(p.size()) for p in interpolated_dx]
            interpolated_dy = [p.new_zeros(p.size()) for p in interpolated_dy]
            interpolated_delta_x = [
                p.new_ones(p.size()) for p in interpolated_delta_x
            ]  # after exp, should be ones
            interpolated_delta_y = [
                p.new_ones(p.size()) for p in interpolated_delta_y
            ]  # after exp, should be ones
        if self.fixed_temporal_sampling:
            batch_dt = [p.new_zeros(p.size()) for p in batch_dt]
            batch_delta_t = [p.new_zeros(p.size()) for p in batch_delta_t
                             ]  # before exp, should be zeros.
        """ temporal sampling """
        # (B,T,C,H,W) -> (B,C,H,W,T)
        log_mu_t = []
        x = [_x.permute(1, 2, 3, 0).contiguous() for _x in x]
        for i, _x in enumerate(x):
            dt = batch_dt[i]
            delta_t = batch_delta_t[i]
            sigma_t = batch_sigma_t[i]

            val_num = meta['in_frame_num'][i]
            in_t = val_num

            _x = _x.unsqueeze(0)  # (C,H,W,T) -> (B=1,C,H,W,T)
            _x = _x[:, :, :, :, :val_num]  # take the valid frames as input
            """ get temporal steps """
            anchor_t = (in_t - 1) / 2.0  # int(round(in_t / 2.0))

            # time center
            if notanh_centers:
                dt = dt
            else:
                dt = torch.tanh(dt)
            if not cfg.EVALUATE:
                mean_dt += [dt.detach()]
            dt = dt * (in_t - 1) / 2.0 + anchor_t

            # time stride
            if self.cfg.ABLATION.TIME_STRIDE == 'exp':
                if not cfg.EVALUATE:
                    mean_stride_t += [torch.exp(delta_t).detach()]
                delta_t = self.cfg.S2D.DEFAULT_TIME_RANGE * (
                    in_t / (atten_out_t - 1)) * torch.exp(delta_t)
            else:
                if not cfg.EVALUATE:
                    mean_stride_t += [torch.tanh(delta_t).detach()]
                delta_t = self.bounded_range(delta_t,
                                             low_bound=0.,
                                             up_bound=in_t / atten_out_t)

            # time sigma
            sigma_t = torch.exp(sigma_t).view(1, 1, 1).view(1)

            grid_t_i = torch.arange(0, atten_out_t).type(dtype).to(device)
            mu_t = dt + (grid_t_i - (atten_out_t - 1) / 2.0) * delta_t
            mu_t = mu_t.view(1, -1)
            """ temporal sampling """
            if cfg.ABLATION.TEMPORAL_SAMPLE_METHOD == 'gaussian':
                if not cfg.EVALUATE:
                    log_mu_t.append(mu_t / (in_t - 1) * 2 - 1)

                t = torch.arange(0, in_t).view(
                    in_t, 1).type(dtype).to(device).detach()  # (T,1)
                eps_tensor_t = 1e-7 * torch.ones(atten_out_t).to(
                    device).detach().view(1, -1)  # (1,t)

                Ft = torch.exp(-1 * torch.pow(t - mu_t, 2) /
                               (2 * sigma_t)).float()  # (T,t)
                Ft = Ft / torch.max(torch.sum(Ft, 0, keepdim=True),
                                    eps_tensor_t)  # (T,t)

                _x = torch.matmul(_x.view(1, -1, in_t), Ft).view(
                    _x.size()[:-1] + (-1, ))  # (1,CHW,T) * (T,t) = (1,CHW,t)

            else:
                mu_t = mu_t / (in_t - 1) * 2 - 1  # (1,t) \in [-1, 1]
                if not cfg.EVALUATE:
                    log_mu_t.append(mu_t)

                mu_t = mu_t.view(1, atten_out_t, 1, 1)
                mu_s = torch.zeros(1, atten_out_t, 1,
                                   1).type(dtype).to(device).detach()
                grid = torch.cat([mu_s, mu_t], -1)  # (1,t,1,2)

                _, C, H, W, T = _x.size()
                _x = F.grid_sample(_x.view(1, -1, in_t, 1),
                                   grid,
                                   align_corners=True)
                _x = _x.view(1, C, H, W, atten_out_t)  # (B,C,h,w,t)

            tmp_x.append(_x)

        x = torch.cat(tmp_x, dim=0).permute(
            0, 4, 1, 2, 3).contiguous()  # (B,C,h,w,t) -> (B,t,C,h,w)

        if spatial_sample:
            new_x = []
            for i, _x in enumerate(x):
                dx = interpolated_dx[i]
                dy = interpolated_dy[i]
                if cfg.S2D.USE_PG_OFFSET:
                    dx = ProgressiveOffset(dx.view(1, -1, 1))
                    dy = ProgressiveOffset(dy.view(1, -1, 1))
                dx = dx.view(-1, 1)
                dy = dy.view(-1, 1)
                if not cfg.EVALUATE:
                    mean_dx += [dx.detach().mean()]
                    mean_dy += [dy.detach().mean()]
                dx = dx * (in_w - 1) / 2.0 + anchor_x  # (val_num,1)
                dy = dy * (in_h - 1) / 2.0 + anchor_y  # (val_num,1)

                delta_x = interpolated_delta_x[i].view(-1, 1)
                delta_y = interpolated_delta_y[i].view(-1, 1)
                sigma_x = interpolated_sigma_x[i].view(-1, 1, 1)
                sigma_y = interpolated_sigma_y[i].view(-1, 1, 1)

                # log only the unpad (|t|<=1) delta
                if not cfg.EVALUATE:
                    log_delta_x = delta_x.view(-1).detach()
                    log_delta_y = delta_y.view(-1).detach()
                    log_mu_t[i] = log_mu_t[i].view(-1)
                    mean_stride_x += [log_delta_x[log_mu_t[i]**2 < 1].mean()]
                    mean_stride_y += [log_delta_y[log_mu_t[i]**2 < 1].mean()]

                grid_x_i = torch.arange(0, atten_out_w).view(
                    1, -1).type(dtype).to(device).detach()
                grid_y_i = torch.arange(0, atten_out_h).view(
                    1, -1).type(dtype).to(device).detach()
                mu_x = dx + (grid_x_i - (atten_out_w - 1) / 2.0) * delta_x
                mu_y = dy + (grid_y_i - (atten_out_h - 1) / 2.0) * delta_y

                if cfg.ABLATION.SPATIAL_SAMPLE_METHOD == 'gaussian':
                    """ get gaussian dist for each step  """
                    mu_x = mu_x.view(-1, atten_out_w, 1)  # (t,w,1)
                    mu_y = mu_y.view(-1, atten_out_h, 1)  # (t,h,1)
                    a = torch.arange(0, in_w).view(
                        1, 1, -1).type(dtype).to(device).detach()
                    b = torch.arange(0, in_h).view(
                        1, 1, -1).type(dtype).to(device).detach()
                    eps_tensor_w = 1e-7 * torch.ones(in_w).to(device).detach()
                    eps_tensor_h = 1e-7 * torch.ones(in_h).to(device).detach()

                    Fx = torch.exp(
                        -1 * torch.pow(a - mu_x, 2) /
                        (2 * sigma_x)).float()  # gaussian over x, (t,w,W)
                    Fy = torch.exp(
                        -1 * torch.pow(b - mu_y, 2) /
                        (2 * sigma_y)).float()  # gaussian over y, (t,h,H)
                    Fx = Fx / torch.max(torch.sum(Fx, 2, keepdim=True),
                                        eps_tensor_w)  # (t,w,W)
                    Fy = Fy / torch.max(torch.sum(Fy, 2, keepdim=True),
                                        eps_tensor_h)  # (t,h,H)
                    """ spatial sampling """
                    Fyv = Fy.view(Fy.size(0), 1, Fy.size(1),
                                  Fy.size(2))  # (t,1,h,H)
                    Fxv = Fx.view(Fx.size(0), 1, Fx.size(1),
                                  Fx.size(2))  # (t,1,w,W)
                    Fxt = torch.transpose(Fxv, 2, 3)  # (t,1,W,w)
                    _x = _x.view(atten_out_t, -1, in_h, in_w)  # (t,C,H,W)
                    _x = torch.matmul(
                        Fyv, torch.matmul(_x, Fxt)
                    )  # (t,1,h, <<H>>)  * ((t, C, <<H>>, <W>) * (t,1,<W>,w)) = (val_num,C,h,w)
                    _x = _x.view(atten_out_t, -1, atten_out_h, atten_out_w)
                else:
                    mu_x = mu_x.view(-1, atten_out_w)  # (t,w)
                    mu_y = mu_y.view(-1, atten_out_h)  # (t,h)
                    mu_x = mu_x / (in_w - 1) * 2 - 1  # (t,w) \in [-1, 1]
                    mu_y = mu_y / (in_h - 1) * 2 - 1  # (t,h) \in [-1, 1]
                    mu_x = mu_x[:, None, :, None].repeat(1, atten_out_h, 1,
                                                         1)  # (B,h,w,1)
                    mu_y = mu_y[:, :, None, None].repeat(1, 1, atten_out_w,
                                                         1)  # (B,h,w,1)
                    grid = torch.cat([mu_x, mu_y], -1)  # should be (x,y)
                    _x = F.grid_sample(
                        _x.view(atten_out_t, -1, in_h, in_w),
                        grid,
                        align_corners=False)  # same as spatial interpolate
                    _x = _x.view(atten_out_t, -1, atten_out_h, atten_out_w)

                new_x.append(_x)

            x = new_x
        x = torch.stack(x, dim=0)  # (B,t,C,h,w)

        if tranpose_flag:
            x = x.transpose(1, 2).contiguous()  # (B,t,C,h,w)  -> (B,C,t,h,w)

        reg_term = x.new_zeros((x.size(0), 4))

        if is_master_rank(
                cfg) and not cfg.EVALUATE and cfg.DEBUG.PRINT_SAMPLING_INFO:
            idx = 0
            logging.info(
                '[sample] dx/y/t: {:.2f}/{:.2f}/{:.2f}, deltax/y/t: {:.2f}/{:.2f}/{:.2f}'
                .format(
                    mean_dx[idx].squeeze(),
                    mean_dy[idx].squeeze(),
                    mean_dt[idx].squeeze(),
                    mean_stride_x[idx].squeeze(),
                    mean_stride_y[idx].squeeze(),
                    mean_stride_t[idx].squeeze(),
                ))

            mean_dx = torch.stack(mean_dx).mean().item()
            mean_dy = torch.stack(mean_dy).mean().item()
            mean_dt = torch.stack(mean_dt).mean().item()
            mean_stride_x = torch.stack(mean_stride_x).mean().item()
            mean_stride_y = torch.stack(mean_stride_y).mean().item()
            mean_stride_t = torch.stack(mean_stride_t).mean().item()
            logging.info(
                '[batch] dx/y/t: {:.2f}/{:.2f}/{:.2f}; deltax/y/t: {:.2f}/{:.2f}/{:.2f}'
                .format(mean_dx, mean_dy, mean_dt, mean_stride_x,
                        mean_stride_y, mean_stride_t))

        return x, reg_term

    def get_sampling_params(self, cues, meta=None, dim=1):
        """Get Time or Space-time Location Parameters using snet_targest and snet_features.

        Parameters
        ----------
        cues : list
            [0] snet_targets: torch.Tensor
                    top-5 softmax probs, (B, A)
            [1] snet_features: torch.Tensor
                    features of CNet, (B,T,C,H,W)
        dim : specify the dim index of T.
        """
        cfg = self.cfg
        targets, snet_feature_map, _ = cues
        assert (dim == 1 or dim == 2)
        if dim == 1:
            B, T, _, H, W = snet_feature_map.size()
            snet_feature_map = snet_feature_map.permute(
                0, 2, 1, 3, 4).contiguous()  # (B,T,C,H,W) -> (B,C,T,H,W)
        else:
            B, _, T, H, W = snet_feature_map.size()
        """ Embed snet target into snet features """
        targets = self.target_encoder(targets)
        targets = targets.view(B, -1, 1, 1, 1).expand(-1, -1, T, H,
                                                      W)  # (B,A,T,H,W)
        snet_feature_map = torch.cat([snet_feature_map, targets],
                                     dim=1)  # (B,C+A,T,H,W)
        """ Prepare features """
        params = []
        if cfg.S2D.SPACE_TIME_SHARE_CONVS:
            tx = sx = self.space_time_convs(snet_feature_map)  #  (B,C,t,h,w)
        else:
            sx = self.spatial_convs(snet_feature_map)  #  (B,C,t,h,w)
            tx = self.temporal_convs(snet_feature_map)  #  (B,C,t,h,w)

        # TODO: check if necessary
        # sx = torch.transpose(sx, 1, 2).contiguous() # (B,C,T,H,W) -> (B,T,C,H,W)
        """ Get temporal parameters """
        tx = self.pool_temporal(tx)  # (B,C,4,4,4)
        tx = tx.view(B, -1)

        if cfg.S2D.INPUT_VIDEO_LEN:  # put in fc
            vlen = meta['in_frame_num'] / cfg.DATA.MAX_FRAME_NUM
            vlen = vlen.view(-1, 1)
            tx = torch.concat([tx, vlen], dim=1)
            sx = torch.concat([sx, vlen], dim=1)

        temporal_offset = self.fc_temporal_offset(tx).view(B,
                                                           -1).float()  # (B,1)
        temporal_stride = self.fc_temporal_stride(tx).view(B,
                                                           -1).float()  # (B,1)
        """ Get spatial parameters """
        sx = sx.view(B, -1)  # Videowise
        if cfg.S2D.SPACE_FOLLOW_TIME:
            sx = torch.cat([sx, temporal_offset, temporal_stride], dim=-1)
        spatial_offset = self.fc_spatial_offset(sx).view(B, -1, 2).float()
        spatial_stride = self.fc_spatial_stride(sx).view(B, -1, 2).float()

        # dx, dy, delta_x, delta_y, dt, delta_t
        params += torch.split(spatial_offset, split_size_or_sections=1, dim=2)
        params += torch.split(spatial_stride, split_size_or_sections=1, dim=2)
        params += torch.split(temporal_offset, 1, 1)
        params += torch.split(temporal_stride, 1, 1)

        if cfg.S2D.PREDICT_SIGMA:
            spatial_sigma = self.fc_spatial_sigma(sx).view(
                B, -1, 2).float()  # (B,1,2)
            temporal_sigma = self.fc_temporal_sigma(tx).view(
                B, -1).float()  # (B,1)
            params += torch.split(spatial_sigma,
                                  split_size_or_sections=1,
                                  dim=2)
            params += torch.split(temporal_sigma,
                                  split_size_or_sections=1,
                                  dim=1)

        return params

    def forward(self, x, cues, meta):
        """Forward the adaptive sampling

        Parameters
        ----------
        x : torch.Tensor
            all input frames (possibly with padding),  (B,C,T,H,W) or (B,T,C,H,W)
        cues : tuple
            [0] snet_targets: torch.Tensor
                    top-5 softmax probs, (B, A)
            [1] snet_features: torch.Tensor
                    features of CNet, (B,T,C,H,W)
        meta : dict
            the meta information of the video, 
                - in_frame_num: num of valid frames
        """
        cfg = self.cfg
        with torch.cuda.amp.autocast(enabled=False):
            # with torch.cuda.amp.autocast(enabled=cfg.AMP):
            # cues = [c.float() for c in cues]
            # x = [_x.float() for _x in x]
            if cfg.S2D.CODEBASE == 'slowfast':
                """SlowFast"""
                params = self.get_sampling_params(cues, meta, dim=2)
                x[0], reg_term = self.sample(x[0], params, meta)
            else:
                """TSM"""
                params = self.get_sampling_params(cues, meta, dim=1)
                x, reg_term = self.sample(x, params, meta)

        return x, reg_term
