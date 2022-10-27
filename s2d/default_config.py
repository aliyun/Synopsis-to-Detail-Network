'''
Description        : Configs for S2DNet (used with TSM or SlowFast Codebase).
'''

from fvcore.common.config import CfgNode


def _assert_in(opt, opts):
    assert(opt in opts), '{} not in options {}'.format(opt, opts)
    return opt


_C = CfgNode()

# -----------------------------------------------------------------------------
# S2D options
# -----------------------------------------------------------------------------
_C.S2D = CfgNode()
_C.S2D.ENABLE = True
_C.S2D.CODEBASE = 'TSM'

_C.S2D.SNET_MODEL = 'TSM'
_C.S2D.SNET_CFG_FILE = ''
_C.S2D.SNET_INPUT_SIZE = 112
_C.S2D.SNET_FRAME_NUM = 32
_C.S2D.SNET_FEATURE_DIM = 2048 # should be set manually
_C.S2D.SNET_LR_RATIO = 1.
_C.S2D.SNET_EARLY_DECAY_EPOCH = -1
_C.S2D.SNET_LOSS = 'CE'
_C.S2D.SNET_SMOOTH_ALPHA = 0.
_C.S2D.SNET_LOSS_WEIGHT = 0.1
_C.S2D.SNET_DROPOUT = 0.5
_C.S2D.SNET_FEATURE_LAYER = 'layer4'
_C.S2D.USE_SNET_DROPOUT = True
_C.S2D.CONTEXT_POOL = True
_C.S2D.TARGET_ENCODE = False
_C.S2D.TARGET_EMBEDDING_DIM = 256

_C.S2D.DNET_MODEL = 'TSM'
_C.S2D.DNET_CFG_FILE = ''
_C.S2D.DNET_INPUT_SIZE = 224
_C.S2D.DNET_FRAME_NUM = 16
_C.S2D.DNET_FEATURE_DIM = 2048 # should be set manually
_C.S2D.DNET_LOSS = 'CE'
_C.S2D.GATHER_DNET_LOSS = False
_C.S2D.NAIVE_DNET_LOSS = True
_C.S2D.DNET_SMOOTH_ALPHA = 0.1
_C.S2D.DNET_DROPOUT = 0.5
_C.S2D.USE_DNET_DROPOUT = True
_C.S2D.FUSION_LAYERS = 'layer2,layer3'
_C.S2D.FUSION_TYPE = 'concat-multilevel-3d'
_C.S2D.USE_PG_OFFSET = False

_C.S2D.HIDDEN_DIM = 64

_C.S2D.SPACE_SAMPLING = False
_C.S2D.SPACE_SAMPLING_SIZE = 112


_C.S2D.TIME_SAMPLING = True
_C.S2D.SAMPLING_RATE = 1
_C.S2D.DETAIL_SAMPLING_DIM_RATE = 16
_C.S2D.DEFAULT_TIME_RANGE = 1.
_C.S2D.DEFAULT_SPACE_RANGE = 1.
_C.S2D.CTX_STOP_GRADIENTS = False
_C.S2D.SAMPLER_STOP_GRADIENTS = False
_C.S2D.BN_SAMPLE_PARAM = False
_C.S2D.INPUT_VIDEO_LEN = False
_C.S2D.CVECTOR_DROPOUT = 0.5

_C.S2D.LATERAL_FUSION = True 
_C.S2D.DETAIL_SAMPLING = True 
_C.S2D.SAMPLER_LR_RATIO = 1.0
_C.S2D.SPATIAL_LR_RATIO = 1.
_C.S2D.TEMPORAL_LR_RATIO = 1.


_C.S2D.SNET_RESUME = ''
_C.S2D.DNET_RESUME = ''
_C.S2D.TRAIN_ONLY_PARAMS = ''

_C.S2D.SNET_OPTS = [] # placeholder
_C.S2D.DNET_OPTS = [] # placeholder

_C.S2D.TWOSTAGE_TRAINING = True
_C.S2D.TWOSTAGE_TRAINING_STAGE = 'WARMUP'
_C.S2D.DNET_LOSS_WEIGHT = 1.0

_C.S2D.SPACE_FOLLOW_TIME = True
_C.S2D.SPACE_TIME_SHARE_CONVS = True
_C.S2D.SPACE_PARAM_INIT_FACTOR = 0.05
_C.S2D.TIME_PARAM_INIT_FACTOR = 0.05
_C.S2D.SAMPLER_OPTIM = 'adamw'
_C.S2D.ITERS_TO_ACCUMULATE = 1

_C.S2D.TOPK_SNET_LOSS = False
_C.S2D.PREDICT_SIGMA = False
_C.S2D.DRAW_CENTERS = False # if True, no tanh, use raw outputs

_C.S2D.MANUAL_T_SIGMA = 1.
_C.S2D.MANUAL_S_SIGMA = 1.

_C.S2D.CTX_FEATURE_DIM = 256

_C.S2D.SAMPLER_FC_BIAS = False



# -----------------------------------------------------------------------------
# Coarse Net options
# -----------------------------------------------------------------------------
_C.SNET = CfgNode()  # placeholder

# -----------------------------------------------------------------------------
# Fine Net options
# -----------------------------------------------------------------------------
_C.DNET = CfgNode()  # placeholder


# -----------------------------------------------------------------------------
# Ablation options
# -----------------------------------------------------------------------------
_C.ABLATION = CfgNode()
_C.ABLATION.SNET_NUM_CLASS = 'A->5'
_C.ABLATION.SNET_ACTION_TYPE = 'SOMEHOT'
_C.ABLATION.LOCATOR_AVG_AXIS = 'C'
_C.ABLATION.LOCATOR_ATT_INPUT = 'ACTIONS'
_C.ABLATION.SPACE_STRIDE = 'exp'
_C.ABLATION.TIME_STRIDE = 'exp'
_C.ABLATION.PARAM_BN_TIMES = 1.0
_C.ABLATION.USE_3D_SPACE_CONV = True
_C.ABLATION.NEW_DNET = False
_C.ABLATION.FREEZE_SAMPLING_AUGMENTATION = False
_C.ABLATION.TEMPORAL_SAMPLE_METHOD = 'gaussian'
_C.ABLATION.SPATIAL_SAMPLE_METHOD = 'gaussian'


# -----------------------------------------------------------------------------
# DEBUG options
# -----------------------------------------------------------------------------
_C.DEBUG = CfgNode()  
_C.DEBUG.CHECK_GRAD = False  
_C.DEBUG.SAVE_CKPT_FREQ = 9999
_C.DEBUG.PRINT_SAMPLING_INFO = True



def _assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
