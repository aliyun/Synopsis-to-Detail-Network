'''
Description        : Configs for TSM codebase.
'''

from fvcore.common.config import CfgNode

def _assert_in(opt, opts):
    assert(opt in opts), '{} not in options {}'.format(opt, opts)
    return opt


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()
_C.EVALUATE = False 
_C.ROOT_LOG = 'experiments' 
_C.ROOT_MODEL = 'experiments'
_C.MODEL_SUFFIX = 'test' 
_C.PRINT_FREQ = 20 
_C.EVAL_FREQ = 1 
_C.NUM_GPUS = 2
_C.DIST = True 
_C.SHARD_ID = 0 
_C.NUM_SHARDS = 2
_C.INIT_METHOD = 'tcp://localhost:9997'
_C.AMP = True
_C.DEBUG = False

# -----------------------------------------------------------------------------
# DATA options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.NUM_CLASS = 400 
_C.DATA.IN_S_SCALE = 256 # image size
_C.DATA.IMAGE_TEMPLATE = '{:06d}.jpg'
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]
_C.DATA.MODALITY = 'RGB'
_C.DATA.DATASET_REPEAT_TIMES = 1
_C.DATA.AUGMENTATION_SCALES = [1, .875, .75, .66]
_C.DATA.RAW_INPUT_SAMPLING_RATE = 1 # e.g., if 2, use <30/2=15>fps in <30>fps videos

# -----------------------------------------------------------------------------
# TRAIN options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.DATASET = 'kinetics'
_C.TRAIN.OPTIM = 'sgd'
_C.TRAIN.EPOCHS = 50
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.BASE_LR = 0.01
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.WORKERS = 8
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 5e-4
_C.TRAIN.CLIP_GRADIENT = 20.0
_C.TRAIN.LR_DECAY_TYPES = 'step'
_C.TRAIN.LR_DECAY_STEPS = [50,100]
_C.TRAIN.TUNING = False
_C.TRAIN.FREEZE_BN = False
_C.TRAIN.SYNC_BN = True
_C.TRAIN.DIST_BN = 'reduce'
_C.TRAIN.TUNE_FROM = ''
_C.TRAIN.RESUME = '' 
_C.TRAIN.RESUME_TRAINING = True # If true, resume optimizer, epcoh and best_prec1
_C.TRAIN.RESUME_IGNORE_FLAGS = ''
_C.TRAIN.WARMUP_STEPS = 200 # deprecated
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WARMUP_LR = 1e-8


# -----------------------------------------------------------------------------
# TSM options
# -----------------------------------------------------------------------------
# configures TSM
_C.TSM = CfgNode()
_C.TSM.ARCH = 'resnet50'
_C.TSM.CONSENSUS_TYPE = 'avg'
_C.TSM.DROPOUT = 0.5
_C.TSM.IMG_FEATURE_DIM = 2048 
_C.TSM.PARTIAL_BN = False
_C.TSM.PRETRAIN = 'imagenet'
_C.TSM.SHIFT = True
_C.TSM.SHIFT_DIV = 8
_C.TSM.SHIFT_PLACE = 'blockres'
_C.TSM.TEMPORAL_POOL = False
_C.TSM.NON_LOCAL = False
_C.TSM.TUNE_FROM = False



def _assert_and_infer_cfg(cfg):
    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
