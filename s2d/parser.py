'''
Description        : Parsing running argument for S2DNet.
'''


import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of S2DNet")

# CUSTOMIZED
parser.add_argument('dataset', type=str, help="The registered name of the dataset")
parser.add_argument('cfg_file', type=str, help="Path to the config file")

# below settings are deprecated
parser.add_argument(
        '--local_rank', 
        dest='local_rank',
        default = 0, 
        type=int
    )
parser.add_argument(
        '--gpus', 
        dest="gpus",
        # nargs='+', 
        type=int, 
        default=0
    )
parser.add_argument(
        "--exp", 
        dest="EXPERIMENTAL_FEATURES",
        help="customized experimental features",
        default="",
        type=str
    )
parser.add_argument(
        "opts",
        help="See s2d/default_config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )


