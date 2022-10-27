'''
Description        : The PyTorch Dataset for S2DNet. 
'''

import torch.utils.data as data

from PIL import Image
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import randint
from random import randrange

import logging

logformat = '%(asctime)s-{%(filename)s:%(lineno)d}-%(levelname)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=logformat)


class VideoRecord(object):

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class S2DDataSet(data.Dataset):

    def __init__(self,
                 cfg,
                 ann_file,
                 random_shift=False,
                 transform=None,
                 test_mode=False,
                 remove_missing=False):
        """The dataset function for S2DNet (modified based on Dataset in TSM codebase).

        Parameters
        ----------
        cfg : CfgNode
            config
        ann_file : str
            path to annotation file
        random_shift : bool
            use random shift or not (for training), by default False
        transform : _type_, optional
            torchvision transform function, by default None
        test_mode : bool, optional
            _description_, by default False
        remove_missing : bool, optional
            remove the videos with #frames<3, by default False
        """
        self.cfg = cfg
        self.list_file = ann_file
        self.random_shift = random_shift
        self.transform = transform
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.new_length = 1  # deprecated setting

        self.root_path = cfg.DATA.ROOT_PATH
        self.modality = cfg.DATA.MODALITY
        self.image_tmpl = cfg.DATA.IMAGE_TEMPLATE
        self.dense_sample = cfg.DATA.DENSE_SAMPLE  # deprecated, using dense sample as I3D
        self.twice_sample = cfg.DATA.TWICE_SAMPLE  # deprecated, twice sample for more validation
        self.max_fnum = cfg.DATA.MAX_FRAME_NUM

        self._parse_list()
        self.vid2idx = None
        self.all_records = self.video_list
        self.pth2vid = {}
        for i, record in enumerate(self.all_records):
            self.pth2vid[record.path] = i

        assert (self.modality == 'RGB'), 'support RGB only'

    def _load_image(self, directory, idx):
        return [
            Image.open(
                os.path.join(self.root_path, directory,
                             self.image_tmpl.format(idx))).convert('RGB')
        ]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]  # num_frames
        self.video_list = [VideoRecord(item) for item in tmp]
        raw_input_sr = self.cfg.DATA.RAW_INPUT_SAMPLING_RATE

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        logging.info('video number:%d' % (len(self.video_list)))
        times = self.cfg.DATA.DATASET_REPEAT_TIMES
        if not self.test_mode and times > 1:
            self.video_list = self.video_list * times
            logging.info(
                'Repeat the dataset for {} times at each epcoh'.format(times))
        else:
            logging.info('Repeat the dataset for 1 times at each epcoh')

    def _get_global_uniform_indices(self, record, t_stride):
        '''
        Across the video, sample a frame every $t_stride$ frame.
        '''

        assert (t_stride >= 1)
        offsets = np.array([i for i in range(0, record.num_frames, t_stride)])
        if not self.random_shift:
            offsets = np.array(
                [i for i in range(0, record.num_frames, t_stride)])
        else:
            average_duration = record.num_frames // max(
                self.cfg.S2D.SNET_FRAME_NUM, self.cfg.S2D.DNET_FRAME_NUM)
            if average_duration <= 1:
                t_stride = 1
            rand_start = 0 + randrange(average_duration + 1)
            rand_end = record.num_frames - randrange(average_duration + 1)
            offsets = np.array(
                [i for i in range(rand_start, rand_end, t_stride)])

        return offsets + 1

    def return_random(self):
        """returns a random sample
        """
        index = np.random.randint(len(self.video_list))
        return self.__getitem__(index)

    def __getitem__(self, index):
        record = self.video_list[index]
        file_name = self.image_tmpl.format(1)
        full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            logging.info('################## Not Found: {}'.format(
                os.path.join(self.root_path, record.path, file_name)))
            return self.return_random()

        segment_indices = self._get_global_uniform_indices(
            record, t_stride=self.cfg.S2D.SAMPLING_RATE)
        frames, label = self.get(record, segment_indices)
        if frames is None:
            return self.return_random()

        frames = frames.view((-1, 3) + frames.size()[1:])
        assert (len(frames.shape) == 4), 'shape should be (N,3,H,W)'
        meta = {}
        meta['in_frame_num'] = min(frames.size(0), self.max_fnum)
        meta['vname'] = record.path
        meta['vid'] = self.pth2vid[record.path]
        frames = frames[:self.max_fnum]

        return frames, label, meta

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    seg_imgs = self._load_image(record.path, p)
                except:
                    logging.info('fail reading: {}, {}'.format(record.path, p))
                    return None, None
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)

        return process_data, record.label

    def get_index_by_video_id(self, spec_vid):
        if self.vid2idx is None:
            vid2idx = {}
            for i, vobj in enumerate(self.video_list):
                vid = vobj.path
                if '/' in vid:
                    vid = vid.split('/')[-1]
                vid2idx[vid] = i
            self.vid2idx = vid2idx
        assert (spec_vid in self.vid2idx.keys()
                ), 'the given vid {} does not exists'.format(spec_vid)
        idx = self.vid2idx[spec_vid]
        return idx

    def __len__(self):
        return len(self.video_list)
