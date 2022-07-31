# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import numbers
import os.path as osp
import random

import cv2
import mmcv
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

import torch.nn.functional as F


class GenerateUVSRSegmentIndices:
    """Generate frame indices for a segment. It also performs temporal
    augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """

    def __init__(self, interval_list, start_idx=0, filename_tmpl='{:08d}.png'):
        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        # key example: '000', 'calendar' (sequence name)
        clip_name = results['key']
        interval = np.random.choice(self.interval_list)

        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        # assert num_input_frames%5==0
        # start_frame_idx = 5 * random.randint(0, 100-num_input_frames//5) 
        lq_start_frame_idx = np.random.randint(
            0, self.sequence_length // 5 - num_input_frames * interval + 1)
        lq_end_frame_idx = lq_start_frame_idx + num_input_frames * interval
        lq_neighbor_list = list(range(lq_start_frame_idx, lq_end_frame_idx, interval))
        lq_neighbor_list = [v + self.start_idx for v in lq_neighbor_list]

        gt_start_frame_idx = lq_start_frame_idx * 5
        gt_end_frame_idx = gt_start_frame_idx + num_input_frames * interval * 5
        gt_neighbor_list = list(range(gt_start_frame_idx, gt_end_frame_idx, interval))
        gt_neighbor_list = [v + 5*self.start_idx for v in gt_neighbor_list]

        # add the corresponding file paths
        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_path = [
            osp.join(lq_path_root, clip_name, self.filename_tmpl.format(v))
            for v in lq_neighbor_list
        ]
        gt_path = [
            osp.join(gt_path_root, clip_name, self.filename_tmpl.format(v))
            for v in gt_neighbor_list
        ]

        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')
        return repr_str
