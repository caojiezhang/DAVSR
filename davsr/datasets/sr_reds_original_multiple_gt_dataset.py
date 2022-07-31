# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.datasets.base_sr_dataset import BaseSRDataset
# from mmedit.datasets.registry import DATASETS

# @DATASETS.register_module()
class SRREDSOrigMultipleGTDataset(BaseSRDataset):
    """REDS dataset (orig, 120 fps) (1 for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for REDS dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # generate keys
        keys = [f'{i:03d}' for i in range(0, 270)]

        if self.val_partition == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif self.val_partition == 'REDS8': # solve the bug on testing REDS4 on 8 gpus
            val_partition = ['000', '011', '015', '020', '000', '011', '015', '020']
        elif self.val_partition == 'official':
            val_partition = [f'{i:03d}' for i in range(240, 270)]
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["official", "REDS4", "REDS8"]')

        if self.test_mode:
            # keys = [v for v in keys if v in val_partition]
            keys = val_partition # to support REDS8 as well
        else:
            keys = [v for v in keys if v not in val_partition]

        data_infos = []
        totoal_frames = 0
        for key in keys:
            totoal_frames += 500
            if self.num_input_frames is None:
                num_input_frames = 500
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=500,  # REDS_orig has 500 frames for each clip
                    num_input_frames=num_input_frames))

        print(f'-----this train/val/test dataset has in total {totoal_frames} images')
        assert totoal_frames==133000 or totoal_frames==2000 or totoal_frames==4000, print(f'expect 133000/2000/4000, but got {totoal_frames} frames')

        return data_infos
