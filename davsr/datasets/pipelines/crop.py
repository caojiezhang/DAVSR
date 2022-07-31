# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from torch.nn.modules.utils import _pair


# @PIPELINES.register_module()
class RandomCrop:
    """random crop.

    It crops gt images with corresponding locations.
    It also supports accepting gt list.
    Required keys are "scale" and "gt",
    added or modified keys is "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        gt_is_list = isinstance(results["gt"], list)
        if not gt_is_list:
            results["gt"] = [results["gt"]]

        h_gt, w_gt, _ = results["gt"][0].shape

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_gt - self.gt_patch_size + 1)
        left = np.random.randint(w_gt - self.gt_patch_size + 1)
        # crop corresponding gt patch
        results["gt"] = [
            v[top : top + self.gt_patch_size, left : left + self.gt_patch_size, ...]
            for v in results["gt"]
        ]

        if not gt_is_list:
            results["gt"] = results["gt"][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(gt_patch_size={self.gt_patch_size})"
        return repr_str
