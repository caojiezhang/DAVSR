import io
import logging
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmedit.datasets.pipelines import blur_kernels as blur_kernels

try:
    import socket
    use_euler = True if 'eu-' in socket.gethostname() else False
except Exception:
    use_euler = False
    
if not use_euler:
    from iopath.common.file_io import g_pathmgr
    from on_device_ai.cvg.interns.projects.mmediting_standalone.mmedit_standalone.models.common import basicblock as B
    from on_device_ai.cvg.interns.projects.mmediting_standalone.mmedit_standalone.utils import utils_image as util
else:
    from mmedit_standalone.models.common import basicblock as B
    from mmedit_standalone.utils import utils_image as util

try:
    import av
    has_av = True
except ImportError:
    has_av = False


# @PIPELINES.register_module()
class UVSRDegradation:
    """Apply uvsr degradation to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def downsample1D(self, x, sf=2):
        '''s-fold downsampler

        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

        x: tensor image, NxCxWxH
        '''
        st = 0
        return x[st::sf, ...]

    def _apply_uvsr_degradation(self, imgs):
        # get kernel and blur the input

        T, C, H, W = imgs.shape             #[100, 3, 256, 256] 
        N = self.params['fuse_frames']

        assert N == 5 
        assert T > N
        t = T // N
        out_imgs = []

        padding = 'reflection'
        if padding == 'reflection':
            imgs = torch.cat((imgs[1:2, ...], imgs[0:1, ...], imgs, imgs[-2:-1, ...], imgs[-3:-2, ...]), dim=0)
            for i in range(len(imgs)):
                if i>=2 and i+3<=len(imgs):
                    out_imgs.append(imgs[i-2:i+3, ...].mean(0, keepdim=True))
        else:
            for i in range(t):
                out_imgs.append(imgs[i*N: i*N+N, ...].mean(0, keepdim=True))
        # # imgs = imgs.mean(0, keepdim=True)

        out_imgs = torch.cat(out_imgs, dim=0)

        # F.conv3d

        # downsampling
        if self.params['down']:
            sf = self.params['scale']
            out_imgs = self.downsample1D(out_imgs, sf=sf[0])
            # out_imgs = F.interpolate(out_imgs.view(-1, C, H, W), size=(H//sf[1], W//sf[2]), mode='bicubic')

            assert sf[1] == sf[2]
            down_imgs = []
            for i in range(out_imgs.shape[0]):
                down_img = util.imresize(out_imgs[i].permute(1,2,0), 1./sf[1])
                down_imgs.append(down_img)
            out_imgs = torch.stack(down_imgs, dim=0).permute(0,3,1,2)
        # noise

        return out_imgs #imgs

    def __call__(self, results):
        for key in self.keys:
            results[key] = self._apply_uvsr_degradation(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str

