import torch
import torch.nn as nn
import torchvision
import torch.fft
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import warnings

from mmedit.models.common import (PixelShufflePack, flow_warp)
from mmedit.models.backbones.sr_backbones.basicvsr_net import (ResidualBlocksWithInputConv, SPyNet)

from mmedit.utils import get_root_logger

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d

from davsr.models.common import basicblock as B
import davsr.models.backbones.sr_backbones.slomo as slomo



"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
class BasicVSRPP(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 img_channels=3,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 vsr_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.img_channels = img_channels

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        if isinstance(vsr_pretrained, str):
            load_net = torch.load(vsr_pretrained)
            for k, v in load_net['state_dict'].items():
                if k.startswith('generator.'):
                    k = k.replace('generator.', '')
                    load_net[k] = v
                    load_net.pop(k)
            self.load_state_dict(load_net, strict=False)
        elif vsr_pretrained is not None:
            raise TypeError('[vsr_pretrained] should be str or None, '
                            f'but got {type(vsr_pretrained)}.')

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(img_channels, mid_channels, 5)     
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(img_channels, mid_channels, 3, 2, 1),       
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                'Deformable alignment module is not added. '
                'Probably your CUDA is not configured correctly. DCN can only '
                'be used with CUDA enabled. Alignment is skipped now.')
            


    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1).permute(0,2,1,3,4)

    def forward(self, lqs_ab):    #TODO
    # def forward(self, lqs):         #TODO
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        lqs_ab = lqs_ab.permute(0,2,1,3,4).contiguous()

        if lqs_ab.shape[2] == 4:
            lqs = lqs_ab[:,:,:-1,:,:] #TODO
        else:
            lqs = lqs_ab
        
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                if lqs_ab.shape[2] == 4:
                    feat = self.feat_extract(lqs_ab[:, i, :, :, :]).cpu()  # TODO
                else:
                    feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()   
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            if lqs_ab.shape[2] == 4:
                feats_ = self.feat_extract(lqs_ab.view(n*t, -1, h, w))     # TODO
            else:
                feats_ = self.feat_extract(lqs.view(n*t, -1, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


def splits3D(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxTxWxH
        sf: 3x1 split factor

    Returns:
        b: NxCxWxHx(sf0*sf1*sf2)
    '''
    
    b = torch.stack(torch.chunk(a, sf[0], dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf[1], dim=3), dim=5)
    b = torch.cat(torch.chunk(b, sf[2], dim=4), dim=5)

    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxTxCxHxW
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)   # [1, 1, 100, 1, 1]
    otf[:, :, :psf.shape[2], ...].copy_(psf)                # [1, 1, 100, 1, 1]
    # for axis, axis_size in enumerate(psf.shape[2:]):
    otf = torch.roll(otf, -int(psf.shape[2]/2), dims=2)     # [1, 1, 100, 1, 1]
    otf = torch.fft.fftn(otf, dim=(2))                      # [1, 1, 100, 1, 1]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def ps2ot(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = ps2ot(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)               # [1, 1, 100, 256, 256]
    otf[:, :, :psf.shape[2], :psf.shape[3], :psf.shape[4]].copy_(psf)   # [1, 1, 100, 256, 256]
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)         # [1, 1, 100, 256, 256]
    otf = torch.fft.fftn(otf, dim=(-3, -2, -1))                         # [1, 1, 100, 256, 256]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample3D(x, sf=(5,4,4)):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    b, c, t, h, w = x.shape
    z = torch.zeros((b, c, t*sf[0], h*sf[1], w*sf[2])).type_as(x)
    z[:, :, st::sf[0], st::sf[1], st::sf[2]].copy_(x)                  # 
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample3D(x, sf=4):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[:, :, :, st::sf, st::sf]


def compute_flow(lqs, spynet_pretrained, cpu_cache):
    """Compute optical flow using SPyNet for feature alignment.

    Note that if the input is an mirror-extended sequence, 'flows_forward'
    is not needed, since it is equal to 'flows_backward.flip(1)'.

    Args:
        lqs (tensor): Input low quality (LQ) sequence with
            shape (n, t, c, h, w).

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = lqs.size()
    lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
    lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
    
    spynet = SPyNet(pretrained=spynet_pretrained)

    flows_backward = spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
    flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

    if cpu_cache:
        flows_backward = flows_backward.cpu()
        flows_forward = flows_forward.cpu()

    return flows_forward, flows_backward

"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""
class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        b, c, t, h, w = x.shape                     # [B, 4, 5, 256, 256]
        x = x.permute(0,2,1,3,4).view(-1, c, w, h)  # [B*5, 4, 256, 256]

        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]
        x = x.view(b, t, -1, h, w).permute(0,2,1,3,4)

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""
class DataNet3D(nn.Module):
    def __init__(self):
        super(DataNet3D, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        
        FR = FBFy + torch.fft.fftn(alpha * x, dim=(2,3,4))              # [1, 3, 100, 256, 256]
        x1 = FB.mul(FR)                                                 # [1, 3, 100, 256, 256]
        if sf == (1, 1, 1):
            FBR = splits3D(x1, sf).squeeze(-1)
            invW = splits3D(F2B, sf).squeeze(-1)
        else:
            FBR = torch.mean(splits3D(x1, sf), dim=-1, keepdim=False)       # [1, 3, 20, 256, 256]
            invW = torch.mean(splits3D(F2B, sf), dim=-1, keepdim=False)     # [1, 1, 20, 1, 1]
        invWBR = FBR.div(invW + alpha)                                  # [1, 3, 20, 256, 256]
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf[0], sf[1], sf[2])      # [1, 3, 100, 256, 256]
        FX = (FR - FCBinvWBR) / alpha                                   # [1, 3, 100, 256, 256]
        Xest = torch.real(torch.fft.ifftn(FX, dim=(2,3,4)))             # [1, 3, 100, 256, 256]

        return Xest

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""
class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())
        
    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main network
# --------------------------------------------
"""
class DAVSRNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='convtranspose',
                 img_channels=3, mid_channels=64, num_blocks=7, max_residue_magnitude=10, is_low_res_input=True,
                 spynet_pretrained=None, vsr_pretrained=None, cpu_cache_length=100,
                 interpolation_mode='nearest', sigma_max=0, noise_level=10, sf=(2,4,4), fix_ab=0,
                 slomo_pretrained=None, pre_denoise_iters=0, use_cuda=True,
                 ):
        super(DAVSRNet, self).__init__()
        self.use_cuda = use_cuda
        self.d = DataNet3D()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)

        if fix_ab == 0:
            self.h = HyPaNet(in_nc=3, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

        self.pre_denoise_iters = pre_denoise_iters
        if self.pre_denoise_iters:
            self.pre_vsr = BasicVSRPP(img_channels=img_channels, mid_channels=mid_channels, num_blocks=num_blocks, max_residue_magnitude=max_residue_magnitude,
                 is_low_res_input=is_low_res_input, spynet_pretrained=spynet_pretrained, vsr_pretrained=vsr_pretrained, cpu_cache_length=cpu_cache_length)

        self.vsr = BasicVSRPP(img_channels=img_channels, mid_channels=mid_channels, num_blocks=num_blocks, max_residue_magnitude=max_residue_magnitude,
                 is_low_res_input=is_low_res_input, spynet_pretrained=spynet_pretrained, vsr_pretrained=vsr_pretrained, cpu_cache_length=cpu_cache_length)

        # self.pixel_unshuffle = nn.PixelUnshuffle(3)
        self.interpolation_mode = interpolation_mode
        self.cpu_cache_length = cpu_cache_length

        if interpolation_mode == 'flow':
            self.spynet_pretrained = spynet_pretrained
            # optical flow
            self.spynet = SPyNet(pretrained=spynet_pretrained)
            # check if the sequence is augmented by flipping
            self.is_mirror_extended = False

        self.sigma_max = sigma_max
        self.noise_level = noise_level

        self.img_channels = img_channels

        self.sf = sf
        self.fix_ab = fix_ab

        # data and slomo
        mean = [0.429, 0.431, 0.397]
        mea0 = [-m for m in mean]
        std = [1] * 3
        self.trans_forward = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        self.trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std)])
        self.slomo_pretrained = slomo_pretrained

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def interpolate_batch(self, frame0, frame1, factor, flow, interp, back_warp):

        # frame0 = torch.stack(frames[:-1])
        # frame1 = torch.stack(frames[1:])

        i0 = frame0
        i1 = frame1
        ix = torch.cat([i0, i1], dim=1)

        flow_out = flow(ix)
        f01 = flow_out[:, :2, :, :]
        f10 = flow_out[:, 2:, :, :]

        frame_buffer = []
        for i in range(1, factor):
            t = i / factor
            temp = -t * (1 - t)
            co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

            ft0 = co_eff[0] * f01 + co_eff[1] * f10
            ft1 = co_eff[2] * f01 + co_eff[3] * f10

            gi0ft0 = back_warp(i0, ft0)
            gi1ft1 = back_warp(i1, ft1)

            iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
            io = interp(iy)

            ft0f = io[:, :2, :, :] + ft0
            ft1f = io[:, 2:4, :, :] + ft1
            vt0 = F.sigmoid(io[:, 4:5, :, :])
            vt1 = 1 - vt0

            gi0ft0f = back_warp(i0, ft0f)
            gi1ft1f = back_warp(i1, ft1f)

            co_eff = [1 - t, t]

            ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
                (co_eff[0] * vt0 + co_eff[1] * vt1)

            frame_buffer.append(ft_p)

        return frame_buffer

    def forward(self, x, k=None, sf=(2,4,4), sigma=None):
        '''
        x: tensor, NxCxTxWxH
        k: tensor, Nx(1,3)Txwxh
        sf: integer, 3
        sigma: tensor, Nx1x1x1x1
        '''

        # reset sf by config
        sf = self.sf
        # initialization & pre-calculation

        b, t, c, w, h = x.shape  # [B, 20, 3, 64, 64]
        x = x.permute(0,2,1,3,4) # [B, 3, 20, 64, 64]
        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        # fix k, sf, sigma
        # ker_x2 = [[-2.73790210e-05,  1.06823636e-05,  5.27729389e-06, -1.59196134e-05,  3.84176519e-06, -1.46503317e-05, 1.89987732e-05, -6.14984383e-05,  2.56804487e-04, 9.66024891e-05, -1.87971722e-03,  1.55689311e-03, 1.57579291e-03, -1.89221010e-03,  9.33783085e-05, 2.59155029e-04, -6.32125302e-05,  2.13026888e-05, -1.11838026e-05, -6.86106659e-06,  9.16862609e-06, -2.52397899e-06, -9.50806225e-06,  1.64881512e-05, -6.11949554e-06], [-6.64541767e-06,  4.26588031e-06, -1.97189735e-07, -1.48234303e-05,  6.46434682e-06, -1.54692370e-05, 7.61442038e-07, -4.50876469e-05,  1.22050318e-04, 1.99884162e-04, -9.60699341e-04,  6.62715873e-04, 6.60090998e-04, -9.55613272e-04,  1.95418004e-04, 1.33476497e-04, -5.95480087e-05,  1.89000802e-05, -9.83723112e-06, -6.05016794e-06, -6.36695131e-06, -6.42165060e-06, -7.52769574e-06,  1.38459909e-05, -1.25416855e-05], [ 4.37294875e-06,  1.75654241e-05,  1.62465603e-05, 6.92083631e-07,  2.05218093e-05,  1.36240842e-05, 2.45285737e-05, -2.52698683e-05,  1.85233963e-04, 1.70230720e-04, -1.21328747e-03,  9.38735087e-04, 9.30643640e-04, -1.22007180e-03,  1.62485914e-04, 1.93002052e-04, -3.93937444e-05,  1.35458222e-05, -5.80504457e-06,  2.73095270e-06,  5.93858658e-06, 4.88662295e-07, -1.80260145e-06,  6.17076830e-06, -9.67090909e-06], [-1.25609495e-05,  7.43713281e-06, -8.56773477e-06, -1.28780985e-05,  5.46749561e-06, -5.64485754e-06, 2.53236976e-06, -4.36493465e-05,  1.51698536e-04, 1.60707568e-04, -1.19257171e-03,  9.25026427e-04, 9.19952057e-04, -1.18824514e-03,  1.67263759e-04, 1.60417156e-04, -4.67360915e-05,  1.41139217e-05, 3.29341765e-06,  1.61315995e-06,  1.38434871e-05, -4.36995197e-06,  1.20590848e-05,  1.80157494e-05, 1.58937655e-05], [-1.93978904e-05,  9.51796210e-06,  8.38011510e-06, -1.30840599e-05, -3.63981417e-06, -8.69802898e-06, 1.38014821e-05, -5.47838608e-05,  1.48947380e-04, 1.71991662e-04, -1.17365574e-03,  8.39259010e-04, 8.36949388e-04, -1.18260039e-03,  1.78835297e-04, 1.64763231e-04, -6.20811770e-05,  8.27202621e-06, -1.01489641e-05, -9.44849944e-06, -1.62161189e-06, 2.41922544e-06, -3.91319418e-06,  1.13924978e-06, -9.98393807e-06], [-1.47085229e-05, -3.03750744e-06, -9.17256602e-06, -2.39395922e-05,  1.05040353e-05, -2.85425576e-05, -1.03774403e-06, -5.76227139e-05,  1.64697019e-04, 1.65235149e-04, -1.39667070e-03,  1.09480391e-03, 1.09168433e-03, -1.40404829e-03,  1.68683328e-04, 1.65955556e-04, -6.85455525e-05,  2.61878722e-06, -2.77685740e-05, -1.56555707e-05, -9.09770642e-06, -1.37821280e-05, -1.53134060e-05, -5.49930019e-06, -3.39435210e-05], [ 2.01187086e-05,  2.94445435e-05,  2.05816832e-05, 1.60791078e-05,  2.71592580e-05,  2.19250815e-05, 3.74306073e-05, -1.65956772e-05,  1.91983141e-04, 2.53421051e-04, -1.31093955e-03,  9.65936750e-04, 9.63562168e-04, -1.31377252e-03,  2.53428996e-04, 2.03979085e-04, -1.70401727e-05,  3.16669611e-05, 5.55467386e-06,  2.61693949e-05,  1.81295327e-05, 1.44122550e-05,  2.00669274e-05,  1.37796560e-05, 4.12298978e-05], [-7.25966238e-05, -4.20080869e-05, -5.50271361e-05, -5.87633695e-05, -4.38200914e-05, -7.48037346e-05, -4.36040245e-05, -1.07893939e-04,  1.40624950e-04, 1.66314217e-04, -1.73082622e-03,  1.31388137e-03, 1.31216680e-03, -1.74232968e-03,  1.78936069e-04, 1.38253308e-04, -1.11566253e-04, -3.36937264e-05, -6.16883044e-05, -5.26943040e-05, -5.10655154e-05, -4.91020219e-05, -5.19694659e-05, -4.14920469e-05, -6.91315581e-05], [ 3.03838431e-04,  1.33412934e-04,  2.01676958e-04, 1.63335993e-04,  1.74875298e-04,  1.73708817e-04, 1.98297974e-04,  1.40646516e-04,  4.88762511e-04, 8.93966411e-04, -2.75404286e-03, -3.54247028e-03, -3.55785154e-03, -2.74608820e-03,  8.92681419e-04, 5.13300591e-04,  1.36379240e-04,  1.98606635e-04, 1.68241691e-04,  1.78030925e-04,  1.75683395e-04, 1.62718759e-04,  2.12299507e-04,  1.29061737e-04, 3.18056293e-04], [ 7.64203505e-05,  2.43523522e-04,  1.76654852e-04, 1.77222435e-04,  2.37513668e-04,  1.89328784e-04, 2.75978935e-04,  1.96968802e-04,  9.22847888e-04, 1.86817558e-03, -5.69051830e-03, -1.26665998e-02, -1.26821557e-02, -5.68657368e-03,  1.87156000e-03, 9.17577127e-04,  1.78599279e-04,  2.70087476e-04, 1.69917083e-04,  2.22638177e-04,  1.44115169e-04, 1.97965608e-04,  1.31968307e-04,  2.41999514e-04, 5.71660166e-05], [-2.34287302e-03, -1.04016997e-03, -1.43838418e-03, -1.34511874e-03, -1.32291438e-03, -1.54945324e-03, -1.44432322e-03, -1.85500889e-03, -2.80551519e-03, -5.68942167e-03,  7.03457836e-03,  4.70049195e-02, 4.70014736e-02,  7.03807268e-03, -5.69001725e-03, -2.79533397e-03, -1.83562422e-03, -1.42671249e-03, -1.54074037e-03, -1.24190864e-03, -1.40941236e-03, -1.13229372e-03, -1.47237605e-03, -9.09873866e-04, -2.27709534e-03], [ 1.99382403e-03,  7.29038671e-04,  1.08967174e-03, 1.02556858e-03,  9.63960309e-04,  1.20824052e-03, 1.03630999e-03,  1.41976378e-03, -3.52574373e-03, -1.25294188e-02,  4.66264375e-02,  1.94717303e-01, 1.94697693e-01,  4.65921983e-02, -1.25411032e-02, -3.53065669e-03,  1.42801669e-03,  1.01704185e-03, 1.22513424e-03,  8.92544398e-04,  1.12497131e-03, 7.93116167e-04,  1.20094954e-03,  5.81475091e-04, 1.93557504e-03], [ 1.97561947e-03,  7.42905831e-04,  1.09767390e-03, 1.04799995e-03,  9.73290997e-04,  1.22036855e-03, 1.04602973e-03,  1.43459777e-03, -3.52052227e-03, -1.25503857e-02,  4.65960838e-02,  1.94719896e-01, 1.94746971e-01,  4.66157384e-02, -1.25539312e-02, -3.52564966e-03,  1.41251297e-03,  1.00189028e-03, 1.21047674e-03,  8.78418563e-04,  1.10861985e-03, 7.71719438e-04,  1.19178079e-03,  5.62349567e-04, 1.95396389e-03], [-2.34582485e-03, -1.04657898e-03, -1.45219720e-03, -1.35150494e-03, -1.32756925e-03, -1.54559675e-03, -1.45416951e-03, -1.85191282e-03, -2.82116211e-03, -5.68750128e-03,  7.02426303e-03,  4.69261184e-02, 4.69413251e-02,  7.02217175e-03, -5.67274448e-03, -2.80293613e-03, -1.84660731e-03, -1.41967926e-03, -1.53858599e-03, -1.24887063e-03, -1.40101265e-03, -1.12363952e-03, -1.46833737e-03, -9.09203256e-04, -2.26110546e-03], [ 7.14002672e-05,  2.32391540e-04,  1.60823925e-04, 1.75606838e-04,  2.19653230e-04,  1.77893657e-04, 2.68715492e-04,  1.86267018e-04,  9.31657152e-04, 1.86216237e-03, -5.67332422e-03, -1.26436725e-02, -1.26355393e-02, -5.67269931e-03,  1.85519119e-03, 9.39068617e-04,  1.89856815e-04,  2.79743341e-04, 1.77371185e-04,  2.46513053e-04,  1.53065310e-04, 2.11547798e-04,  1.30771063e-04,  2.52344849e-04, 5.29337631e-05], [ 3.27722955e-04,  1.51425033e-04,  2.07831021e-04, 1.65778009e-04,  1.79980329e-04,  1.77673617e-04, 2.03578253e-04,  1.49035724e-04,  4.92425286e-04, 9.11799551e-04, -2.72671529e-03, -3.56522761e-03, -3.55933281e-03, -2.73191906e-03,  8.98807426e-04, 5.06086682e-04,  1.30560176e-04,  1.89463841e-04, 1.71241642e-04,  1.69718216e-04,  1.71104839e-04, 1.52038410e-04,  2.06082390e-04,  1.16630028e-04, 3.10478412e-04], [-7.80986375e-05, -4.17460324e-05, -6.05467321e-05, -4.64987279e-05, -4.52517343e-05, -6.30418945e-05, -4.51321685e-05, -1.17250675e-04,  1.30635482e-04, 1.61290867e-04, -1.73326826e-03,  1.31162116e-03, 1.31093233e-03, -1.73161877e-03,  1.54900437e-04, 1.35589115e-04, -1.13583876e-04, -4.39200012e-05, -7.22785844e-05, -4.37078379e-05, -6.27767804e-05, -5.26952863e-05, -4.92708459e-05, -5.12603474e-05, -6.14264864e-05], [ 4.43111785e-05,  1.42509307e-05,  2.46002819e-05, 1.29074569e-05,  2.48342567e-05,  8.13482893e-06, 4.30920591e-05, -2.35917560e-05,  1.98117879e-04, 2.54554121e-04, -1.27058919e-03,  9.25375498e-04, 9.36906319e-04, -1.28346705e-03,  2.56848056e-04, 1.98574446e-04, -1.80717343e-05,  3.08902381e-05, 1.99238602e-05,  3.01405616e-05,  2.11545030e-05, 2.41093057e-05,  3.21055595e-05,  1.56252936e-05, 3.85099884e-05], [-3.10455507e-05, -1.77404581e-05, -1.79895778e-05, -1.91718591e-05,  3.97473656e-07, -2.88564133e-05, 4.14428132e-06, -6.95225535e-05,  1.73122884e-04, 1.48673658e-04, -1.39362202e-03,  1.09192391e-03, 1.10284693e-03, -1.39653275e-03,  1.57899995e-04, 1.69453153e-04, -6.12561489e-05, -4.22620951e-06, -1.65424171e-05,  8.22425682e-06, -1.86974830e-05, -1.40873099e-05,  8.56863892e-07, -1.58086077e-05, -9.74401610e-06], [-1.04868486e-05,  4.74151966e-06, -4.29840838e-06, -7.96445022e-07, -1.12011930e-05, -5.29243152e-06, 1.20977165e-05, -4.83263721e-05,  1.44151898e-04, 1.98559675e-04, -1.10873976e-03,  7.71494757e-04, 7.61959760e-04, -1.09891128e-03,  1.85769532e-04, 1.41629615e-04, -5.38070308e-05,  1.37137192e-06, -1.46556695e-05, -3.14711838e-06, -1.82525600e-05, -1.23930334e-07, -1.39305075e-06, -1.19962042e-05, 5.34274704e-06], [ 2.19623544e-05,  7.97118173e-06,  6.10645293e-06, 9.08991751e-06,  1.12109910e-05, -2.60121965e-06, 2.29621073e-05, -5.22511837e-05,  1.71522857e-04, 1.38477757e-04, -1.21770578e-03,  9.72478942e-04, 9.78251919e-04, -1.23212696e-03,  1.41478173e-04, 1.60976939e-04, -4.39173491e-05, -7.70615134e-06, 1.85248393e-06,  9.33729552e-06, -1.71828633e-05, -5.42726502e-06,  1.87042224e-05, -2.08984929e-05, 6.55476060e-06], [ 4.07135694e-08, -9.48505345e-08, -1.02016315e-06, -1.05977097e-06, -6.81535721e-06, -3.35334425e-06, 7.80172650e-06, -3.78480436e-05,  1.45965474e-04, 1.82368560e-04, -9.65867890e-04,  6.91024354e-04, 6.86893705e-04, -9.60888108e-04,  1.92766471e-04, 1.49390893e-04, -2.69174780e-05,  1.96130077e-05, 6.61751164e-06,  1.82956337e-05,  1.15052962e-05, 1.05052095e-05,  1.77299862e-05,  9.33916090e-06, 2.48851429e-05], [-4.13689168e-06, -2.18234368e-06, -1.55320995e-05, 6.22986590e-06, -3.24416715e-06, -3.16861042e-06, 1.68840452e-05, -5.13368723e-05,  1.79821043e-04, 1.22134268e-04, -1.25209708e-03,  9.98256728e-04, 9.88062937e-04, -1.24467665e-03,  1.12416834e-04, 1.86583216e-04, -5.51166995e-05, -2.56689282e-06, -5.56072018e-06,  5.86551778e-06, -1.92668613e-05, -3.17886224e-06,  3.06903280e-06, -1.27824151e-05, 2.52837731e-06], [-1.34809753e-07,  2.12675289e-07, -6.90108607e-07, 3.61151501e-06, -8.79038089e-06,  3.87692467e-07, 1.09410630e-05, -4.60121228e-05,  1.11860347e-04, 2.05623306e-04, -8.08959536e-04,  5.15313644e-04, 5.04500233e-04, -8.02553899e-04,  2.02182171e-04, 1.10343681e-04, -5.16045075e-05,  4.26646557e-06, -8.29305918e-06, -1.30085414e-06, -1.33882286e-05, 1.21123173e-06, -5.61368597e-06, -1.45425856e-05, 2.87523221e-06], [ 8.59723968e-06, -1.34293132e-05, -7.37832488e-06, 8.99252518e-06, -7.41926942e-06, -1.39037675e-05, 2.19359372e-05, -6.31832881e-05,  2.54434126e-04, 7.58602764e-05, -1.81955611e-03,  1.53282168e-03, 1.52461079e-03, -1.82736886e-03,  9.12981704e-05, 2.58950051e-04, -6.42658924e-05,  1.12216312e-05, -6.36591312e-06,  1.21175499e-05, -1.72667096e-05, 3.95190136e-06,  1.27823423e-05, -1.68311435e-05, 7.88476518e-06]]
        # ker_x3 = [[ 7.73586123e-07,  1.33932519e-06, -2.39053725e-06, 9.79036395e-06,  2.35812222e-05, -1.14350732e-04, 3.05547437e-04, -3.73053073e-04,  3.07764887e-04, -4.71562409e-04,  4.93091997e-04, -3.62898660e-04, 4.94741602e-04, -4.73646534e-04,  3.07975191e-04, -3.73508199e-04,  3.04881280e-04, -1.13323731e-04, 2.42157821e-05,  9.84435155e-06, -2.30754404e-06, 2.22664448e-06, -1.65437632e-06,  8.18472870e-07, 2.45909746e-07], [ 1.01575972e-06, -1.12414205e-06, -6.20949493e-07, -2.40288205e-06,  2.17867455e-05, -8.83573011e-05, 1.84916134e-04, -2.17579858e-04,  2.06320241e-04, -2.69804907e-04,  2.66449584e-04, -2.33610524e-04, 2.66376475e-04, -2.68765842e-04,  2.04857977e-04, -2.17666922e-04,  1.85438766e-04, -8.75637925e-05, 2.20179663e-05, -2.00324757e-06, -1.12013186e-06, -2.06786331e-06,  9.51426273e-07, -1.62904519e-06, 1.78427683e-06], [-2.62154026e-06, -7.72342787e-07, -4.27504165e-06, 5.70990915e-06,  1.89232578e-05, -9.91508859e-05, 2.46995362e-04, -2.98776169e-04,  2.71959230e-04, -3.84676125e-04,  4.10213106e-04, -3.15336860e-04, 4.12497204e-04, -3.87928420e-04,  2.74699240e-04, -2.99256382e-04,  2.45230447e-04, -9.82557031e-05, 1.97066984e-05,  6.15638828e-06, -6.17096521e-06, 3.34451693e-06, -4.28681460e-06,  8.99140389e-07, -2.40627173e-06], [ 1.22733663e-05, -1.87014246e-06,  4.37162453e-06, -3.99446662e-06,  2.99976073e-05, -1.04995641e-04, 2.31087644e-04, -2.72958307e-04,  2.51288613e-04, -3.47367924e-04,  3.33103555e-04, -3.00152315e-04, 3.30611976e-04, -3.43422929e-04,  2.46328971e-04, -2.70706805e-04,  2.31586208e-04, -1.05960797e-04, 3.01183572e-05, -3.68035035e-06,  3.94292783e-06, -2.97687484e-06,  4.28538897e-06, -4.26551105e-06, 1.22236052e-05], [ 2.86300383e-05,  2.37514996e-05,  2.80381082e-05, 3.27619964e-05,  5.88069488e-05, -7.85106167e-05, 3.10062431e-04, -2.82626454e-04,  3.39034887e-04, -3.58388934e-04,  5.01400209e-04, -2.69659824e-04, 5.03070827e-04, -3.60517937e-04,  3.44558241e-04, -2.86057650e-04,  3.10014613e-04, -7.57913949e-05, 5.49999313e-05,  3.71156311e-05,  2.14043939e-05, 3.18219318e-05,  2.14861484e-05,  2.49304903e-05, 2.59912413e-05], [-1.40700795e-04, -9.72163252e-05, -1.27717431e-04, -1.13129936e-04, -1.08860877e-04, -2.57204316e-04, 1.34641479e-04, -4.95884509e-04,  1.27578372e-04, -6.11344818e-04,  1.98440699e-04, -5.74693200e-04, 1.96748952e-04, -6.09628041e-04,  1.24216182e-04, -4.96743072e-04,  1.41684039e-04, -2.65856972e-04, -9.68919630e-05, -1.24721453e-04, -1.08652544e-04, -1.17701413e-04, -1.03253136e-04, -9.24142078e-05, -1.29175634e-04], [ 3.85800289e-04,  2.03825213e-04,  3.07322392e-04, 2.53335835e-04,  3.58981517e-04,  1.77340684e-04, 7.86906923e-04,  3.26401205e-04,  7.22619065e-04, -1.36179593e-03, -2.23621493e-03, -3.95923853e-03, -2.23626196e-03, -1.36144133e-03,  7.18969328e-04, 3.30562878e-04,  7.71628285e-04,  1.95078901e-04, 3.32321069e-04,  2.79362081e-04,  2.56227941e-04, 2.60384812e-04,  2.53066130e-04,  1.86994395e-04, 3.51356721e-04], [-4.73769207e-04, -2.33258092e-04, -3.77762277e-04, -2.86581344e-04, -3.60405364e-04, -5.03059360e-04, 2.50381127e-04, -2.29448575e-04, -1.13196882e-04, -3.74968746e-03, -6.49978081e-03, -9.23100114e-03, -6.49539800e-03, -3.75397876e-03, -1.06004642e-04, -2.38465844e-04,  2.72636418e-04, -5.29674115e-04, -3.19406332e-04, -3.28660157e-04, -3.06325703e-04, -3.10250674e-04, -3.04734654e-04, -2.13331892e-04, -4.32794506e-04], [ 3.88197834e-04,  2.19845926e-04,  3.52853240e-04, 2.58877699e-04,  4.15245013e-04,  1.48964173e-04, 7.65419332e-04, -7.14470953e-05,  8.65264039e-04, -1.14203867e-04,  1.14363700e-03,  4.31302033e-05, 1.13981473e-03, -1.06787455e-04,  8.56478524e-04, -5.83722103e-05,  7.39235373e-04,  1.81700845e-04, 3.69317655e-04,  3.06807342e-04,  2.84461130e-04, 2.83372297e-04,  2.81689252e-04,  2.19640890e-04, 3.47409456e-04], [-6.02925487e-04, -2.81579647e-04, -4.93707252e-04, -3.58797464e-04, -4.59881121e-04, -6.26074732e-04, -1.44437014e-03, -3.76903708e-03, -1.57047849e-04, 1.11523727e-02,  2.86425650e-02,  3.58470716e-02, 2.86472477e-02,  1.11421337e-02, -1.45396465e-04, -3.78893036e-03, -1.41038350e-03, -6.65644824e-04, -4.02943289e-04, -4.21799603e-04, -3.97907104e-04, -3.90411034e-04, -4.04845166e-04, -2.66264309e-04, -5.54843689e-04], [ 6.31662144e-04,  2.75643077e-04,  5.28678240e-04, 3.37078207e-04,  6.15859462e-04,  2.03650270e-04, -2.13506282e-03, -6.49629533e-03,  1.20457809e-03, 2.86230110e-02,  6.88361153e-02,  8.64541978e-02, 6.88305348e-02,  2.86331624e-02,  1.19340443e-03, -6.47953479e-03, -2.16651964e-03,  2.44070718e-04, 5.54519938e-04,  4.06137900e-04,  4.28100087e-04, 3.88634682e-04,  4.29641019e-04,  2.67805328e-04, 5.78577165e-04], [-4.62130265e-04, -2.45000265e-04, -4.11871384e-04, -3.09606083e-04, -3.58146382e-04, -5.97337552e-04, -4.02041338e-03, -9.26845521e-03,  1.79700764e-05, 3.58212888e-02,  8.64953026e-02,  1.09942488e-01, 8.64987075e-02,  3.58137004e-02,  2.58177242e-05, -9.28357244e-03, -3.99168627e-03, -6.32158422e-04, -3.04591173e-04, -3.68571258e-04, -3.31326330e-04, -3.26783484e-04, -3.37559264e-04, -2.40583613e-04, -4.17708448e-04], [ 6.33009884e-04,  2.78673018e-04,  5.26285847e-04, 3.43997177e-04,  6.11266529e-04,  2.11587016e-04, -2.13785213e-03, -6.49307389e-03,  1.20559731e-03, 2.86251660e-02,  6.88412860e-02,  8.64545479e-02, 6.88409656e-02,  2.86278483e-02,  1.19921402e-03, -6.48034969e-03, -2.16350053e-03,  2.44547060e-04, 5.58410597e-04,  4.05871542e-04,  4.30363900e-04, 3.89701891e-04,  4.30429762e-04,  2.68348609e-04, 5.80603431e-04], [-6.06549380e-04, -2.86588387e-04, -4.89534228e-04, -3.66436783e-04, -4.55034926e-04, -6.36240060e-04, -1.43676903e-03, -3.78049258e-03, -1.51803091e-04, 1.11443121e-02,  2.86387168e-02,  3.58421691e-02, 2.86419597e-02,  1.11422352e-02, -1.48783336e-04, -3.78936599e-03, -1.41375232e-03, -6.67363463e-04, -4.05276689e-04, -4.22466081e-04, -4.01897239e-04, -3.89116292e-04, -4.07856802e-04, -2.65206821e-04, -5.57466061e-04], [ 3.92793998e-04,  2.24152667e-04,  3.49184469e-04, 2.68756441e-04,  4.09066270e-04,  1.61027652e-04, 7.61489617e-04, -6.37862904e-05,  8.66561488e-04, -1.08711196e-04,  1.15095812e-03,  4.48649844e-05, 1.14943588e-03, -1.09408640e-04,  8.61374312e-04, -5.59954387e-05,  7.40644173e-04,  1.85718542e-04, 3.70614551e-04,  3.08581395e-04,  2.87246803e-04, 2.84363574e-04,  2.84186361e-04,  2.19990863e-04, 3.49390379e-04], [-4.80366172e-04, -2.39006928e-04, -3.77682940e-04, -2.96749029e-04, -3.56771197e-04, -5.19048946e-04, 2.58212385e-04, -2.46004638e-04, -1.09008382e-04, -3.76636186e-03, -6.49638893e-03, -9.24573652e-03, -6.49554143e-03, -3.76570667e-03, -1.04802581e-04, -2.52255239e-04,  2.78305088e-04, -5.40071924e-04, -3.17208818e-04, -3.35283199e-04, -3.05405731e-04, -3.15561629e-04, -3.05919995e-04, -2.16329980e-04, -4.36319067e-04], [ 3.94938426e-04,  2.07452482e-04,  3.10778967e-04, 2.57483887e-04,  3.61386454e-04,  1.88252612e-04, 7.83601252e-04,  3.37584177e-04,  7.22565223e-04, -1.35390402e-03, -2.23093317e-03, -3.94992158e-03, -2.23193527e-03, -1.35327189e-03,  7.16941315e-04, 3.42859305e-04,  7.64294353e-04,  2.08217520e-04, 3.24593653e-04,  2.90951575e-04,  2.50624260e-04, 2.68159230e-04,  2.51206919e-04,  1.90851890e-04, 3.54179589e-04], [-1.49945670e-04, -9.73654824e-05, -1.31701716e-04, -1.13132759e-04, -1.11450616e-04, -2.64589733e-04, 1.38353935e-04, -5.04754600e-04,  1.30518863e-04, -6.23321917e-04,  2.04058699e-04, -5.87413961e-04, 2.02795316e-04, -6.21051411e-04,  1.32315530e-04, -5.09961450e-04,  1.54728608e-04, -2.82699009e-04, -8.33007580e-05, -1.38151649e-04, -9.89753244e-05, -1.27401014e-04, -9.62494596e-05, -9.75810035e-05, -1.26193816e-04], [ 3.45936533e-05,  2.19280500e-05,  3.08828094e-05, 2.94661186e-05,  6.34338867e-05, -7.81994022e-05, 3.09922500e-04, -2.81242974e-04,  3.38974467e-04, -3.57498589e-04,  5.01307775e-04, -2.66815681e-04, 5.01874252e-04, -3.59111727e-04,  3.38589918e-04, -2.77340529e-04,  2.98689760e-04, -6.39044010e-05, 4.18648997e-05,  5.00047863e-05,  8.01988699e-06, 4.37178023e-05,  1.11548115e-05,  3.09344869e-05, 2.20287875e-05], [ 1.05386225e-05, -7.79207525e-08,  1.72495970e-06, 2.06526852e-06,  2.46244053e-05, -1.01437035e-04, 2.34468258e-04, -2.77603423e-04,  2.56136380e-04, -3.56610079e-04,  3.43501510e-04, -3.07194452e-04, 3.44498258e-04, -3.54362128e-04,  2.54704850e-04, -2.80948792e-04,  2.43690927e-04, -1.16024145e-04, 4.31226435e-05, -1.49618663e-05,  1.77516176e-05, -1.29836517e-05,  1.46969514e-05, -9.66317475e-06, 1.76882095e-05], [-9.42907604e-07, -2.64067876e-06, -2.84443877e-06, 1.21680830e-06,  2.02335868e-05, -9.51931361e-05, 2.21917144e-04, -2.66198418e-04,  2.48715252e-04, -3.45665438e-04,  3.69719433e-04, -2.86301074e-04, 3.71196802e-04, -3.49378010e-04,  2.50180019e-04, -2.65725917e-04,  2.15554188e-04, -8.29772325e-05, 6.92903131e-06,  1.66658774e-05, -1.76682352e-05, 1.07042870e-05, -1.12996504e-05,  4.88040951e-06, -6.18648255e-06], [ 1.00828061e-06,  6.43523094e-07, -1.78580808e-06, 1.19257106e-06,  2.32857856e-05, -9.65210565e-05, 2.19917871e-04, -2.65196664e-04,  2.36731590e-04, -3.34055949e-04,  3.32502794e-04, -2.74072256e-04, 3.30156938e-04, -3.29998526e-04,  2.33974046e-04, -2.65375478e-04,  2.26075033e-04, -1.05710998e-04, 3.37439269e-05, -1.00948137e-05,  9.93309186e-06, -8.23455775e-06,  6.23833830e-06, -3.88943226e-06, 4.85228111e-06], [-2.07215109e-07, -1.15308512e-06, -1.65905692e-06, 2.28422823e-06,  1.92554271e-05, -8.60758810e-05, 2.09876351e-04, -2.52476690e-04,  2.37960077e-04, -3.34824377e-04,  3.54365329e-04, -2.82222609e-04, 3.56762612e-04, -3.36049095e-04,  2.39593937e-04, -2.52102618e-04,  2.06660436e-04, -8.19053967e-05, 1.32207842e-05,  9.68719996e-06, -8.92831849e-06, 5.50940149e-06, -5.22098253e-06,  1.83236739e-06, -1.61806099e-06], [-6.17320438e-07,  6.67945812e-07, -2.30952787e-06, 7.50241099e-07,  1.69660161e-05, -7.34230998e-05, 1.54137248e-04, -1.79041439e-04,  1.77888331e-04, -2.25018099e-04,  2.27690674e-04, -1.97698217e-04, 2.25470358e-04, -2.22315095e-04,  1.75687950e-04, -1.79721348e-04,  1.58108640e-04, -7.72954008e-05, 2.18408895e-05, -5.11899498e-06,  1.89792468e-06, -2.90804724e-06,  1.53492056e-06, -2.29557145e-06, 1.40164093e-06], [ 1.16688364e-06,  1.24371172e-06, -1.34454251e-06, 8.77513685e-06,  2.24372889e-05, -1.05267129e-04, 2.81182089e-04, -3.47500929e-04,  2.86547467e-04, -4.45890706e-04,  4.64120676e-04, -3.41809500e-04, 4.64405573e-04, -4.46184305e-04,  2.86796101e-04, -3.48789385e-04,  2.82447087e-04, -1.06070525e-04, 2.19803533e-05,  1.10647825e-05, -3.23109680e-06, 2.93892299e-06, -1.43408818e-06,  1.49014841e-06, 3.08541530e-07]]
        ker_x4 = [[-6.62296952e-06, -1.43531806e-05,  7.71780906e-05, -1.71278414e-04,  4.48358012e-04, -4.35484835e-04, 6.00204476e-05,  1.72932487e-04, -6.59890880e-04, 6.63316052e-04, -1.29075677e-04, -1.32539615e-04, 6.65061933e-04, -6.57583529e-04,  1.72624437e-04, 5.85416637e-05, -4.35113558e-04,  4.47460392e-04, -1.68691287e-04,  7.48491948e-05, -1.20825425e-05, -6.16945181e-06,  1.40647523e-06, -2.46621721e-06, -1.89478260e-06], [-1.57257091e-05, -4.14571550e-05,  3.42346466e-05, -1.73117092e-04,  2.75364990e-04, -3.03023058e-04, 2.78934094e-05,  1.25176040e-04, -4.78930044e-04, 3.73299612e-04, -1.87901940e-04, -1.90068182e-04, 3.75959906e-04, -4.78251721e-04,  1.18706637e-04, 3.30086950e-05, -3.05971625e-04,  2.75636732e-04, -1.65608712e-04,  3.10237883e-05, -3.31510455e-05, -2.40114514e-05, -1.54131249e-05, -2.07109570e-05, -1.25366314e-05], [ 8.79877261e-05,  4.48250794e-05,  1.43474914e-04, -8.13370716e-05,  4.46069986e-04, -2.51096324e-04, 1.68041937e-04,  2.82216643e-04, -4.16284049e-04, 6.57742261e-04,  5.42777002e-07, -3.69401528e-06, 6.61203521e-04, -4.13602858e-04,  2.84677109e-04, 1.66339727e-04, -2.53320148e-04,  4.44667967e-04, -8.76248087e-05,  1.30069660e-04,  5.17768203e-05, 5.41626141e-05,  6.42609593e-05,  4.86184363e-05, 8.17263572e-05], [-1.97389381e-04, -1.94303153e-04, -1.13850823e-04, -3.61691375e-04,  1.82534612e-04, -5.27508499e-04, -1.55936519e-04, -1.29739608e-04, -9.89535823e-04, -1.25785678e-04, -8.99035716e-04, -9.06590605e-04, -1.23707752e-04, -9.87760257e-04, -1.27587555e-04, -1.51980901e-04, -5.24035189e-04,  1.87726633e-04, -3.42782645e-04, -1.10211164e-04, -1.84603006e-04, -1.53397850e-04, -1.49407264e-04, -1.39940108e-04, -1.75328663e-04], [ 5.53600083e-04,  3.17559112e-04,  4.92999156e-04, 2.56536092e-04,  9.47497436e-04,  3.44920816e-04, 7.36473070e-04,  5.37106011e-04, -9.44029307e-04, -7.17143354e-04, -2.01520137e-03, -2.00945209e-03, -7.20593613e-04, -9.43417777e-04,  5.21528418e-04, 7.37398921e-04,  3.29374365e-04,  9.26103967e-04, 2.47579796e-04,  4.35025868e-04,  3.67166678e-04, 3.02359578e-04,  3.52910836e-04,  2.49822013e-04, 4.81858966e-04], [-5.44751005e-04, -3.35610297e-04, -3.12026648e-04, -5.53822261e-04,  2.57063075e-04, -5.57883643e-04, -1.78515082e-04, -7.74280983e-04, -3.02986428e-03, -3.41906445e-03, -5.31860953e-03, -5.32733742e-03, -3.41347419e-03, -3.02830292e-03, -7.65440869e-04, -1.71034655e-04, -5.48122509e-04,  2.81811052e-04, -5.46014286e-04, -2.51284830e-04, -3.87486536e-04, -2.68345058e-04, -3.32601747e-04, -2.11314007e-04, -4.75095061e-04], [ 7.09585001e-05,  4.18588024e-05,  1.67012768e-04, -1.25738865e-04,  7.24959245e-04, -1.43978832e-04, 3.76816170e-04,  7.09050728e-05, -1.63729477e-03, -1.36717036e-03, -2.86972895e-03, -2.86799134e-03, -1.36989472e-03, -1.63232943e-03,  6.25513348e-05, 3.80185200e-04, -1.57744900e-04,  7.22191471e-04, -1.45817728e-04,  1.65092526e-04,  1.68122333e-05, 6.79298028e-05,  4.19494318e-05,  6.21088911e-05, 4.45288824e-05], [ 2.14053944e-04,  1.55249145e-04,  3.15109006e-04, -9.39263991e-05,  5.47674368e-04, -7.34235626e-04, 9.68308959e-05,  9.93094640e-04,  1.74057961e-03, 5.13418857e-03,  5.55444276e-03,  5.54880314e-03, 5.14267059e-03,  1.73651369e-03,  9.93132242e-04, 1.00239566e-04, -7.32561864e-04,  5.48743410e-04, -8.96907441e-05,  3.12769960e-04,  1.57679460e-04, 1.99063375e-04,  1.75503345e-04,  1.85952114e-04, 1.91494139e-04], [-8.33386788e-04, -5.49032586e-04, -5.07539778e-04, -1.06966426e-03, -1.01934304e-03, -3.11078108e-03, -1.69244420e-03,  1.67598017e-03,  7.88008701e-03, 1.75587516e-02,  2.22854838e-02,  2.22803839e-02, 1.75612923e-02,  7.88581278e-03,  1.67268072e-03, -1.68787350e-03, -3.11866286e-03, -9.99479322e-04, -1.08888245e-03, -4.38772287e-04, -6.55522686e-04, -4.78970935e-04, -5.76296239e-04, -3.98336182e-04, -7.76139379e-04], [ 8.47557909e-04,  4.35938098e-04,  7.62687647e-04, -5.77692408e-05, -6.17786020e-04, -3.36047029e-03, -1.29651721e-03,  5.17373439e-03,  1.76013876e-02, 3.49443704e-02,  4.45243642e-02,  4.45200689e-02, 3.49482298e-02,  1.75940301e-02,  5.17483568e-03, -1.30213180e-03, -3.36014689e-03, -6.37859222e-04, -3.88037770e-05,  6.88977481e-04,  5.45008399e-04, 4.89238359e-04,  5.55366103e-04,  3.87062290e-04, 7.71155697e-04], [-1.69815918e-04, -2.18344649e-04, -2.21612809e-05, -9.38849698e-04, -2.00746651e-03, -5.37591428e-03, -2.87766312e-03,  5.53244352e-03,  2.22622342e-02, 4.45263647e-02,  5.75122349e-02,  5.75120337e-02, 4.45260294e-02,  2.22679544e-02,  5.53486682e-03, -2.87880329e-03, -5.37305139e-03, -2.00803089e-03, -9.31822578e-04, -2.41083799e-05, -2.12080122e-04, -1.42975681e-04, -1.42997713e-04, -1.48685474e-04, -1.47800747e-04], [-1.58008785e-04, -2.20213420e-04, -2.49124987e-05, -9.40598780e-04, -2.01905868e-03, -5.37448563e-03, -2.88040726e-03,  5.53052593e-03,  2.22669058e-02, 4.45207581e-02,  5.75119779e-02,  5.75034432e-02, 4.45257016e-02,  2.22618692e-02,  5.52939530e-03, -2.88189040e-03, -5.37443440e-03, -2.00954010e-03, -9.36773140e-04, -2.21936552e-05, -2.14422282e-04, -1.45715894e-04, -1.54002031e-04, -1.48140462e-04, -1.54624955e-04], [ 8.41734291e-04,  4.39786090e-04,  7.64512457e-04, -5.01856593e-05, -6.14856894e-04, -3.35174706e-03, -1.29923481e-03,  5.18318405e-03,  1.75996777e-02, 3.49550471e-02,  4.45260629e-02,  4.45347801e-02, 3.49522084e-02,  1.76057480e-02,  5.17884130e-03, -1.29504339e-03, -3.35397781e-03, -6.39995153e-04, -2.98883297e-05,  6.91780238e-04,  5.46498981e-04, 4.95493179e-04,  5.64463960e-04,  3.90185334e-04, 7.76677974e-04], [-8.44855735e-04, -5.48137992e-04, -5.17587876e-04, -1.06950570e-03, -1.03276374e-03, -3.11213103e-03, -1.69876206e-03,  1.67016091e-03,  7.87680782e-03, 1.75525062e-02,  2.22780760e-02,  2.22723745e-02, 1.75588578e-02,  7.87489023e-03,  1.67177862e-03, -1.69695402e-03, -3.11630010e-03, -1.00974995e-03, -1.08485261e-03, -4.46302700e-04, -6.57052209e-04, -4.86649194e-04, -5.82089007e-04, -4.01840021e-04, -7.81816256e-04], [ 2.20901216e-04,  1.54273032e-04,  3.14960664e-04, -8.90729498e-05,  5.54220926e-04, -7.33066408e-04, 9.89281252e-05,  9.96466959e-04,  1.73565838e-03, 5.14443079e-03,  5.54698193e-03,  5.55603346e-03, 5.14005916e-03,  1.73831673e-03,  9.93110589e-04, 1.04838342e-04, -7.32817047e-04,  5.57456224e-04, -1.00352911e-04,  3.18755949e-04,  1.53792425e-04, 2.01663090e-04,  1.80350034e-04,  1.85500350e-04, 1.96015768e-04], [ 6.80312296e-05,  4.67211248e-05,  1.67404462e-04, -1.26764338e-04,  7.17099989e-04, -1.39176409e-04, 3.78375495e-04,  7.70669430e-05, -1.63196120e-03, -1.36380037e-03, -2.86062155e-03, -2.86648283e-03, -1.36091013e-03, -1.63337926e-03,  7.43566634e-05, 3.75896314e-04, -1.48836014e-04,  7.13720219e-04, -1.29924010e-04,  1.64013400e-04,  2.48319393e-05, 6.64570471e-05,  4.10560206e-05,  6.33243035e-05, 4.05774481e-05], [-5.45703631e-04, -3.43761698e-04, -3.12373304e-04, -5.55901264e-04,  2.58315587e-04, -5.63259004e-04, -1.89779050e-04, -7.85009935e-04, -3.04622995e-03, -3.42933508e-03, -5.33473352e-03, -5.33901295e-03, -3.41859250e-03, -3.04469489e-03, -7.76268018e-04, -1.82858930e-04, -5.51534817e-04,  2.73532642e-04, -5.58369618e-04, -2.51840771e-04, -3.98721168e-04, -2.66345829e-04, -3.37610429e-04, -2.15057604e-04, -4.78444214e-04], [ 5.67275973e-04,  3.24470515e-04,  5.02358191e-04, 2.61400215e-04,  9.38397250e-04,  3.65285261e-04, 7.49175902e-04,  5.44011069e-04, -9.18668928e-04, -7.11209315e-04, -1.99727667e-03, -1.98949291e-03, -7.21541233e-04, -9.20234655e-04,  5.39022731e-04, 7.34888134e-04,  3.58505407e-04,  9.16481775e-04, 2.61462701e-04,  4.42512159e-04,  3.73992341e-04, 3.06948525e-04,  3.56516335e-04,  2.54016195e-04, 4.89348255e-04], [-2.13831081e-04, -1.93390282e-04, -1.25380873e-04, -3.53034324e-04,  1.79654700e-04, -5.38106658e-04, -1.67460195e-04, -1.39865500e-04, -1.01759122e-03, -1.17336975e-04, -9.09379276e-04, -9.09572060e-04, -1.12089809e-04, -1.01564266e-03, -1.29799577e-04, -1.71492764e-04, -5.28148550e-04,  1.81944168e-04, -3.42864310e-04, -1.16412935e-04, -1.86264180e-04, -1.59471107e-04, -1.43489378e-04, -1.46315157e-04, -1.75701032e-04], [ 9.86208252e-05,  4.58713112e-05,  1.34083530e-04, -7.13232366e-05,  4.01340396e-04, -1.96186505e-04, 1.66365484e-04,  2.88085139e-04, -3.57542915e-04, 6.16980193e-04,  1.15134583e-06,  7.62918989e-06, 6.15735538e-04, -3.58529476e-04,  2.83650123e-04, 1.64830781e-04, -1.99025308e-04,  4.03412967e-04, -7.59382310e-05,  1.28919011e-04,  5.98033548e-05, 5.52197635e-05,  6.18129852e-05,  4.97701039e-05, 8.39975401e-05], [-1.99451788e-05, -3.31935407e-05,  3.95913230e-05, -1.68362312e-04,  3.14572651e-04, -3.49850015e-04, 1.06234475e-05,  1.22595913e-04, -5.72570891e-04, 4.58886905e-04, -1.84707867e-04, -1.90342587e-04, 4.58817725e-04, -5.69112890e-04,  1.24198428e-04, 9.29201997e-06, -3.51323106e-04,  3.14522476e-04, -1.64944271e-04,  3.88640401e-05, -3.53931464e-05, -2.12568339e-05, -8.58892872e-06, -2.45825486e-05, -1.05292356e-05], [-9.00222312e-06, -1.98616726e-05,  4.21185614e-05, -1.23659498e-04,  2.51632213e-04, -2.23101437e-04, 5.72576610e-05,  1.64446668e-04, -4.00483987e-04, 4.06739826e-04, -1.18534343e-04, -1.12464768e-04, 4.06901614e-04, -4.00473684e-04,  1.65002872e-04, 5.49562719e-05, -2.20523259e-04,  2.50615441e-04, -1.24957500e-04,  4.32211928e-05, -1.65310466e-05, -4.98800637e-06, -2.85803117e-06, -3.67808548e-06, -6.34900243e-06], [-1.77956721e-07, -1.01589767e-05,  5.13609484e-05, -1.24955608e-04,  2.97146733e-04, -2.79870292e-04, 3.80024503e-05,  1.45534897e-04, -4.76902613e-04, 4.59446310e-04, -1.26425191e-04, -1.23541555e-04, 4.59672912e-04, -4.76756628e-04,  1.48665858e-04, 3.31575102e-05, -2.75740458e-04,  2.95688311e-04, -1.30191984e-04,  5.70978700e-05, -1.24289973e-05, -1.27969145e-06,  3.63751792e-06,  6.45288026e-07, 2.65192944e-06], [-2.00507111e-06, -1.61474363e-05,  3.89097513e-05, -1.13249087e-04,  2.08590413e-04, -1.84651391e-04, 5.38416571e-05,  1.43155528e-04, -3.37289792e-04, 3.32014955e-04, -1.22051191e-04, -1.14295792e-04, 3.26063979e-04, -3.39099526e-04,  1.49085303e-04, 5.05394964e-05, -1.89832150e-04,  2.16426371e-04, -1.12529029e-04,  3.91345893e-05, -1.57909017e-05, -3.37711867e-06, -1.28724935e-06, -3.23299014e-06, -3.96142786e-06], [ 7.00379701e-07, -1.14050572e-05,  6.83115868e-05, -1.42903358e-04,  3.87923239e-04, -3.78625060e-04, 4.03025588e-05,  1.51877452e-04, -6.00269414e-04, 5.96299767e-04, -1.21738311e-04, -1.19786884e-04, 6.02350628e-04, -6.07174530e-04,  1.57549570e-04, 3.34923061e-05, -3.69006040e-04,  3.82823084e-04, -1.46364197e-04,  6.91332971e-05, -9.03777436e-06, -2.87651778e-06,  2.54614224e-06, -4.18355597e-07, 4.72551346e-06]]

        k = (torch.tensor(ker_x4).repeat(b, 1, 5, 1, 1)/5)
        if self.use_cuda:
            k = k.cuda()       # TODO
        # k = (torch.ones(b, 1, 5, 25, 25)/5).cuda()  # TODO

        if self.sigma_max:
            if self.training:
                noise_level = torch.randint(0, self.sigma_max, (1,))/255.0
            else:
                assert self.noise_level < self.sigma_max
                noise_level = (torch.FloatTensor([self.noise_level])/255.)
                if self.use_cuda:
                    noise_level = noise_level.cuda()
            sigma = noise_level.repeat(b, 1, 1, 1, 1)
            if self.use_cuda:
                sigma = sigma.cuda()
            # add Gaussian noise
            x = x + torch.normal(0, noise_level[0], x.shape)
            if self.use_cuda:
                x = x.cuda()
        else:
            sigma = (torch.zeros(b, 1, 1, 1, 1))
            if self.use_cuda:
                sigma = sigma.cuda()

        FB = ps2ot(k, (sf[0]*t, sf[1]*w, sf[2]*h))                              # [1, 1, 100, 256, 256]
        FBC = torch.conj(FB)                                                    # [1, 1, 100, 256, 256]
        F2B = torch.pow(torch.abs(FB), 2)                                       # [1, 1, 100, 256, 256]
        STy = upsample3D(x, sf)                                                 # [1, 3, 100, 256, 256]
        FBFy = FBC * torch.fft.fftn(STy, dim=(2,3,4))                           # [1, 3, 100, 256, 256]

        if self.interpolation_mode == 'nearest':
            x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')   # [1, 3, 100, 256, 256]
        elif self.interpolation_mode == 'trilinear':
            x = nn.functional.interpolate(x, scale_factor=sf, mode='trilinear', align_corners=True)
        elif self.interpolation_mode == 'flow':
            x = nn.functional.interpolate(x, scale_factor=(1,sf[1],sf[2]), mode='blinear')
            x_inter = x.permute(0,2,1,3,4)
            if sf[0] == 2:
                flows = 0.5*self.compute_flow(x_inter)[0]
                flows = torch.cat([flows, flows[:,-1,...].unsqueeze(1)], dim=1)
                flows = flows.view(b*t, 2, w*sf[1], h*sf[2]).permute(0,2,3,1)
                x_inter = flow_warp(x_inter.view(b*t, -1, w*sf[1], h*sf[2]), flows)
                x_inter = x_inter.view(b, -1, c, w*sf[1], h*sf[2]).permute(0,2,1,3,4)
                x = torch.stack([x, x_inter], dim=3).view(b,c,-1,w*sf[1], h*sf[2])
            elif sf[0] == 5:
                fw = torch.linspace(0,1,sf[0]+2)[1:-1].view(1,sf[0],1,1,1)
                if self.use_cuda:
                    fw = fw.cuda()
                flows = self.compute_flow(x_inter)[0]
                out_x = []
                for i in range(flows.shape[1]):
                    x_warp = flow_warp(x_inter[:,i,...].unsqueeze(1).repeat(1,sf[0],1,1,1).view(-1, c, w*sf[1], h*sf[2]),
                                      (fw*flows[:,i,...].unsqueeze(1)).view(-1, 2, w*sf[1], h*sf[2]).permute(0,2,3,1))
                    out_x.append(x_inter[:,i,...].unsqueeze(1))
                    out_x.append(x_warp.view(b, -1, c, w*sf[1], h*sf[2]))
                out_x.append(x_inter[:,-1,...].unsqueeze(1))
                x = torch.cat(out_x, dim=1).permute(0,2,1,3,4)

        elif self.interpolation_mode == 'slomo':
            # model
            flow = slomo.UNet(6, 4)
            if self.use_cuda:
                flow = flow.cuda()
            interp = slomo.UNet(20, 5)
            if self.use_cuda:
                interp = interp.cuda()
            states = torch.load(self.slomo_pretrained, map_location='cpu')
            with torch.set_grad_enabled(False):
                flow.load_state_dict(states['state_dictFC'])
                interp.load_state_dict(states['state_dictAT'])
                if self.use_cuda:
                    back_warp = slomo.backWarp(h, w, "cuda")
                else:
                    back_warp = slomo.backWarp(h, w, "cpu")

            x0 = x.permute(0,2,1,3,4).view(-1,c,w,h)
            x0 = self.trans_forward(x0).view(b,t,c,w,h)

            frame0 = x0[:,:-1,:,:,:].reshape(-1,c,w,h)
            frame1 = x0[:,1:,:,:,:].reshape(-1,c,w,h)
            x_inter = self.interpolate_batch(frame0, frame1, sf[0], flow, interp, back_warp)
            x_inter = torch.stack(x_inter, dim=1).view(-1,c,w,h)  # [20, 3, 64, 64]
            x_inter = self.trans_backward(x_inter).view(b, t-1, sf[0]-1, c, w, h) # [b, 4, 5, 3, 64, 64]
            x0 = self.trans_backward(x0.view(-1,c,w,h)).view(b, t, c, w, h)

            out_x = []
            out_x.append(x0[:,0,:,:,:].unsqueeze(1).repeat(1,2,1,1,1))
            for i in range(t-1):
                out_x.append(x0[:,i,:,:,:].unsqueeze(1))
                out_x.append(x_inter[:,i,...])
                # x_inter = self.interpolate_batch(x0[:,i,:,:,:], x0[:,i+1,:,:,:], sf[0]+1, flow, interp, back_warp)
                # out_x.append(torch.stack(x_inter, dim=1))
            out_x.append(x0[:,-1,:,:,:].unsqueeze(1).repeat(1,3,1,1,1))
            x = torch.cat(out_x, dim=1)  # [b, 25, 3, 64, 64]

            # x = self.trans_backward(out_x).view(-1,c,w,h)
            x = nn.functional.interpolate(x.view(-1,c,w,h), scale_factor=sf[1:], mode='bilinear', align_corners=True)
            x = x.view(b,t*sf[0],c,w*sf[1],h*sf[2]).permute(0,2,1,3,4)

        # hyper-parameter, alpha & beta
        if self.fix_ab == 0:
            ab = self.h(torch.cat((sigma, torch.tensor(sf[0]).type_as(sigma).expand_as(sigma), torch.tensor(sf[1]).type_as(sigma).expand_as(sigma)), dim=1))  # [1, 12, 1, 1, 1]
        else:
            ab = torch.tensor(self.fix_ab).repeat(b,1,1,1,1).permute(0,4,2,3,1)
            if self.use_cuda:
                ab = ab.cuda()

        # unfolding
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)

            if i < self.pre_denoise_iters:
                x = self.pre_vsr(x)
            else:
                if self.img_channels == 4:
                    x = self.vsr(torch.cat((x, ab[:, i + self.n:i + self.n + 1, ...].repeat(1, 1, x.size(2), x.size(3), x.size(4))), dim=1))
                elif self.img_channels == 3:
                    x = self.vsr(x)

            if not self.training:
                if self.use_cuda:
                    x = x.cuda()

        x = x.permute(0,2,1,3,4)

        return x


    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
