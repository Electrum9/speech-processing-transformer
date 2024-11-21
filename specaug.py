"""
This implementation is adapted from ESPnet:
    https://github.com/espnet/espnet/blob/master/espnet/utils/spec_augment.py
"""

import random
import torch


def specaug(
    spec, F=30, T=40, num_freq_masks=2, num_time_masks=2, replace_with_zero=False
):
    """SpecAugment

    Reference:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        (https://arxiv.org/pdf/1904.08779.pdf)

    This implementation modified from https://github.com/zcaceres/spec_augment

    :param torch.Tensor spec: input tensor with the shape (T, dim)
    :param int F: maximum width of each freq mask
    :param int T: maximum width of each time mask
    :param int num_freq_masks: number of frequency masks
    :param int num_time_masks: number of time masks
    :param bool replace_with_zero: if True, masked parts will be filled with 0,
        if False, filled with mean
    """
    return time_mask(
        freq_mask(
            spec,
            F=F,
            num_masks=num_freq_masks,
            replace_with_zero=replace_with_zero,
        ),
        T=T,
        num_masks=num_time_masks,
        replace_with_zero=replace_with_zero,
    )


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    """Frequency masking

    :param torch.Tensor spec: input tensor with shape (T, dim)
    :param int F: maximum width of each mask
    :param int num_masks: number of masks
    :param bool replace_with_zero: if True, masked parts will be filled with 0,
        if False, filled with mean
    """
    cloned = spec.unsqueeze(0).clone()
    num_mel_channels = cloned.shape[2]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            return cloned.squeeze(0)

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[0][:, f_zero:mask_end] = 0
        else:
            cloned[0][:, f_zero:mask_end] = cloned.mean()
    return cloned.squeeze(0)


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    """Time masking

    :param torch.Tensor spec: input tensor with shape (T, dim)
    :param int T: maximum width of each mask
    :param int num_masks: number of masks
    :param bool replace_with_zero: if True, masked parts will be filled with 0,
        if False, filled with mean
    """
    cloned = spec.unsqueeze(0).clone()
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return cloned.squeeze(0)

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[0][t_zero:mask_end, :] = 0
        else:
            cloned[0][t_zero:mask_end, :] = cloned.mean()
    return cloned.squeeze(0)
