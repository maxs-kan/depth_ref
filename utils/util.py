"""This module contains simple helper functions """
# from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import numbers
import math
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def data_to_meters(input, opt):
    return input

def tensor2mm(input, opt):
    """"Converts a Tensor array into a numpy image array in meters.

    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    if isinstance(input, torch.Tensor):  # get the data from a variable
        tensor = input.data
        tensor = tensor * 1000.
        numpy = tensor.cpu().permute(0,2,3,1).numpy().astype(np.uint16)[:,:,:,0]
    return numpy

def tensor2im(input, opt, input_type='depth'):
    """"Converts a Tensor array into a numpy image array in meters.

    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    if not isinstance(input, np.ndarray):
        if isinstance(input, torch.Tensor):  # get the data from a variable
            tensor = input.data
        else:
            return input
        if input_type == 'depth':
            tensor = data_to_meters(tensor, opt)
            numpy = tensor.cpu().permute(0,2,3,1).numpy()[:,:,:,0]
        elif input_type == 'normals':
            tensor = (tensor + 1.) / 2.
            numpy = tensor.cpu().permute(0,2,3,1).numpy()
        else:
            raise ValueError('Unknown input type {}'.format(input_type))
    else:  # if it is a numpy array, do nothing
        numpy = input
    return numpy

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)