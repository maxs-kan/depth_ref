import numpy as np
import matplotlib
import imageio
import matplotlib.pyplot as plt
import torch
from utils import util
import os
import logging

def get_normals(depth):
    norm = np.zeros(( depth.shape[0], depth.shape[1], depth.shape[2], 3))
    dzdx = np.gradient(depth, 1, axis=1)
    dzdy = np.gradient(depth, 1, axis=2)
    norm[:, :, :, 0] = -dzdx
    norm[:, :, :, 1] = -dzdy
    norm[:, :, :, 2] = np.ones_like(depth)
    n = np.linalg.norm(norm, axis = 3, ord=2, keepdims=True)
    norm = norm/(n + 1e-15)
    return (norm + 1.) / 2.

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.logger = logging.getLogger()
        
        
    
    def plot_imgScannet(self, img_dict):
        A_depth = util.tensor2im(img_dict['depth_A'], self.opt, input_type = 'depth')
        A_norm = get_normals(A_depth*1000)
#         A_norm = util.tensor2im(img_dict['norm_A'], self.opt, input_type = 'normals')
        B_depth = util.tensor2im(img_dict['depth_B'], self.opt, input_type = 'depth')
        B_norm = get_normals(B_depth*1000)
        B_depth_fake = util.tensor2im(img_dict['d_4'], self.opt, input_type = 'depth')
        B_norm_fake = get_normals(B_depth_fake*1000)
        
        max_dist = self.opt.max_distance/1000
        batch_size = A_depth.shape[0]
        n_pic = min(batch_size, self.opt.n_pic)
        n_col = 6
        fig_size = (50,30)
        n_row = n_pic
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
        
        self.logger.setLevel(100)
        old_level = self.logger.level
        for i in range(n_pic):
            axes[i, 0].set_title('Real Depth')
            axes[i, 1].set_title('R-S Depth')
            axes[i, 2].set_title('Syn Depth')
            axes[i, 3].set_title('Real Norm')
            axes[i, 4].set_title('R-S Norm')
            axes[i, 5].set_title('Syn Norm')

            axes[i, 0].imshow(A_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[i, 1].imshow(B_depth_fake[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[i, 2].imshow(B_depth[i],cmap=plt.get_cmap('RdYlBu'), vmin=0, vmax=max_dist)
            axes[i, 3].imshow(A_norm[i])
            axes[i, 4].imshow(B_norm_fake[i])
            axes[i, 5].imshow(B_norm[i])

        self.logger.setLevel(old_level)                        
        return fig
            