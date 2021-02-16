from utils.visualizer import Visualizer
from dataloader import Dataloader
from models.model import Model
from utils import util

import argparse
import wandb
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import os
import copy
from collections import OrderedDict 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/all_data/Scannet_ssim', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='trainLap', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode, no wandb')
    parser.add_argument('--max_distance', type=float, default=5100.0, help='all depth bigger will seted to this value')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.') 
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--use_half', action='store_true', default=False, help='use semantic')
    parser.add_argument('--use_full', action='store_true', default=False, help='use semantic')
    parser.add_argument('--load_size_h', type=int, default=320, help='scale images to this size')#480
    parser.add_argument('--load_size_w', type=int, default=320, help='scale images to this size')#640
    parser.add_argument('--crop_size_h', type=int, default=128, help='then crop to this size')
    parser.add_argument('--crop_size_w', type=int, default=128, help='then crop to this size')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic of cudnn, if true maybe slower')
    parser.add_argument('--load_epoch', type=str, default='last', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--n_pic', type=int, default=3, help='# of picture pairs for vis.')
    
    parser.add_argument('--img_freq', type=int, default=80, help='frequency of showing training results on screen')
    parser.add_argument('--loss_freq', type=int, default=30, help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true', default=False, help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    # training parameters
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs with the initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='initial learning rate for adam')
    return check_args(parser.parse_args())

def check_args(args):
    args.isTrain = args.phase == 'train'
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
        
    expr_dir = os.path.join(args.checkpoints_dir, args.name)
    util.mkdirs(expr_dir)
    return args

def acc_loss(d_acc, d):
    output = OrderedDict([(key, d_acc[key] + d[key]) for key in d_acc.keys()])
    return output
def div_loss(d_acc, n, epoch):
    output = OrderedDict([(key, d_acc[key] / n) for key in d_acc.keys()])
    output['Epoch'] = epoch
    return output

if __name__ == '__main__':
    seed_value = 101
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    opt = opt = parse_args()   
    opt_v = copy.deepcopy(opt)
    opt_v.isTrain = False
    opt_v.phase = 'val'
    torch.cuda.set_device(opt.gpu_ids[0])
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.backends.cudnn.benchmark = not opt.deterministic

    vis = Visualizer(opt)
    if not opt.debug:
        wandb.init(project="depth_refine", name=opt.name)
        wandb.config.update(opt)  
    dataset = Dataloader(opt) 
    dataset_size = len(dataset)    
    print('The number of training images = {}'.format(dataset_size))
    dataset_v = Dataloader(opt_v) 
    dataset_size_v = len(dataset_v)    
    print('The number of test images = {}'.format(dataset_size_v))
    model = Model(opt)    
    model.setup()
    if not opt.debug:
        wandb.watch(model)
    global_iter = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.train_mode()
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            global_iter += 1
            model.set_input(data)
            model.optimize_param()
            iter_finish_time = time.time()
            if global_iter % opt.loss_freq == 0:
                if not opt.debug:
                    wandb.log(model.get_current_losses(), step = global_iter)
            if global_iter % opt.img_freq == 0:
                print('{} img procesed out of {}, taken {:04.2f} sec per 1 batch'.format((i+1)*opt.batch_size, dataset_size, iter_finish_time - iter_start_time))
                fig = vis.plot_imgScannet(model.get_current_vis())
                if not opt.debug:
                    wandb.log({"chart": fig}, step=global_iter)
                plt.close(fig)
        print('Validation')
        n_b = 0
        for i, data in enumerate(dataset_v):
            if (i+1) % 200 == 0:
                print('{} img procesed out of {}'.format((i+1)*opt.batch_size, dataset_size_v))
            n_b += 1
            model.set_input(data)
            model.test()
            model.calc_test_loss()
            if i == 0:
                mean_loss = model.get_current_losses_test()
            else:
                mean_loss = acc_loss(mean_loss, model.get_current_losses_test())
        fig = vis.plot_imgScannet(model.get_current_vis())
        if not opt.debug:
            wandb.log({"chart_val": fig, 'Epoch':epoch}, step=global_iter)
        plt.close(fig)
        mean_loss = div_loss(mean_loss, n_b, epoch)
        if not opt.debug:
            wandb.log(mean_loss, step = global_iter)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch {}, iters {}'.format(epoch, global_iter))
            model.save_net(epoch)
        print('End of epoch {} / {} \t Time Taken: {:04.2f} sec'.format(epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.save_net('last')
    print('Finish')
