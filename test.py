import argparse
import torch
from dataloader import Dataloader
from ulapsrn import Net_Quarter_Half_Original
import multiprocessing
import os
from tqdm import tqdm
import numpy as np
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/all_data/Scannet_ssim', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--name', type=str, default='testLap', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode, no wandb')
#     parser.add_argument('--max_distance', type=float, default=5100.0, help='all depth bigger will seted to this value')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--img_dir', type=str, default='/workspace/results/', help='saves results here.')
    parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.') 
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
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
        
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def tensor2mm(input):
    if isinstance(input, torch.Tensor):  # get the data from a variable
        tensor = input.data
        tensor = tensor * 1000
        numpy = tensor.cpu().squeeze(dim=1).numpy().astype(np.uint16)
    return numpy

def do_work(item):
    depth, path = item
    imageio.imwrite(path, depth)

def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        do_work(item)

q = multiprocessing.Queue()
n_processes = 16
        
if __name__ == '__main__':
    opt = parse_args()   # get training options
    dataset = Dataloader(opt) 
    print('Dataset len: ',len(dataset))   
    
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()) if torch.cuda.is_available else 'cpu')
    model = Net_Quarter_Half_Original()
    model = model.to(device)
    print("===> Setting model")
    weights = torch.load('./model_epoch_99.pth', map_location=device)
    model.load_state_dict(weights['model'].state_dict())
    model.eval()
    print("load model")
    
    a2b_path = os.path.join(opt.img_dir, opt.name, opt.phase,'A2B', 'depth')
    mkdirs(a2b_path)
    
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=worker, args=(q,))
        processes.append(p)
        p.start()
    with torch.no_grad():    
        for data in tqdm(dataset.dataloader):
            d = data['lq_depth'].to(device)
            d_quarter = data['lq_quarter_depth'].to(device)
            pred_original, pred_half, pred_quarter = model((d, d_quarter))
            res = pred_original
            B_depth_fake = tensor2mm(res)
            A_name = data['name']
            for depth_pred, depth_name in zip(B_depth_fake, A_name):
                path_to_save = os.path.join(a2b_path, depth_name+'.png')
                q.put_nowait((depth_pred, path_to_save))
            
    for i in range(n_processes):
        q.put_nowait(None)

    q.close()
    q.join_thread()

    for p in processes:
        p.join()
    print("done!")