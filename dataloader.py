import os
import glob
import numpy as np
import imageio
import albumentations as A
import torch.utils.data as data
import torch
from PIL import Image

class Dataloader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = BaseDataset(opt)
        print('Dataset {} was created'.format(type(self.dataset).__name__))
        if (min(len(self.dataset), self.opt.max_dataset_size) % opt.batch_size != 0) and opt.phase == 'train':
            print('Warning, drop last batch')
            drop_last = True
        else:
            drop_last = False
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle= opt.phase == 'train',
            num_workers=int(opt.num_workers),
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available())
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

class BaseDataset(data.Dataset):
    
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.root = opt.dataroot
        self.IMG_EXTENSIONS = ['.png', '.jpg']
        self.base_transforms = []
        if self.opt.phase != 'test':
            self.add_base_transform()
        if opt.phase == 'test':
            self.dir = os.path.join(self.root, self.opt.phase + 'A', 'full_size')
            self.dir_hq = os.path.join(self.root, self.opt.phase + 'B', 'full_size','depth')
        else:
            self.dir = os.path.join(self.root, self.opt.phase + 'A')
            self.dir_hq = os.path.join(self.dir, 'render')

        self.dir_lq = os.path.join(self.dir, 'depth') 

        self.lq_depths = self.get_paths(self.dir_lq)
        self.hq_depths = self.get_paths(self.dir_hq)
        assert (len(self.lq_depths) == len(self.hq_depths)), 'not pair raw-render'
        self.is_image_files(self.lq_depths + self.hq_depths)
        
        self.size = len(self.lq_depths)
        
    def __getitem__(self, index):
        
        lq_path = self.lq_depths[index]
        hq_path = self.hq_depths[index]

        lq_n = self.get_name(lq_path)
        hq_n = self.get_name(hq_path)
        assert (lq_n == hq_n), 'not pair raw render '
        
        lq_depth = self.read_data(lq_path)
        hq_depth = self.read_data(hq_path)
        
        lq_depth, lq_quarter_depth, hq_depth, depth_hq_quarter = self.transform(lq_depth, hq_depth)
        
        return {'lq_depth': lq_depth, 'lq_quarter_depth':lq_quarter_depth, 'hq_depth': hq_depth, 'hq_quarter_depth':depth_hq_quarter,'name':lq_n}
    
    def apply_transformer(self, transformations, depth_lq, depth_hq):
        
        target = {
            'image':'image',
            'depth_hq':'image'
        }
        res = A.Compose(transformations, p=1, additional_targets=target)(image=depth_lq, depth_hq=depth_hq)
        return res
    
    def transform(self, depth_lq, depth_hq):
        depth_lq = self.normalize_depth(depth_lq)
        depth_hq = self.normalize_depth(depth_hq)
        if self.isTrain:
            transformed = self.apply_transformer(self.base_transforms, depth_lq, depth_hq)
            depth_lq = transformed['image']
            depth_hq = transformed['depth_hq']
        h, w = depth_lq.shape
        depth_lq_quarter = np.array(Image.fromarray(depth_lq, mode='F').resize((w // 4, h // 4), Image.NEAREST))
        depth_hq_quarter = np.array(Image.fromarray(depth_hq, mode='F').resize((w // 4, h // 4), Image.NEAREST))
        depth_lq /= 1000.
        depth_hq /= 1000.
        depth_lq_quarter /= 1000.
        depth_hq_quarter /=1000.
        depth_lq = torch.from_numpy(depth_lq).unsqueeze(0)
        depth_lq_quarter = torch.from_numpy(depth_lq_quarter).unsqueeze(0)
        depth_hq_quarter = torch.from_numpy(depth_hq_quarter).unsqueeze(0)
        depth_hq = torch.from_numpy(depth_hq).unsqueeze(0)
        return depth_lq, depth_lq_quarter, depth_hq, depth_hq_quarter
        
        
    def add_base_transform(self):
        self.base_transforms.append(A.Resize(height=self.opt.load_size_h, width=self.opt.load_size_w, interpolation=4, p=1))
        if self.opt.isTrain:
            self.base_transforms.append(A.RandomCrop(height=self.opt.crop_size_h, width=self.opt.crop_size_w, p=1))
    
    def is_image_files(self, files):
        for f in files:
            assert any(f.endswith(extension) for extension in self.IMG_EXTENSIONS), 'not implemented file extntion type {}'.format(f.split('.')[1])
        
    def get_paths(self, dir, reverse=False):
        files = []
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
        files = sorted(glob.glob(os.path.join(dir, '**/*.*'), recursive=True), reverse=reverse)
        return files[:min(self.opt.max_dataset_size, len(files))]
    
    def get_name(self, file_path):
        img_n = os.path.basename(file_path).split('.')[0]
        return img_n
    
    def normalize_depth(self, depth):
        if isinstance(depth, np.ndarray):
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32)
                return depth
            else:
                print(depth.dtype)
                raise AssertionError('Depth datatype')
        else:
            raise AssertionError('Depth filetype')

    
    def read_data(self, path):
        return imageio.imread(path)
    
    def __len__(self):
        return self.size





