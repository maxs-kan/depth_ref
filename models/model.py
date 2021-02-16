import itertools
from .base_model import BaseModel
from .network import Net_Quarter_Half_Original
from . import network
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from utils import util

class Model(BaseModel, nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__(opt)
        if self.isTrain:
            self.loss_names = ['Ld', 'Lg', 'Ln', 'Lb', 'depth_dif_full', 'depth_dif_4']
        self.loss_names_test = ['depth_dif_full', 'depth_dif_4'] 
                
        self.visuals_names = ['depth_A', 
                              'depth_B', 
                              'depth_A2B', 
                              'd_4',
                              'name']

        self.model_names = ['netG_A']
        self.netG_A = Net_Quarter_Half_Original()
        self.netG_A = nn.DataParallel(self.netG_A, opt.gpu_ids).cuda()
        self.surf_normals = network.SurfaceNormals() 
        self.L1_ = nn.L1Loss()
        if self.isTrain:
            self.criterionL1 = network.L1_Charbonnier_loss()
            self.criterionGrad =  network.L1_Gradient_loss()
            self.criterionBorder = network.Patch_Discontinuity_loss()
                
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G])
            self.opt_names = ['optimizer_G']

    
    def set_input(self, input):
        self.name = input['name']
        self.depth_A = input['lq_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        self.depth_A_4 = input['lq_quarter_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        self.depth_B = input['hq_depth']
        self.depth_B_4 = input['hq_quarter_depth'].to(self.device, non_blocking=torch.cuda.is_available())
        self.norm_B_4 = self.surf_normals(self.depth_B_4)
        
        if self.opt.use_full:
            self.depth_B = self.depth_B.to(self.device, non_blocking=torch.cuda.is_available())
            self.norm_A = self.surf_normals(self.depth_A)
            self.norm_B = self.surf_normals(self.depth_B)
        
        if self.opt.use_half:
            self.depth_B_2 = interpolate(self.depth_B, scale_factor=0.5, align_corners=False, mode='nearest').to(self.device, non_blocking=torch.cuda.is_available())
            self.depth_A_2 = interpolate(self.depth_A, scale_factor=0.5, align_corners=False, mode='nearest').to(self.device, non_blocking=torch.cuda.is_available())
            self.norm_B_2 = self.surf_normals(self.depth_B_2)

    
    def forward(self): 
        ###Fake depth
        self.depth_A2B, self.d_2, self.d_4 = self.netG_A((self.depth_A, self.depth_A_4))
        ###Normals
        self.norm_A2B = self.surf_normals(self.depth_A2B)

    def backward_G(self):
        
        self.loss_Lb = 0.
        L_b_4 = self.criterionBorder(self.depth_A_4, self.d_4)
        self.loss_Lb += L_b_4
        if self.opt.use_half:
            L_b_2 = self.criterionBorder(self.depth_A_2, self.d_2)
            self.loss_Lb += L_b_2
        if self.opt.use_full:
            L_b = self.criterionBorder(self.depth_A, self.depth_A2B)
            self.loss_Lb += L_b
        
        self.loss_Ld = 0.
        L_d_4 = self.criterionL1(self.d_4, self.depth_B_4)
        self.loss_Ld += L_d_4
        if self.opt.use_half:
            L_d_2 = self.criterionL1(self.d_2, self.depth_B_2)
            self.loss_Ld += L_d_2
        if self.opt.use_full:
            L_d = self.criterionL1(self.depth_A2B, self.depth_B)
            self.loss_Ld += L_d
            
        self.loss_Lg = 0.
        L_g_4 = self.criterionGrad(self.d_4, self.depth_B_4)
        self.loss_Lg += L_g_4
        if self.opt.use_half:
            L_g_2 = self.criterionGrad(self.d_2, self.depth_B_2)
            self.loss_Lg += L_g_2
        if self.opt.use_full:
            L_g = self.criterionGrad(self.depth_A2B, self.depth_B)
            self.loss_Lg += L_g
            
        self.loss_Ln = 0.
        L_n_4 = self.criterionL1(self.surf_normals(self.d_4), self.norm_B_4)
        self.loss_Ln += L_n_4
        if self.opt.use_half:
            L_n_2 = self.criterionL1(self.surf_normals(self.d_2), self.norm_B_2)
            self.loss_Ln += L_n_2
        if self.opt.use_full:
            L_n = self.criterionL1(self.norm_A2B, self.norm_B)
            self.loss_Ln += L_n
            
        self.loss = self.loss_Ld + 2. * self.loss_Lg + 2. * self.loss_Ln + 10. * self.loss_Lb
        self.loss.backward()
        
        # visualization
        with torch.no_grad():
            self.loss_depth_dif_full = self.L1_(self.depth_B.cuda(), self.depth_A2B.cuda())
            self.loss_depth_dif_4 = self.L1_(self.depth_B_4, self.d_4)
        
    def optimize_param(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def calc_test_loss(self):
        self.test_depth_dif_full = self.L1_(self.depth_B.cuda(), self.depth_A2B.cuda())
        self.test_depth_dif_4 = self.L1_(self.depth_B_4, self.d_4)
