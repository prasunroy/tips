import numpy as np
import torch
from models.base_model import BaseModel
from models.netG import NetG
from models.netD import NetD
from utils.visualization import translate_heatmap, visualize_heatmap


class Text2PoseModel(BaseModel):
    
    def __init__(self, gpuids=None, noise_dim=128, embed_dim=84, heatmap_channels=18, gradient_penalty=True):
        super(Text2PoseModel, self).__init__()
        self.models = ['netG', 'netD']
        self.losses = ['lossG1', 'lossG2', 'lossG', 'lossD1', 'lossD2', 'penalty', 'lossD']
        self.gpuids = gpuids if isinstance(gpuids, list) or isinstance(gpuids, tuple) else []
        self.device = None
        
        self.setup(verbose=True)
        
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.heatmap_channels = heatmap_channels
        self.gradient_penalty = gradient_penalty
        self.lossG1 = torch.zeros(1)
        self.lossG2 = torch.zeros(1)
        self.lossG = torch.zeros(1)
        
        self.netG = NetG(noise_dim, embed_dim, heatmap_channels)
        self.netD = NetD(heatmap_channels, embed_dim)
        
        self.init_networks(verbose=True)
        
        if self.gradient_penalty:
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0, 0.9))
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0, 0.9))
        else:
            self.optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=0.00005)
            self.optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=0.00005)
        
        self.iters = 0
    
    def set_inputs(self, inputs):
        self.real_pose_x = inputs['poseA'].to(self.device)
        self.real_text_h1 = inputs['textA'].to(self.device)
        self.real_text_h2 = inputs['textB'].to(self.device)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def optimize_parameters(self, d_iters=5, c=0.01, lambda_gp=10):
        self.iters += 1
        batch_size = self.real_pose_x.size(0)
        
        z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
        
        # update netD
        self.set_requires_grad(['netD'], True)
        self.optimizerD.zero_grad()
        
        fake_pose_zh1 = self.netG(z, self.real_text_h1).detach()
        
        pred_fake_zh1 = self.netD(fake_pose_zh1, self.real_text_h1).squeeze()
        pred_real_xh1 = self.netD(self.real_pose_x, self.real_text_h1).squeeze()
        # pred_real_xh2 = self.netD(self.real_pose_x, self.real_text_h2).squeeze()
        
        self.lossD1 = -(torch.mean(pred_real_xh1) - torch.mean(pred_fake_zh1))
        self.lossD2 = torch.zeros(1) # -(torch.mean(pred_real_xh1) - torch.mean(pred_real_xh2))
        
        if self.gradient_penalty:
            self.penalty = self.compute_gradient_penalty(self.real_pose_x, self.real_text_h1, fake_pose_zh1)
            self.lossD = self.lossD1 + lambda_gp * self.penalty # self.lossD1 + self.lossD2 + lambda_gp * self.penalty
        else:
            self.penalty = torch.zeros(1)
            self.lossD = self.lossD1 # self.lossD1 + self.lossD2
        
        self.lossD.backward()
        self.optimizerD.step()
        
        if not self.gradient_penalty:
            for param in self.netD.parameters():
                param.data.clamp_(-c, c)
        
        # update netG once every d_iters updates of netD
        if self.iters % d_iters == 0:
            self.set_requires_grad(['netD'], False)
            self.optimizerG.zero_grad()
            
            fake_pose_zh1 = self.netG(z, self.real_text_h1)
            pred_fake_zh1 = self.netD(fake_pose_zh1, self.real_text_h1).squeeze()
            
            interp_real_text_h1h2 = 0.5 * (self.real_text_h1 + self.real_text_h2)
            interp_fake_pose_zh1h2 = self.netG(z, interp_real_text_h1h2)
            pred_fake_zh1h2 = self.netD(interp_fake_pose_zh1h2, interp_real_text_h1h2)
            
            self.lossG1 = -torch.mean(pred_fake_zh1)
            self.lossG2 = -torch.mean(pred_fake_zh1h2)
            self.lossG = self.lossG1 + self.lossG2
            
            self.lossG.backward()
            self.optimizerG.step()
    
    def compute_visuals(self, padding=1, confidence_cutoff=0.2):
        mode = self.netG.training
        self.netG.eval()
        batch_size = self.real_pose_x.size(0)
        z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
        with torch.no_grad():
            fake_pose_zh1 = self.netG(z, self.real_text_h1)
        real_pose = self.real_pose_x.detach().cpu().permute(0, 2, 3, 1).numpy()
        real_pose = (real_pose + 1.0) / 2.0
        fake_pose = fake_pose_zh1.detach().cpu().permute(0, 2, 3, 1).numpy()
        fake_pose = (fake_pose + 1.0) / 2.0
        grid_image = np.zeros((batch_size*256 + (batch_size+1)*padding, 2*256 + 3*padding, 3), dtype=np.uint8)
        for i, (real, fake) in enumerate(zip(real_pose, fake_pose)):
            real = translate_heatmap(real, (256, 256))
            real = visualize_heatmap(real, confidence_cutoff=confidence_cutoff)
            fake = translate_heatmap(fake, (256, 256))
            fake = visualize_heatmap(fake, confidence_cutoff=confidence_cutoff)
            grid_image[padding + i*(256+padding) : (i+1)*(256+padding), padding : 256+padding] = real
            grid_image[padding + i*(256+padding) : (i+1)*(256+padding), 256+2*padding : 2*(256+padding)] = fake
        self.netG.train(mode)
        return grid_image
    
    def compute_gradient_penalty(self, real_poses, real_texts, fake_poses):
        batch_size = real_poses.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interp_poses = (alpha * real_poses + (1 - alpha) * fake_poses).requires_grad_(True)
        pred = self.netD(interp_poses, real_texts).view(-1, 1)
        ones = torch.ones(batch_size, 1).to(self.device).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interp_poses,
            grad_outputs=ones,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=False
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
