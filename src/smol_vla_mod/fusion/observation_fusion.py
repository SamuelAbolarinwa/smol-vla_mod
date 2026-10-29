"""Initial two-camera observation fusion. No implicit model downloads."""
import torch
from torch import nn
from torch.nn import functional as F


class ObservationFusion(nn.Module):
    """Three spatial scales per camera -> 98 tokens of width 256 by default.

    neck maps: tuple of [B,V,C,H,W], C=(128,256,512).
    instruction: ordinary detached CLIP [B,512]; state: normalised [B,8].
    Camera order must match the data pipeline (external, wrist).
    Adaptive 4x4 pooling retains a coarse grid; learned position, camera and
    scale embeddings distinguish tokens. Two bidirectional transformer layers
    return all tokens for the future action expert's cross-attention.
    This is a trainable initial design, not a pretrained fusion network.
    """
    def __init__(self, width=256, cameras=2, grid=4, state_dim=8, layers=2, heads=8):
        super().__init__()
        if width<=0 or heads<=0 or width%heads or min(cameras,grid,state_dim,layers)<=0:
            raise ValueError('Invalid fusion dimensions')
        self.width,self.cameras,self.grid,self.state_dim=width,cameras,grid,state_dim
        self.channels=(128,256,512)
        self.visual=nn.ModuleList([nn.Linear(c,width) for c in self.channels])
        self.position=nn.Parameter(torch.randn(1,1,grid*grid,width)*.02)
        self.camera=nn.Parameter(torch.randn(1,cameras,1,width)*.02)
        self.scale=nn.Parameter(torch.randn(3,width)*.02)
        self.modality=nn.Parameter(torch.randn(3,width)*.02)
        self.instruction=nn.Sequential(nn.Linear(512,width),nn.LayerNorm(width))
        self.state=nn.Sequential(nn.Linear(state_dim,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width))
        # Build layers separately so their initial parameters are independent.
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,heads,dim_feedforward=4*width,
            dropout=0.0,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.norm=nn.LayerNorm(width)

    @property
    def token_count(self):return self.cameras*3*self.grid*self.grid+2

    def forward(self, neck, instruction, state):
        if not isinstance(neck,(tuple,list)) or len(neck)!=3:
            raise ValueError('Expected three neck feature maps')
        if not isinstance(instruction,torch.Tensor) or instruction.ndim!=2 or instruction.shape[1]!=512:
            raise ValueError('Expected instruction [B,512]')
        b=instruction.shape[0]
        if b<1 or not isinstance(state,torch.Tensor) or state.shape!=(b,self.state_dim):
            raise ValueError('Expected nonempty state [B,state_dim]')
        tensors=[*neck,instruction,state]
        for x in tensors:
            if not isinstance(x,torch.Tensor) or not x.is_floating_point():raise TypeError('Floating tensors required')
            if x.device!=instruction.device:raise ValueError('Input devices must match')
        tokens=[]
        for i,(x,c,projection) in enumerate(zip(neck,self.channels,self.visual)):
            if x.ndim!=5 or tuple(x.shape[:3])!=(b,self.cameras,c) or min(x.shape[-2:])<self.grid:
                raise ValueError(f'Invalid feature map at scale {i}')
            pooled=F.adaptive_avg_pool2d(x.reshape(b*self.cameras,c,*x.shape[-2:]),(self.grid,self.grid))
            t=projection(pooled.flatten(2).transpose(1,2)).reshape(b,self.cameras,self.grid*self.grid,self.width)
            t=t+self.position+self.camera+self.scale[i]+self.modality[0]
            tokens.append(t.flatten(1,2))
        tokens += [(self.instruction(instruction)+self.modality[1]).unsqueeze(1),
                   (self.state(state)+self.modality[2]).unsqueeze(1)]
        x=torch.cat(tokens,dim=1)
        for block in self.blocks:x=block(x)
        return self.norm(x)


def flatten_camera_inputs(images, instructions):
    """Sample-major [B,V,3,H,W] -> [B*V,3,H,W] and [B*V,1,512]."""
    if images.ndim!=5 or images.shape[2]!=3:raise ValueError('Expected images [B,V,3,H,W]')
    b,v=images.shape[:2]
    if instructions.shape!=(b,512):raise ValueError('Expected instructions [B,512]')
    return images.reshape(b*v,*images.shape[2:]),instructions[:,None,:].repeat_interleave(v,dim=0)


def restore_camera_maps(neck,batch,cameras):
    if any(x.shape[0]!=batch*cameras for x in neck):raise ValueError('Camera batch mismatch')
    return tuple(x.reshape(batch,cameras,*x.shape[1:]) for x in neck)
