"""Conditional flow matching: x_t=(1-t)*noise+t*action, velocity=action-noise."""
import math
import torch
from torch import nn
from smol_vla_mod.fusion.observation_fusion import flatten_camera_inputs,restore_camera_maps


class FlowActionExpert(nn.Module):
    def __init__(self,width=256,horizon=16,action_dim=7,layers=2,heads=8):
        super().__init__()
        if min(width,horizon,action_dim,layers,heads)<=0 or width%heads or width%2:
            raise ValueError('Invalid action expert dimensions')
        self.width,self.horizon,self.action_dim=width,horizon,action_dim
        self.action_in=nn.Linear(action_dim,width)
        self.position=nn.Parameter(torch.randn(1,horizon,width)*.02)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([nn.TransformerDecoderLayer(width,heads,4*width,
             dropout=0,batch_first=True,norm_first=True,activation='gelu') for _ in range(layers)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,action_dim))
        self.register_buffer('frequencies',torch.exp(-math.log(10000)*torch.arange(width//2)/(width//2)),persistent=False)

    def forward(self,x,t,context,valid=None):
        b=x.shape[0]
        if x.shape!=(b,self.horizon,self.action_dim) or b<1:raise ValueError('Invalid action shape')
        if t.shape!=(b,) or context.ndim!=3 or context.shape[0]!=b or context.shape[-1]!=self.width:
            raise ValueError('Invalid time or context shape')
        if valid is None:valid=torch.ones((b,self.horizon),device=x.device,dtype=torch.bool)
        if valid.shape!=(b,self.horizon) or valid.dtype!=torch.bool or not valid.any(1).all():
            raise ValueError('Each sample needs a boolean mask with at least one valid action')
        # Invalid targets cannot influence valid predictions through self-attention.
        x=torch.where(valid[...,None],x,torch.zeros_like(x))
        phase=t[:,None]*self.frequencies[None,:]*1000
        time=self.time(torch.cat((phase.sin(),phase.cos()),dim=-1))
        h=self.action_in(x)+self.position+time[:,None,:]
        for block in self.blocks:h=block(h,context,tgt_key_padding_mask=~valid)
        return self.output(h)

    def loss(self,context,actions,valid,noise=None,t=None):
        if valid.shape!=actions.shape[:2] or valid.dtype!=torch.bool or not valid.any(1).all():
            raise ValueError('Invalid action mask')
        if noise is None:noise=torch.randn_like(actions)
        if t is None:t=torch.rand(actions.shape[0],device=actions.device,dtype=actions.dtype)
        if noise.shape!=actions.shape:raise ValueError('Noise shape mismatch')
        if t.shape!=(actions.shape[0],) or not ((t>=0)&(t<=1)).all():raise ValueError('Time must lie in [0,1]')
        clean=torch.where(valid[...,None],actions,torch.zeros_like(actions))
        eps=torch.where(valid[...,None],noise,torch.zeros_like(noise))
        xt=(1-t[:,None,None])*eps+t[:,None,None]*clean
        target=clean-eps
        prediction=self(xt,t,context,valid)
        squared=torch.where(valid[...,None],(prediction-target).square(),torch.zeros_like(prediction))
        return squared.sum()/(valid.sum()*self.action_dim)

    @torch.no_grad()
    def sample(self,context,steps=10,noise=None):
        if not isinstance(steps,int) or steps<1:raise ValueError('Positive integer sampling steps required')
        shape=(context.shape[0],self.horizon,self.action_dim)
        if noise is None:noise=torch.randn(shape,device=context.device,dtype=context.dtype)
        if noise.shape!=shape:raise ValueError('Noise shape mismatch')
        x=noise.clone()
        for i in range(steps):
            t=torch.full((x.shape[0],),i/steps,device=x.device,dtype=x.dtype)
            x=x+self(x,t,context)/steps
        return x


class RobotPolicy(nn.Module):
    """Policy accepts normalised state and ordinary cached CLIP embeddings.

    CLIP remains frozen outside this module; use the validated instruction encoder.
    Outputs are normalised actions; invert training statistics before rollout.
    """
    def __init__(self,extractor,fusion,expert):
        super().__init__();self.extractor=extractor;self.fusion=fusion;self.expert=expert
    def encode(self,images,instructions,state):
        flat,guides=flatten_camera_inputs(images,instructions)
        neck=self.extractor(flat,guides).neck
        neck=restore_camera_maps(neck,images.shape[0],images.shape[1])
        return self.fusion(neck,instructions,state)
    def forward(self,images,instructions,state,actions,valid,noise=None,t=None):
        return self.expert.loss(self.encode(images,instructions,state),actions,valid,noise,t)
    @torch.no_grad()
    def sample(self,images,instructions,state,steps=10,noise=None):
        if self.training:raise RuntimeError('Call policy.eval() before sampling')
        return self.expert.sample(self.encode(images,instructions,state),steps,noise)
