import torch
from torch import nn 
from torch.nn import init  
from torch.nn import functional as F  
from torchvision import models  
from collections import namedtuple  
import numpy as np  


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class TimeEmbedding(nn.Module):
    def __init__(self,embed_dim,scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim//2)*scale,requires_grad=False)

    def forward(self,x):
        x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class DownSample(nn.Module):
    def __init__(self,in_ch):
        super().__init__()

        self.main = nn.Conv2d(in_ch,in_ch,3,stride=2,padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self,x,temb):
        x = self.main(x)
        return x
    

class UpSample(nn.Module):
    def __init__(self,in_ch):
        super().__init__()

        self.main = nn.Conv2d(in_ch,in_ch,3,stride=1,padding=1)
        self.initialize() 

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self,x,temb):
        _, _, H, W = x.shape
        x = F.interpolate(x,scale_factor=2,mod='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self,in_ch):
        super().__init__()

        self.group_norm = nn.GroupNorm(32,in_ch)
        self.proj_q = nn.Conv2d(in_ch,in_ch,1,stride=1,padding=0)
        self.proj_k = nn.Conv2d(in_ch,in_ch,1,stride=1,padding=0)
        self.proj_v = nn.Conv2d(in_ch,in_ch,1,stride=1,padding=0)
        self.proj = nn.Conv2d(in_ch,in_ch,1,stride=1,padding=1)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_normal_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self,x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0,2,3,1).view(B, H * W, C)
        k = k.view(B, C, H * W)

        w = torch.bmm(q,k)*(int(C)**(-0.5))
        w = F.softmax(w,dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w,v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x+h
    

class CrossAttention(nn.Module):
    def __init__(self,in_ch,re_ch):
        super().__init__()
        self.in_ch = in_ch
        self.re_ch = re_ch
        self.proj_q = nn.Conv2d(in_ch,re_ch,1)
        self.proj_k = nn.Conv2d(in_ch,re_ch,1)
        self.proj_v = nn.Conv2d(in_ch,re_ch,1)
        self.proj = nn.Conv2d(re_ch,in_ch,1)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_normal_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self,x,y,shape):
        B, C, H, W = x.shape

        q = self.proj_q(x)
        k = self.proj_k(y)
        v = self.proj_v(y)
        norm = nn.LayerNorm([self.in_ch, shape, shape]).cuda()   

        q = q.view(B,self.re_ch,-1)
        k = k.view(B,self.re_ch,-1)
        v = v.view(B,self.re_ch,-1)  

        attention = F.softmax(torch.bmm(q.transpose(1,2),k),dim=-1)

        out = torch.bmm(v,attention.transpose(1,2))
        out = out.view(B,self.re_ch,x.shape[2],x.shape[3])
        out = self.final_conv(out)

        out += x
        out = norm(x)

        return out


class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,tdim,dropout,attn=False):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim,out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,1,stride=1,padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.intialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self,x,temb):
        h = self.block1(x)
        h1 = self.temb_proj(temb)[:,:,None,None]

        h = h+h1
        h = self.block2(h)
        h = h+self.shortcut(x)
        h = self.attn(h)

        return h

class MambaUnet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        self.T = T
        self.ch = ch
        self.ch_mult = ch_mult
        self.attn = attn  
        self.num_res_blocks = num_res_blocks

        self.time_emb = TimeEmbedding(ch*4)

        self.downs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        in_ch = ch

        for mult in ch_mult:
            self.downs.append(DownSample(in_ch))
            for _ in range(num_res_blocks):
                self.res_blocks.append(ResBlock(in_ch,in_ch*mult,self.T,dropout,attn=1 if len(self.res_blocks) in attn else 0))  
            in_ch *= mult

        self.ups = nn.ModuleList()
        for mult in ch_mult[::-1]:
            self.ups.append(UpSample(in_ch))
            for _ in range(num_res_blocks):
                self.res_blocks.append(ResBlock(in_ch,in_ch//mult,self.T,dropout,attn=1 if len(self.res_blocks) in attn else 0))
            in_ch //= mult

        self.final_conv = nn.Conv2d(ch,1,1,stride=1,padding=0)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)  
                init.zeros_(module.bias) 

    def forward(self,x,t):
        temb = self.time_emb(t)
        skips = []

        for down in self.downs:
            x = down(x,temb)
            skips.append(x)

        for res_block in self.res_blocks:
            x = res_block(x,temb)

        for up,skip in zip(self.ups,skip[::-1]):
            x = up(x,temb)
            x += skip

        return self.final_conv(x)