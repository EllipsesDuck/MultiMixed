from collections import OrderedDict
from os.path import join
import pdb
import ot

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from nystrom_attention import NystromAttention
import admin_torch
from utils.utils_model import *


class MIL_Sum_FC_ill(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg="small", dropout=0.25, n_classes=4):
        super(MIL_Sum_FC_ill, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(*[nn.Linear(size[0],size[1]),nn.ReLU(),nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1],size[2]),nn.ReLU(),nn.Dropout(dropout)])

        if self.fusion is not None:
            hidden = [256,256]
            fc_omic = [SNN_Block(dim1=omic_input_dim,dim2=hidden[0])]
            for i,_ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to('cuda:0')
        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)
        
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self,**kwargs):
        x_path = kwargs['x_path']

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic).squeeze(dim=0)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path 

        logits = self.classifier(h).unsqueeze()
        Y_hat = torch.topk(logits,1,dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1-hazards,dim=1)

        return hazards,S,Y_hat,None,None


class MIL_Attention_FC_ill(nn.Module): 
    def __init(self,omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4):
        super(MIL_Attention_FC_ill,self).__init__()

        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        if self.fusion is not None:
            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None
        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0) 
        A_raw = A  
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)  
        h_path = self.rho(h_path).squeeze() 

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic) 
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()  
                h = self.mm(torch.cat([h_path, h_omic], axis=0)) 
            h = h_path 

        logits  = self.classifier(h).unsqueeze(0)  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        hazards = torch.sigmoid(logits) 
        S = torch.cumprod(1 - hazards, dim=1)  
        
        return hazards, S, Y_hat, None, None


class MIL_Cluster_FC_ill(nn.Module): 
    def __init__(self, fusion=None, num_clusters=10, size_arg="small", dropout=0.25, n_classes=4):
        super(MIL_Cluster_FC_ill, self).__init__()
        
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.num_clusters = num_clusters 
        self.fusion = fusion

        size = self.size_dict_path[size_arg]
        phis = []

        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        if fusion is not None:
            hidden = self.size_dict_omic['small']
            omic_sizes = [100, 200, 300, 400, 500, 600]
            sig_networks = []
            for input_dim in omic_sizes:
                fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
                for i, _ in enumerate(hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.sig_networks = nn.ModuleList(sig_networks)

            if fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(size[2]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None
        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        if self.fusion is not None:
            self.sig_networks = self.sig_networks.to(device)
            self.mm = self.mm.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        cluster_id = kwargs['cluster_id'].detach().cpu().numpy()

        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id == i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1, 512)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        A, h_path = self.attention_net(h_cluster)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1)

        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, None, None  


class OT_Attn_assem(nn.Module): 
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5):
        super().__init__()

        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau

    def normalize_feature(self,x):
        x = x-x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1,weight2)**2
        
            src_weight = weight1.sum(dim=1)/weight1.sum()
            dst_weight = weight2.sum()/weight2.sum()

            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                    M=cost_map_detach / cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map*flow
            dist = torch.sum(dist)
        elif self.impl == "pot-uot-12":
            a,b = ot.unif(weight1.size()[0]).astype('float64'), ot.unif(weight2.size()[0]).astype('float64')
            self.cost_map = torch.cdist(weight1, weight2)**2

            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double().cpu().numpy(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = torch.from_numpy(flow).type(torch.FloatTensor).cuda()

            dist = self.cost_map*flow
            dist = torch.sum(dist)
            return flow,dist
        else:
            raise NotImplementedError
        
    def forward(self,x,y):
        x = x.sequeeze()
        y = y.sequeeze()
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        pi,dist = self.OT(x,y)
        return pi.T.unsqueeze(0),dist
    

class MOTCAT_ill(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25, ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2"):
        super(MOTCAT_ill, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)   

        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i,_ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.coattn = OT_Attn_assem(impl=ot_impl,ot_reg=ot_reg,ot_tau=ot_tau)

        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        self.classifier = nn.Linear(size[2],n_classes)

    def forward(self,**kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]   
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) 
        
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]  # 每个基因组特征通过其对应的全连接层
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)  
        
        A_coattn, _ = self.coattn(h_path_bag, h_omic_bag) 
        h_path_coattn = torch.mm(A_coattn.squeeze(), h_path_bag.squeeze()).unsqueeze(1)  

        h_path_trans = self.path_transformer(h_path_coattn)  
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1)) 
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)  
        h_path = self.path_rho(h_path).squeeze()  
        
        h_omic_trans = self.omic_transformer(h_omic_bag) 
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))  
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic) 
        h_omic = self.omic_rho(h_omic).squeeze()  
        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))
                
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1] 
        hazards = torch.sigmoid(logits)  
        S = torch.cumprod(1 - hazards, dim=1)  
        
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        
        return hazards, S, Y_hat, attention_scores 


class RMSNorm(torch.nn.Module): 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.translayer = TransLayer(norm_layer, dim)  # 初始化变换层

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1) 
        x = self.translayer(x)  
        return x[:, :x1.shape[1], :] 
    

class BottleneckTransFusion(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)  
        self.n_bottlenecks = n_bottlenecks 
        self.attn1 = TransLayer(nn.LayerNorm, dim=dim)  
        self.attn2 = TransLayer(nn.LayerNorm, dim=dim)
        self.bottleneck = torch.rand((1, n_bottlenecks, dim)).cuda()  

    def forward(self, x1, x2):
        b, seq, dim_len = x1.shape
        bottleneck = torch.cat([self.bottleneck, x2], dim=1) 
        bottleneck = self.attn2(bottleneck)[:, :self.n_bottlenecks, :]  
        x = torch.cat([x1, bottleneck], dim=1)  
        x = self.attn1(x) 
        return x[:, :seq, :] 

class AddFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)  
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)  
        self.norm1 = norm_layer(dim) 
        self.norm2 = norm_layer(dim) 

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1
    
    def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
        y_soft = (logits / tau).softmax(dim)
        
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0) 
            ret = y_hard - y_soft.detach() + y_soft 
        else:
            ret = y_soft
        
        return ret
    

class RoutingNetwork(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(RoutingNetwork, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),  
                norm_layer(dim),    
                nn.GELU(),         
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),  
                norm_layer(dim),    
                nn.GELU(),          
            ]
        )
        self.clsfer = nn.Linear(dim, branch_num) 

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1, x2 = self.fc1(x1), self.fc2(x2) 
        x = x1.mean(dim=1) + x2.mean(dim=1) 
        logits = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1)  
        return logits

class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256):
        super().__init__()
        self.TransFusion = TransFusion(norm_layer, dim)
        self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        self.AddFusion = AddFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        self.routing_network = RoutingNetwork(4, dim=dim)  
        self.routing_dict = {
            0: self.TransFusion,
            1: self.BottleneckTransFusion,
            2: self.AddFusion,
            3: self.DropX2Fusion,
        }

    def forward(self, x1, x2, hard=False):
        logits = self.routing_network(x1, x2, hard=hard) 
        if hard:
            corresponding_net_id = torch.argmax(logits, dim=1).item() 
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            x = torch.zeros_like(x1)
            for branch_id, branch in self.routing_dict.items():
                x += branch(x1, x2)  
        return x

class TransLayer(nn.Module): 
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim) 
        self.residual_attn = admin_torch.as_module(8)  
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,   
            pinv_iterations = 6,    
            residual = True,        
            dropout = 0.1
        )

    def forward(self, x):
        x = self.residual_attn(x, self.attn(self.norm(x)))  
        return x
    

class MoMETransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MoMETransformer, self).__init__()
        self.omic_sizes = omic_sizes  
        self.n_classes = n_classes  
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}  
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}  

        hidden = self.size_dict_omic[model_size_omic]  
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])] 
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout)) 
            sig_networks.append(nn.Sequential(*fc_omic))  
        self.sig_networks = nn.ModuleList(sig_networks)

        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()] 
        fc.append(nn.Dropout(dropout)) 
        self.wsi_net = nn.Sequential(*fc)  

        self.MoME_genom1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_genom2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])

        self.multi_layer1 = TransLayer(dim=size[2])  
        self.cls_multimodal = torch.rand((1, size[2])).cuda()
        self.classifier = nn.Linear(size[2], n_classes) 
    
    def forward(self, **kwargs):
        x_path = kwargs['x_path']  
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)] 

        h_path_bag = self.wsi_net(x_path)  

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] 

        h_path_bag = h_path_bag.unsqueeze(0)  
        h_omic_bag = h_omic_bag.unsqueeze(0)
        h_path_bag = self.MoME_patho1(h_path_bag, h_omic_bag, hard=True)
        h_omic_bag = self.MoME_genom1(h_omic_bag, h_path_bag, hard=True)
        h_path_bag = self.MoME_patho2(h_path_bag, h_omic_bag, hard=True)
        h_omic_bag = self.MoME_genom2(h_omic_bag, h_path_bag, hard=True)

        h_path_bag = h_path_bag.squeeze() 
        h_omic_bag = h_omic_bag.squeeze()

        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        h = self.multi_layer1(h_multi)[:, 0, :]  

        logits = self.classifier(h)  
        Y_hat = torch.topk(logits, 1, dim=1)[1]  
        hazards = torch.sigmoid(logits) 
        S = torch.cumprod(1 - hazards, dim=1) 

        attention_scores = {} 

        return hazards, S, Y_hat, attention_scores  
    

class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes 
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}  # 模型尺寸字典
        
        hidden = self.size_dict_omic[model_size_omic] 
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]  
        for i, _ in enumerate(hidden[1:]):  
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)  
        init_max_weights(self)  

    def forward(self, **kwargs):
        x = kwargs['x_omic']  
        features = self.fc_omic(x)  

        logits = self.classifier(features).unsqueeze(0)  
        Y_hat = torch.topk(logits, 1, dim=1)[1] 
        hazards = torch.sigmoid(logits) 
        S = torch.cumprod(1 - hazards, dim=1)  
        return hazards, S, Y_hat, None, None
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))  
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')  # 将 fc_omic 模型包装为 DataParallel，并将其移动到主设备
        else:
            self.fc_omic = self.fc_omic.to(device)  

        self.classifier = self.classifier.to(device) 


class MCAT_ill(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MCAT_ill, self).__init__()
        
        self.fusion = fusion  
        self.omic_sizes = omic_sizes  
        self.n_classes = n_classes  
        
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25)) 
        self.wsi_net = nn.Sequential(*fc) 
        
        hidden = self.size_dict_omic[model_size_omic]  
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])] 
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic)) 
        self.sig_networks = nn.ModuleList(sig_networks)  
        
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1) 
        
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)  
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])  
        
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2) 
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1) 
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)]) 
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        self.classifier = nn.Linear(size[2], n_classes)  

    def forward(self, **kwargs):
        x_path = kwargs['x_path']  
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)] 
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1)
    
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)
        
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)
        
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()
        
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()
        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))
        
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        
        return hazards, S, Y_hat, attention_scores

    def captum(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]
        
        h_path_bag = self.wsi_net(x_path)
        h_path_bag = torch.reshape(h_path_bag, (500, 10, 256))
        
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic)

        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)
        
        h_path_trans = self.path_transformer(h_path_coattn)
        h_path_trans = torch.reshape(h_path_trans, (10, 6, 256))
        A_path, h_path = self.path_attention_head(h_path_trans)
        A_path = F.softmax(A_path.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_path = torch.bmm(A_path, h_path).squeeze(dim=1)

        h_omic_trans = self.omic_transformer(h_omic_bag)
        h_omic_trans = torch.reshape(h_omic_trans, (10, 6, 256))
        A_omic, h_omic = self.omic_attention_head(h_omic_trans)
        A_omic = F.softmax(A_omic.squeeze(dim=2), dim=1).unsqueeze(dim=1)
        h_omic = torch.bmm(A_omic, h_omic).squeeze(dim=1)

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))
        
        logits = self.classifier(h).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        
        return hazards