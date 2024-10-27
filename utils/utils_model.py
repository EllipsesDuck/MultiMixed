import math
import pdb
import collections
import numpy as np
from os.path import join
from itertools import islice
import sklearn.metrics as skm
from collections import OrderedDict
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")    


class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img): 
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): 
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): 
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F,image_A)+ssim(image_F,image_B)


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB


class SiLU(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)
    

class GroupNorm32(nn.GroupNorm):
    def fowward(self,x):
        return super().forward(x.float()).type(x.dtype)
    

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args,**kwargs)
    elif dims == 2:
        return nn.Conv2d(*args,**kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs) 
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args,**kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args,**kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs) 
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    for tag,src in zip(target_params,source_params):
        tag.detach().mul_(rate).add_(src,alpha=1-rate)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mu_(scale)
    return module


def mean_flat(tensors):
    return tensors.mean(dim=list(range(1,len(tensors.shape()))))


def normalizations(channels):
    return GroupNorm32(32,channels)


def get_optim(model,args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def nll_loss(hazards,S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()

    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    uncensored_loss = -(1-c)*(torch.log(torch.gather(S_padded,1,Y).clamp(min=eps))+torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = uncensored_loss+censored_loss
    loss = (1-alpha)*neg_l+alpha*uncensored_loss
    loss = loss.mean()

    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) 
    c = c.view(batch_size, 1).float()
    
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) 
    
    S_padded = torch.cat([torch.ones_like(c), S], 1) 
    
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg 
    loss = loss.mean()
    
    return loss    


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.Dropout(p=dropout, inplace=False))


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class BilinearFusion(nn.Module):
    def __init__(self,skip=0,use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion,self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og,dim1),nn.ReLU())  
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)                             
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        if self.skip:
            self.encoder2 = nn.Sequential(nn.Linear(256 + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        else:
            self.encoder2 = nn.Sequential(nn.Linear(256, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) 
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out
    

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
    
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A) 
        return A, x


class CrossEntropySurvLoss(object):
    def __init__(self,alpha=0.15):
        self.alpha = alpha
    
    def __call__(self,hazards,S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards,S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self,alpha=0.15):
        self.alpha = alpha
    def __call__(self, hazards, S, Y, c, alpha=None):    
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(self, hazards, S, c, **kwargs):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)  
        theta = hazards.reshape(-1)  
        exp_theta = torch.exp(theta) 
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))  # 计算 Cox 损失
        
        return loss_cox


def l1_reg_all(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() 
    return l1_reg


def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg += l1_reg_all(model.fc_omic) 
    l1_reg += l1_reg_all(model.mm) 
    return l1_reg


class SelfContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=T1):
        super(SelfContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())

    def forward(self, q, k):
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)   # (bs,)
        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk/self.temperature), dim=1)

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq/self.temperature), dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss

class FullContrastiveLoss(nn.Module):
    def __init__(self, batch_size, num_r, num_nr, device='cuda', temperature=T2):
        super(FullContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.num_r = num_r
        self.num_nr = num_nr

        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("rumor_mask", (~torch.eye(num_r, num_r, dtype=torch.bool).to(device)).float())
        self.register_buffer("nonrumor_mask", (~torch.eye(num_nr, num_nr, dtype=torch.bool).to(device)).float())

    def compute_loss(self, feature, label):
        """
        feature: (batch, dim)
        r: rumor nr: non-rumor
        """
        index_r = torch.nonzero(label).squeeze()
        index_nr = torch.nonzero(label == 0).squeeze()
        ft_r = torch.index_select(feature, dim=0, index=index_r)
        ft_nr = torch.index_select(feature, dim=0, index=index_nr)

        similarity_matrix_r = F.cosine_similarity(ft_r.unsqueeze(1), ft_r.unsqueeze(0), dim=2)
        similarity_matrix_nr = F.cosine_similarity(ft_nr.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_r_nr = F.cosine_similarity(ft_r.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_nr_r = F.cosine_similarity(ft_nr.unsqueeze(1), ft_r.unsqueeze(0), dim=2)

        nominator_r = torch.sum(self.rumor_mask * torch.exp(similarity_matrix_r / self.temperature), dim=1)
        nominator_nr = torch.sum(self.nonrumor_mask * torch.exp(similarity_matrix_nr / self.temperature), dim=1)

        denominator_r = nominator_r + torch.sum(torch.torch.exp(similarity_matrix_r_nr / self.temperature), dim=1)
        denominator_nr = nominator_nr + torch.sum(torch.torch.exp(similarity_matrix_nr_r / self.temperature), dim=1)

        loss_r = torch.sum(-torch.log(nominator_r / denominator_r + eps)) / self.num_r
        loss_nr = torch.sum(-torch.log(nominator_nr / denominator_nr + eps)) / self.num_nr
        loss = loss_r + loss_nr
        return loss

    def forward(self, text, image, label):
        text = F.normalize(text, dim=1)
        image = F.normalize(image, dim=1)

        loss = self.compute_loss(text, label) + self.compute_loss(image, label)

        return loss


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()  

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]

        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)

        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad 

        return loss_total, loss_in, loss_grad 


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps 
    N, C, _, _ = img1.shape 

    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)

    return cc.mean()  


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim//2
    freqs = torch.exp(
        -math.log(max_period)*torch.arange(start=0,end=half,dtype=torch.float32)/half
    ).to(device=timesteps.device)
    args = timesteps[:,None].float()*freqs[None]
    embedding = torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
    if dim%2:
        embedding = torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs)+tuple(params)
        return CheckpointFunction.apply(func,len(input),*args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function  
        ctx.input_tensors = list(args[:length])  
        ctx.input_params = list(args[length:])  
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors) 
        return output_tensors 

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]  
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]  
            output_tensors = ctx.run_function(*shallow_copies)  
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        ) 
        del ctx.input_tensors 
        del ctx.input_params  
        del output_tensors  
        return (None, None) + input_grads  
    

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net) 
    for param in net.parameters():
        n = param.numel() 
        num_params += n
        if param.requires_grad:
            num_params_train += n
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)