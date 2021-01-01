import math
import numpy as np

import torch
import torch.nn.functional as F

from cirtorch.utils.general import similarity_matrix, upper_triangular_mask, pairwise_distances
from cirtorch.utils.whiten import whitenapply
# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative

def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    _, idx = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b).int() - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b).int() - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2,i_,wl).narrow(3,j_,wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)

def klw(x, alphas):
    
    # x       # size: 1, #reg, D, 1, 1
    # alphas  # size: 1, 1, #reg, 1
    x = x.squeeze(-1).squeeze(-1).unsqueeze(0)  # size: 1, 1, #reg, D
    #print(x.size())
    o = F.conv2d(x,alphas)                      # size: 1, 1, 1, D 
    return o.squeeze(0).squeeze(0)              # size: 1, D


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


# --------------------------------------
# loss
# --------------------------------------
def triplet_loss(x, label, Lw=None, margin=0.7, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    # x is D x N
    # we are assuming only one tuple (q,p,n1,n2...nM) in x.
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples (has to be 1!)
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n
    
    # anchor and positive are shared for all negatives
    a = x[:,0].repeat(1,S-2).view((S-2)*nq,dim).permute(1,0) # anchor
    p = x[:,1].repeat(1,S-2).view((S-2)*nq,dim).permute(1,0) # positive
    n = x[:,2::]                                             # negatives
    
    Dap = torch.pow((a-p)+eps, 2).sum(dim=0).sqrt()
    Dan = torch.pow((a-n)+eps, 2).sum(dim=0).sqrt()
    
    y = torch.clamp( torch.pow(Dap,2) - torch.pow(Dan,2) + margin, min=0)
    y = torch.sum(y)
    return y

def contrastive_loss(x, label, Lw=None, margin=0.7, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def root_contrastive_loss(x, label, Lw=None, margin=0.7, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*D + 0.5*(1-lbl)*torch.clamp(margin-D, min=0)
    y = torch.sum(y)
    return y

def weighted_contrastive_loss(x, label, avgPosDist, avgNegDist,alpha, Lw=None, margin=0.7, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    # get the num positives and num negatives in the margin
    numPos = nq*1.0
    numNeg = torch.sum(D[lbl==0]<margin).item()*1.0
    
    # if this batch has no negatives we won't train on it
    if numNeg == 0:
        y = 0.0*D
        y = torch.sum(y)
        return y
    
    # finally, compute the weight to balance gradients and compute wLoss
    W = alpha*(numPos / numNeg)#0.5*(avgPosDist / (margin - avgNegDist)) * (numPos / numNeg)
    y = 0.5*lbl*D + W*0.5*(1-lbl)*torch.clamp(margin-D, min=0)
    y = torch.sum(y)
    return y


def exponential_loss(x, label, Lw=None, gamma=1.0, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    dpos = D[0]
    dneg = D[1::]
    
    dd = dneg - dpos
    y = torch.exp(-gamma*dd)
    y = torch.sum(y)
    return y


    
def angle_exponential_loss(x, label, Lw=None, eps=1e-3):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()
        
    dot = torch.mm(x[:,0].unsqueeze(0), x[:,1::]).squeeze(0)        
    dot[0] = dot[0] - eps
    ang = torch.acos(dot)
    angPos = ang[0]
    angNeg = ang[1::]
    y = torch.exp(-(angNeg - angPos))
    y = torch.sum(y)
    return y


def aggregate_exponential_loss(x, label, Lw=None, gamma=1.0, alpha=1.0, beta=0.0, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()

    numPos = torch.sum(label!=0).item()
    #numNeg = 1.*torch.sum(label==0).item()
    xp = x[:, 0:numPos]
    xn = x[:, numPos::]
    
    # compute all unique pairwise positive distances    
    iup,jup = np.triu_indices(numPos, k=1)
    inp = np.hstack((iup,jup))
    jnp = np.hstack((jup,iup))
    
    i = torch.LongTensor(inp).cuda()
    j = torch.LongTensor(jnp).cuda()
    
    difp = xp[:,i] - xp[:,j]
    dpos = torch.pow(difp+eps, 2).sum(dim=0).sqrt()
    
    difn = xp[:,i] - xn[:,i]
    dneg = torch.pow(difn+eps, 2).sum(dim=0).sqrt()
    
    Wp = torch.exp(-beta*dpos)
    Wp = Wp / torch.sum(Wp)
    
    Wn = Wp.clone()
    if( beta < 0 ):
        Wn = Wn*0 + 1./Wn.shape[0]
    
    Dp = torch.sum( Wp * dpos)
    Dn = torch.sum( Wn * dneg)

    dda = Dn - alpha*Dp
    y1 = torch.exp(-gamma*dda)
    y1 = torch.sum(y1)
    return y1

def ranking_triplet_exponential( x, label, Lw=None, gamma=1.0, alpha=1.0, beta=0.0, eps=1e-6):
    
    x1 = x[:,0].repeat(x.size(1)-1,1).permute(1,0)
    x2 = x[:,1::]
    D = torch.pow((x1 - x2) +eps, 2).sum(dim=0).sqrt()
    
    Ds, idx = D.sort()
    l = label[1::][idx]
    
    y = torch.sum(Ds)*0.0 
    for i,li in enumerate(l):
        if( li.item() == 0 ):
            for j,lj in enumerate(l[i::]):
                if( lj.item() == 1 ):
                    y += torch.exp(-gamma * (Ds[i] - alpha*Ds[i+j]) ) 
                
    return y
    

    

#def mahalanobis_aggregate_exponential_loss(x, label, Lw, gamma=1.0, eps=1e-6):
#    
#    x_lw = whitenapply(x, Lw['m'], Lw['P'])
#    x_lw = np.dot(P[:dimensions, :], X-m)
#    x_lw = x_lw / (np.linalg.norm(x_lw, ord=2, axis=0, keepdims=True) + 1e-6)
#    
#    numPos = torch.sum(label!=0).item()
#    numNeg = 1.*torch.sum(label==0).item()
#    xp = x_lw[:, 0:numPos]
#    xn = x_lw[:, numPos::]
#    
#    # compute all unique pairwise positive distances
#    inp,jnp = np.triu_indices(numPos, k=1)
#    i = torch.LongTensor(inp).cuda()
#    j = torch.LongTensor(jnp).cuda()
#    
#    dif = xp[:,i] - xp[:,j]
#    dpos = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
#    
#    dif = xp[:,i] - xn[:,i]
#    dneg = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
#    
#    expW = torch.exp(-gamma*dpos)
#    expW = expW / torch.sum(expW)
#    dposWAvg = torch.sum( expW * dpos)
#    dnegWAvg = torch.sum( expW * dneg)
#    
#    dd = dnegWAvg - dposWAvg
#    y = torch.exp(-dd)# + 5e-2*dposWAvg # 1e-1*dposWavg
#    y = torch.sum(y)/numNeg
#    return y

#    
#import os
#filename = os.path.join('results/loss_data_input.pth.tar')
#d = torch.load(filename)
#x = d['x']
#label = d['label']
#
#eps = 1e-6
#
#x1 = x[:,0].repeat(x.size(1)-1,1).permute(1,0)
#x2 = x[:,1::]
#D = torch.pow((x1 - x2) +eps, 2).sum(dim=0).sqrt()
#
#Ds, idx = D.sort()
#l = label[1::][idx]
#
#negpos = []
#for i,li in enumerate(l):
#    if( li.item() == 0 ):
#        for j,lj in enumerate(l[i::]):
#            if( lj.item() == 1 ):
#                negpos.append((i,j+i))
#                
#y = 0.0             
#if( len(negpos) == 0 ):
#    return y
#
#
#for np in negpos:
#    y += torch.exp(-1.0 * (Ds[np[0]] - Ds[np[1]]) ) 
#
#
#cpde = torch.sum( torch.pow( torch.diag(covP),2 ) )
#cnde = torch.sum( torch.pow( torch.diag(covN),2 ) )
#mask = (torch.ones_like(covP) - torch.eye(covP.size()[0]).cuda()).byte()
#cpnde = torch.sum( torch.pow( covP[mask],2 ) )
#cnnde = torch.sum( torch.pow( covN[mask],2 ) )









#def aggregate_exponential_loss(x, label, Lw=None, gamma=1.0, alpha=1.0, beta=0.0, eps=1e-6):
#    
#    if( Lw is not None ):
#        x = torch.mm(Lw['P'], x-Lw['m'])
#        x = l2n(x.t()).t()
#
#    numPos = torch.sum(label!=0).item()
#    #numNeg = 1.*torch.sum(label==0).item()
#    xp = x[:, 0:numPos]
#    xn = x[:, numPos::]
#    
#    # compute all unique pairwise positive distances    
#    iup,jup = np.triu_indices(numPos, k=1)
#    inp = np.hstack((iup,jup))
#    jnp = np.hstack((jup,iup))
#    
#    i = torch.LongTensor(inp).cuda()
#    j = torch.LongTensor(jnp).cuda()
#    
#    difp = xp[:,i] - xp[:,j]
#    dpos = torch.pow(difp+eps, 2).sum(dim=0).sqrt()
#    
#    difn = xp[:,i] - xn[:,i]
#    dneg = torch.pow(difn+eps, 2).sum(dim=0).sqrt()
#    
#    expW = torch.exp(-beta*dpos)
#    expW = expW / torch.sum(expW)
#    expW_ = torch.exp(beta*dpos)
#    
#    dposWAvg = torch.sum( expW * dpos)
#    dnegWAvg = torch.sum( expW_ * dneg)
#    
#
#    dda = dnegWAvg - alpha*dposWAvg
#    y1 = torch.exp(-gamma*dda)
#    y1 = torch.sum(y1)
#    return y1







