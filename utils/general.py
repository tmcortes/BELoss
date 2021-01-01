import os
import hashlib
import torch

def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def get_data_root():
    return os.path.join(get_root(), 'data')


def htime(c):
    c = round(c)
    
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


def sha256_hash(filename, block_size=65536, length=8):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()[:length-1]

def similarity_matrix(x, eps=1e-6):
    
    """ 
    This function expects the input to have vectors as colums. So, a 8x2 matrix
    is actually 2 eight-dimensional vectors, hence, the output similarity mat
    is 2x2 sized.
    It is computing the pair-wise L2 distance between all column vectors. If x
    is a cuda tensor the distance is computed on the GPU.
    """  
    r = torch.mm(x.t(), x)
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    D = diag.t() + diag + eps - 2*r
    return D.sqrt()

def pairwise_distances(x, y=None, eps=1e-6):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm + eps - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, min=0.0)

def upper_triangular_mask( mat ):
    
    n = mat.size(-1)
    coords = torch.arange(n)
    return coords > coords.view(n, 1)