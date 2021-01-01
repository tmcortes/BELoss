import torch
import torch.nn as nn

import cirtorch.layers.functional as LF

# --------------------------------------
# Loss/Error layers
# --------------------------------------

class TripletLoss(nn.Module):
    

    def __init__(self, margin=0.7, eps=1e-6):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        print('Creating triplet loss with margin: {}'.format(self.margin))
        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        return LF.triplet_loss(x, label, Lw=Lw, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'


class ContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        p: the norm degree. Default: 2

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = autograd.Variable(torch.randn(128, 35))
    >>> label = autograd.Variable(torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5))
    >>> output = contrastive_loss(input, label)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        print('Creating contrastive loss with margin: {}'.format(self.margin))
        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        return LF.contrastive_loss(x, label, Lw=Lw, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    

class RootContrastiveLoss(nn.Module):
    
    r"""
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(RootContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        return LF.root_contrastive_loss(x, label, Lw=Lw, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
class WeightedContrastiveLoss(nn.Module):
    
    r"""
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.oAvgPosDist = margin / 2.0
        self.oAvgNegDist = margin / 2.0
        self.alpha = 1.0
        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        
        # check if this is a new epoch
        if( avgPosDist != self.oAvgPosDist ):
            
            self.oAvgPosDist = avgPosDist
            self.oAvgNegDist = avgNegDist
            
            self.alpha = self.margin / (self.oAvgPosDist + self.oAvgNegDist)
            self.alpha = max(0.85, self.alpha)
            self.alpha = min(self.alpha, 1.15)
        
            print("\n\nIts a new EPOCH in WEIGTHED LOSS, new alpha = {:.4f}\n\n".format(self.alpha))
            
        return LF.weighted_contrastive_loss(x, label, avgPosDist, avgNegDist, self.alpha, 
                                            margin=self.margin, eps=self.eps, Lw=Lw)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
    
class ExponentialLoss(nn.Module):
    
    r"""
    """

    def __init__(self, gamma=1.0, eps=1e-6):
        super(ExponentialLoss, self).__init__()      
        self.eps = eps
        self.gamma = gamma

        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
            
        return LF.exponential_loss(x, label, Lw, gamma=self.gamma, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
class AngleExponentialLoss(nn.Module):
    
    r"""
    """

    def __init__(self, eps=1e-3):
        super(AngleExponentialLoss, self).__init__()      
        self.eps = eps

        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
            
        return LF.angle_exponential_loss(x, label, Lw, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
class AggregateExponentialLoss(nn.Module):
    
    r"""
    The loss from the paper:
        Tomás Martínez-Cortés, Iván González-Díaz, Fernando Díaz-de-María,
        Training Deep Retrieval Models with Noisy Datasets: Bag Exponential Loss,
        Pattern Recognition,
        2021,
        107811,
        ISSN 0031-3203,
        https://doi.org/10.1016/j.patcog.2020.107811.
    """

    def __init__(self, alpha=1.0, beta=0.0, gamma=0.8, drop_loss=0, drop_loss_freq=0, eps=1e-6):
        super(AggregateExponentialLoss, self).__init__()      
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.drop_loss = drop_loss
        self.drop_loss_freq = drop_loss_freq
        self.eps = eps
        self.count = 0
        self.idx = None
        print('Creating aggExpLoss with gamma: {}, alpha: {}, beta: {}'.format(
            self.gamma, self.alpha, self.beta))

        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        
        updateDropped = False
        if( (self.count % self.drop_loss_freq) == 0 ):
            updateDropped = True
            self.count = 0
        
        self.count += 1
        
        if( self.drop_loss > 0 ):
            if( updateDropped ):
                numFeat = x.size()[0]
                self.idx = torch.rand(numFeat) > self.drop_loss

            x = x[self.idx, :]        

            
        return LF.aggregate_exponential_loss(x, label, Lw, gamma=self.gamma, 
            alpha=self.alpha, beta=self.beta, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
    
class RankingTripletExponential(nn.Module):
    
    r"""
    """

    def __init__(self, gamma=1.0, alpha=1.0, beta=0.0, drop_loss=0, drop_loss_freq=0, eps=1e-6):
        super(RankingTripletExponential, self).__init__()      
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.drop_loss = drop_loss
        self.drop_loss_freq = drop_loss_freq
        self.eps = eps
        self.count = 0
        self.idx = None
        print('Creating rankingTripletExponential with gamma: {}, alpha: {}, beta: {}'.format(
            self.gamma, self.alpha, self.beta))

        
    def forward(self, x, label, avgPosDist=None, avgNegDist=None, Lw=None):
        
        updateDropped = False
        if( (self.count % self.drop_loss_freq) == 0 ):
            updateDropped = True
            self.count = 0
        
        self.count += 1
        
        if( self.drop_loss > 0 ):
            if( updateDropped ):
                numFeat = x.size()[0]
                self.idx = torch.rand(numFeat) > self.drop_loss

            x = x[self.idx, :]        

            
        return LF.ranking_triplet_exponential(x, label, Lw, gamma=self.gamma, 
            alpha=self.alpha, beta=self.beta, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + str(self.margin) + ')'
    
    

    
    
    
    
    
    
