import os
import pickle
import itertools

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root
from cirtorch.utils.whiten import whitenlearn, pcawhitenlearn

#import matplotlib.pyplot as plt
from PIL import Image
from cirtorch.networks.imageretrievalnet import init_network
import torchvision.transforms as transforms

import sys

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
class TuplesDatasetSfM(data.Dataset):
    """Data loader that loads training and validation tuples of 
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, pnum=1, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        # setting up paths
        data_root = get_data_root()
        db_root = os.path.join(data_root, 'train', name)
        ims_root = os.path.join(db_root, 'ims')

        # loading db
        db_fn = os.path.join(db_root, '{}.pkl'.format(name))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)[mode]

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']

        ## If we want to keep only unique q-p pairs 
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.pnum = pnum
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.avgPosDist = None
        self.avgNegDist = None
        
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))
        
        # positive images
        #output.append(self.loader(self.images[self.pidxs[index]]))
        for i in range(len(self.pidxs[index])):
            output.append(self.loader(self.images[self.pidxs[index][i]]))
            
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1] + [1]*len(self.pidxs[index]) + [0]*len(self.nidxs[index]))

        return output, target

    def __len__(self):
        if not self.qidxs:
            return 0
        return len(self.qidxs)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def create_epoch_tuples(self,net, numEpoch=None):
        
        if( self.pnum == 1 ):
            self.create_epoch_tuples_single_positive(net, numEpoch)
        else:
            self.create_epoch_tuples_multiple_positives(net)
        return

    def create_epoch_tuples_single_positive(self, net, numEpoch, print_freq=1000):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.qidxs = []
        self.pidxs = []
        #self.qidxs = [self.qpool[i] for i in idxs2qpool]
        #self.pidxs = [self.ppool[i] for i in idxs2qpool]
        
        # find all positives to each query
        for q in qsel:
            
            pos = []
            for i,x in enumerate(self.qpool):
                if( q == x ):
                    pos.append(self.ppool[i])
                    
            # select pnum positives
            self.qidxs.append(q)
            self.pidxs.append([pos[i] for i in torch.randperm(len(pos))[:self.pnum]])
            
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------
    
        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return
    
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for query images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(self.qidxs)).cuda()
        
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        print('>> Extracting descriptors for the first positive image...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i[0]] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        pvecs = torch.Tensor(net.meta['outputdim'], len(self.pidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
            pvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        ##############
        # WHITENING ##
        ##############
        #print("\ncomputing WHITENING inside TuplesDatasetSfM\n")
        #q = []
        #p = list( range(qvecs.size()[1], qvecs.size()[1]+pvecs.size()[1]) )
        #for i in range(len(self.qidxs)):
        #    numPos = len(self.pidxs[i])
        #    q.extend( [i]*numPos )
        #
        #X = np.hstack( (qvecs.cpu().numpy(), pvecs.cpu().numpy(), poolvecs.cpu().numpy()) )
        #try:
        #    m,P = whitenlearn(X, q, p)
        #except:
        #    m,P = pcawhitenlearn(X)
        #
        #net.meta['Lw']['m'] = torch.from_numpy(m).cuda()
        #net.meta['Lw']['P'] = torch.from_numpy(P).cuda()
        
        
        
        
    
#        print('>> {}: Learning whitening...'.format('retrieval-SfM-30k'))
#
#        # loading db
#        db_root = os.path.join(get_data_root(), 'train', 'retrieval-SfM-30k')
#        ims_root = os.path.join(db_root, 'ims')
#        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format('retrieval-SfM-30k'))
#        with open(db_fn, 'rb') as f:
#            db = pickle.load(f)
#        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
#        
#        # extract whitening vectors
#        print('>> {}: Extracting...'.format('retrieval-SfM-30k'))
#        wvecs = extract_vectors(net, images, self.imsize, self.transform)
#
#        # learning whitening 
#        print('>> {}: Learning...'.format('retrieval-SfM-30k'))
#        wvecs = wvecs.numpy()        
#        #m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
#    
#        #net.meta['Lw']['m'] = torch.from_numpy(m).cuda()
#        #net.meta['Lw']['P'] = torch.from_numpy(P).cuda()    
#        print('finished learning whitening')
#        np.save('results/wvecs/wvecs_epoch_{}.npy'.format(numEpoch), wvecs)
        
        
    
        
        
        
        
        ##############################
    
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        for q in range(len(self.qidxs)):
            qcluster = self.clusters[self.qidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < self.nnum:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters and potential not in nidxs:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs)
            
        # Finally, update the average pos and negative dist
        dif = qvecs - pvecs
        D = torch.pow(dif+1e-6, 2).sum(dim=0).sqrt()
        self.avgPosDist = D.mean()
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {:.4f}'.format(self.avgPosDist))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
        return
    
    def create_epoch_tuples_multiple_positives(self,net, print_freq=1000):
        
        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = []
     
        # find all positives to each query
        for q in qsel:
            
            pos = []
            for i,x in enumerate(self.qpool):
                if( q == x ):
                    pos.append(self.ppool[i])
                    
            # select pnum positives
            self.pidxs.append([q]+[pos[i] for i in torch.randperm(len(pos))[:self.pnum-1]])
            
        
        # now, lets unroll the list of lists in idx into a single list to find 
        # a hard negative for each positive. We need to keep the info about the
        # positives grouping
        lengths = [len(i) for i in self.pidxs]
        epidxs = list(itertools.chain(*self.pidxs))
        
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for positive images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in epidxs], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(epidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(epidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        ##############
        # WHITENING ##
        ##############
        #print("\ncomputing WHITENING inside TuplesDatasetSfM\n")
        #q = []
        #p = list( range(qvecs.size()[1], qvecs.size()[1]+pvecs.size()[1]) )
        #for i in range(len(self.qidxs)):
        #    numPos = len(self.pidxs[i])
        #    q.extend( [i]*numPos )
        #
        #X = np.hstack( (qvecs.cpu().numpy(), pvecs.cpu().numpy(), poolvecs.cpu().numpy()) )
        #try:
        #    m,P = whitenlearn(X, q, p)
        #except:
        #    m,P = pcawhitenlearn(X)
        #
        #net.meta['Lw']['m'] = torch.from_numpy(m).cuda()
        #net.meta['Lw']['P'] = torch.from_numpy(P).cuda()
    
        #print('>> {}: Learning whitening...'.format('retrieval-SfM-30k'))
        #
        ## loading db
        #db_root = os.path.join(get_data_root(), 'train', 'retrieval-SfM-30k')
        #ims_root = os.path.join(db_root, 'ims')
        #db_fn = os.path.join(db_root, '{}-whiten.pkl'.format('retrieval-SfM-30k'))
        #with open(db_fn, 'rb') as f:
        #    db = pickle.load(f)
        #images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        #
        ## extract whitening vectors
        #print('>> {}: Extracting...'.format('retrieval-SfM-30k'))
        #wvecs = extract_vectors(net, images, self.imsize, self.transform)
        #
        ## learning whitening 
        #print('>> {}: Learning...'.format('retrieval-SfM-30k'))
        #wvecs = wvecs.numpy()        
        #m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        #
        #net.meta['Lw']['m'] = torch.from_numpy(m).cuda()
        #net.meta['Lw']['P'] = torch.from_numpy(P).cuda()    
        #print('finished learning whitening')
        
        
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        
        self.nidxs = []
        numIm = 0
        
        for group in self.pidxs:
            npos = len(group)
            queries = [ numIm+i for i in range(npos) ]
            
            nidxs = []
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            clusters = [self.clusters[epidxs[q]] for q in queries] 
            
            for q in queries:
                added = 0
                r = 0
                
                while added == 0:
                    potential = idxs2images[ranks[r, q]]
                    if not self.clusters[potential] in clusters and potential not in nidxs:
                        nidxs.append(potential)
                        added += 1
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
            self.nidxs.append(nidxs)
            numIm = numIm + npos
        
        
    
        # for compatibility with other code, lets label the first positive as query
        self.qidxs = []
        for l in self.pidxs:
            self.qidxs.append(l.pop(0))
        
        # print some info
        self.avgPosDist = -1.0
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {}'.format('Not computed'))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
        return
    
    
    

    
    
    def create_epoch_tuples_multiple_positivesOld(self, net, print_freq=1000):
        
        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = []
     
        # find all positives to each query
        for q in qsel:
            
            pos = []
            for i,x in enumerate(self.qpool):
                if( q == x ):
                    pos.append(self.ppool[i])
                    
            # select pnum positives
            self.pidxs.append([q]+[pos[i] for i in torch.randperm(len(pos))[:self.pnum-1]])
            
        
        # now, lets unroll the list of lists in idx into a single list to find 
        # a hard negative for each positive. We need to keep the info about the
        # positives grouping
        lengths = [len(i) for i in self.pidxs]
        epidxs = list(itertools.chain(*self.pidxs))
        
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for positive images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in epidxs], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(epidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(epidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        for q in range(len(epidxs)):
            qcluster = self.clusters[epidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < 1:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs[0])
        
        # re-nest the negatives list 
        self.nidxs = self.renestList(self.nidxs, lengths)
        
        # for compatibility with other code, lets label the first positive as query
        self.qidxs = []
        for l in self.pidxs:
            self.qidxs.append(l.pop(0))
        
        # print some info
        self.avgPosDist = -1.0
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {}'.format('Not computed'))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
        
        return
    
    def renestList( self, extendedList, lengths):
        
        rebuild = []
        start = 0
        for l in lengths:
            rebuild.append(extendedList[start:l+start])
            start += l
        
        return rebuild
        

    
class TuplesDatasetGoogleLandmarks(data.Dataset):
    
    def __init__(self, name, imsize=None, pnum=10, nnum=1, qsize=2000, poolsize=20000, 
                 transform=None, loader=default_loader):

        # setting up paths
        data_root = get_data_root()
        ims_root = os.path.join(data_root, 'train', name)
    
        images = []
        clusters = []
        nClusters = 0
        nImages = 0
        db = {}
        
        categories = sorted(listdir_nohidden(ims_root)) 
        #break_ = False
        for c,cat in enumerate(categories):
            
            #if( break_ ):
            #    break
            
            subcategories = sorted(listdir_nohidden(os.path.join(ims_root, cat)))
            for i,subcat in enumerate(subcategories):
                
                # list all images in a particular subcategory and build paths
                imfiles = os.listdir(os.path.join(ims_root, cat, subcat))
                impaths = [ os.path.join(ims_root, cat, subcat, i) for i in imfiles]
                
                # store paths and cluster id (or subcategory id)
                images.extend(impaths)
                clusters.extend( [nClusters]*len(impaths) )
                
                # keep an indexed dict
                db[nClusters] = [i for i in range(nImages, nImages+len(impaths))]
                nClusters += 1
                nImages = nImages+len(impaths)
                
                #if( nImages > 110000 ):
                #    break_ = True
                #    break
                
        # initializing tuples dataset
        self.name = name
        self.imsize = imsize
        self.images = images
        self.clusters = clusters
        self.db = db
        self.qpool = None
        self.ppool = None

        # size of training subset for an epoch
        self.pnum = pnum
        self.nnum = nnum 
        self.qsize = min(qsize, 10000)
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.avgPosDist = None
        self.avgNegDist = None
        
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        try:
            output.append(self.loader(self.images[self.qidxs[index]]))
        except:
            print(len(self.qidxs))
            print("index: {}".format(index))
            
        # positive images
        for i in range(len(self.pidxs[index])):
            output.append(self.loader(self.images[self.pidxs[index][i]]))
            
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

#        if self.imsize is not None:
#            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1] + [1]*len(self.pidxs[index]) + [0]*len(self.nidxs[index]))
        
        return output, target

    def __len__(self):
        if not self.qidxs:
            return 0
        return len(self.qidxs)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {}\n'.format(self.name)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of positives per tuple: {}\n'.format(self.pnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def create_epoch_tuples(self,net, numEpoch=None):
        
        if( self.pnum == 1 ):
            self.create_epoch_tuples_single_positive(net)
        else:
            self.create_epoch_tuples_multiple_positives(net)
        return
    
    def create_epoch_tuples_single_positive(self, net, print_freq=1000):

        print('>> Creating tuples for an epoch of {}...'.format(self.name))

        # pick qsize (q,p) pairs randomly from the clusters
        randomCats = torch.LongTensor(self.qsize).random_(0,self.clusters[-1])        
        self.qidxs = []
        self.pidxs = []
        
        for rc in randomCats:
            rc = rc.item()
            
            if( len(self.db[rc]) < 2 ):
                continue
            
            pnum = min(self.pnum, len(self.db[rc])-1 )
            qpos = torch.randperm(len(self.db[rc]))[:pnum+1]
            self.qidxs.append(self.db[rc][qpos[0]])
            self.pidxs.append([self.db[rc][qpos[i+1]] for i in range(pnum)])
            
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for query images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(self.qidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        print('>> Extracting descriptors for the first positive image...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i[0]] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        pvecs = torch.Tensor(net.meta['outputdim'], len(self.pidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
            pvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        for q in range(len(self.qidxs)):
            qcluster = self.clusters[self.qidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < self.nnum:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs)
                        
            
        # Finally, update the average pos and negative dist
        dif = qvecs - pvecs
        D = torch.pow(dif+1e-6, 2).sum(dim=0).sqrt()
        self.avgPosDist = D.mean()
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {:.4f}'.format(self.avgPosDist))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> The avg pos distance is computed using only one positive (p1)')
        print('>>>> Done')
        
        return
    
    def create_epoch_tuples_multiple_positives(self, net, print_freq=1000):
        
        # create epoch tuples for self.pnum > 1
        if self.pnum != self.nnum:
            print(Warning('For num positives > 1, num negatives is forced to be equal to num positives'))
            self.nnum = self.pnum
        
        print('>> Creating tuples for an epoch of {}...'.format(self.name))
    
        if(self.name == 'Landmarks' ):
            # select qsize random clusters, and pick from each pnum images randomly (if available)
            counts = np.array([len(self.db[i]) for i in self.db.keys()])
            weights = counts / np.sum(counts)
            randomCats = np.random.choice(range(0,self.clusters[-1]+1), size=self.qsize, replace=True, p=weights)
            randomCats = torch.LongTensor(randomCats)
        else:
            randomCats = torch.LongTensor(self.qsize).random_(0,self.clusters[-1])        
        
#        counts = np.array([len(self.db[i]) for i in self.db.keys()])
#        weights = counts / np.sum(counts)
#        randomCats = np.random.choice(range(0,self.clusters[-1]+1), size=self.qsize, replace=True, p=weights)
#        randomCats = torch.LongTensor(randomCats)        
        
        self.pidxs = []
        
        for rc in randomCats:
            rc = rc.item()
            # at least we need two positives (and one negative) to compute a cost
            if( len(self.db[rc]) < 2 ):
                continue
            
            maxPos = len(self.db[rc])
            if( self.pnum * 10 < maxPos ):
                pnum = self.pnum
            else:
                pnum = max(min(3,maxPos), int(maxPos*0.1))
                
            #pnum = min(self.pnum, len(self.db[rc]))
            selpos = torch.randperm(len(self.db[rc]))[:pnum]
            self.pidxs.append([self.db[rc][selpos[i]] for i in range(pnum)])
            
            
        # now, lets unroll the list of lists in idx into a single list to find 
        # a hard negative for each positive. We need to keep the info about the
        # positives grouping
        lengths = [len(i) for i in self.pidxs]
        epidxs = list(itertools.chain(*self.pidxs))
        
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for positive images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in epidxs], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(epidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(epidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], 
            imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        mblist = []
        cs = np.cumsum(lengths)
        for q in range(len(epidxs)):
            
            # this list is to check that for the same minibatch we do not get 
            # the same negative sample several times
            if(np.sum(q==cs)):
                mblist = []
            
            qcluster = self.clusters[epidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < 1:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters and not potential in mblist:
                    
                    mblist.append(potential)
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs[0])
        
        # re-nest the negatives list 
        self.nidxs = self.renestList(self.nidxs, lengths)
        
        # for compatibility with other code, lets label the first positive as query
        self.qidxs = []
        for l in self.pidxs:
            self.qidxs.append(l.pop(0))
        
        # print some info
        self.avgPosDist = -1.0
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {}'.format('Not computed'))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
    
    def renestList( self, extendedList, lengths):
        
        rebuild = []
        start = 0
        for l in lengths:
            rebuild.append(extendedList[start:l+start])
            start += l
        
        return rebuild
    

class TuplesDatasetRGBD(data.Dataset):
    
    def __init__(self, name, imsize=None, pnum=10, nnum=1, qsize=2000, poolsize=20000, 
                 transform=None, loader=default_loader):

        # setting up paths
        data_root = get_data_root()
        ims_root = os.path.join(data_root, 'train', name)

        images = []
        clusters = []
        imCategories = []
        nClusters = 0
        nImages = 0
        db = {}
        
        categories = sorted(os.listdir(ims_root))        
        subcategories = [ sorted(os.listdir(os.path.join(ims_root, cat))) for cat in categories]
        for c,cat in enumerate(categories):
            
            subcategories = sorted(os.listdir(os.path.join(ims_root, cat)))
            for i,subcat in enumerate(subcategories):
                
                # list all images in a particular subcategory and build paths
                imfiles = os.listdir(os.path.join(ims_root, cat, subcat))
                imfiles = [ imf for imf in imfiles if '_crop' in imf]
                impaths = [ os.path.join(ims_root, cat, subcat, i) for i in imfiles]
                
                # store paths and cluster id (or subcategory id)
                images.extend(impaths)
                clusters.extend( [nClusters]*len(impaths) )
                imCategories.extend( [c]*len(impaths) )
                
                # keep an indexed dict
                db[nClusters] = [i for i in range(nImages, nImages+len(impaths))]
                nClusters += 1
                nImages = nImages+len(impaths)
                
        # initializing tuples dataset
        self.name = name
        self.imsize = imsize
        self.images = images
        self.clusters = clusters
        self.imCategories = imCategories
        self.db = db
        self.qpool = None
        self.ppool = None

        # size of training subset for an epoch
        self.nnum = nnum
        self.pnum = pnum
        self.qsize = min(qsize, 10000)
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.avgPosDist = None
        self.avgNegDist = None
        
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))
        
        # positive images
        for i in range(len(self.pidxs[index])):
            output.append(self.loader(self.images[self.pidxs[index][i]]))
            
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1] + [1]*len(self.pidxs[index]) + [0]*len(self.nidxs[index]))

        return output, target

    def __len__(self):
        if not self.qidxs:
            return 0
        return len(self.qidxs)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {}\n'.format(self.name)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of positives per tuple: {}\n'.format(self.pnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def create_epoch_tuples(self, net):

        print('>> Creating tuples for an epoch of {}...'.format(self.name))

        # pick qsize (q,p) pairs randomly from the clusters
        randomCats = torch.LongTensor(self.qsize).random_(0,self.clusters[-1])        
        self.qidxs = []
        self.pidxs = []
        
        for rc in randomCats:
            rc = rc.item()
            # we need q + pnum images from each category
            qpos = torch.randperm(len(self.db[rc]))[:self.pnum+1]
            self.qidxs.append(self.db[rc][qpos[0]])
            self.pidxs.append([self.db[rc][qpos[i+1]] for i in range(self.pnum)])
            
        # pick poolsize negatives too
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for query images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qvecs = torch.Tensor(net.meta['outputdim'], len(self.qidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        print('>> Extracting descriptors for the first positive image...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i[0]] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        pvecs = torch.Tensor(net.meta['outputdim'], len(self.pidxs)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
            pvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        for q in range(len(self.qidxs)):
            qcluster = self.clusters[self.qidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < self.nnum:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs)
                        
            
        # Finally, update the average pos and negative dist
        dif = qvecs - pvecs
        D = torch.pow(dif+1e-6, 2).sum(dim=0).sqrt()
        self.avgPosDist = D.mean()
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {:.4f}'.format(self.avgPosDist))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> The avg pos distance is computed using only one positive (p1)')
        print('>>>> Done')
    

def mmByParts( poolvecs, qvecs, qsize, maxRank=1000 ):
    
    step = 1000 if qsize > 1000 else qsize
    start = 0
    end = int(qvecs.size()[1])
    scoresR = torch.cuda.FloatTensor(maxRank, end)
    ranksR = torch.cuda.LongTensor(maxRank, end)
    
    for i in range(int(np.ceil(end/step))):
        
        e = start+step
        e = e if e < end else end
        #print('[:,{}:{}] '.format(start, e))
        
        # compute scores and ranks for the portion of queries
        scores_i = torch.mm(poolvecs.t(), qvecs[:,start:e])
        scores_i, ranks_i = torch.sort(scores_i, dim=0, descending=True)
        
        # store the top maxRank results
        scoresR[:,start:e] = scores_i[0:maxRank]
        ranksR[:,start:e] = ranks_i[0:maxRank]
        
        # update the start point
        start = e
        
    return scoresR, ranksR

class TuplesDatasetSfMGL(data.Dataset):
    
    def __init__(self, name, mode, imsize=None, pnum=1, nnum=5, qsize=2000, poolsize=20000, 
                transform=None, loader=default_loader):
            
        self.SfM = TuplesDatasetSfM(name[0], mode, imsize=imsize, pnum=pnum[0], nnum=nnum[0], 
            qsize=int(qsize/2), poolsize=int(poolsize/2), transform=transform, loader=loader)
        
        self.GL = TuplesDatasetGoogleLandmarks(name[1], imsize=imsize, pnum=pnum[1], nnum=nnum[1], 
            qsize=int(qsize/2), poolsize=int(poolsize/2), transform=transform, loader=loader)
        
        self.SfMTurn = True
        self.avgPosDist = self.SfM.avgPosDist
        self.avgNegDist = self.SfM.avgNegDist
        return
    
    
    def __getitem__(self, index):
        
        #print("index: {}".format(index))
        if( index % 2 == 0 ):
            
            #print("Getting {} from SfM".format(int(index/2)))
            #sys.stdout.flush()
            return self.SfM.__getitem__(int(index/2))
        else:
            
            #print("Getting {} from GL".format(int(index/2)))
            #sys.stdout.flush()
            return self.GL.__getitem__(int(index/2))

    
    def __len__(self):
        #print("lens: {}".format(self.SfM.__len__()))
        #print("leng: {}".format(self.GL.__len__()))
        return min(self.SfM.__len__()*2, self.GL.__len__()*2)

    def __repr__(self):
    
        if( self.SfMTurn ):
            return self.SfM.__repr__()
        else:
            return self.GL.__repr__()
        
    def create_epoch_tuples(self, net, numEpoch=None):
        
        self.SfM.create_epoch_tuples( net, numEpoch=None)
        self.GL.create_epoch_tuples( net, numEpoch=None)
        

if( __name__ == '__main__' ):
    

    # Google-Landmarks test
    # rgbd_test
    # retrieval-SfM-120k test
    
    
    kargs = {'model': 'resnet101', 'pooling': 'gem', 'whitening': False }
    net = init_network(**kargs, pretrained=True)
        
    normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    
    self = TuplesDatasetGoogleLandmarks('Google-Landmarks', imsize=362, pnum=10, nnum=10, 
        qsize=200, poolsize=2000, transform=transform)
    
    # randomcrop
    from PIL import Image
    from cirtorch.datasets.customTransforms import CustomCenterCrop, CustomResize
    img = Image.open(self.images[3])


    transform = transforms.Compose([
        transforms.RandomAffine(20, shear=20, resample=Image.BICUBIC),
        CustomCenterCrop(0.2),
        CustomResize(362),
        transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.5 ,hue=0.05),
        transforms.RandomGrayscale(p=0.05)
        #transforms.ToTensor(),
        #normalize,
    ])

    nimg = transform(img)
    nimg
    
    
#    transform2 = transforms.Compose([
#        transforms.RandomAffine(20, shear=20, resample=Image.BICUBIC),
#        CustomCenterCrop(0.2),
#        CustomResize(100),
#        #transforms.Resize(362),
#
#        transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.5 ,hue=0.05),
#        transforms.RandomGrayscale(p=0.05),
#        transforms.ToTensor(),
#        normalize,
#    ])
#    
#    nnimg = transform2(img)
#    output = net(nnimg.unsqueeze_(0))
    
    #print(nimg.size)
#    lengths = []
#    for key in db.keys():
#        l = len(db[key])
#        if(l > 100):
#            l = 100
#        lengths.append(l)
#    
#    plt.hist(lengths)
#    plt.xlim([0,100])
    #self = TuplesDatasetSfM('retrieval-SfM-120k', 'train', imsize=362, pnum=1, nnum=1, 
    #    qsize=20, poolsize=1001, transform=transform)
    
#    self = TuplesDatasetSfMGL(['retrieval-SfM-120k', 'Google-Landmarks'], 'train', 
#        imsize=362, pnum=1, nnum=1, qsize=20, poolsize=1001, transform=transform)
        
#    self.create_epoch_tuples(net)
    
    
    

#    for i in range(len(self)):
#
#        q = self.qidxs[i]
#        p = self.pidxs[i]
#        n = self.nidxs[i]
#        
#        pnum = len(p)
#        nnum = len(n)
#        
#        d = 0
#        plt.figure(figsize=(28,14))
#        ax = []
#        for z in range(nnum+pnum+1):        
#            #ax.append(fig.add_subplot(1,7,i+1))
#            ax.append(plt.subplot(1,nnum+pnum+1,z+1))
#            
#        
#        
#        # plot the query
#        ax[d].imshow(Image.open(self.images[q]))
#        print(self.images[q])
#        ax[0].set_title("Query")
#        d = d + 1
#        
#        # plot the positives
#        for j in range(pnum):
#            ax[j+d].imshow(Image.open(self.images[p[j]]))
#            print(self.images[p[j]])
#            ax[j+d].set_title("P_{}".format(j+1))
#        d = d + j + 1
#        
#        # plot the negatives
#        for j in range(nnum):
#            ax[j+d].imshow(Image.open(self.images[n[j]]))
#            print(self.images[n[j]])
#            ax[j+d].set_title("N_{}".format(j+1))
#        d = d + j + 1
#        
#        plt.show()
#        input("press to continue")
#    
#    
#    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    

    
    
    
    

































































#
#
#
#class TuplesDatasetSfM(data.Dataset):
#    """Data loader that loads training and validation tuples of 
#        Radenovic etal ECCV16: CNN image retrieval learns from BoW
#
#    Args:
#        name (string): dataset name: 'retrieval-sfm-120k'
#        mode (string): 'train' or 'val' for training and validation parts of dataset
#        imsize (int, Default: None): Defines the maximum size of longer image side
#        transform (callable, optional): A function/transform that  takes in an PIL image
#            and returns a transformed version. E.g, ``transforms.RandomCrop``
#        loader (callable, optional): A function to load an image given its path.
#        nnum (int, Default:5): Number of negatives for a query image in a training tuple
#        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
#        poolsize (int, Default:10000): Pool size for negative images re-mining
#
#     Attributes:
#        images (list): List of full filenames for each image
#        clusters (list): List of clusterID per image
#        qpool (list): List of all query image indexes
#        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool
#
#        qidxs (list): List of qsize query image indexes to be processed in an epoch
#        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
#        nidxs (list): List of qsize tuples of negative images
#                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs
#
#        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
#            ie new q-p pairs are picked and negative images are remined
#    """
#
#    def __init__(self, name, mode, imsize=None, pnum=1, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader):
#
#        if not (mode == 'train' or mode == 'val'):
#            raise(RuntimeError("MODE should be either train or val, passed as string"))
#
#        # setting up paths
#        data_root = get_data_root()
#        db_root = os.path.join(data_root, 'train', name)
#        ims_root = os.path.join(db_root, 'ims')
#
#        # loading db
#        db_fn = os.path.join(db_root, '{}.pkl'.format(name))
#        with open(db_fn, 'rb') as f:
#            db = pickle.load(f)[mode]
#
#        # initializing tuples dataset
#        self.name = name
#        self.mode = mode
#        self.imsize = imsize
#        self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
#        self.clusters = db['cluster']
#        self.qpool = db['qidxs']
#        self.ppool = db['pidxs']
#
#        ## If we want to keep only unique q-p pairs 
#        ## However, ordering of pairs will change, although that is not important
#        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
#        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
#        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]
#
#        # size of training subset for an epoch
#        self.pnum = pnum
#        self.nnum = nnum
#        self.qsize = min(qsize, len(self.qpool))
#        self.poolsize = min(poolsize, len(self.images))
#        self.qidxs = None
#        self.pidxs = None
#        self.nidxs = None
#
#        self.avgPosDist = None
#        self.avgNegDist = None
#        
#        self.transform = transform
#        self.loader = loader
#
#    def __getitem__(self, index):
#        
#        """
#        Args:
#            index (int): Index
#
#        Returns:
#            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
#        """
#        if self.__len__() == 0:
#            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))
#
#        output = []
#        # query image
#        output.append(self.loader(self.images[self.qidxs[index]]))
#        
#        # positive images
#        #output.append(self.loader(self.images[self.pidxs[index]]))
#        for i in range(len(self.pidxs[index])):
#            output.append(self.loader(self.images[self.pidxs[index][i]]))
#            
#        # negative images
#        for i in range(len(self.nidxs[index])):
#            output.append(self.loader(self.images[self.nidxs[index][i]]))
#
#        if self.imsize is not None:
#            output = [imresize(img, self.imsize) for img in output]
#        
#        if self.transform is not None:
#            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]
#
#        target = torch.Tensor([-1] + [1]*len(self.pidxs[index]) + [0]*len(self.nidxs[index]))
#
#        return output, target
#
#    def __len__(self):
#        if not self.qidxs:
#            return 0
#        return len(self.qidxs)
#
#    def __repr__(self):
#        fmt_str = self.__class__.__name__ + '\n'
#        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
#        fmt_str += '    Number of images: {}\n'.format(len(self.images))
#        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
#        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
#        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
#        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
#        tmp = '    Transforms (if any): '
#        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#        return fmt_str
#
#    def create_epoch_tuples(self, net):
#
#        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
#
#        ## ------------------------
#        ## SELECTING POSITIVE PAIRS
#        ## ------------------------
#    
#        # draw qsize random queries for tuples
#        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
#        qsel = [self.qpool[i] for i in idxs2qpool]
#        self.qidxs = []
#        self.pidxs = []
#        #self.qidxs = [self.qpool[i] for i in idxs2qpool]
#        #self.pidxs = [self.ppool[i] for i in idxs2qpool]
#        
#        # find all positives to each query
#        for q in qsel:
#            
#            pos = []
#            for i,x in enumerate(self.qpool):
#                if( q == x ):
#                    pos.append(self.ppool[i])
#                    
#            # select pnum positives
#            self.qidxs.append(q)
#            self.pidxs.append([pos[i] for i in torch.randperm(len(pos))[:self.pnum]])
#            
#        ## ------------------------
#        ## SELECTING NEGATIVE PAIRS
#        ## ------------------------
#    
#        # if nnum = 0 create dummy nidxs
#        # useful when only positives used for training
#        if self.nnum == 0:
#            self.nidxs = [[] for _ in range(len(self.qidxs))]
#            return
#    
#        # pick poolsize negatives too
#        idxs2images = torch.randperm(len(self.images))[:self.poolsize]
#            
#        # prepare network
#        net.cuda()
#        net.eval()
#    
#        print('>> Extracting descriptors for query images...')
#        loader = torch.utils.data.DataLoader(
#            ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
#            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
#        )
#        qvecs = torch.Tensor(net.meta['outputdim'], len(self.qidxs)).cuda()
#        for i, input_ in enumerate(loader):
#            print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
#            qvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
#        print('')
#        
#        print('>> Extracting descriptors for the first positive image...')
#        loader = torch.utils.data.DataLoader(
#            ImagesFromList(root='', images=[self.images[i[0]] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
#            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
#        )
#        pvecs = torch.Tensor(net.meta['outputdim'], len(self.pidxs)).cuda()
#        for i, input_ in enumerate(loader):
#            print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
#            pvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
#        print('')
#    
#        print('>> Extracting descriptors for negative pool...')
#        loader = torch.utils.data.DataLoader(
#            ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
#            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
#        )
#        poolvecs = torch.Tensor(net.meta['outputdim'], len(idxs2images)).cuda()
#        for i, input_ in enumerate(loader):
#            print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
#            poolvecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
#        print('')
#    
#        print('>> Searching for hard negatives...')
#        scores = torch.mm(poolvecs.t(), qvecs)
#        scores, ranks = torch.sort(scores, dim=0, descending=True)
#        self.nidxs = []
#        for q in range(len(self.qidxs)):
#            qcluster = self.clusters[self.qidxs[q]]
#            clusters = [qcluster]
#            nidxs = []
#            r = 0
#            avg_ndist = torch.Tensor([0]).cuda()
#            n_ndist = torch.Tensor([0]).cuda()
#            while len(nidxs) < self.nnum:
#                potential = idxs2images[ranks[r, q]]
#                # take at most one image from the same cluster
#                if not self.clusters[potential] in clusters:
#                    nidxs.append(potential)
#                    clusters.append(self.clusters[potential])
#                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
#                    n_ndist += 1
#                r += 1
#            self.nidxs.append(nidxs)
#            
#        # Finally, update the average pos and negative dist
#        dif = qvecs - pvecs
#        D = torch.pow(dif+1e-6, 2).sum(dim=0).sqrt()
#        self.avgPosDist = D.mean()
#        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
#        print('>>>> Average positive distance: {:.4f}'.format(self.avgPosDist))
#        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
#        print('>>>> Done')
#        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
