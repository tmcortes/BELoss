import argparse
import os
import shutil
import time
import math
import pickle
import copy
import subprocess
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.layers.loss import (TripletLoss, ContrastiveLoss, RootContrastiveLoss, 
                                  WeightedContrastiveLoss, ExponentialLoss, 
                                  AngleExponentialLoss, AggregateExponentialLoss, 
                                  RankingTripletExponential)
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.datasets.customTransforms import CustomCenterCrop, CustomResize
from cirtorch.datasets.traindataset import TuplesDatasetSfM, TuplesDatasetRGBD, TuplesDatasetGoogleLandmarks, TuplesDatasetSfMGL
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply, pcawhitenlearn
from cirtorch.utils.evaluate import compute_map_and_print, compute_ap
from cirtorch.utils.general import get_data_root, htime, similarity_matrix, upper_triangular_mask, pairwise_distances


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import OrderedDict

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

markers = Line2D.markers

def save_results( results, directory):
    
    # results can come with one or several test datasets performances
    if( len(results) == 1 ):
        filename = os.path.join(directory, results[0][0])
        with open(filename, 'a') as ofile:
            ofile.write('{}\n'.format(results[0][1]))
                
            
    else:
        for r in results:
            filename = os.path.join(directory, r[0])
            with open(filename, 'a') as ofile:
                ofile.write('{}\n'.format(r[1]))
            
    return

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std
        m.eval()
        # freeze parameters
        #for p in m.parameters():
        #   p.requires_grad = False

def setRandomSeeds( s ):
    
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    return

def createExportDir( eConfig, expName ):
    
    directory = "{}".format(expName)
    directory += "_{}".format(time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime()))
    
    directory = os.path.join(eConfig['resultsFolder'], directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(directory))
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return directory
 
def loadModelToCuda( eConfig ):
    
    kargs = {'model': eConfig['arch'], 'pooling': eConfig['pooling'], 
            'whitening': False}#eConfig['whitening'] }
    
    if( eConfig['pretrained'] == 'imageNet' ):
        model = init_network(**kargs, pretrained=True)
    else:
        model = init_network(**kargs, pretrained=False)
  
    return model.cuda()

def defineLossFunction( eConfig ):


    if eConfig['loss'] == 'contrastive':
        criterion = ContrastiveLoss(margin=eConfig['loss-margin']).cuda()
        
    elif eConfig['loss'] == 'rootContrastive':
        criterion = RootContrastiveLoss(margin=eConfig['loss-margin']).cuda()
        
    elif eConfig['loss'] == 'wContrastive':
        criterion = WeightedContrastiveLoss( margin=eConfig['loss-margin'] ).cuda()
        
    elif eConfig['loss'] == 'exp':
        criterion = ExponentialLoss(gamma=eConfig['exp-loss-gamma']).cuda()
        
    elif eConfig['loss'] == 'angExp':
        criterion = AngleExponentialLoss().cuda()
        
    elif eConfig['loss'] == 'aggExp':
        criterion = AggregateExponentialLoss(alpha=eConfig['exp-loss-alpha'], 
            beta=eConfig['exp-loss-beta'], drop_loss=eConfig['drop-loss'], 
            drop_loss_freq=eConfig['drop-loss-freq']).cuda()
    
    elif eConfig['loss'] == 'rankExp':
        criterion = RankingTripletExponential(gamma=eConfig['exp-loss-gamma'],
            alpha=eConfig['exp-loss-alpha'], beta=eConfig['exp-loss-beta'],
            drop_loss=eConfig['drop-loss'], drop_loss_freq=eConfig['drop-loss-freq']).cuda()
        
    elif eConfig['loss'] == 'triplet':
        criterion = TripletLoss(margin=eConfig['loss-margin']).cuda()
        
    else:
        raise(RuntimeError("Loss {} not available!".format(eConfig['loss'])))
    
    return criterion

def defineOptimizer( eConfig, model ):
    
    # parameters split into features and pool (no weight decay for pooling layer)
    parameters = [
        {'params': model.features.parameters()},
        {'params': model.pool.parameters(), 'lr': eConfig['lr']*10, 'weight_decay': 0}
    ]

    # define optimizer
    if eConfig['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, eConfig['lr'], momentum=eConfig['momentum'], weight_decay=eConfig['weight-decay'])
        
    elif eConfig['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, eConfig['lr'], weight_decay=eConfig['weight-decay'])
        
    else:
        raise(RuntimeError("Optimizer {} not available!".format(eConfig['optimizer'])))
        
    return optimizer

def createDataLoading( eConfig, model ):
    
    if( eConfig['training-set'] == 'retrieval-SfM-120k'):
        return createDataLoadingRetrievalSfM( eConfig, model )
    
    elif( eConfig['training-set'] == 'Google-Landmarks' or 
          eConfig['training-set'] == 'Landmarks'):
        return createDataLoadingGoogleLandmarks( eConfig, model )
    
    elif( eConfig['training-set'] == 'rgbd-dataset'):
        return createDataLoadingRGBD( eConfig, model )
    
    elif( eConfig['training-set'] == ['retrieval-SfM-120k', 'Google-Landmarks']):
        return createDataLoadingSfMGL( eConfig, model )

    else:
        pass
    return

def createDataLoadingSfMGL( eConfig, model ):
    
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TuplesDatasetSfMGL(
        name=eConfig['training-set'],
        mode='train',
        imsize=eConfig['image-size'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        qsize= eConfig['query-size'],
        poolsize = eConfig['pool-size'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )

    
    return train_dataset, train_loader, None, None
    
def createDataLoadingRetrievalSfM( eConfig, model ):
    
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TuplesDatasetSfM(
        name=eConfig['training-set'],
        mode='train',
        imsize=eConfig['image-size'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        qsize= eConfig['query-size'],
        poolsize = eConfig['pool-size'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )

    val_dataset = TuplesDatasetSfM(
        name=eConfig['training-set'],
        mode='val',
        imsize=eConfig['image-size'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        qsize=float('Inf'),
        poolsize=float('Inf'),
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=True,
        drop_last=True, collate_fn=collate_tuples
    )
    
    return train_dataset, train_loader, val_dataset, val_loader

def createDataLoadingGoogleLandmarks( eConfig, model ):
    
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])

    transform = transforms.Compose([
        #transforms.RandomAffine(20, shear=20, resample=Image.BICUBIC),
        #CustomCenterCrop(eConfig['crop-scale']),
        CustomResize(eConfig['image-size']),
        transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.5 ,hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
    ])


    train_dataset = TuplesDatasetGoogleLandmarks(
        name=eConfig['training-set'],
        imsize=eConfig['image-size'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        qsize= eConfig['query-size'],
        poolsize = eConfig['pool-size'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )

    
    return train_dataset, train_loader, None, None

def createDataLoadingRGBD( eConfig, model ):
    
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TuplesDatasetRGBD(
        name=eConfig['training-set'],
        imsize=eConfig['image-size'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        qsize= eConfig['query-size'],
        poolsize = eConfig['pool-size'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )

    
    return train_dataset, train_loader, None, None
    
def keepGradsFromColumn(i):
            
    def hook(grad):
        g = grad[:,i].clone()
        grad *= 0
        grad[:,i] = g

    return hook
    
def switch2Train(model, pretrained):
        
        model.train()
        if pretrained is not None:
            model.apply(set_batchnorm_eval)
        return 

def train(train_loader, model, criterion, optimizer, epoch, pretrained, accumulate):
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    train_loader.dataset.create_epoch_tuples(model)    

    # switch to train mode
    model.train()
    if pretrained is not None:
        model.apply(set_batchnorm_eval)

    end = time.time()
    optimizer.zero_grad()
    for i, (input_, target) in enumerate(train_loader):
    
        # measure data loading time
        data_time.update(time.time() - end)
        nq = len(input_) # number of training tuples

        for q in range(nq):
            
            ni = len(input_[q]) # number of images in qth tuple
            output = torch.Tensor(model.meta['outputdim'], ni).cuda()
            target_var = target[q].cuda()
            
            # first, pass all tuple images in test mode to fill output
            with torch.no_grad():
                for imi in range(ni):
                    output[:, imi] = model(input_[q][imi].cuda()).squeeze()
                
            # second, pass images in train mode one by one generating the gradients
            switch2Train(model, pretrained)
            for imi in range(ni):
                
                # clone the test output, and fill one particular column at a time
                outputGrad = output.clone()
                outputGrad[:, imi] = model(input_[q][imi].cuda()).squeeze()
                outputGrad.register_hook(keepGradsFromColumn(imi))
                
                loss = criterion(outputGrad, target_var, train_loader.dataset.avgPosDist, train_loader.dataset.avgNegDist)#, Lw=model.meta['Lw'])
                losses.update(loss.item())
                loss.backward()

        # accumulated gradients for multiple batches
        if (i+1) % accumulate == 0 or (i+1) == len(train_loader):
            print("Batch {}, doing a gradient update".format(i+1))
            optimizer.step()
            optimizer.zero_grad()
       

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % eConfig['print-freq'] == 0:
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
    return losses.avg

def validate(val_loader, model, criterion, epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):

        nq = len(input_) # number of training tuples
        ni = len(input_[0]) # number of images per tuple
        output = torch.autograd.Variable(torch.Tensor(model.meta['outputdim'], nq*ni).cuda(), volatile=True)

        for q in range(nq):
            for imi in range(ni):
                # target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input_[q][imi].cuda())

                # compute output
                output[:, q*ni + imi] = model(input_var)

        target_var = torch.autograd.Variable(torch.cat(target).cuda())
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % eConfig['print-freq'] == 0:
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

def test(net, eConfig):
    
    td = eConfig['test-datasets']
    if( td != 'ukbench'):
        return testOxfordParisHolidays(net, eConfig)
    elif( eConfig['test-datasets'] == 'ukbench' ):
        return testUkbench(net, eConfig)
    else:
        pass
    return

def testUkbench(net, eConfig):
    
    #datasets = eConfig['test-datasets'].split(',')
    #results = []
    #
    #for dataset in datasets:
    #    results.append((dataset, np.random.rand(1)[0]))
    #    
    #return results
    
    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    dataset = 'ukbench'
    image_size = 362

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
        
    dbpath = os.path.join(get_data_root(), 'test', 'ukbench', 'full')
    images = [os.path.join(dbpath, 'ukbench{:05d}.jpg'.format(i)) for i in range(10200)]
    labels = np.arange(10200, dtype=np.int) // 4
    
    # extract database and query vectors
    print('>> {}: database images...'.format(dataset))
    X = extract_vectors(net, images, image_size, transform)
    
    print('>> {}: Evaluating...'.format(dataset))
    
    # rank the similarities
    X = X.numpy()
    scores = np.dot(X.T, X)
    ranks = np.argsort(-scores, axis=1)
    ranks = ranks[:,0:4]
    
    # compute the average accuracy for the first 4 entries
    ranksLabel = labels[ranks]
    accs = np.sum(ranksLabel == np.repeat(labels[:,np.newaxis], 4, axis=1), axis=1)
    avgAcc = np.mean(accs)
    print('avgAcc: {:.6f}'.format(avgAcc))
    
    return [('ukbench', avgAcc)]

def testHolidays(net, eConfig, dataset, Lw):
    
    print('>> Evaluating network on test dataset: {}'.format(dataset))

    # for testing we use image size of max 1024
    image_size = 1024

    ms = [1]
    msp = 1
    if(eConfig['multiscale']):
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]
            
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
        
    # read the images and generate paths and queries-positive indexes
    dbpath = os.path.join(get_data_root(), 'test', 'holidays')
    ext = 'jpg' if dataset=='holidays' else 'rjpg'
    images = sorted(os.listdir(os.path.join(dbpath, ext)))
    with open(os.path.join(dbpath, 'straight_gnd_holidays.pkl'), 'rb') as f:
        queries = pickle.load(f)
        positives = pickle.load(f)
    
    qidx = []
    pidx = []
    for i in range(len(queries)):
        
        qidx.append(images.index(queries[i]))
    
        aux = []
        for j in range(len(positives[i])):
            aux.append(images.index(positives[i][j]))
        pidx.append(aux)
    
    # extract database and query vectors
    print('>> {}: database images...'.format(dataset))
    X = extract_vectors(net, [os.path.join(dbpath, ext, n) for n in images], 
                              image_size, transform, ms=ms, msp=msp)
    
    print('>> {}: Evaluating...'.format(dataset))
    
    # rank the similarities
    X = X.numpy()
    
    if( Lw is not None):
        X  = whitenapply(X, Lw['m'], Lw['P'])
        
    scores = np.dot(X.T, X)
    ranks = np.argsort(-scores, axis=1)
    ranks = ranks[qidx, 1::]
    
    APs = []
    for i,r in enumerate(ranks):
        trueRanks = np.isin(r, pidx[i])
        trueRanks = np.where(trueRanks == True)[0]
        APs.append( compute_ap(trueRanks, len(pidx[i])) )
    
    mAP = np.mean(APs)
    print(">> {}: mAP {:.2f}".format(dataset,mAP*100))

    # return the average mAP 
    return (dataset + ('+ multiscale' if eConfig['multiscale'] else ''), mAP)

def testOxfordParisHolidays(net, eConfig):
    
    #datasets = eConfig['test-datasets'].split(',')
    #results = []
    #
    #for dataset in datasets:
    #    results.append((dataset, np.random.rand(1)[0]))
    #    
    #return results

    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if(eConfig['multiscale']):
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if eConfig['whitening']:
    
        start = time.time()

        print('>> {}: Learning whitening...'.format(eConfig['test-whiten']))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', eConfig['test-whiten'])
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(eConfig['test-whiten']))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        
        # extract whitening vectors
        print('>> {}: Extracting...'.format(eConfig['test-whiten']))
        wvecs = extract_vectors(net, images, image_size, transform, ms=ms, msp=msp)

        # learning whitening 
        print('>> {}: Learning...'.format(eConfig['test-whiten']))
        wvecs = wvecs.numpy()        
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}
        
        print('>> {}: elapsed time: {}'.format(eConfig['test-whiten'], htime(time.time()-start)))
        

    else:
        Lw = None

    # evaluate on test datasets
    datasets = eConfig['test-datasets'].split(',')
    results = []
    
    for dataset in datasets: 
        start = time.time()

        if( dataset != 'holidays' and dataset != 'rholidays' ):
            print('>> {}: Extracting...'.format(dataset))
    
            # prepare config structure for the test dataset
            cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
            images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
            
            if(dataset == 'oxford105k' or dataset == 'paris106k'):
                images.extend(cfg['distractors'])
                
            # extract database and query vectors
            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, image_size, transform, ms=ms, msp=msp)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, image_size, transform, bbxs, ms=ms, msp=msp)
            print('>> {}: Evaluating...'.format(dataset))
    
            # convert to numpy
            vecs = vecs.numpy()
            qvecs = qvecs.numpy()
                    
            # search, rank, and print
            scores = np.dot(vecs.T, qvecs)
            ranks = np.argsort(-scores, axis=0)
            results.append(compute_map_and_print(dataset + 
                ('+ multiscale' if eConfig['multiscale'] else ''), ranks, cfg['gnd']))
        
            if Lw is not None:
                # whiten the vectors
                vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
                qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
    
                # search, rank, and print
                scores = np.dot(vecs_lw.T, qvecs_lw)
                ranks = np.argsort(-scores, axis=0)
                results.append(compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd']))
                
        else:
            results.append(testHolidays(net, eConfig, dataset, Lw))
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    return results

def trainAndTest( eConfig, saveResults=False ):

    setRandomSeeds(0)

    # load initial model to cuda
    model = loadModelToCuda( eConfig )

    # create save dir
    if saveResults:
        saveDIR = createExportDir( eConfig, 'Loss_{}_m_{}_g_{}_a_{}_b_{}_numpos_{}_dl_{}_dlf_{}_trSet_{}_lr_{}_accumulate_{}_wd_{}_cs_{}_is_{}'.format(
            eConfig['loss'], eConfig['loss-margin'], eConfig['exp-loss-gamma'],
            eConfig['exp-loss-alpha'], eConfig['exp-loss-beta'],
            eConfig['pos-num'], eConfig['drop-loss'], eConfig['drop-loss-freq'],
            eConfig['training-set'], eConfig['lr'], eConfig['accumulate'],
            eConfig['weight-decay'], eConfig['crop-scale'], eConfig['image-size']))
    
    # define loss function, optimizer and lr schedule
    criterion = defineLossFunction( eConfig )
    optimizer = defineOptimizer( eConfig, model )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(eConfig['lr-expDecay']))
    
    # create dataset and loader
    train_dataset, train_loader, val_dataset, val_loader = createDataLoading( eConfig, model )

    # evaluate the network before starting
    results = {}
#    results[0] = {'mAPs': test(model, eConfig), 'lossTr': 0.0, 'lossVal': 0.0, 
#           'avgPosDist': train_dataset.avgPosDist, 'avgNegDist': train_dataset.avgNegDist}
#    
    for epoch in range(eConfig['epochs']):

        # set manual seeds per epoch
        setRandomSeeds(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        lr_feat = optimizer.param_groups[0]['lr']
        lr_pool = optimizer.param_groups[1]['lr']
        print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))
            
        # train for one epoch on train set
        lossTr = train(train_loader, model, criterion, optimizer, epoch, eConfig['pretrained'], eConfig['accumulate'])

        # evaluate on validation set
        if eConfig['run-validation']:
            lossVal = validate(val_loader, model, criterion, epoch)
        else:
            lossVal = 0.0

        # evaluate on test datasets
        if (epoch+1) % 20 == 0 or (epoch+1) == eConfig['epochs']:
            results[epoch+1] = {'mAPs': test(model, eConfig), 'lossTr': lossTr, 'lossVal': lossVal, 
               'avgPosDist': train_dataset.avgPosDist, 'avgNegDist': train_dataset.avgNegDist}
    
            save_results(results[epoch+1]['mAPs'], saveDIR)
            
            if saveResults:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'meta': model.meta,
                    'state_dict': model.state_dict(),
                    'min_loss': lossTr,
                    'optimizer' : optimizer.state_dict(),
                }, False, saveDIR)
        
        
        
        # save results
        if saveResults:
            with open( '{}/results.pkl'.format(saveDIR), 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(eConfig, f, pickle.HIGHEST_PROTOCOL)
    
    ## load results
    #with open('{}/results.pkl'.format(saveDIR), 'rb') as f:
    #    results2 = pickle.load(f)
    #    eConfig2 = pickle.load(f) 
        
    return results

def testModel(modelPath, eConfig):
    
    # load original model
    model = loadModelToCuda( eConfig )
    
    # load the learned weigths
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])
            
    # test the model
    r = test(model, eConfig)
    return r
    

if __name__ == '__main__':
           
    torch.cuda.device(0)
    cd = torch.cuda.current_device()
    print('count: {}'.format(torch.cuda.device_count()))
    print('current_device: {}'.format(cd))
    print('device name: {}'.format(torch.cuda.get_device_name(cd)))
      
    eConfig = {
        'resultsFolder': 'results',
        'training-set': 'Google-Landmarks',# 'Google-Landmarks'], #options: 'retrieval-SfM-120k', 'rgbd-dataset', 'Google-Landmarks', 'Landmarks'
        'test-datasets': 'oxford5k,paris6k,roxford5k,rparis6k',  #options: 'oxford5k,roxford5k,oxford105k,paris6k,rparis6k,paris106k,holidays,rholidays', 'ukbench'
        'whitening': True,
        'multiscale': True,
        'test-whiten': 'retrieval-SfM-120k',
        'run-validation': False,
        'arch': 'resnet101', # this refers to the backbone (resnet, vgg, densenet)
        'pooling': 'gem', # options: 'mac', 'spoc', 'gem', 'rmac'
        'pretrained': 'imageNet', #options: 'None', 'imageNet', 'retrievalSfM120k'
        'loss': 'aggExp', # options: 'triplet', 'contrastive', 'rootContrastive', 'wContrastive', 'exp', 'angExp', 'aggExp'
        'drop-loss': 0,
        'drop-loss-freq': 5,
        'loss-margin': 0.85,   # for margin based losses
        'exp-loss-gamma': 1.0,  # for the exp and rankExp losses
        'exp-loss-alpha': 1.05, # for the margin ratio in our Bag Exponential 
        'exp-loss-beta': 1.5,   # the paper /beta parameter e^(-beta(d(pi,pj)))
        'optimizer': 'adam', # adam
        'lr': 1e-5, # 1e-6 - 1e-7
        'lr-expDecay': -0.01,
        'momentum': 0.9, #0.9
        'weight-decay': 1e-4,#1e-4 seems to work
        'epochs': 60,
        'batch-size': 10, # the number of tuples (q,p1,p2..,n1,n2,...) per batch
        'accumulate': 50, # the number of batches to accum for one param update
        'image-size': 1024, #362
        'neg-num': 5,  # the number of negatives in a training tuple
        'pos-num': 5, # the number of positives in a training tuple
        'query-size': 2000, # the number of tuples per epoch
        'pool-size': 20000, # the negative pool size 
        'workers': 8,
        'print-freq': 500,
        'crop-scale': 1.0}
    
    trainAndTest(eConfig, saveResults=True)
    
    # # load results
    # with open('results.pkl', 'rb') as f:
    #     results2 = pickle.load(f)
    #     eConfig2 = pickle.load(f)
    
    
