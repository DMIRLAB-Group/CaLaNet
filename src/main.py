######################################################################################
# NOTE： Our code partly follows the code from NetEst and TNet. Thanks for their code!
#
# NetEst: https://github.com/songjiang0909/Causal-Inference-on-Networked-Data
# TNET: https://github.com/DMIRLAB-Group/TNet
#
######################################################################################


import sys
sys.path.append("..")

import argparse
import torch
import time
import utils as utils
import numpy as np
from model import NetEsimator
# from src.data import dataGenUtils
from baselineModels import GCN_DECONF, CFR, GCN_DECONF_INTERFERENCE, CFR_INTERFERENCE, CFR_INTERFERENCE_GCN
from netEst_crl_sm_ipm import netEstCRLSM_IPM
from experiment import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=7, help='Use CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed. RIP KOBE')  # 24
# parser.add_argument('--dataset', type=str, default='BC')
# parser.add_argument('--flipRate', type=float, default=1)
# ["BC","Flickr"]
parser.add_argument('--alpha', type=float, default=.5, help='trade-off of p(t|x).')
parser.add_argument('--gamma', type=float, default=.5, help='trade-off of p(z|x).')
parser.add_argument('--lrT', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--lrTR_TZ', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--beta', type=float, default=20, help='trade-off of targeted regur in TargetedModel')
parser.add_argument('--tr_knots', type=float, default=0.1, help='trade-off of targeted regur in TargetedModel')

parser.add_argument('--loss_2step_with_ly', type=int, default=0,
                    help='loss in 2 step contains loss of y, 0 means no, 1 means yes')
parser.add_argument('--loss_2step_with_ltz', type=int, default=0,
                    help='loss in 2 step contains loss of tz, 0 means no, 1 means yes')
parser.add_argument('--lr_1step', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--lr_2step', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=.9, help='momentum for optimizer 1')
parser.add_argument('--pre_train_step', type=int, default=0, help='momentum for optimizer 1')

parser.add_argument('--pstep', type=int, default=1, help='epoch of training')  # default 1
parser.add_argument('--iter_2step', type=int, default=50, help='epoch of training fluctation param')

parser.add_argument('--weight_decay_tr', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')  # 1e-5

# iter_1step = pstep
parser.add_argument('--lrD', type=float, default=1e-3, help='Initial learning rate of Discriminator.')
parser.add_argument('--lrD_z', type=float, default=1e-3, help='Initial learning rate of Discriminator_z.')
parser.add_argument('--dstep', type=int, default=50, help='epoch of training discriminator')
parser.add_argument('--d_zstep', type=int, default=50, help='epoch of training discriminator_z')
parser.add_argument('--save_intermediate', type=int, default=1,
                    help='Save training curve and imtermediate embeddings')  # default 1
parser.add_argument('--alpha_base', type=float, default=0.5, help='trade-off of balance for baselines.')
parser.add_argument('--printDisc', type=int, default=0, help='Print discriminator result for debug usage')
parser.add_argument('--printDisc_z', type=int, default=0, help='Print discriminator_z result for debug usage')
parser.add_argument('--printPred', type=int, default=1, help='Print encoder-predictor result for debug usage')
parser.add_argument('--search', type=int, default=0, help='parameter searching')
parser.add_argument('--num_grid', type=int, default=20, help='Number of epochs to train.')

parser.add_argument('--model', type=str, default='NetEsimator', help='Models or baselines')
parser.add_argument('--dataset', type=str, default='BC_uncon_decom')
parser.add_argument('--flipRate', type=float, default=0)
parser.add_argument('--expID', type=int, default=0)

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden units.') # default 3  2 for simulation
parser.add_argument('--ZI_dim', type=int, default=3, help='Number of hidden units.')  # 32
parser.add_argument('--ZC_dim', type=int, default=4, help='Number of hidden units.')  # 32
parser.add_argument('--ZN_dim', type=int, default=3, help='Number of hidden units.')  # 32

parser.add_argument('--vae_alpha', type=float, default=1., help='trade-off elbo+alpha*ly')
parser.add_argument('--vae_beta', type=float, default=1., help='trade-off beta* elbo+alpha*ly')
parser.add_argument('--balancing_eta', type=float, default=1., help='trade-off beta*elbo+alpha*ly + eta*IPM ')
parser.add_argument('--vae_gamma', type=float, default=1, help='trade-off beta* beta*elbo+alpha*ly + eta*IPM + vae_gamma*SM')

parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.') # default 1e-3  simulation 1e-2
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')  # 32
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--savelatents', type=int, default=0, help='1 means save latents')

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    # if print_log:
    #     print('using cuda')
    cuda = True
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    # if print_log:
    #     print('using cpu')
    torch.set_default_dtype(torch.float)

startTime = time.time()

args = parser.parse_args()

print(args)

args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset == 'SIMULATION':
    from data.dataGenUtils import load_data
    trainA, trainX, trainT, \
        cfTrainT, POTrain, cfPOTrain, \
        valA, valX, valT, cfValT, POVal, cfPOVal, \
        testA, testX, testT, cfTestT, POTest, cfPOTest, \
        train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, \
        val_t1z1, val_t1z0, val_t0z0, val_t0z7, val_t0z2, \
        test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2 = load_data(1000)
else:
    trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT, POVal, cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest, \
    train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7, val_t0z2,\
    test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2 = utils.load_data(args)

print(trainX.shape, trainA.shape, trainT.shape, POTrain.shape, train_t0z7.shape)



if args.model == "NetEsimator":
    model = NetEsimator(Xshape=trainX.shape[1], hidden=args.hidden, dropout=args.dropout)
elif args.model == "ND":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.model == "TARNet":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.model == "CFR":
    model = CFR(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)

elif args.model == "CFR_INTERFERENCE":
    model = CFR_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.model == "ND_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.model == "TARNet_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)

elif args.model == "CFR_INTERFERENCE_GCN":
    model = CFR_INTERFERENCE_GCN(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)


elif args.model == "netEstCRLSM_IPM":
    model = netEstCRLSM_IPM(Xshape=trainX.shape[1], hidden=args.hidden, dropout=args.dropout,
                      n_layers=args.n_layers,
                      ZI_dim=args.ZI_dim, ZC_dim=args.ZC_dim, ZN_dim=args.ZN_dim, num_grid=args.num_grid)
else:
    raise Exception("NO SUCH MODEL")

exp = Experiment(args, model, trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT, POVal,
                 cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest, \
                 train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7,
                 val_t0z2, test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2)

"""Train the model"""
try:
    exp.train()
    # exp.predict()
except KeyboardInterrupt:
    exp.predict()
    # utils.savelatents(model,trainA,trainX,testA,testX,args)

    pass

if args.savelatents:
    utils.savelatents(model,trainA,trainX,testA,testX,args)

if args.dataset == 'SIMULATION':
    model = exp.model
    XN_prior, encoder_params, decoder_params, \
        ZI_hat, ZC_hat, ZN_hat, Z_hat = model.vae_module(trainA, trainX)
    zi, zc, zn = ZI_hat.detach().cpu().numpy(), ZC_hat.detach().cpu().numpy(), ZN_hat.detach().cpu().numpy()

    import warnings
    warnings.simplefilter("ignore")

    from data.dataGenUtils import load_latent
    trainU, _,_, = load_latent(1000)
    ui, uc, un = trainU[:,0][:,None],trainU[:,1][:,None],trainU[:,2][:,None]
    from src.results.visul_u import visual_given

    est_z = np.concatenate((zi, zc, zn), axis=1)
    visual_given(z=trainU, est_z=est_z, fn='1new.pdf')

    est_z = (est_z - np.mean(est_z, axis=0)) / np.std(est_z, axis=0)
    trainU = (trainU - np.mean(trainU, axis=0)) / np.std(trainU, axis=0)
    visual_given(z=trainU, est_z=est_z, fn='2new.pdf')

print("Time usage:{:.4f} mins".format((time.time() - startTime) / 60))
print("================================Setting again================================")
print("Model:{} Dataset:{}, expID:{}, filpRate:{}, alpha:{}, gamma:{}".format(args.model, args.dataset, args.expID,
                                                                              args.flipRate, args.alpha, args.gamma))
print(args)
print("================================BYE================================")
