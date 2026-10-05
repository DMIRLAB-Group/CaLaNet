import os

import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import utils as utils
import numpy as np

def mape(gt, est):
    return torch.mean(torch.abs(((gt-est)/gt)))

def criterion(out, y):
    return ((out - y) ** 2).mean()

class Experiment():

    def __init__(self, args, model, trainA, trainX, trainT, cfTrainT, POTrain, cfPOTrain, valA, valX, valT, cfValT,
                 POVal, cfPOVal, testA, testX, testT, cfTestT, POTest, cfPOTest,
                 train_t1z1, train_t1z0, train_t0z0, train_t0z7, train_t0z2, val_t1z1, val_t1z0, val_t0z0, val_t0z7,
                 val_t0z2, test_t1z1, test_t1z0, test_t0z0, test_t0z7, test_t0z2):
        super(Experiment, self).__init__()

        self.args = args
        self.model = model
        if self.args.model == "NetEsimator":
            self.optimizerD = optim.Adam([{'params': self.model.discriminator.parameters()}], lr=self.args.lrD,
                                         weight_decay=self.args.weight_decay)
            self.optimizerD_z = optim.Adam([{'params': self.model.discriminator_z.parameters()}], lr=self.args.lrD_z,
                                           weight_decay=self.args.weight_decay)
            self.optimizerP = optim.Adam(
                [{'params': self.model.encoder.parameters()}, {'params': self.model.predictor.parameters()}],
                lr=self.args.lr, weight_decay=self.args.weight_decay)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif args.model == "netEstCRLSM_IPM":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        else:
            self.optimizerB = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.cuda:
            self.model = self.model.cuda()
        print("================================Model================================")
        print(self.model)

        self.Tensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        self.trainA = trainA
        self.trainX = trainX
        self.trainT = trainT
        # print(torch.sum(trainT))
        self.trainZ = self.compute_z(self.trainT, self.trainA)
        # print(self.trainZ )
        self.cfTrainT = cfTrainT
        self.POTrain = POTrain
        self.cfPOTrain = cfPOTrain
        self.valA = valA
        self.valX = valX
        self.valT = valT
        self.valZ = self.compute_z(self.valT, self.valA)
        self.cfValT = cfValT
        self.POVal = POVal
        self.cfPOVal = cfPOVal
        self.testA = testA
        self.testX = testX
        self.testT = testT
        self.testZ = self.compute_z(self.testT, self.testA)
        self.cfTestT = cfTestT
        self.POTest = POTest
        self.cfPOTest = cfPOTest

        self.z_1 = 0.7
        self.z_2 = 0.2
        self.train_t1z1 = self.Tensor(train_t1z1)
        self.train_t1z0 = self.Tensor(train_t1z0)
        self.train_t0z0 = self.Tensor(train_t0z0)
        self.train_t0z7 = self.Tensor(train_t0z7)
        self.train_t0z2 = self.Tensor(train_t0z2)

        self.val_t1z1 = self.Tensor(val_t1z1)
        self.val_t1z0 = self.Tensor(val_t1z0)
        self.val_t0z0 = self.Tensor(val_t0z0)
        self.val_t0z7 = self.Tensor(val_t0z7)
        self.val_t0z2 = self.Tensor(val_t0z2)

        self.test_t1z1 = self.Tensor(test_t1z1)
        self.test_t1z0 = self.Tensor(test_t1z0)
        self.test_t0z0 = self.Tensor(test_t0z0)
        self.test_t0z7 = self.Tensor(test_t0z7)
        self.test_t0z2 = self.Tensor(test_t0z2)

        """PO normalization if any"""
        self.YFTrain, self.YCFTrain = utils.PO_normalize(self.args.normy, self.POTrain, self.POTrain, self.cfPOTrain)
        self.YFVal, self.YCFVal = utils.PO_normalize(self.args.normy, self.POTrain, self.POVal, self.cfPOVal)
        self.YFTest, self.YCFTest = utils.PO_normalize(self.args.normy, self.POTrain, self.POTest, self.cfPOTest)

        self.loss = nn.MSELoss(reduction='mean')
        # self.loss = criterion
        self.d_zLoss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.peheLoss = nn.MSELoss(reduction='mean')
        # kernel_mul = 1, kernel_num = 45, fix_sigma=1
        self.mmd_loss = utils.MMD_loss(kernel_mul = 1, kernel_num = 45, fix_sigma=1)

        self.alpha = self.Tensor([self.args.alpha])
        self.gamma = self.Tensor([self.args.gamma])
        self.alpha_base = self.Tensor([self.args.alpha_base])
        self.vae_alpha = self.Tensor([self.args.vae_alpha])
        self.vae_beta = self.Tensor([self.args.vae_beta])
        self.balancing_eta = self.Tensor([self.args.balancing_eta])
        self.vae_gamma = self.Tensor([self.args.vae_gamma])
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            self.loss = self.loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.peheLoss = self.peheLoss.cuda()

        self.lossTrain = []
        self.lossVal = []
        self.lossTest = []
        self.lossCFTrain = []
        self.lossCFVal = []
        self.lossCFTest = []

        self.dissTrain = []
        self.dissVal = []
        self.dissTrainHalf = []
        self.dissValHalf = []
        self.diss_zTrain = []
        self.diss_zVal = []
        self.diss_zTrainHalf = []
        self.diss_zValHalf = []

        self.labelTrain = []
        self.labelVal = []
        self.labelTest = []
        self.labelTrainCF = []
        self.labelValCF = []
        self.labelTestCF = []

        self.predT = []
        self.labelT = []

    def loss_func_y(self, y, y_pred, w):
        y = y.reshape(-1, 1)
        loss = (w * (y - y_pred) ** 2).mean()
        return loss

    def loss_func_mmd(self, x, y):
        loss = self.mmd_loss(x, y)
        return loss

    def loss_func_hsic(self, x, y):
        loss = utils.hsic_normalized(x, y)
        return loss

    def loss_func_wass(self, x, y):
        loss, _ = utils.wasserstein(x, y)
        return loss

    # def get_peheLoss(self, y1pred, y0pred, y1gt, y0gt):
    #     pred = y1pred - y0pred
    #     gt = y1gt - y0gt
    #     print(pred.shape, gt.shape)
    #     return torch.sqrt(self.peheLoss(pred, gt))

    def compute_z(self, T, A):
        # print ("A has identity?: {}".format(not (A[0][0]==0 and A[24][24]==0 and A[8][8]==0)))
        neighbors = torch.sum(A, 1)
        neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        return neighborAverageT

    def get_peheLoss(self, y1pred, y0pred, y1gt, y0gt):
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        # print(pred.shape, gt[:,None].shape)
        return torch.sqrt(self.peheLoss(pred, gt))

    def get_ateLoss(self, y1pred, y0pred, y1gt, y0gt):
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.abs(torch.mean(pred) - torch.mean(gt))

    def compute_z(self, T, A):
        # print ("A has identity?: {}".format(not (A[0][0]==0 and A[24][24]==0 and A[8][8]==0)))
        neighbors = torch.sum(A, 1)
        neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        return neighborAverageT



    def train_one_step_discriminator(self, A, X, T):

        self.model.train()
        self.optimizerD.zero_grad()
        pred_treatmentTrain, _, _, _, _ = self.model(A, X, T)
        # print(T.shape)
        # print(pred_treatmentTrain.shape)
        discLoss = self.bce_loss(pred_treatmentTrain.reshape(-1), T)
        num = pred_treatmentTrain.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
        discLoss.backward()
        self.optimizerD.step()

        return discLoss, discLosshalf

    def eval_one_step_discriminator(self, A, X, T):

        self.model.eval()
        pred_treatment, _, _, _, _ = self.model(A, X, T)
        discLossWatch = self.bce_loss(pred_treatment.reshape(-1), T)
        num = pred_treatment.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))

        return discLossWatch, discLosshalf, pred_treatment, T

    def train_discriminator(self, epoch):

        for ds in range(self.args.dstep):
            discLoss, discLossTrainhalf = self.train_one_step_discriminator(self.trainA, self.trainX, self.trainT)
            discLossVal, discLossValhalf, _, _ = self.eval_one_step_discriminator(self.valA, self.valX, self.valT)
            discLossTest, discLossTesthalf, _, _ = self.eval_one_step_discriminator(self.testA, self.testX, self.testT)

            if ds == self.args.dstep - 1:
                if self.args.printDisc:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                          'dLoss:{:05f}'.format(discLoss),
                          'dLossVal:{:05f}'.format(discLossVal),
                          'dLossTest:{:05f}'.format(discLossTest),
                          'dLoss0.5:{:05f}'.format(discLossTrainhalf),
                          'dLossVal0.5:{:05f}'.format(discLossValhalf),
                          'dLossTest0.5:{:05f}'.format(discLossTesthalf),
                          )
                self.dissTrain.append(discLoss.detach().cpu().numpy())
                self.dissVal.append(discLossVal.detach().cpu().numpy())
                self.dissTrainHalf.append(discLossTrainhalf.detach().cpu().numpy())
                self.dissValHalf.append(discLossValhalf.detach().cpu().numpy())

    def train_fluctuation_param(self, epoch):
        for ds in range(self.args.iter_2step):
            # self.model.zero_grad()
            self.model.train()
            self.optimizer2step.zero_grad()
            A, X, T, Y = self.trainA, self.trainX, self.trainT, self.YFTrain

            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            Loss_TR = self.loss(
                Q_hat.reshape(-1) +
                epsilon.reshape(-1) * (1 / (g_T_hat.reshape(-1).detach() * g_Z_hat.reshape(-1).detach() + 1e-6))
                #  - (1 - T) / (  (1 - g_T_hat.reshape(-1)) * g_Z_hat.reshape(-1) + 1e-6))
                , Y)
            loss_train = self.args.beta * Loss_TR

            if self.args.loss_2step_with_ly == 1:
                Q_Loss = self.loss(Q_hat.reshape(-1), Y)
                loss_train = loss_train + Q_Loss

            if self.args.loss_2step_with_ltz == 1:
                g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
                g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
                loss_train = loss_train + self.args.alpha * g_T_Loss + self.args.gamma * g_Z_Loss

            loss_train.backward()
            self.optimizer2step.step()

            g_T_hat_val, g_Z_hat_val, Q_hat_val, epsilon_val, _, _ = self.model(self.valA, self.valX,
             self.valT)

            pLoss_val = self.loss(
                Q_hat_val.reshape(-1) +
                epsilon_val.reshape(-1) * (1 / (g_T_hat_val.reshape(-1).detach() * g_Z_hat_val.reshape(-1).detach() + 1e-6))
                , self.YFVal
            ) * self.args.beta


        individual_effect_train, peer_effect_train, total_effect_train, \
            ate_individual_train, ate_peer_train, ate_total_train, _,_,_,_,_,_ \
            = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                       self.train_t0z7, self.train_t0z2, self.train_t0z0)

        individual_effect_te, peer_effect_te, total_effect_te, \
            ate_individual_te, ate_peer_te, ate_total_te, _,_,_,_,_,_ \
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)


        if self.args.printPred:
            print('2_Epoch: {:04d}'.format(epoch + 1),
                  'tLossTrain:{:.4f}'.format(loss_train.item()),
                  'tLossVal:{:.4f}'.format(pLoss_val.item()),

                  'iE_train:{:.4f}'.format(individual_effect_train.item()),
                  'PE_train:{:.4f}'.format(peer_effect_train.item()),
                  'TE_train:{:.4f}'.format(total_effect_train.item()),

                  '\t',
                  'iE_te:{:.4f}'.format(individual_effect_te.item()),
                  'PE_te:{:.4f}'.format(peer_effect_te.item()),
                  'TE_te:{:.4f}'.format(total_effect_te.item()),

                  'AiE_te:{:.4f}'.format(ate_individual_te.item()),
                  'APE_te:{:.4f}'.format(ate_peer_te.item()),
                  'ATE_te:{:.4f}'.format(ate_total_te.item()),

                  )



    def train_one_step_discriminator_z(self, A, X, T):

        self.model.train()
        self.optimizerD_z.zero_grad()
        _, pred_zTrain, _, _, labelZ = self.model(A, X, T)
        discLoss_z = self.d_zLoss(pred_zTrain.reshape(-1), labelZ)
        num = pred_zTrain.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf_z = self.loss(pred_zTrain.reshape(-1), self.Tensor(target))
        discLoss_z.backward()
        self.optimizerD_z.step()

        return discLoss_z, discLosshalf_z

    def eval_one_step_discriminator_z(self, A, X, T):

        self.model.eval()
        _, pred_z, _, _, labelZ = self.model(A, X, T)
        discLossWatch = self.d_zLoss(pred_z.reshape(-1), labelZ)
        num = pred_z.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf = self.loss(pred_z.reshape(-1), self.Tensor(target))

        return discLossWatch, discLosshalf, pred_z, labelZ

    def train_discriminator_z(self, epoch):

        for dzs in range(self.args.d_zstep):
            discLoss_z, discLoss_zTrainRandom = self.train_one_step_discriminator_z(self.trainA, self.trainX,
                                                                                    self.trainT)
            discLoss_zVal, discLoss_zValRandom, _, _ = self.eval_one_step_discriminator_z(self.valA, self.valX,
                                                                                          self.valT)
            discLoss_zTest, discLoss_zTestRandom, _, _ = self.eval_one_step_discriminator_z(self.testA, self.testX,
                                                                                            self.testT)

            if dzs == self.args.d_zstep - 1:
                if self.args.printDisc_z:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                          'd_zLoss:{:05f}'.format(discLoss_z),
                          'd_zLossVal:{:05f}'.format(discLoss_zVal),
                          'd_zLossTest:{:05f}'.format(discLoss_zTest),
                          'd_zLRanTrain:{:05f}'.format(discLoss_zTrainRandom),
                          'd_zLRanVal:{:05f}'.format(discLoss_zValRandom),
                          'd_zLRanTest:{:05f}'.format(discLoss_zTestRandom),
                          )
                self.diss_zTrain.append(discLoss_z.detach().cpu().numpy())
                self.diss_zVal.append(discLoss_zVal.detach().cpu().numpy())
                self.diss_zTrainHalf.append(discLoss_zTrainRandom.detach().cpu().numpy())
                self.diss_zValHalf.append(discLoss_zValRandom.detach().cpu().numpy())

    def train_one_step_encoder_predictor(self, A, X, T, Y):

        if self.args.model == "NetEsimator":
            self.model.zero_grad()
            self.model.train()
            self.optimizerP.zero_grad()
            pred_treatmentTrain, pred_zTrain, pred_outcomeTrain, _, _ = self.model(A, X, T)
            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
            num = pred_treatmentTrain.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLoss = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
            num = pred_zTrain.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLoss = self.d_zLoss(pred_zTrain.reshape(-1), target)
            loss_train = pLoss + dLoss * self.alpha + d_zLoss * self.gamma
            loss_train.backward()
            self.optimizerP.step()

        elif self.args.model == 'TargetedModel_DoubleBSpline':
            # self.model.zero_grad()
            self.model.train()
            self.optimizerT.zero_grad()
            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            Q_Loss = self.loss(Q_hat.reshape(-1), Y)
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
            g_Z_Loss = - torch.log(g_Z_hat + 1e-6).mean()
            Loss_base = Q_Loss + self.args.alpha * g_T_Loss + self.args.gamma * g_Z_Loss
            Loss_TR = self.loss(
                Q_hat.reshape(-1) +
                epsilon.reshape(-1) * (1 / (g_T_hat.detach().reshape(-1) * g_Z_hat.detach().reshape(-1) + 1e-6) )
                ,Y)

            loss_train = Loss_base + self.args.beta * Loss_TR
            loss_train.backward()
            self.optimizerT.step()
            pLoss, dLoss, d_zLoss = Q_Loss, g_T_Loss, g_Z_Loss

        elif self.args.model == "netEstCRL":
            # do not use
            self.model.zero_grad()
            self.model.train()
            self.optimizer.zero_grad()
            y1_mean, y0_mean, elbo, log_py_zzn = self.model(A, X, T, Y, alpha=self.vae_alpha)
            loss_train = -elbo - self.vae_alpha*log_py_zzn
            loss_train.backward()
            self.optimizer.step()
            pLoss, dLoss, d_zLoss = -elbo, -log_py_zzn, self.Tensor([0])


        elif self.args.model == "netEstCRLSM_IPM":
            self.model.zero_grad()
            self.model.train()

            self.model.zero_grad()
            self.optimizer.zero_grad()
            y1_mean, y0_mean, elbo, log_py_zzn, loss_sm, loss_balance = self.model(A, X, T, Y, alpha=self.vae_alpha)
            loss_train = - self.vae_beta * elbo - self.vae_alpha * log_py_zzn - self.vae_gamma * loss_sm \
                         + self.balancing_eta * loss_balance

            loss_train.backward()

            self.optimizer.step()


            pLoss, dLoss, d_zLoss = -self.vae_beta * elbo, -self.vae_alpha * log_py_zzn, - self.vae_gamma * loss_sm

        else:
            self.model.zero_grad()
            self.model.train()
            self.optimizerB.zero_grad()
            if self.args.model in set(["CFR_INTERFERENCE_GCN"]):
                _, _, pred_outcomeTrain, rep, _, gcn_rep = self.model(A, X, T)
                pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
                dLoss = self.loss_func_hsic(rep, T.reshape(-1, 1))
                loss_train = pLoss + self.alpha_base * dLoss

            else:
                _, _, pred_outcomeTrain, rep, _ = self.model(A, X, T)
                pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
                if self.args.model in set(["TARNet", "TARNet_INTERFERENCE"]):
                    loss_train = pLoss
                    dLoss = self.Tensor([0])

                else:
                    rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                    dLoss, _ = utils.wasserstein(rep_t1, rep_t0, cuda=self.args.cuda)
                    loss_train = pLoss + self.alpha_base * dLoss
            d_zLoss = self.Tensor([-1])
            loss_train.backward()
            self.optimizerB.step()

        return loss_train, pLoss, dLoss, d_zLoss

    def eval_one_step_encoder_predictor(self, A, X, T, Y):

        self.model.eval()
        if self.args.model == "NetEsimator":
            pred_treatment, pred_z, pred_outcome, _, _ = self.model(A, X, T)
            pLossV = self.loss(pred_outcome.reshape(-1), Y)
            num = pred_treatment.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLossV = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))
            num = pred_z.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLossV = self.d_zLoss(pred_z.reshape(-1), target)
            loss_val = pLossV + dLossV * self.alpha + d_zLossV * self.gamma

        elif self.args.model == 'TargetedModel_DoubleBSpline':
            g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = self.model(A, X, T)
            Q_Loss = self.loss(Q_hat.reshape(-1), Y)
            g_T_Loss = self.bce_loss(g_T_hat.reshape(-1), T)
            g_Z_Loss = -torch.log(g_Z_hat + 1e-6).mean()
            Loss_base = Q_Loss + 0.5 * g_T_Loss + 0.5 * g_Z_Loss
            Loss_TR = self.loss(Y, Q_hat.reshape(-1) + epsilon.reshape(-1) / (
                    g_T_hat.reshape(-1) * g_Z_hat.reshape(-1) + 1e-6))
            loss_val = Loss_base + 1. * Loss_TR
            pLossV, dLossV, d_zLossV = Q_Loss, g_T_Loss, g_Z_Loss

        elif self.args.model == "netEstCRLSM_IPM":
            loss_val = self.Tensor([-1])
            pLossV, dLossV, d_zLossV =  self.Tensor([-1]), self.Tensor([-1]), self.Tensor([-1])
        else:
            if self.args.model in set(["CFR_INTERFERENCE_GCN"]):
                _, _, pred_outcomeTrain, rep, _, gcn_rep = self.model(A, X, T)
                pLossV = self.loss(pred_outcomeTrain.reshape(-1), Y)
                dLossV = self.loss_func_hsic(rep, T.reshape(-1, 1))
                loss_val = pLossV + self.alpha_base * dLossV
            else:
                _, _, pred_outcome, rep, _ = self.model(A, X, T)
                pLossV = self.loss(pred_outcome.reshape(-1), Y)
                if self.args.model in set(["TARNet", "TARNet_INTERFERENCE"]):
                    loss_val = pLossV
                    dLossV = self.Tensor([0])
                else:
                    rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                    dLossV, _ = utils.wasserstein(rep_t1, rep_t0, cuda=self.args.cuda)
                    loss_val = self.args.alpha_base * dLossV
                    loss_val = pLossV + self.args.alpha_base * dLossV
            d_zLossV = self.Tensor([-1])

        return loss_val, pLossV, dLossV, d_zLossV

    def compute_effect_pehe(self, A, X, gt_t1z1, gt_t1z0, gt_t0z7, gt_t0z2, gt_t0z0):

        num = X.shape[0]
        z_1s = self.Tensor(np.ones(num))
        z_0s = self.Tensor(np.zeros(num))
        z_07s = self.Tensor(np.zeros(num) + self.z_1)
        z_02s = self.Tensor(np.zeros(num) + self.z_2)
        t_1s = self.Tensor(np.ones(num))
        t_0s = self.Tensor(np.zeros(num))

        if self.args.model == 'TargetedModel_DoubleBSpline' :
            pred_outcome_t1z1 = self.model.infer_potential_outcome(A, X, t_1s, z_1s)
            pred_outcome_t1z0 = self.model.infer_potential_outcome(A, X, t_1s, z_0s)
            pred_outcome_t0z0 = self.model.infer_potential_outcome(A, X, t_0s, z_0s)
            pred_outcome_t0z7 = self.model.infer_potential_outcome(A, X, t_0s, z_07s)
            pred_outcome_t0z2 = self.model.infer_potential_outcome(A, X, t_0s, z_02s)
        elif self.args.model == "CFR_INTERFERENCE_GCN":
            _, _, pred_outcome_t1z1, _, _, _ = self.model(A, X, t_1s, z_1s)
            _, _, pred_outcome_t1z0, _, _, _ = self.model(A, X, t_1s, z_0s)
            _, _, pred_outcome_t0z0, _, _, _ = self.model(A, X, t_0s, z_0s)
            _, _, pred_outcome_t0z7, _, _, _ = self.model(A, X, t_0s, z_07s)
            _, _, pred_outcome_t0z2, _, _, _ = self.model(A, X, t_0s, z_02s)
        elif self.args.model == "netEstCRLSM_IPM":
            # print('compute')
            Y = self.Tensor(np.ones(num))
            pred_outcome_t1z1,_,_,_,_ = self.model.infer_potential_outcome(A, X, t_1s, Y, z_1s)
            pred_outcome_t1z0,_,_,_,_ = self.model.infer_potential_outcome(A, X, t_1s, Y, z_0s)
            pred_outcome_t0z0,_,_,_,_ = self.model.infer_potential_outcome(A, X, t_0s, Y, z_0s)
            pred_outcome_t0z7,_,_,_,_ = self.model.infer_potential_outcome(A, X, t_0s, Y, z_07s)
            pred_outcome_t0z2,_,_,_,_ = self.model.infer_potential_outcome(A, X, t_0s, Y, z_02s)
        else:
            _, _, pred_outcome_t1z1, _, _ = self.model(A, X, t_1s, z_1s)
            _, _, pred_outcome_t1z0, _, _ = self.model(A, X, t_1s, z_0s)
            _, _, pred_outcome_t0z0, _, _ = self.model(A, X, t_0s, z_0s)
            _, _, pred_outcome_t0z7, _, _ = self.model(A, X, t_0s, z_07s)
            _, _, pred_outcome_t0z2, _, _ = self.model(A, X, t_0s, z_02s)

        pred_outcome_t1z1 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t1z1)
        pred_outcome_t1z0 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t1z0)
        pred_outcome_t0z0 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z0)
        pred_outcome_t0z7 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z7)
        pred_outcome_t0z2 = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome_t0z2)

        # print(pred_outcome_t0z7, pred_outcome_t0z2)

        individual_effect = self.get_peheLoss(pred_outcome_t1z0, pred_outcome_t0z0, gt_t1z0, gt_t0z0)
        peer_effect = self.get_peheLoss(pred_outcome_t0z7, pred_outcome_t0z2, gt_t0z7, gt_t0z2)
        total_effect = self.get_peheLoss(pred_outcome_t1z1, pred_outcome_t0z0, gt_t1z1, gt_t0z0)

        ate_individual = self.get_ateLoss(pred_outcome_t1z0, pred_outcome_t0z0, gt_t1z0, gt_t0z0)
        ate_peer = self.get_ateLoss(pred_outcome_t0z7, pred_outcome_t0z2, gt_t0z7, gt_t0z2)
        ate_total = self.get_ateLoss(pred_outcome_t1z1, pred_outcome_t0z0, gt_t1z1, gt_t0z0)


        ate_mape_indi = mape(est=torch.mean(pred_outcome_t1z0-pred_outcome_t0z0), gt=torch.mean(gt_t1z0-gt_t0z0))
        ate_mape_peer = mape(est=torch.mean(pred_outcome_t0z7-pred_outcome_t0z2), gt=torch.mean(gt_t0z7-gt_t0z2))
        ate_mape_total = mape(est=torch.mean(pred_outcome_t1z1-pred_outcome_t0z0), gt=torch.mean(gt_t1z1-gt_t0z0))

        ite_mape_indi = mape(est=(pred_outcome_t1z0 - pred_outcome_t0z0), gt=(gt_t1z0 - gt_t0z0))
        ite_mape_peer = mape(est=(pred_outcome_t0z7 - pred_outcome_t0z2), gt=(gt_t0z7 - gt_t0z2))
        ite_mape_total = mape(est=(pred_outcome_t1z1 - pred_outcome_t0z0), gt=(gt_t1z1 - gt_t0z0))

        return individual_effect, peer_effect, total_effect, ate_individual, ate_peer, ate_total, \
            ite_mape_indi, ite_mape_peer, ite_mape_total, ate_mape_indi, ate_mape_peer, ate_mape_total

    def train_encoder_predictor(self, epoch):

        for epoch_num in range(self.args.pstep):
            loss_train, pLoss_train, dLoss_train, d_zLoss_train = self.train_one_step_encoder_predictor(self.trainA,
                                                                                                        self.trainX,
                                                                                                        self.trainT,
                                                                                                        self.YFTrain)
            loss_val, pLoss_val, dLoss_val, d_zLoss_val = self.eval_one_step_encoder_predictor(self.valA, self.valX,
                                                                                               self.valT, self.YFVal)
            self.lossTrain.append(loss_train.cpu().detach().numpy())
            self.lossVal.append(loss_val.cpu().detach().numpy())


        individual_effect_train, peer_effect_train, total_effect_train,\
            ate_individual_train, ate_peer_train, ate_total_train, _,_,_,_,_,_\
            = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                       self.train_t0z7, self.train_t0z2, self.train_t0z0)

        individual_effect_te, peer_effect_te, total_effect_te,\
            ate_individual_te, ate_peer_te, ate_total_te, \
            ite_mape_indi, ite_mape_peer, ite_mape_total, ate_mape_indi, ate_mape_peer, ate_mape_total\
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)

        if self.args.printPred:
            print('p_Epoch: {:04d}'.format(epoch + 1),
                  'pLossTrain:{:.4f}'.format(pLoss_train.item()),
                  'pLossVal:{:.4f}'.format(pLoss_val.item()),

                  'iE_train:{:.4f}'.format(individual_effect_train.item()),
                  'PE_train:{:.4f}'.format(peer_effect_train.item()),
                  'TE_train:{:.4f}'.format(total_effect_train.item()),

                  '\t',
                  'iE_te:{:.4f}'.format(individual_effect_te.item()),
                  'PE_te:{:.4f}'.format(peer_effect_te.item()),
                  'TE_te:{:.4f}'.format(total_effect_te.item()),
                  'AiE_te:{:.4f}'.format(ate_individual_te.item()),
                  'APE_te:{:.4f}'.format(ate_peer_te.item()),
                  'ATE_te:{:.4f}'.format(ate_total_te.item()),

                  '\t'
                  'mape_AiE_te:{:.4f}'.format(ate_mape_indi.item()),
                  'mape_APE_te:{:.4f}'.format(ate_mape_peer.item()),
                  'mape_ATE_te:{:.4f}'.format(ate_mape_total.item()),
                  'mape_iE_te:{:.4f}'.format(ite_mape_indi.item()),
                  'mape_PE_te:{:.4f}'.format(ite_mape_peer.item()),
                  'mape_TE_te:{:.4f}'.format(ite_mape_total.item()),
                  )

    def train(self):
        print("================================Training Start================================")

        if self.args.model == "NetEsimator":
            print("******************NetEsimator******************")
            for epoch in range(self.args.epochs):
                self.train_discriminator(epoch)
                self.train_discriminator_z(epoch)
                self.train_encoder_predictor(epoch)

        elif self.args.model == "TargetedModel_DoubleBSpline" :
            print("******************" + str(self.args.model) + "******************")

            print("******************" + "train 1 step for specified epochs" + "******************")
            for epoch in range(self.args.pre_train_step):
                # 1 step
                self.train_encoder_predictor(epoch)

            print("******************" + "train iteratively for specified epochs" + "******************")
            for epoch in range(self.args.epochs):
                # iterative training step
                self.train_encoder_predictor(epoch)
                # if self.args.beta != 0:
                self.train_fluctuation_param(epoch)
                # self.train_fluctuation_param(epoch)
        elif self.args.model == "netEstCRLSM_IPM":
            print("******************" + str(self.args.model) + "******************")
            for epoch in range(self.args.epochs):
                self.train_encoder_predictor(epoch)
        else:
            print("******************" + str(self.args.model) + "******************")
            for epoch in range(self.args.epochs):
                self.train_encoder_predictor(epoch)

    def one_step_predict(self, A, X, T, Y):
        self.model.eval()
        if self.args.model == 'TargetedModel_DoubleBSpline':
            pred_outcome = self.model.infer_potential_outcome(A, X, T)
            pred_outcome = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome)
            Y = utils.PO_normalize_recover(self.args.normy, self.POTrain, Y)
            pLoss = self.loss(pred_outcome.reshape(-1), Y)
        elif self.args.model == "CFR_INTERFERENCE_GCN":
            pred_treatment, _, pred_outcome, _, _, _ = self.model(A, X, T)
            pLoss = self.Tensor([0])
            save_p_t = self.Tensor([0] * pred_treatment.shape[0])
        elif self.args.model == "netEstCRLSM_IPM":
            pred_outcome,_,_,_,_ = self.model.infer_potential_outcome(A, X, Y, T)
            pred_outcome = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome)
            Y = utils.PO_normalize_recover(self.args.normy, self.POTrain, Y)
            pLoss = self.loss(pred_outcome.reshape(-1), Y)
        else:
            pred_treatment, _, pred_outcome, _, _ = self.model(A, X, T)
            pred_outcome = utils.PO_normalize_recover(self.args.normy, self.POTrain, pred_outcome)
            Y = utils.PO_normalize_recover(self.args.normy, self.POTrain, Y)
            pLoss = self.loss(pred_outcome.reshape(-1), Y)

        return pLoss, pred_outcome, Y

    def predict(self):
        print("================================Predicting================================")
        factualLossTrain, pred_train, YFTrainO = self.one_step_predict(self.trainA, self.trainX, self.trainT,
                                                                       self.YFTrain)
        factualLossVal, pred_val, YFValO = self.one_step_predict(self.valA, self.valX, self.valT, self.YFVal)
        factualLossTest, pred_test, YFTestO = self.one_step_predict(self.testA, self.testX, self.testT, self.YFTest)

        cfLossTrain, cfPred_train, YCFTrainO = self.one_step_predict(self.trainA, self.trainX, self.cfTrainT,
                                                                     self.YCFTrain)
        cfLossVal, cfPred_val, YCFValO = self.one_step_predict(self.valA, self.valX, self.cfValT, self.YCFVal)
        cfLossTest, cfPred_test, YCFTestO = self.one_step_predict(self.testA, self.testX, self.cfTestT, self.YCFTest)

        individual_effect_train, peer_effect_train, total_effect_train, \
                ate_individual_train, ate_peer_train, ate_total_train, \
            mindividual_effect_train, mpeer_effect_train, mtotal_effect_train, \
            mate_individual_train, mate_peer_train, mate_total_train\
                = self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0,
                                           self.train_t0z7, self.train_t0z2, self.train_t0z0)
        individual_effect_val, peer_effect_val, total_effect_val, \
            ate_individual_val, ate_peer_val, ate_total_val, \
            mindividual_effect_val, mpeer_effect_val, mtotal_effect_val, \
            mate_individual_val, mate_peer_val, mate_total_val\
            = self.compute_effect_pehe(self.valA, self.valX, self.val_t1z1, self.val_t1z0,
                                        self.val_t0z7, self.val_t0z2, self.val_t0z0)
        individual_effect_test, peer_effect_test, total_effect_test, \
            ate_individual_test, ate_peer_test, ate_total_test, \
            mindividual_effect_test, mpeer_effect_test, mtotal_effect_test, \
            mate_individual_test, mate_peer_test, mate_total_test\
            = self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0,
                                       self.test_t0z7, self.test_t0z2, self.test_t0z0)

        print(

              'IiE_train:{:.4f}'.format(individual_effect_train.item()),
              'IPE_train:{:.4f}'.format(peer_effect_train.item()),
              'ITE_train:{:.4f}'.format(total_effect_train.item()),

              'IiE_test:{:.4f}'.format(individual_effect_test.item()),
              'IPE_test:{:.4f}'.format(peer_effect_test.item()),
              'ITE_test:{:.4f}'.format(total_effect_test.item()),
            'ate_ie_train{:.4f}'.format(ate_individual_train.detach().cpu().numpy()),
            'ate_pe_train{:.4f}'.format(ate_peer_train.detach().cpu().numpy()),
            'ate_te_train{:.4f}'.format(ate_total_train.detach().cpu().numpy()),

            'ate_ie_test{:.4f}'.format(ate_individual_test.detach().cpu().numpy()),
            'ate_pe_test{:.4f}'.format(ate_peer_test.detach().cpu().numpy()),
            'ate_te_test{:.4f}'.format(ate_total_test.detach().cpu().numpy()),

            'mape_IiE_train:{:.4f}'.format(mindividual_effect_train.item()),
            'mape_IPE_train:{:.4f}'.format(mpeer_effect_train.item()),
            'mape_ITE_train:{:.4f}'.format(mtotal_effect_train.item()),
            'mape_IiE_test:{:.4f}'.format(mindividual_effect_test.item()),
            'mape_IPE_test:{:.4f}'.format(mpeer_effect_test.item()),
            'mape_ITE_test:{:.4f}'.format(mtotal_effect_test.item()),

            'mape_ate_ie_train{:.4f}'.format(mate_individual_train.detach().cpu().numpy()),
            'mape_ate_pe_train{:.4f}'.format(mate_peer_train.detach().cpu().numpy()),
            'mape_ate_te_train{:.4f}'.format(mate_total_train.detach().cpu().numpy()),
            'mape_ate_ie_test{:.4f}'.format(mate_individual_test.detach().cpu().numpy()),
            'mape_ate_pe_test{:.4f}'.format(mate_peer_test.detach().cpu().numpy()),
            'mape_ate_te_test{:.4f}'.format(mate_total_test.detach().cpu().numpy()),
              )

        data = {
            "pred_train_factual": pred_train,
            "PO_train_factual": YFTrainO,
            "pred_val_factual": pred_val,
            "PO_val_factual": YFValO,
            "pred_test_factual": pred_test,
            "PO_test_factual": YFTestO,

            "pred_train_cf": cfPred_train,
            "PO_train_cf": YCFTrainO,
            "pred_val_cf": cfPred_val,
            "PO_val_cf": YCFValO,
            "pred_test_cf": cfPred_test,
            "PO_test_cf": YCFTestO,

            "factualLossTrain": factualLossTrain.detach().cpu().numpy(),
            "factualLossVal": factualLossVal.detach().cpu().numpy(),
            "factualLossTest": factualLossTest.detach().cpu().numpy(),
            "cfLossTrain": cfLossTrain.detach().cpu().numpy(),
            "cfLossVal": cfLossVal.detach().cpu().numpy(),
            "cfLossTest": cfLossTest.detach().cpu().numpy(),

            "individual_effect_train": individual_effect_train.detach().cpu().numpy(),
            "peer_effect_train": peer_effect_train.detach().cpu().numpy(),
            "total_effect_train": total_effect_train.detach().cpu().numpy(),
            "individual_effect_val": individual_effect_val.detach().cpu().numpy(),
            "peer_effect_val": peer_effect_val.detach().cpu().numpy(),
            "total_effect_val": total_effect_val.detach().cpu().numpy(),
            "individual_effect_test": individual_effect_test.detach().cpu().numpy(),
            "peer_effect_test": peer_effect_test.detach().cpu().numpy(),
            "total_effect_test": total_effect_test.detach().cpu().numpy(),

            "ate_individual_train": ate_individual_train.detach().cpu().numpy(),
            "ate_peer_train": ate_peer_train.detach().cpu().numpy(),
            "ate_total_train": ate_total_train.detach().cpu().numpy(),
            "ate_individual_val": ate_individual_val.detach().cpu().numpy(),
            "ate_peer_val":ate_peer_val.detach().cpu().numpy(),
            "ate_total_val":ate_total_val.detach().cpu().numpy(),
            "ate_individual_test":ate_individual_test.detach().cpu().numpy(),
            "ate_peer_test":ate_peer_test.detach().cpu().numpy(),
            "ate_total_test":ate_total_test.detach().cpu().numpy(),

            "mape_individual_effect_train": mindividual_effect_train.detach().cpu().numpy(),
            "mape_peer_effect_train": mpeer_effect_train.detach().cpu().numpy(),
            "mape_total_effect_train": mtotal_effect_train.detach().cpu().numpy(),
            "mape_individual_effect_val": mindividual_effect_val.detach().cpu().numpy(),
            "mape_peer_effect_val": mpeer_effect_val.detach().cpu().numpy(),
            "mape_total_effect_val": mtotal_effect_val.detach().cpu().numpy(),
            "mape_individual_effect_test": mindividual_effect_test.detach().cpu().numpy(),
            "mape_peer_effect_test": mpeer_effect_test.detach().cpu().numpy(),
            "mape_total_effect_test": mtotal_effect_test.detach().cpu().numpy(),

            "mape_ate_individual_train": mate_individual_train.detach().cpu().numpy(),
            "mape_ate_peer_train": mate_peer_train.detach().cpu().numpy(),
            "mape_ate_total_train": mate_total_train.detach().cpu().numpy(),
            "mape_ate_individual_val": mate_individual_val.detach().cpu().numpy(),
            "mape_ate_peer_val": mate_peer_val.detach().cpu().numpy(),
            "mape_ate_total_val": mate_total_val.detach().cpu().numpy(),
            "mape_ate_individual_test": mate_individual_test.detach().cpu().numpy(),
            "mape_ate_peer_test": mate_peer_test.detach().cpu().numpy(),
            "mape_ate_total_test": mate_total_test.detach().cpu().numpy(),

            "args": str(self.args),
                }

        if self.args.model == "NetEsimator":
            print("================================Save prediction...================================")
            file = "./results/baselines/" + self.args.model + '/' + self.args.dataset + "/perf2k/" + self.args.dataset + "_prediction_expID_" + str(
                    self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
                    self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl"


        elif self.args.model == 'TargetedModel_DoubleBSpline':
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + \
                       str(self.args.expID) + \
                       "_alpha_" + str(self.args.alpha) + "_gamma_" + str(self.args.gamma) + "_beta_" + '{:.4f}'.format(self.args.beta)  + \
                       "_flipRate_" + str(self.args.flipRate) + "_numGrid_" + str(self.args.num_grid) + \
                       '_kn_' + str(self.args.tr_knots) + "_lr_1_" + str(self.args.lr_1step) + "_lr_2_" + str(self.args.lr_2step) + ".pkl"

        elif self.args.model == "netEstCRLSM_IPM":
            print("================================Save netEst_crl prediction...================================")
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + str(
                    self.args.expID) + "_alpha_" + str(self.args.vae_alpha) + "_numgrid_" + str(
                    self.args.num_grid) + '_hd_'+str(self.args.hidden)
            # if self.args.ZI_dim != 16:
            file +=  "_dim_" + str(self.args.ZI_dim)+ str(self.args.ZC_dim)+ str(self.args.ZN_dim)
            file += "_flipRate_" + str(self.args.flipRate) + ".pkl"

        elif self.args.model == "CFR_INTERFERENCE_GCN":
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + str(
                self.args.expID) + "_alphabase_" + str(self.args.alpha_base) + "_lr_" + str(
                self.args.lr) + "_flipRate_" + str(self.args.flipRate)  + ".pkl"

        else:
            print("================================Save Bseline prediction...================================")
            file = "./results/baselines/" + self.args.model + "/" + self.args.dataset + "/perf/prediction_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
                self.args.gamma) + "_flipRate_" + str(self.args.flipRate)  + ".pkl"


        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            # print('not save')
            pkl.dump(data, f)
        print("================================Save prediction done!================================")

    def save_curve(self):
        print("================================Save curve...================================")
        data = {"dissTrain": self.dissTrain,
                "dissVal": self.dissVal,
                "dissTrainHalf": self.dissTrainHalf,
                "dissValHalf": self.dissValHalf,
                "diss_zTrain": self.diss_zTrain,
                "diss_zVal": self.diss_zVal,
                "diss_zTrainHalf": self.diss_zTrainHalf,
                "diss_zValHalf": self.diss_zValHalf}

        with open("../results/" + str(self.args.dataset) + "/curve/" + "curve_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
            self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl", "wb") as f:
            pkl.dump(data, f)
        # print("================================Save curve done!================================")

    def save_embedding(self):
        print("================================Save embedding...================================")
        _, _, _, embedsTrain, _ = self.model(self.trainA, self.trainX, self.trainT)
        _, _, _, embedsTest, _ = self.model(self.testA, self.testX, self.testT)
        data = {"embedsTrain": embedsTrain.cpu().detach().numpy(), "embedsTest": embedsTest.cpu().detach().numpy(),
                "trainT": self.trainT.cpu().detach().numpy(), "testT": self.testT.cpu().detach().numpy(),
                "trainZ": self.trainZ.cpu().detach().numpy(), "testZ": self.testZ.cpu().detach().numpy()}
        with open("../results/" + str(self.args.dataset) + "/embedding/" + "embeddings_expID_" + str(
                self.args.expID) + "_alpha_" + str(self.args.alpha) + "_gamma_" + str(
            self.args.gamma) + "_flipRate_" + str(self.args.flipRate) + ".pkl", "wb") as f:
            pkl.dump(data, f)
        print("================================Save embedding done!================================")
