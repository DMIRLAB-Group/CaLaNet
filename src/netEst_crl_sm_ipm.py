import torch
import torch.nn as nn
from modules import GCN, NN, Predictor, Discriminator, Density_Estimator, Discriminator_simplified
from model_utils import Normal, MLP, weights_init, Vcnet, UnnormalizedExponentialFamily
import utils as utils


class netEstCRLSM_IPM(nn.Module):
    # networked effect estimator with causal representation learning

    def __init__(self, Xshape, hidden, dropout, n_layers=3,
                 ZI_dim=8, ZC_dim=8, ZN_dim=8,
                 init_weight=True, num_grid=20,
                 device='cuda',activation='lrelu', slope=.1,):
        super(netEstCRLSM_IPM, self).__init__()
        self.ZN_dim, self.ZI_dim, self.ZC_dim = ZN_dim, ZI_dim, ZC_dim
        self.continuous_dist = Normal(device=device)

        self.gcn_vae = GCN(nfeat=Xshape, nclass=int(hidden/2), dropout=dropout, init_weight=init_weight)
        self.encoder_vae_mean = MLP(Xshape + int(hidden/2), ZI_dim+ZC_dim+ZN_dim, hidden, n_layers,
                               activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.encoder_vae_logv = MLP(Xshape + int(hidden/2), ZI_dim+ZC_dim+ZN_dim, hidden, n_layers,
                               activation=activation, slope=slope, device=device, init_weight=init_weight)

        self.decoder_vae_mean = MLP(ZI_dim+ZC_dim+ZN_dim, Xshape, hidden, n_layers,
                               activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.decoder_vae_logv = MLP(ZI_dim+ZC_dim+ZN_dim, Xshape, hidden, n_layers,
                               activation=activation, slope=slope, device=device, init_weight=init_weight)

        # The prior network
        # whether it use same gcn?
        self.gcn_prior = GCN(nfeat=Xshape, nclass=int(hidden / 2), dropout=dropout, init_weight=init_weight)
        self.t_f = lambda z: torch.cat([z, torch.pow(z, 2)], axis=-1)
        self.t_nf = MLP(ZI_dim + ZC_dim + ZN_dim, 1, hidden, n_layers=1,
                        activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.lambd_nf = MLP(int(hidden / 2), 1, hidden, n_layers=1,
                            activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.lambd_f = MLP(int(hidden / 2), 2 * (ZI_dim + ZC_dim + ZN_dim), hidden, n_layers=1,
                           activation=activation, slope=slope, device=device, init_weight=init_weight)

        # outcome prediction
        self.encoder = GCN(nfeat=ZC_dim+ZN_dim, nclass=int(hidden/2), dropout=dropout, init_weight=init_weight)
        self.Z_ZN = MLP(ZI_dim + ZC_dim + int(hidden / 2), 16, hidden, n_layers,
                           activation=activation, slope=slope, device=device, init_weight=init_weight)

        self.y1_mean = MLP(16+1, 1, hidden, n_layers,
                         activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.y0_mean = MLP(16+1, 1, hidden, n_layers,
                         activation=activation, slope=slope, device=device, init_weight=init_weight)
        # self.y1_mean = Vcnet(16, 1, hidden, n_layers,
        #                      activation=activation, slope=slope, device=device, init_weight=init_weight)
        # self.y0_mean = Vcnet(16, 1, hidden, n_layers,
        #                    activation=activation, slope=slope, device=device, init_weight=init_weight)
        self.y1_logv = .01 * torch.ones(1).to(device)
        self.y0_logv = .01 * torch.ones(1).to(device)

        self.continuous_dist = Normal(device=device)
        self.prior_dist = UnnormalizedExponentialFamily(self.t_f, self.t_nf, self.lambd_f, self.lambd_nf)

        self.to(device)

    def loss_func_wass(self, x, y):
        loss, _ = utils.wasserstein(x, y)
        return loss

    def loss_IPM(self, phi, t, z):
        t_random = t.clone().detach()
        z_random = z.clone().detach()
        ran_index = torch.randperm(t_random.size(0))
        t_random = t_random[ ran_index,:]
        z_random = z_random[ ran_index,:]
        phi_fix = phi.clone().detach()

        loss = self.loss_func_wass(
            torch.cat((phi,t,z), dim=1), torch.cat((phi_fix,t_random,z_random), dim=1)
        )
        return loss

    def score_matching(self, Z, XN):
        return self.prior_dist.log_pdf(Z, XN)

    def encoder_params(self, X, XN):
        Z_mean = self.encoder_vae_mean(torch.cat((X, XN), dim=1))
        Z_logl = self.encoder_vae_logv(torch.cat((X, XN), dim=1))
        return Z_mean, Z_logl.exp()

    def decoder_params(self, Z):
        X_mean = self.decoder_vae_mean(Z)
        return X_mean, .01 * torch.ones(1)
            # X_logv.exp()

    def vae_module(self, A, X):
        XN_prior = self.gcn_prior(X, A)
        XN = self.gcn_vae(X, A)

        encoder_params = self.encoder_params(X, XN)
        Z_hat = self.continuous_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(Z_hat)
        # split Z into ZI ZC ZN
        ZI_hat, ZC_hat, ZN_hat = \
            Z_hat[:,:self.ZI_dim],\
            Z_hat[:,self.ZI_dim:self.ZI_dim+self.ZC_dim],\
            Z_hat[:,self.ZI_dim+self.ZC_dim:self.ZI_dim+self.ZC_dim+self.ZN_dim]

        return XN_prior, encoder_params, decoder_params,\
            ZI_hat, ZC_hat, ZN_hat, Z_hat

    def predictor(self, confounder, Z, T):
        y0_mean = self.y0_mean(torch.cat((Z, confounder), dim=1))
        y1_mean = self.y1_mean(torch.cat((Z, confounder), dim=1))
        y_mean = torch.where(T[:,None]==1, y1_mean, y0_mean)

        return (y1_mean, self.y1_logv.exp()), (y0_mean, self.y0_logv.exp()), (y_mean, self.y1_logv.exp())

    def elbo(self, X, XN_prior, encoder_params, decoder_params, Z_hat):
        log_pX_Z = self.continuous_dist.log_pdf(X, *decoder_params)
        log_qZ_XNX = self.continuous_dist.log_pdf(Z_hat, *encoder_params)
        loss_sm = self.score_matching(Z_hat, XN_prior)
        return (log_pX_Z - log_qZ_XNX).mean(), loss_sm.mean()

    def forward(self, A, X, T, Y, neighborT=None, alpha=1.):
        XN_prior, encoder_params, decoder_params, \
            ZI_hat, ZC_hat, ZN_hat, Z_hat = self.vae_module(A, X)

        embeddings = self.encoder(torch.cat((ZC_hat,ZN_hat), dim=1), A)
        confounder = self.Z_ZN(torch.cat((embeddings, ZI_hat, ZC_hat), dim=1))

        if neighborT is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = neighborT

        # ZNi ZI ZC Z
        y1_params, y0_params, y_params = self.predictor(confounder, neighborAverageT.reshape(-1, 1), T)
        y1_mean, _ = y1_params
        y0_mean, _ = y0_params

        Y = Y[:,None]
        elbo, loss_sm = self.elbo(X, XN_prior, encoder_params, decoder_params, Z_hat)
        log_py_zzn = self.continuous_dist.log_pdf(Y, *y_params)

        # for balancing loss
        ZI_hat, ZC_hat, ZN_hat, Z_hat = ZI_hat.clone().detach(), ZC_hat.clone().detach(), \
            ZN_hat.clone().detach(), Z_hat.clone().detach()
        embeddings_b = self.encoder(torch.cat((ZC_hat,ZN_hat), dim=1), A)
        confounder_b = self.Z_ZN(torch.cat((embeddings_b, ZI_hat, ZC_hat), dim=1))

        loss_balance = self.loss_IPM(phi=confounder_b, t=T.reshape(-1, 1), z=neighborAverageT.reshape(-1, 1))

        return y1_mean, y0_mean, elbo, log_py_zzn.mean(), loss_sm, loss_balance

    def infer_potential_outcome(self, A, X, T, Y, neighborT=None, alpha=1.):
        XN_prior, encoder_params, decoder_params, \
            _, _, _, Z_hat = self.vae_module(A, X)

        Z_mean, _ = encoder_params
        ZI_mean = Z_mean[:,:self.ZI_dim]
        ZC_mean = Z_mean[:,self.ZI_dim:self.ZI_dim+self.ZC_dim]
        ZN_mean = Z_mean[:,self.ZI_dim+self.ZC_dim:self.ZI_dim+self.ZC_dim+self.ZN_dim]

        embeddings = self.encoder(torch.cat((ZC_mean,ZN_mean), dim=1), A)
        confounder = self.Z_ZN(torch.cat((embeddings, ZI_mean, ZC_mean), dim=1))

        if neighborT is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = neighborT

        # ZNi ZI ZC Z
        y1_params, y0_params, y_params = self.predictor(confounder,neighborAverageT.reshape(-1, 1), T)
        y1_mean, _ = y1_params
        y0_mean, _ = y0_params
        y, _ = y_params
        Y = Y[:,None]
        elbo, loss_sm = self.elbo(X, XN_prior, encoder_params, decoder_params, Z_hat)
        log_py_zzn = self.continuous_dist.log_pdf(Y, *y_params).mean()

        return y, y1_mean, y0_mean,  elbo, log_py_zzn#, loss_sm