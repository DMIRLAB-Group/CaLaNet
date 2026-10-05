from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
import random
from torch.autograd import grad


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


def log_laplace(x, mu, b, broadcast_size=False):
    """compute the log-pdf of a laplace distribution with diagonal covariance"""
    # b might not have batch_dimension. This case is handled by _check_inputs
    if broadcast_size:
        mu, b = _check_inputs(x.size(), mu, b)
    else:
        mu, b = _check_inputs(None, mu, b)
    return -torch.log(2 * b) - (x - mu).abs().div(b)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers,
                 activation='none', slope=.1, device='cpu', init_weight=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(lambda x: F.relu(x))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

        if init_weight:
            self._initialize_weights()

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h

    def _initialize_weights(self, ini_normal_variance=0.1):
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                self.fc[c].weight.data.normal_(0, ini_normal_variance)


class cleanIVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
        return elbo, z


class cleanVAE(nn.Module):

    def __init__(self, data_dim, latent_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1)
        self.prior_var = torch.ones(1)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def forward(self, x):
        g, v = self.encoder(x)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z = self.forward(x)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps = log_normal(z, None, None, broadcast_size=True).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps)).mean()
        return elbo, z


class Discriminator(nn.Module):
    def __init__(self, z_dim=5, hdim=1000):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, 2),
        )
        self.hdim = hdim

    def forward(self, z):
        return self.net(z).squeeze()


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class UnnormalizedExponentialFamily(Dist):
    # from https://github.com/yaochenzhu/CiVAE/blob/main/lib/dist.py
    def __init__(self, t_f, t_nf, lambd_f, lambd_nf, *args, **kwargs):
        super().__init__()
        self.t_f = t_f
        self.t_nf = t_nf
        self.lambd_f = lambd_f
        self.lambd_nf = lambd_nf


    def log_pdf(self, value, XN):
        lambd_f_XN = self.lambd_f(XN)       # shape [n, 2 * d]
        lambd_nf_XN = self.lambd_nf(XN)     # shape [n, 1]

        ### Infer the batch shape and event shape
        self.batch_shape = torch.Size([lambd_f_XN.shape[0]])
        self.event_shape = torch.Size([lambd_f_XN.shape[1] // 2])

        ### Used for score matching purpose
        ### Calculate the negative match score
        ### which will be optimized normally in the Elbo
        ### should be maximizing
        n_samples, d = value.shape
        event_shape = d
        # z = value.detach().requires_grad_(True)  # z: [n, d]
        z = value.requires_grad_(True)  # z: [n, d]

        # z.retain_grad()

        # Compute t_nf(z): shape [n, k]
        t_nf_z = self.t_nf(z)

        # Compute unnormalized log-density (non-factorized part): p_hat = <t(z), λ(x)>
        pnf_hat = (t_nf_z * lambd_nf_XN).sum(dim=1, keepdim=True)  # shape [n, 1]

        # Compute gradients w.r.t. z: shape [n, d]
        grads = grad(pnf_hat.sum(), z, create_graph=True)[0]  # ∇_z ⟨t(z), λ⟩

        # Compute squared norm of gradients per sample
        squared_norm = grads.pow(2).sum(dim=1)  # shape [n]

        # Factorized components
        lambda1 = lambd_f_XN[:, :event_shape]  # shape [n, d]
        lambda2 = lambd_f_XN[:, event_shape:]  # shape [n, d]

        neg_match = torch.zeros(n_samples, device=value.device)
        neg_match -= 2 * lambda2.sum(dim=-1)  # linear term
        neg_match -= 0.5 * (lambda1 ** 2).sum(dim=-1)  # quadratic term
        neg_match -= 0.5 * squared_norm  # non-factorized score term

        return neg_match  # this should be maximized

    def log_pdf_old(self, value, XN):
        # the old version is very slow, use the new one instead
        lambd_f_XN = self.lambd_f(XN)       # shape [n, 2 * d]
        lambd_nf_XN = self.lambd_nf(XN)     # shape [n, 1]

        ### Infer the batch shape and event shape
        self.batch_shape = torch.Size([lambd_f_XN.shape[0]])
        self.event_shape = torch.Size([lambd_f_XN.shape[1] // 2])

        ### Used for score matching purpose
        ### Calculate the negative match score
        ### which will be optimized normally in the Elbo
        ### should be maximizing
        z = value                     # shape [n, d]
        n_samples = z.shape[0]

        ### Calculate the sufficient statistics
        # t_f = self.t_f(z)
        t_nf = self.t_nf(z)

        ### Calculate the unnormalized log-density corresponds to
        ### the nonfactorized part
        pnf_hat = torch.mul(t_nf, lambd_nf_XN).sum(axis=1, keepdims=True)


        ### Calculate the negative match score
        neg_match = torch.zeros((n_samples))

        ### Calculate the factorized part
        neg_match -= 2 * lambd_f_XN[:, self.event_shape[0]:].sum(axis=-1)
        neg_match -= 0.5 * torch.pow(lambd_f_XN[:, :self.event_shape[0]], 2).sum(axis=-1)

        ### Calculate the non-factorized part
        for i in range(n_samples):
            grads = grad(pnf_hat[i], z, create_graph=True, retain_graph=True)[0][i]
            # neg_match[i] -= 0.5 * torch.pow(grads, 2).sum() * 0.1
            neg_match[i] -= 0.5 * torch.pow(grads, 2).sum()

        return neg_match

    def sample(self):
        pass

class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        # eps = self._dist.sample(mu.size()).squeeze()
        eps = self._dist.sample(mu.size()).squeeze(dim=-1)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        if x.shape != mu.shape:
            print(x.shape, mu.shape)
        assert x.shape == mu.shape
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class Bernoulli(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.bernoulli.Bernoulli(0.5 * torch.ones(1).to(self.device))
        self.name = 'bernoulli'

    def sample(self, p):
        eps = self._dist.sample(p.size())
        return eps

    def log_pdf(self, x, f, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            f = f.view(param_shape)
        # 1e-8 to avoid nan
        lpdf = x * torch.log(f+1e-8) + (1 - x) * torch.log(1 - f+1e-8)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class RelaxedBernoulli(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.relaxed_bernoulli.RelaxedBernoulli
        self.name = 'bernoulli'

    def sample(self, p, temperature=1.):
        eps = self._dist(temperature=temperature, probs=p).rsample()
        return eps

    def log_pdf(self, x, f, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            f = f.view(param_shape)
        # 1e-8 to avoid nan
        lpdf = x * torch.log(f+1e-8) + (1 - x) * torch.log(1 - f+1e-8)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf



class DiscreteIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv

    def decoder_params(self, z):
        f = self.f(z)
        return torch.sigmoid(f)

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.reparameterize(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def elbo(self, x, u):
        f, (g, logv), z, (h, logl) = self.forward(x, u)
        BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        l = logl.exp()
        v = logv.exp()
        KLD = -0.5 * torch.sum(logl - logv - 1 + (g - h).pow(2) / l + v / l)
        return (BCE + KLD) / x.size(0), z  # average per batch


class VAE(nn.Module):

    def __init__(self, latent_dim, data_dim, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder
        self.prior_dist = Normal(device)
        self.prior_mean = torch.zeros(1).to(device)
        self.prior_var = torch.ones(1).to(device)

        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def forward(self, x):
        encoder_params = self.encoder_params(x)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, (self.prior_mean, self.prior_var)

    def elbo(self, x):
        decoder_params, encoder_params, z, prior_params = self.forward(x)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_x = self.encoder_dist.log_pdf(z, *encoder_params)
        log_pz = self.prior_dist.log_pdf(z, *prior_params)

        return (log_px_z + log_pz - log_qz_x).mean(), z


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, data_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv

    def decoder_params(self, z):
        f = self



class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, device='cuda'):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=.1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.to(device)

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


class Vcnet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none',
                 degree=2, knots=None, slope=.1, device='cuda'):
        print('vcnet')
        print(input_dim, output_dim, hidden_dim, n_layers)
        super(Vcnet, self).__init__()
        if knots is None:
            knots = [0.33, 0.66]
            # knots = list(np.arange(0.1, 1, 0.1))
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        """
        cfg = []
        for i in range(n_layers):
            if i == 0:
                cfg.append((input_dim, hidden_dim, 1, activation))
            elif i == n_layers-1:
                cfg.append((hidden_dim, output_dim, 1, 'id'))
            else:
                cfg.append((hidden_dim, hidden_dim, 1, activation))
        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)
        self.Q.to(device)

    def forward(self, x):
        # x = torch.concatenate((t,x), dim=1)
        # x include t,x
        Q = self.Q(x)
        return Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
