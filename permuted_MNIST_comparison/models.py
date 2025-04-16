import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch import special 
    HAS_TORCH_SPECIAL = True
except ImportError:
    HAS_TORCH_SPECIAL = False
    def erf_approx(x: torch.Tensor):
        p = 0.3275911
        a1,a2,a3,a4,a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        sign = torch.sign(x)
        t = 1.0/(1.0 + p*torch.abs(x))
        y = 1.0 - (a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5)*torch.exp(-x*x)
        return sign*y
else:
    def erf_approx(x: torch.Tensor):
        return special.erf(x)

def Eabs_gaussian(mu, sigma, mu_p):
    diff = mu - mu_p
    exp_val = torch.exp(-(diff**2)/(2*sigma**2))
    erf_val = erf_approx(diff / (math.sqrt(2)*sigma))

    term1 = sigma * math.sqrt(2.0/math.pi) * exp_val
    term2 = diff * erf_val
    return term1 + term2

def kl_gaussian_laplace_1dim(mu, log_var, mu_p, log_b):
    sigma = torch.exp(0.5*log_var)
    b = torch.exp(log_b)

    kl_val = torch.log(torch.tensor(2.0, device=mu.device)) + log_b
    kl_val -= 0.5*( math.log(2*math.pi) + log_var )
    kl_val -= 0.5
    eabs = Eabs_gaussian(mu, sigma, mu_p)
    kl_val += eabs / b

    return kl_val

def kl_gaussian_laplace_allparams(weight_mean, weight_log_var,
                                  laplace_mean_w, laplace_log_b_w,
                                  bias_mean, bias_log_var,
                                  laplace_mean_b, laplace_log_b_b):
    w_kl = kl_gaussian_laplace_1dim(weight_mean, weight_log_var, laplace_mean_w, laplace_log_b_w)
    w_kl_sum = w_kl.sum()

    b_kl = kl_gaussian_laplace_1dim(bias_mean, bias_log_var, laplace_mean_b, laplace_log_b_b)
    b_kl_sum = b_kl.sum()

    return w_kl_sum + b_kl_sum


class VanillaNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_tasks=1):
        super().__init__()
        self.num_tasks = num_tasks
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.features = nn.Sequential(*layers)

        if self.num_tasks == 1:
            self.output_layer = nn.Linear(in_dim, output_size)
        else:
            self.output_layers = nn.ModuleList([nn.Linear(in_dim, output_size)
                                                for _ in range(num_tasks)])

    def forward(self, x, task_id=0, sample=False):
        x = self.features(x)
        if self.num_tasks == 1:
            return self.output_layer(x)
        else:
            return self.output_layers[task_id](x)


class MFVI_Layer(nn.Module):
    def __init__(self, in_features, out_features,
                 prior_mean=0.0, prior_log_var=0.0):
        super().__init__()
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer('prior_weight_mean',
                             torch.full((out_features, in_features), prior_mean))
        self.register_buffer('prior_weight_log_var',
                             torch.full((out_features, in_features), prior_log_var))
        self.register_buffer('prior_bias_mean',
                             torch.full((out_features,), prior_mean))
        self.register_buffer('prior_bias_log_var',
                             torch.full((out_features,), prior_log_var))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=np.sqrt(5))
        nn.init.constant_(self.weight_log_var, -6.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mean)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mean, -bound, bound)
        nn.init.constant_(self.bias_log_var, -6.0)

    def forward(self, x, sample=True):
        if self.training or sample:
            eps_w = torch.randn_like(self.weight_mean)
            eps_b = torch.randn_like(self.bias_mean)
            w_std = torch.exp(0.5 * self.weight_log_var)
            b_std = torch.exp(0.5 * self.bias_log_var)
            weight = self.weight_mean + eps_w * w_std
            bias = self.bias_mean + eps_b * b_std
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        kl_w = 0.5 * torch.sum(
            self.prior_weight_log_var - self.weight_log_var +
            (torch.exp(self.weight_log_var) +
             (self.weight_mean - self.prior_weight_mean) ** 2)
            / torch.exp(self.prior_weight_log_var) - 1.0
        )
        kl_b = 0.5 * torch.sum(
            self.prior_bias_log_var - self.bias_log_var +
            (torch.exp(self.bias_log_var) +
             (self.bias_mean - self.prior_bias_mean) ** 2)
            / torch.exp(self.prior_bias_log_var) - 1.0
        )
        return kl_w + kl_b

    def set_prior(self, w_mean, w_logvar, b_mean, b_logvar):
        self.prior_weight_mean.data.copy_(w_mean.detach())
        self.prior_weight_log_var.data.copy_(w_logvar.detach())
        self.prior_bias_mean.data.copy_(b_mean.detach())
        self.prior_bias_log_var.data.copy_(b_logvar.detach())


class MFVI_NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_tasks=1,
                 initial_prior_mean=0.0, initial_prior_log_var=0.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h_dim in hidden_sizes:
            self.layers.append(MFVI_Layer(in_dim, h_dim,initial_prior_mean,initial_prior_log_var))
            in_dim = h_dim
        if self.num_tasks == 1:
            self.output_layer = MFVI_Layer(in_dim, output_size,initial_prior_mean,initial_prior_log_var)
        else:
            self.output_layers = nn.ModuleList([
                MFVI_Layer(in_dim, output_size,initial_prior_mean,initial_prior_log_var)
                for _ in range(num_tasks)
            ])

    def forward(self, x, task_id=0, sample=True):
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample))
        if self.num_tasks == 1:
            x = self.output_layer(x, sample=sample)
        else:
            x = self.output_layers[task_id](x, sample=sample)
        return x

    def kl_divergence(self, task_id=0):
        total_kl = sum(layer.kl_divergence() for layer in self.layers)
        if self.num_tasks == 1:
            total_kl += self.output_layer.kl_divergence()
        else:
            total_kl += self.output_layers[task_id].kl_divergence()
        return total_kl

    def set_prior_from_posterior(self):
        for layer in self.layers:
            layer.set_prior(layer.weight_mean.data, layer.weight_log_var.data,
                            layer.bias_mean.data, layer.bias_log_var.data)
        if self.num_tasks == 1:
            self.output_layer.set_prior(self.output_layer.weight_mean.data,
                                        self.output_layer.weight_log_var.data,
                                        self.output_layer.bias_mean.data,
                                        self.output_layer.bias_log_var.data)
        else:
            for out_layer in self.output_layers:
                out_layer.set_prior(out_layer.weight_mean.data,
                                    out_layer.weight_log_var.data,
                                    out_layer.bias_mean.data,
                                    out_layer.bias_log_var.data)

    def initialize_from_vanilla_model(self, vanilla_model):
        sd_vanilla = vanilla_model.state_dict()

        idx_lin = 0
        for layer_idx in range(len(self.layers)):
            wkey = f"features.{idx_lin}.weight"
            bkey = f"features.{idx_lin}.bias"
            self.layers[layer_idx].weight_mean.data.copy_(sd_vanilla[wkey])
            self.layers[layer_idx].bias_mean.data.copy_(sd_vanilla[bkey])
            nn.init.constant_(self.layers[layer_idx].weight_log_var, -6.0)
            nn.init.constant_(self.layers[layer_idx].bias_log_var, -6.0)
            idx_lin += 2

        if self.num_tasks == 1:
            wkey = "output_layer.weight"
            bkey = "output_layer.bias"
            self.output_layer.weight_mean.data.copy_(sd_vanilla[wkey])
            self.output_layer.bias_mean.data.copy_(sd_vanilla[bkey])
            nn.init.constant_(self.output_layer.weight_log_var, -6.0)
            nn.init.constant_(self.output_layer.bias_log_var, -6.0)
        else:
            for i, out_layer in enumerate(self.output_layers):
                wkey = f"output_layers.{i}.weight"
                bkey = f"output_layers.{i}.bias"
                out_layer.weight_mean.data.copy_(sd_vanilla[wkey])
                out_layer.bias_mean.data.copy_(sd_vanilla[bkey])
                nn.init.constant_(out_layer.weight_log_var, -6.0)
                nn.init.constant_(out_layer.bias_log_var, -6.0)
        print("MFVI model means initialized from VanillaNN.")


class MFVI_Layer_AnalyticLaplace(nn.Module):
    def __init__(self, in_features, out_features, global_prior_b=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # posterior
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))

        # laplace prior
        self.register_buffer('laplace_mean_w', torch.zeros(out_features, in_features))
        self.register_buffer('laplace_log_b_w', torch.zeros(out_features, in_features))
        self.register_buffer('laplace_mean_b', torch.zeros(out_features))
        self.register_buffer('laplace_log_b_b', torch.zeros(out_features))

        self.global_prior_b = global_prior_b
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=np.sqrt(5))
        nn.init.constant_(self.weight_log_var, -6.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mean)
        bound = 1./ np.sqrt(fan_in) if fan_in>0 else 0
        nn.init.uniform_(self.bias_mean, -bound, bound)
        nn.init.constant_(self.bias_log_var, -6.0)

        with torch.no_grad():
            self.laplace_mean_w.zero_()
            self.laplace_mean_b.zero_()
            init_log_b = math.log(self.global_prior_b)
            self.laplace_log_b_w.fill_(init_log_b)
            self.laplace_log_b_b.fill_(init_log_b)

    def forward(self, x, sample=True):
        if self.training or sample:
            eps_w = torch.randn_like(self.weight_mean)
            eps_b = torch.randn_like(self.bias_mean)
            std_w = torch.exp(0.5*self.weight_log_var)
            std_b = torch.exp(0.5*self.bias_log_var)
            w_samp = self.weight_mean + eps_w*std_w
            b_samp = self.bias_mean + eps_b*std_b
        else:
            w_samp = self.weight_mean
            b_samp = self.bias_mean
        return F.linear(x, w_samp, b_samp)

    def kl_divergence(self):
        kl_val = kl_gaussian_laplace_allparams(
            self.weight_mean, self.weight_log_var,
            self.laplace_mean_w, self.laplace_log_b_w,
            self.bias_mean, self.bias_log_var,
            self.laplace_mean_b, self.laplace_log_b_b
        )
        return kl_val

    def set_prior_from_posterior(self):
        w_mean_det = self.weight_mean.detach()
        w_logvar_det = self.weight_log_var.detach()
        b_mean_det = self.bias_mean.detach()
        b_logvar_det = self.bias_log_var.detach()

        self.laplace_mean_w.copy_(w_mean_det)
        self.laplace_mean_b.copy_(b_mean_det)

        half_w_logvar = 0.5*w_logvar_det
        log_b_w = half_w_logvar - math.log(math.sqrt(2.0))
        self.laplace_log_b_w.copy_(log_b_w)

        half_b_logvar = 0.5*b_logvar_det
        log_b_b = half_b_logvar - math.log(math.sqrt(2.0))
        self.laplace_log_b_b.copy_(log_b_b)


class MFVI_NN_AnalyticLaplace(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_tasks=1, prior_b=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.layers = nn.ModuleList()
        in_dim = input_size
        for h_dim in hidden_sizes:
            self.layers.append(MFVI_Layer_AnalyticLaplace(in_dim, h_dim, global_prior_b=prior_b))
            in_dim = h_dim

        if self.num_tasks == 1:
            self.output_layer = MFVI_Layer_AnalyticLaplace(in_dim, output_size, global_prior_b=prior_b)
        else:
            self.output_layers = nn.ModuleList([
                MFVI_Layer_AnalyticLaplace(in_dim, output_size, global_prior_b=prior_b)
                for _ in range(num_tasks)
            ])

    def forward(self, x, task_id=0, sample=True):
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample))
        if self.num_tasks == 1:
            return self.output_layer(x, sample=sample)
        else:
            return self.output_layers[task_id](x, sample=sample)

    def kl_divergence(self, task_id=0):
        total_kl = 0.0
        for layer in self.layers:
            total_kl += layer.kl_divergence()
        if self.num_tasks == 1:
            total_kl += self.output_layer.kl_divergence()
        else:
            total_kl += self.output_layers[task_id].kl_divergence()
        return total_kl

    def set_prior_from_posterior(self):
        for layer in self.layers:
            layer.set_prior_from_posterior()
        if self.num_tasks == 1:
            self.output_layer.set_prior_from_posterior()
        else:
            for out_layer in self.output_layers:
                out_layer.set_prior_from_posterior()

    def initialize_from_vanilla_model(self, vanilla_model):
        vdict = vanilla_model.state_dict()

        idx_lin = 0
        for i, layer in enumerate(self.layers):
            wkey = f"features.{idx_lin}.weight"
            bkey = f"features.{idx_lin}.bias"
            layer.weight_mean.data.copy_(vdict[wkey])
            layer.bias_mean.data.copy_(vdict[bkey])
            nn.init.constant_(layer.weight_log_var, -6.)
            nn.init.constant_(layer.bias_log_var, -6.)
            idx_lin += 2

        if self.num_tasks == 1:
            wkey = "output_layer.weight"
            bkey = "output_layer.bias"
            self.output_layer.weight_mean.data.copy_(vdict[wkey])
            self.output_layer.bias_mean.data.copy_(vdict[bkey])
            nn.init.constant_(self.output_layer.weight_log_var, -6.)
            nn.init.constant_(self.output_layer.bias_log_var, -6.)
        else:
            for i, out_layer in enumerate(self.output_layers):
                wkey = f"output_layers.{i}.weight"
                bkey = f"output_layers.{i}.bias"
                out_layer.weight_mean.data.copy_(vdict[wkey])
                out_layer.bias_mean.data.copy_(vdict[bkey])
                nn.init.constant_(out_layer.weight_log_var, -6.)
                nn.init.constant_(out_layer.bias_log_var, -6.)

