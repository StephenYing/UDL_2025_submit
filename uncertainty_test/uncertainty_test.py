import os
import math
import pickle
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.cluster import KMeans

# Data
def load_mnist_pkl(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set


def to_one_hot(labels, num_classes=10):
    N = labels.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float32)
    for i in range(N):
        out[i, labels[i]] = 1.0
    return out


class PermutedMnistRegression:
    def __init__(self, max_iter=10, data_path='data/mnist.pkl.gz'):
        train_set, valid_set, test_set = load_mnist_pkl(data_path)
        x_tr = np.vstack([train_set[0], valid_set[0]]).astype(np.float32)
        y_tr = np.hstack([train_set[1], valid_set[1]]).astype(np.int64)
        x_te = test_set[0].astype(np.float32)
        y_te = test_set[1].astype(np.int64)

        self.X_train = x_tr
        self.Y_train = to_one_hot(y_tr, 10)
        self.X_test  = x_te
        self.Y_test  = to_one_hot(y_te, 10)

        self.max_iter = max_iter
        self.cur_iter = 0
        self.input_dim  = 784
        self.output_dim = 10

    def get_dims(self):
        return self.input_dim, self.output_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise StopIteration

        np.random.seed(self.cur_iter)
        idx = np.arange(self.input_dim)
        np.random.shuffle(idx)

        x_train_perm = self.X_train[:, idx]
        x_test_perm  = self.X_test[:, idx]

        out = (torch.from_numpy(x_train_perm),
               torch.from_numpy(self.Y_train),
               torch.from_numpy(x_test_perm),
               torch.from_numpy(self.Y_test))
        self.cur_iter += 1
        return out

# CoreSet
def random_coreset(x_coresets, y_coresets, x_train, y_train, coreset_size):
    n_data = x_train.size(0)
    if coreset_size<=0 or coreset_size>=n_data:
        return x_coresets, y_coresets, x_train, y_train

    idx = np.random.choice(n_data, coreset_size, replace=False)
    x_c = x_train[idx]
    y_c = y_train[idx]
    x_coresets.append(x_c)
    y_coresets.append(y_c)

    mask = np.ones(n_data, dtype=bool)
    mask[idx] = False
    x_rem = x_train[mask]
    y_rem = y_train[mask]
    return x_coresets, y_coresets, x_rem, y_rem


def kmeans_coreset(x_coresets, y_coresets, x_train, y_train, coreset_size):
    n_data = x_train.size(0)
    if coreset_size<=0:
        return x_coresets, y_coresets, x_train, y_train
    if coreset_size>=n_data:
        x_coresets.append(x_train)
        y_coresets.append(y_train)
        return x_coresets, y_coresets, torch.empty(0, x_train.size(1)), torch.empty(0, y_train.size(1))

    X_np = x_train.numpy()
    km = KMeans(n_clusters=coreset_size, n_init=5, max_iter=100)
    km.fit(X_np)
    centers = km.cluster_centers_
    labels  = km.labels_

    chosen_idx = []
    for c in range(coreset_size):
        idx_c = np.where(labels==c)[0]
        if len(idx_c)==0:
            continue
        c_points = X_np[idx_c]
        dists = np.sum((c_points - centers[c])**2, axis=1)
        nearest_i = idx_c[np.argmin(dists)]
        chosen_idx.append(nearest_i)

    chosen_idx = np.array(chosen_idx)
    x_c = x_train[chosen_idx]
    y_c = y_train[chosen_idx]
    x_coresets.append(x_c)
    y_coresets.append(y_c)

    mask = np.ones(n_data, dtype=bool)
    mask[chosen_idx] = False
    x_rem = x_train[mask]
    y_rem = y_train[mask]
    return x_coresets, y_coresets, x_rem, y_rem

def merge_coresets(x_coresets, y_coresets):
    if len(x_coresets)==0:
        return None, None
    Xc = torch.cat(x_coresets, dim=0)
    Yc = torch.cat(y_coresets, dim=0)
    return Xc, Yc

# Model and evaluation
def rmse_metric(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))

@torch.no_grad()
def predictive_variance(model, x_data, device='cpu', num_samples=20):
    model.eval()
    n_data = x_data.size(0)
    batch_size = 256
    preds_list = []
    for _ in range(num_samples):
        all_preds = []
        start = 0
        while start<n_data:
            end = min(start+batch_size, n_data)
            xb = x_data[start:end].to(device)
            yb = model(xb, sample=True)  
            all_preds.append(yb.cpu())
            start = end
        all_preds = torch.cat(all_preds, dim=0)  
        preds_list.append(all_preds.unsqueeze(0)) 
    pred_stack = torch.cat(preds_list, dim=0) 
    var_tensor = torch.var(pred_stack, dim=0) 
    mean_var_per_sample = var_tensor.mean(dim=1) 
    return mean_var_per_sample.mean().item() 

def gaussian_nll(pred, target):
    return F.mse_loss(pred, target, reduction='mean')


# Gaussian Prior 
class MFVI_Layer_GaussPrior(nn.Module):
    def __init__(self, in_dim, out_dim, prior_sigma=0.1):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.weight_mean   = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias_mean   = nn.Parameter(torch.Tensor(out_dim))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_dim))
        self.prior_logvar = math.log(prior_sigma**2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        nn.init.constant_(self.weight_logvar, -6.0)
        fan_in = self.weight_mean.shape[1]
        bound  = 1./math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mean, -bound, bound)
        nn.init.constant_(self.bias_logvar, -6.0)

    def forward(self, x, sample=True):
        if sample or self.training:
            eps_w = torch.randn_like(self.weight_mean)
            eps_b = torch.randn_like(self.bias_mean)
            std_w = torch.exp(0.5*self.weight_logvar)
            std_b = torch.exp(0.5*self.bias_logvar)
            w_samp = self.weight_mean + eps_w*std_w
            b_samp = self.bias_mean + eps_b*std_b
        else:
            w_samp = self.weight_mean
            b_samp = self.bias_mean
        return F.linear(x, w_samp, b_samp)

    def kl_term(self):
        var_p = torch.exp(torch.tensor(self.prior_logvar, device=self.weight_mean.device))
        kl_w = 0.5*torch.sum(self.prior_logvar - self.weight_logvar + torch.exp(self.weight_logvar)/var_p + (self.weight_mean**2)/var_p - 1.0)
        kl_b = 0.5*torch.sum(self.prior_logvar - self.bias_logvar + torch.exp(self.bias_logvar)/var_p + (self.bias_mean**2)/var_p - 1.0)
        return kl_w + kl_b

class MFVI_Net_GaussPrior(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, prior_sigma=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim]+hidden_sizes
        for i in range(len(dims)-1):
            self.layers.append(MFVI_Layer_GaussPrior(dims[i], dims[i+1],prior_sigma))
        self.out_layer = MFVI_Layer_GaussPrior(dims[-1], output_dim,prior_sigma)

    def forward(self, x, sample=True):
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample))
        x = self.out_layer(x, sample=sample)
        return x

    def kl_divergence(self):
        kl_sum = 0
        for ly in self.layers:
            kl_sum += ly.kl_term()
        kl_sum += self.out_layer.kl_term()
        return kl_sum


# Laplace Prior 
def erf_approx(x):
    a1,a2,a3,a4,a5 = 0.254829592, -0.284496736,1.421413741,-1.453152027,1.061405429
    p = 0.3275911
    sign = torch.sign(x)
    t = 1./(1.+p*torch.abs(x))
    y = 1. - (a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5))*torch.exp(-x*x)
    return sign*y

def Eabs_gaussian(mu, sigma):
     # E|Z| = sigma*sqrt(2/pi)*exp(-mu^2/(2sigma^2)) + mu*erf(mu/(sqrt(2)sigma))
    term1 = sigma*math.sqrt(2./math.pi)*torch.exp(-0.5*(mu**2)/(sigma**2))
    term2 = mu*erf_approx(mu/(math.sqrt(2.)*sigma))
    return term1 + term2

def kl_gauss_laplace(mu, logvar, b):
    # KL( N(mu,sigma^2) || Laplace(0,b) ) = ln(2b) - 0.5 ln(2π sigma^2) - 0.5 + (1/b)* E|X|, X~N(mu, sigma^2).
    sigma = torch.exp(0.5*logvar)
    kl_val = math.log(2.) + math.log(b) - 0.5*(math.log(2.*math.pi)+ logvar) - 0.5
    eabs = Eabs_gaussian(mu, sigma)
    kl_val += eabs/b
    return kl_val

class MFVI_Layer_LaplacePrior(nn.Module):
    def __init__(self, in_dim, out_dim, prior_b=0.1):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.prior_b = prior_b  # scalar
        self.weight_mean   = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias_mean   = nn.Parameter(torch.Tensor(out_dim))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        nn.init.constant_(self.weight_logvar, -6.0)
        fan_in = self.weight_mean.shape[1]
        bound  = 1./math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mean, -bound, bound)
        nn.init.constant_(self.bias_logvar, -6.0)

    def forward(self, x, sample=True):
        if sample or self.training:
            eps_w = torch.randn_like(self.weight_mean)
            eps_b = torch.randn_like(self.bias_mean)
            w_std = torch.exp(0.5*self.weight_logvar)
            b_std = torch.exp(0.5*self.bias_logvar)
            w_samp = self.weight_mean + eps_w*w_std
            b_samp = self.bias_mean + eps_b*b_std
        else:
            w_samp = self.weight_mean
            b_samp = self.bias_mean
        return F.linear(x, w_samp, b_samp)

    def kl_term(self):
        kl_w = kl_gauss_laplace(self.weight_mean, self.weight_logvar,self.prior_b).sum()
        kl_b = kl_gauss_laplace(self.bias_mean, self.bias_logvar,self.prior_b).sum()
        return kl_w + kl_b

class MFVI_Net_LaplacePrior(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, prior_b=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim]+hidden_sizes
        for i in range(len(dims)-1):
            self.layers.append(MFVI_Layer_LaplacePrior(dims[i], dims[i+1], prior_b))
        self.out_layer = MFVI_Layer_LaplacePrior(dims[-1], output_dim, prior_b)

    def forward(self, x, sample=True):
        for layer in self.layers:
            x = F.relu(layer(x, sample=sample))
        x = self.out_layer(x, sample=sample)
        return x

    def kl_divergence(self):
        kl_sum = 0
        for ly in self.layers:
            kl_sum += ly.kl_term()
        kl_sum += self.out_layer.kl_term()
        return kl_sum


def train_epoch_vcl_regression(model, loader, optimizer, kl_weight,device='cpu', mc_samples=1):
    model.train()
    total_loss = 0.; total_mse=0.; total_kl=0.
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        batch_mse=0.
        for _ in range(mc_samples):
            out = model(x_batch, sample=True)
            batch_mse += F.mse_loss(out, y_batch, reduction='mean')
        batch_mse /= mc_samples

        kl = model.kl_divergence()
        loss = batch_mse + kl_weight*kl
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse  += batch_mse.item()
        total_kl   += kl.item()

    nb = len(loader)
    return (total_loss/nb, total_mse/nb, total_kl/nb)


@torch.no_grad()
def evaluate_rmse(model, x_test, y_test, device='cpu', mc_samples=10):
    model.eval()
    n_data = x_test.size(0)
    bs = 256
    preds_list=[]
    for s in range(mc_samples):
        preds = []
        st=0
        while st<n_data:
            en=min(st+bs,n_data)
            xb = x_test[st:en].to(device)
            out= model(xb, sample=True)
            preds.append(out.cpu())
            st=en
        preds = torch.cat(preds, dim=0) 
        preds_list.append(preds.unsqueeze(0)) 
    stack_pred = torch.cat(preds_list, dim=0) 
    mean_pred  = stack_pred.mean(dim=0)  
    rmse_val   = torch.sqrt(F.mse_loss(mean_pred, y_test, reduction='mean'))
    return rmse_val.item()


# run in permuted MNIST
def run_vcl_gaussian_likelihood(data_gen,prior_type="gaussian",hidden_sizes=[200,200],coreset_func=None,coreset_size=0,lr=1e-3,epochs_first=5,epochs_subseq=5,device='cpu'):
    num_tasks = data_gen.max_iter
    x_coresets, y_coresets = [], []
    all_rmse = []
    all_var  = []

    if prior_type=="gaussian":
        def create_model():
            return MFVI_Net_GaussPrior(data_gen.input_dim,hidden_sizes,data_gen.output_dim,prior_sigma=0.1).to(device)
    else:
        def create_model():
            return MFVI_Net_LaplacePrior(data_gen.input_dim,hidden_sizes,data_gen.output_dim,prior_b=0.1).to(device)

    current_model = None

    for t_idx in range(num_tasks):
        print(f"\n--- Task {t_idx+1}/{num_tasks}, prior={prior_type}, coreset_size={coreset_size} ---")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        print(f"  x_train={x_train.shape}, x_test={x_test.shape}")

        # coreset
        if coreset_func is not None and coreset_size>0:
            x_coresets,y_coresets, x_rem, y_rem = coreset_func(x_coresets,y_coresets, x_train,y_train, coreset_size)
            Xc, Yc = merge_coresets(x_coresets, y_coresets)
            if Xc is not None:
                used_x = torch.cat([x_rem, Xc], dim=0)
                used_y = torch.cat([y_rem, Yc], dim=0)
            else:
                used_x = x_rem
                used_y = y_rem
        else:
            used_x, used_y = x_train, y_train

        if t_idx==0:
            current_model = create_model()
            ds = TensorDataset(used_x, used_y)
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
            kl_w = 1.0/len(ds)

            for ep in range(epochs_first):
                loss,mse_,kl_ = train_epoch_vcl_regression(current_model, loader, optimizer, kl_w, device=device, mc_samples=1)
            current_model.set_prior_from_posterior()
        else:
            ds = TensorDataset(used_x, used_y)
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
            kl_w = 1.0/len(ds)

            for ep in range(epochs_subseq):
                _loss,_mse,_kl = train_epoch_vcl_regression(current_model, loader, optimizer, kl_w, device=device, mc_samples=1)
            current_model.set_prior_from_posterior()

        rmse_val = evaluate_rmse(current_model, x_test, y_test, device=device, mc_samples=10)
        var_val  = predictive_variance(current_model, x_test, device=device, num_samples=10)
        print(f"  => RMSE={rmse_val:.4f}, Var={var_val:.4f}")
        all_rmse.append(rmse_val)
        all_var.append(var_val)

    return np.array(all_rmse), np.array(all_var)


def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] device={device}, seed={seed}")

    data_gen = PermutedMnistRegression(max_iter=10, data_path='data/mnist.pkl.gz')
    print("Created PermutedMnistRegression with 10 tasks ...")

    configs = [
        ("gaussian","none", 0),    
        ("gaussian","random",200), 
        ("gaussian","kmeans",200), 
        ("laplace","none",0),  
        ("laplace","random",200),
        ("laplace","kmeans",200),
    ]

    def coreset_none(*args):
        return args[0],args[1],args[2],args[3]
    def coreset_rand(xc,yc, xt,yt, csz):
        return random_coreset(xc,yc, xt,yt, csz)
    def coreset_kmeans(xc,yc, xt,yt, csz):
        return kmeans_coreset(xc,yc, xt,yt, csz)

    method2coreset = {
        "none":   coreset_none,
        "random": coreset_rand,
        "kmeans": coreset_kmeans
    }

    results_dict = {}
    hidden_sizes = [200,200]  

    for (prior, ctype, csize) in configs:
        data_gen_ = PermutedMnistRegression(max_iter=10, data_path='data/mnist.pkl.gz')
        coreset_func = method2coreset[ctype]
        label = f"{prior.capitalize()}-{ctype}"
        print(f"\n RUN: {label}, csize={csize}")
        rmse_arr, var_arr = run_vcl_gaussian_likelihood(data_gen_,prior_type=prior,hidden_sizes=hidden_sizes,coreset_func=coreset_func,coreset_size=csize,lr=1e-3,epochs_first=10,epochs_subseq=10,device=device)
        results_dict[label] = (rmse_arr, var_arr)

    outfig = "permuted_all6_comparison.png"
    tasks_x = np.arange(1, 10+1)
    plt.figure(figsize=(7,6))

    # RMSE
    plt.subplot(2,1,1)
    for label,(rmses,_) in results_dict.items():
        plt.plot(tasks_x, rmses, marker='o', label=label)
    plt.title("Permuted MNIST (Gaussian-likelihood) - RMSE vs Task")
    plt.ylabel("RMSE")
    plt.grid(True, linestyle='--')
    plt.legend()

    # Var
    plt.subplot(2,1,2)
    for label,(_,vars_) in results_dict.items():
        plt.plot(tasks_x, vars_, marker='x', label=label)
    plt.title("Average Predictive Variance")
    plt.xlabel("Task Index")
    plt.ylabel("Variance")
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(outfig, dpi=120)
    plt.close()


if __name__=="__main__":
    main()
