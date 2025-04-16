import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import (VanillaNN, MFVI_NN,MFVI_NN_AnalyticLaplace)
from utils import (merge_coresets, evaluate_task, get_scores_pytorch,concatenate_results)

# VCL
def train_vanilla_epoch(model, loader, optimizer, criterion, device, task_id=0):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb, task_id=task_id)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_mfvi_epoch(model, loader, optimizer, criterion,
                     kl_weight, device, task_id=0, num_samples=1):
    model.train()
    total_loss, total_nll, total_kl = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        batch_nll = 0
        for _ in range(num_samples):
            out = model(xb, task_id=task_id, sample=True)
            batch_nll += criterion(out, yb)
        nll_term = batch_nll / num_samples

        kl_term = model.kl_divergence(task_id=task_id)
        loss = nll_term + kl_weight * kl_term
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_nll += nll_term.item()
        total_kl += kl_term.item()

    nb = len(loader)
    return total_loss / nb, total_nll / nb, total_kl / nb


def run_vcl_pytorch(data_gen, input_dim, hidden_sizes, output_dim, num_tasks,
                    num_epochs_vanilla, num_epochs_mfvi, batch_size, lr,
                    coreset_method=None, coreset_size=0,
                    single_head=True, device='cpu', train_mc_samples=1,
                    results_path='results'):
    
    os.makedirs(results_path, exist_ok=True)
    print(f"--- VCL: single_head={single_head}, coreset={coreset_size}, device={device} ---")

    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    all_results = np.array([])

    current_mfvi_model = None
    start = time.time()

    for task_id in range(num_tasks):
        print(f"\n===== [VCL] Task {task_id+1}/{num_tasks} =====")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        cur_x_train, cur_y_train = x_train, y_train
        if coreset_method is not None and coreset_size > 0 and x_train.shape[0] > 0:
            x_coresets, y_coresets, cur_x_train, cur_y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size
            )

        head_idx = 0 if single_head else task_id
        num_heads = 1 if single_head else num_tasks

        if task_id == 0:
            print(" Training VanillaNN on first task ...")
            vanilla_model = VanillaNN(input_dim, hidden_sizes, output_dim,
                                      num_heads).to(device)
            optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            ds = TensorDataset(cur_x_train, cur_y_train)
            bsz = batch_size if batch_size else len(ds)
            loader = DataLoader(ds, batch_size=bsz, shuffle=True)

            for ep in range(num_epochs_vanilla):
                train_vanilla_epoch(vanilla_model, loader, optimizer, criterion,device, head_idx)

            current_mfvi_model = MFVI_NN(input_dim, hidden_sizes, output_dim,num_heads).to(device)
            current_mfvi_model.initialize_from_vanilla_model(vanilla_model)
            current_mfvi_model.set_prior_from_posterior()
            del vanilla_model

        else:
            if current_mfvi_model is None:
                raise RuntimeError("MFVI not initialized!")
            if len(x_coresets) > 0:
                merged_x, merged_y = merge_coresets(x_coresets, y_coresets)
                combined_x = torch.cat([cur_x_train, merged_x], dim=0)
                combined_y = torch.cat([cur_y_train, merged_y], dim=0)
                print(f"  Using coreset. Train data={cur_x_train.shape[0]} + coreset={merged_x.shape[0]}")
            else:
                combined_x, combined_y = cur_x_train, cur_y_train

            if combined_x.shape[0] > 0:
                ds = TensorDataset(combined_x, combined_y)
                bsz = batch_size if batch_size else len(ds)
                loader = DataLoader(ds, batch_size=bsz, shuffle=True)
                kl_weight = 1.0 / len(ds)

                optimizer = torch.optim.Adam(current_mfvi_model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                for ep in range(num_epochs_mfvi):
                    train_mfvi_epoch(current_mfvi_model, loader, optimizer,criterion, kl_weight, device,head_idx, train_mc_samples)

                current_mfvi_model.set_prior_from_posterior()

        print(" Evaluating ...")
        accs = get_scores_pytorch(current_mfvi_model, x_testsets, y_testsets,device, single_head, num_tasks)
        all_results = concatenate_results(accs, all_results)
        print(f"  Task {task_id+1} done. mean acc={np.nanmean(accs):.4f}")

    total_time = time.time() - start
    final_acc = np.nanmean(all_results[-1, :]) if all_results.size > 0 else 0
    suffix = f"{'single' if single_head else 'multi'}head_{coreset_size}"
    fn = os.path.join(results_path, f"results_vcl_{suffix}.npy")
    np.save(fn, all_results)
    print(f"[VCL] Saved results to {fn}")
    return all_results


# EWC
class EWC:
    def __init__(self, model, dataloader, device='cpu', ewc_lambda=100.0):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda

        self.params = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher(dataloader)

    def compute_fisher(self, dataloader):
        fisher_dict = {}
        self.model.train()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher_dict[n] = torch.zeros_like(p).to(self.device)

        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.model.zero_grad()
            out = self.model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_dict[n] += p.grad.data.pow(2)

        for n in fisher_dict:
            fisher_dict[n] /= len(dataloader)
        return fisher_dict

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return (self.ewc_lambda / 2.0) * loss


def run_ewc_pytorch(data_gen, input_dim, hidden_sizes, output_dim,
                    num_tasks=5, epochs_per_task=10, batch_size=256,
                    lr=1e-3, ewc_lambda=100.0, single_head=True,
                    device='cpu', results_path='results'):
    os.makedirs(results_path, exist_ok=True)
    print(f"--- EWC: single_head={single_head}, lambda={ewc_lambda}, device={device} ---")

    model = VanillaNN(input_dim, hidden_sizes, output_dim,
                      1 if single_head else num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ewc_list = []
    x_testsets = []
    y_testsets = []
    all_results = np.array([])

    for task_id in range(num_tasks):
        print(f"\n=== [EWC] Task {task_id+1}/{num_tasks} ===")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        ds = TensorDataset(x_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        head_idx = 0 if single_head else task_id

        for ep in range(epochs_per_task):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb, task_id=head_idx)
                loss = criterion(out, yb)
                for ewc_obj in ewc_list:
                    loss += ewc_obj.penalty(model)
                loss.backward()
                optimizer.step()

        fisher_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        ewc_obj = EWC(model, fisher_loader, device, ewc_lambda)
        ewc_list.append(ewc_obj)

        accs = []
        for eval_id in range(len(x_testsets)):
            e_id = 0 if single_head else eval_id
            logits_acc = evaluate_task(model, e_id, x_testsets[eval_id],
                                       y_testsets[eval_id],
                                       device, num_samples=1, batch_size=batch_size)
            accs.append(logits_acc)
        print("Acc:", [f"{x:.4f}" for x in accs])
        all_results = concatenate_results(accs, all_results)
        print(f" Task {task_id+1} done. mean acc={np.nanmean(accs):.4f}")

    fn = os.path.join(results_path, "results_ewc.npy")
    np.save(fn, all_results)
    print(f"[EWC] saved to {fn}")
    return all_results


# SI
class SI:
    def __init__(self, model, si_lambda=1.0, device='cpu'):
        self.model = model
        self.si_lambda = si_lambda
        self.device = device
        self.eps = 1e-7
        self._reset_ref()

        self.small_omega = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.small_omega[n] = torch.zeros_like(p).to(self.device)

    def _reset_ref(self):
        self.ref_params = {n: p.clone().detach()
                           for n, p in self.model.named_parameters()
                           if p.requires_grad}

    def initialize_start_params(self):
        self.initial_params = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.initial_params[n] = p.clone().detach()

    def begin_finetuning(self):
        self.small_omega = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.small_omega[n] = torch.zeros_like(p).to(self.device)
        self._reset_ref()

    def update_omega(self, old_params):
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta_theta = p.detach() - old_params[n]
                self.small_omega[n] += p.grad.detach() * delta_theta

    def end_finetuning(self):
        self.big_omega = {}
        self.star_params = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.star_params[n] = p.clone().detach()
                denom = (p.detach() - self.initial_params[n]).pow(2) + self.eps
                self.big_omega[n] = self.small_omega[n] / denom

        if hasattr(self, 'accum_omega'):
            for n in self.big_omega:
                self.big_omega[n] += self.accum_omega[n]
        self.accum_omega = self.big_omega
        self.initial_params = self.star_params.copy()

    def penalty(self, model):
        if not hasattr(self, 'accum_omega'):
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.accum_omega[n] * (p - self.star_params[n]) ** 2
                loss += _loss.sum()
        return (self.si_lambda / 2.0) * loss


def run_si_pytorch(data_gen, input_dim, hidden_sizes, output_dim,
                   num_tasks=5, epochs_per_task=10, batch_size=256,
                   lr=1e-3, si_lambda=100, single_head=True,
                   device='cpu', results_path='results'):
    os.makedirs(results_path, exist_ok=True)
    print(f"--- SI: single_head={single_head}, lambda={si_lambda}, device={device} ---")

    model = VanillaNN(input_dim, hidden_sizes, output_dim,
                      1 if single_head else num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    si_obj = SI(model, si_lambda, device)
    si_obj.initialize_start_params()

    x_testsets = []
    y_testsets = []
    all_results = np.array([])

    for task_id in range(num_tasks):
        print(f"\n=== [SI] Task {task_id+1}/{num_tasks} ===")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        si_obj.begin_finetuning()
        ds = TensorDataset(x_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        head_idx = 0 if single_head else task_id

        for ep in range(epochs_per_task):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                old_params = {n: p.clone().detach()
                              for n, p in model.named_parameters()
                              if p.requires_grad}

                optimizer.zero_grad()
                out = model(xb, task_id=head_idx)
                total_loss = criterion(out, yb) + si_obj.penalty(model)
                total_loss.backward()

                si_obj.update_omega(old_params)

                optimizer.step()

        si_obj.end_finetuning()

        accs = []
        for eid in range(len(x_testsets)):
            e_id = 0 if single_head else eid
            a = evaluate_task(model, e_id, x_testsets[eid], y_testsets[eid],
                              device, num_samples=1, batch_size=batch_size)
            accs.append(a)
        print("Acc:", [f"{x:.4f}" for x in accs])
        all_results = concatenate_results(accs, all_results)
        print(f" Task {task_id+1} done. mean={np.nanmean(accs):.4f}")

    fn = os.path.join(results_path, "results_si.npy")
    np.save(fn, all_results)
    print(f"[SI] saved to {fn}")
    return all_results


# Laplace Propagation
class LP:
    def __init__(self, model, dataloader, device='cpu', alpha=100.0):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.old_params = {n: p.clone().detach()
                           for n, p in model.named_parameters() if p.requires_grad}
        self.hessian = self.compute_hessian(dataloader)

    def compute_hessian(self, dataloader):
        hessian_dict = {}
        self.model.train()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                hessian_dict[n] = torch.zeros_like(p).to(self.device)

        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.model.zero_grad()
            out = self.model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    hessian_dict[n] += p.grad.data.pow(2)

        for n in hessian_dict:
            hessian_dict[n] /= len(dataloader)
        return hessian_dict

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.hessian[n] * (p - self.old_params[n]) ** 2
                loss += _loss.sum()
        return (self.alpha / 2.0) * loss


def run_lp_pytorch(data_gen, input_dim, hidden_sizes, output_dim,
                   num_tasks=5, epochs_per_task=10, batch_size=256,
                   lr=1e-3, alpha=100.0, single_head=True,
                   device='cpu', results_path='results'):
    os.makedirs(results_path, exist_ok=True)
    print(f"--- LP: single_head={single_head}, alpha={alpha}, device={device} ---")

    model = VanillaNN(input_dim, hidden_sizes, output_dim,
                      1 if single_head else num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    lp_list = []
    x_testsets = []
    y_testsets = []
    all_results = np.array([])

    for task_id in range(num_tasks):
        print(f"\n=== [LP] Task {task_id+1}/{num_tasks} ===")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        ds = TensorDataset(x_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        head_idx = 0 if single_head else task_id

        for ep in range(epochs_per_task):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb, task_id=head_idx)
                loss = criterion(out, yb)
                for lpo in lp_list:
                    loss += lpo.penalty(model)
                loss.backward()
                optimizer.step()

        eval_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        lpobj = LP(model, eval_loader, device, alpha)
        lp_list.append(lpobj)

        accs = []
        for eid in range(len(x_testsets)):
            e_id = 0 if single_head else eid
            a = evaluate_task(model, e_id, x_testsets[eid], y_testsets[eid],
                              device, num_samples=1, batch_size=batch_size)
            accs.append(a)
        print("Acc:", [f"{x:.4f}" for x in accs])
        all_results = concatenate_results(accs, all_results)
        print(f" Task {task_id+1} done. mean={np.nanmean(accs):.4f}")

    fn = os.path.join(results_path, "results_lp.npy")
    np.save(fn, all_results)
    print(f"[LP] saved to {fn}")
    return all_results


## Laplace prior
def train_mfvi_epoch_laplace(model, loader, optimizer, criterion,kl_weight, device, task_id=0,num_samples=1):
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        batch_nll = 0.0
        for _ in range(num_samples):
            outputs = model(x_batch, task_id=task_id, sample=True)
            batch_nll += criterion(outputs, y_batch)
        nll_term = batch_nll / num_samples

        kl_term = model.kl_divergence(task_id=task_id)
        loss = nll_term + kl_weight * kl_term

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_nll += nll_term.item()
        total_kl += kl_term.item()

    nb = len(loader)
    return (total_loss/nb, total_nll/nb, total_kl/nb)


def run_vcl_pytorch_laplace(data_gen,input_dim,hidden_sizes,output_dim,num_tasks,num_epochs_vanilla,num_epochs_mfvi,batch_size,lr,
                            coreset_method=None,coreset_size=0,single_head=True,device='cpu',train_num_mc=1,prior_b=0.1,results_path='results_laplace'):
    os.makedirs(results_path, exist_ok=True)
    print(f"=== VCL Laplace: single_head={single_head}, coreset={coreset_size}, b={prior_b} ===")

    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    all_results_matrix = np.array([])

    current_mfvi_model = None
    start_time = time.time()

    for task_id in range(num_tasks):
        print(f"\n--- Task {task_id+1} / {num_tasks} ---")
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)
        print(f"  Task data: train={x_train.shape[0]}, test={x_test.shape[0]}")

        current_x_train, current_y_train = x_train, y_train
        if coreset_method is not None and coreset_size>0 and x_train.shape[0]>0:
            x_coresets, y_coresets, current_x_train, current_y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size
            )

        head_idx = 0 if single_head else task_id
        num_heads = 1 if single_head else num_tasks

        if task_id==0:
            vanilla_model = VanillaNN(input_dim, hidden_sizes, output_dim, num_heads).to(device)
            optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            train_dataset = TensorDataset(current_x_train, current_y_train)
            bsize = batch_size if (batch_size is not None) else len(train_dataset)
            loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

            for ep in range(num_epochs_vanilla):
                train_vanilla_epoch(vanilla_model, loader, optimizer, criterion, device, head_idx)

            current_mfvi_model = MFVI_NN_AnalyticLaplace(input_dim, hidden_sizes, output_dim,num_heads, prior_b=prior_b).to(device)

            current_mfvi_model.initialize_from_vanilla_model(vanilla_model)
            current_mfvi_model.set_prior_from_posterior()
            del vanilla_model

        else:
            if current_mfvi_model is None:
                raise RuntimeError("No MFVI model found!")
            if len(x_coresets)>0:
                merged_xc, merged_yc = merge_coresets(x_coresets, y_coresets)
                combined_x = torch.cat([current_x_train, merged_xc], dim=0)
                combined_y = torch.cat([current_y_train, merged_yc], dim=0)
                print(f"  Using {current_x_train.shape[0]} + coreset={merged_xc.shape[0]}")
            else:
                combined_x = current_x_train
                combined_y = current_y_train
                print(f"  Using {combined_x.shape[0]} data, no coreset")

            if combined_x.shape[0] > 0:
                train_dataset = TensorDataset(combined_x, combined_y)
                bsize = batch_size if (batch_size is not None) else len(train_dataset)
                loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)

                optimizer = torch.optim.Adam(current_mfvi_model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                kl_weight = 1.0 / len(train_dataset)

                for ep in range(num_epochs_mfvi):
                    train_mfvi_epoch_laplace(current_mfvi_model, loader, optimizer, criterion,kl_weight, device, head_idx,num_samples=train_num_mc)

                current_mfvi_model.set_prior_from_posterior()
            else:
                print("  No data. Skipping this task training.")

        accs = get_scores_pytorch(current_mfvi_model, x_testsets, y_testsets,device, single_head, num_tasks)
        all_results_matrix = concatenate_results(accs, all_results_matrix)
        print(f"  Task {task_id+1} done. Mean acc={np.nanmean(accs):.4f}")

    total_time = time.time() - start_time
    final_acc = np.nanmean(all_results_matrix[-1,:]) if all_results_matrix.size>0 else 0
    print(f"\n=== All tasks done in {total_time:.2f}s, final avg acc={final_acc:.4f} ===")

    file_suffix = f"{'single' if single_head else 'multi'}head_laplace_b{prior_b}"
    save_path = f"{results_path}/laplace_results_{file_suffix}.npy"
    np.save(save_path, all_results_matrix)
    print(f"Results saved to {save_path}")
    return all_results_matrix
