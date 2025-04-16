import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

def rand_from_batch(x_coresets, y_coresets, x_train_tensor, y_train_tensor, coreset_size):
    x_train = x_train_tensor.numpy()
    y_train = y_train_tensor.numpy()

    num_examples = x_train.shape[0]
    if coreset_size > num_examples:
        coreset_size = num_examples
        idx = np.arange(num_examples)
    else:
        idx = np.random.choice(num_examples, coreset_size, replace=False)

    x_coresets.append(x_train_tensor[idx])
    y_coresets.append(y_train_tensor[idx])

    mask = np.ones(num_examples, dtype=bool)
    mask[idx] = False
    x_train_remaining = torch.from_numpy(x_train[mask])
    y_train_remaining = torch.from_numpy(y_train[mask])

    return x_coresets, y_coresets, x_train_remaining, y_train_remaining


def k_center(x_coresets, y_coresets, x_train_tensor, y_train_tensor, coreset_size):
    x_train = x_train_tensor.numpy()
    y_train = y_train_tensor.numpy()

    num_examples = x_train.shape[0]
    if coreset_size <= 0:
        return x_coresets, y_coresets, x_train_tensor, y_train_tensor
    if coreset_size >= num_examples:
        x_coresets.append(x_train_tensor)
        y_coresets.append(y_train_tensor)
        return (x_coresets, y_coresets,
                torch.empty((0, ) + x_train_tensor.shape[1:]),
                torch.empty((0, ) + y_train_tensor.shape[1:]))
    min_dists = np.full(num_examples, np.inf, dtype=np.float32)

    current_id = np.random.randint(0, num_examples)
    selected_indices = [current_id]
    center_feats = x_train[current_id:current_id+1]
    dists_sq = np.sum((x_train - center_feats)**2, axis=1)
    min_dists = np.minimum(min_dists, dists_sq)

    for i in range(1, coreset_size):
        if (i % 10 == 0 or i == coreset_size - 1) and i > 0:
            print(f"  K-Center selecting point {i+1}/{coreset_size}")
        current_id = np.argmax(min_dists)
        if current_id in selected_indices:
            print(f"Warning: K-Center selected an already chosen index ({current_id}).")
            break
        if min_dists[current_id] == 0:
            print(f"Warning: K-Center found a point with zero distance ({current_id}).")
            remaining_indices = np.setdiff1d(np.arange(num_examples), selected_indices)
            if len(remaining_indices) == 0:
                break
            current_id = np.random.choice(remaining_indices)

        selected_indices.append(current_id)
        center_feats = x_train[current_id:current_id+1]
        new_dists_sq = np.sum((x_train - center_feats) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists_sq)
        min_dists[selected_indices] = 0

    selected_indices = np.array(selected_indices)
    x_coresets.append(x_train_tensor[selected_indices])
    y_coresets.append(y_train_tensor[selected_indices])

    mask = np.ones(num_examples, dtype=bool)
    mask[selected_indices] = False
    x_train_remaining = torch.from_numpy(x_train[mask])
    y_train_remaining = torch.from_numpy(y_train[mask])
    return x_coresets, y_coresets, x_train_remaining, y_train_remaining


def k_means_coreset(x_coresets, y_coresets,x_train_tensor, y_train_tensor,coreset_size, n_init=10, max_iter=300):

    x_train = x_train_tensor.numpy()
    y_train = y_train_tensor.numpy()

    num_examples = x_train.shape[0]
    if coreset_size <= 0:
        return x_coresets, y_coresets, x_train_tensor, y_train_tensor
    if coreset_size >= num_examples:
        x_coresets.append(x_train_tensor)
        y_coresets.append(y_train_tensor)

        return (x_coresets, y_coresets,
                torch.empty((0, ) + x_train_tensor.shape[1:]),
                torch.empty((0, ) + y_train_tensor.shape[1:]))

    kmeans = KMeans(n_clusters=coreset_size,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=42)
    kmeans.fit(x_train)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_indices = []
    for cluster_id in range(coreset_size):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = x_train[cluster_indices]
        dists = np.sum((cluster_points - centers[cluster_id]) ** 2, axis=1)
        closest_idx = cluster_indices[np.argmin(dists)]
        selected_indices.append(closest_idx)

    selected_indices = np.array(selected_indices)
    x_coresets.append(x_train_tensor[selected_indices])
    y_coresets.append(y_train_tensor[selected_indices])

    mask = np.ones(num_examples, dtype=bool)
    mask[selected_indices] = False
    x_train_remaining = torch.from_numpy(x_train[mask])
    y_train_remaining = torch.from_numpy(y_train[mask])
    return x_coresets, y_coresets, x_train_remaining, y_train_remaining



def merge_coresets(x_coresets, y_coresets):
    if not x_coresets:
        return None, None
    x_cat = torch.cat(x_coresets, dim=0)
    y_cat = torch.cat(y_coresets, dim=0)
    return x_cat, y_cat


def evaluate_task(model, task_id, x_test, y_test,device, num_samples=100, batch_size=256):
    model.eval()

    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_targets = []
    collected_probs = []

    with torch.no_grad():
        for s in range(num_samples):
            batch_probs = []
            batch_tgts = []
            for xb, yb in loader:
                xb = xb.to(device)
                out = model(xb, task_id=task_id, sample=True)
                prob = F.softmax(out, dim=1)
                batch_probs.append(prob.cpu())
                if s == 0:
                    batch_tgts.append(yb)
            collected_probs.append(torch.cat(batch_probs, dim=0))
            if s == 0:
                all_targets = torch.cat(batch_tgts, dim=0)

    avg_probs = torch.mean(torch.stack(collected_probs, dim=0), dim=0)
    _, pred_labels = torch.max(avg_probs, dim=1)
    correct = (pred_labels == all_targets).sum().item()
    total = all_targets.size(0)
    return correct / total if total > 0 else 0.0


def get_scores_pytorch(model, x_testsets, y_testsets, device,
                       single_head, num_tasks,
                       eval_num_samples=100, eval_batch_size=256):
    model.eval()
    accs = []
    for i in range(len(x_testsets)):
        task_eval_id = 0 if single_head else i
        acc = evaluate_task(model,task_eval_id,x_testsets[i],y_testsets[i],device,num_samples=eval_num_samples,batch_size=eval_batch_size)
        accs.append(acc)
    print("Accuracies of tasks:", [f"{a:.4f}" for a in accs])
    return accs


def concatenate_results(score_list, all_score):
    cur_score = np.array(score_list)
    if all_score is None or all_score.size == 0:
        return np.reshape(cur_score, (1, -1))
    else:
        old_cols = all_score.shape[1]
        new_cols = max(old_cols, len(cur_score))
        new_rows = all_score.shape[0] + 1
        new_matrix = np.full((new_rows, new_cols), np.nan)
        new_matrix[:all_score.shape[0], :old_cols] = all_score
        new_matrix[-1, :len(cur_score)] = cur_score
        return new_matrix


def plot_results(filename, results_dict, num_tasks):
    plt.figure(figsize=(7, 4))
    task_axis = np.arange(1, num_tasks + 1)
    for label, arr in results_dict.items():
        if arr is None or arr.size == 0:
            continue
        avg_acc = np.nanmean(arr, axis=1)
        xvals = np.arange(1, len(avg_acc) + 1)
        plt.plot(xvals, avg_acc, marker='o', label=label)
    plt.xticks(task_axis)
    plt.xlabel("Number of Tasks Learned")
    plt.ylabel("Average Accuracy")
    plt.title("Continual Learning Comparison")
    plt.grid(True, linestyle='--')
    plt.legend()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved: {filename}")
    plt.show()
    plt.close()

