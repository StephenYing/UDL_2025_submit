import os
import numpy as np
import torch

from dataset import PermutedMnistGenerator, SplitMnistGenerator
from utils import (rand_from_batch, k_center, k_means_coreset,plot_results)
from baseline import (run_vcl_pytorch,run_vcl_pytorch_laplace,run_ewc_pytorch,run_si_pytorch,run_lp_pytorch)


def run_split_example():
    config = {
        'hidden_sizes': [256, 256],
        'batch_size': 256,
        'num_epochs_vanilla': 10,
        'num_epochs_mfvi': 10,
        'learning_rate': 7e-3,
        'single_head': False,   
        'num_tasks': 5,
        'data_path': 'data/mnist.pkl.gz',
        'results_path': 'results_split_example',
        'seed': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'laplace_b': 0.5,
    }
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed(config['seed'])

    data_gen = SplitMnistGenerator(data_path=config['data_path'])
    input_dim, output_dim = data_gen.get_dims()
    num_tasks = data_gen.max_iter

    results_all = {}

    def kmeans_wrapper(xc, yc, xt, yt, csz):
        return k_means_coreset(xc, yc, xt, yt, csz, n_init=5, max_iter=100)

    print("VCL (Gaussian, no coreset)")
    results_all["VCL_nocoreset"] = run_vcl_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=None,
        coreset_size=0,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("VCL-Random (Gaussian)")
    results_all["VCL+Rand40"] = run_vcl_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=rand_from_batch,
        coreset_size=40,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("VCL-KMeans (Gaussian)")
    results_all["VCL+KMeans40"] = run_vcl_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=kmeans_wrapper,
        coreset_size=40,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("VCL-Laplace (no coreset)")
    results_all["Laplace_nocoreset"] = run_vcl_pytorch_laplace(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=None,
        coreset_size=0,
        single_head=config['single_head'],
        device=config['device'],
        prior_b=config['laplace_b'],
        results_path=config['results_path']
    )

    print("VCL-Laplace-Random")
    results_all["Laplace+Rand40"] = run_vcl_pytorch_laplace(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=rand_from_batch,
        coreset_size=40,
        single_head=config['single_head'],
        device=config['device'],
        prior_b=config['laplace_b'],
        results_path=config['results_path']
    )

    print("VCL-Laplace-KMeans")
    results_all["Laplace+KMeans40"] = run_vcl_pytorch_laplace(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=kmeans_wrapper,
        coreset_size=40,
        single_head=config['single_head'],
        device=config['device'],
        prior_b=config['laplace_b'],
        results_path=config['results_path']
    )

    print("EWC")
    results_all["EWC"] = run_ewc_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        ewc_lambda=1,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("SI")
    results_all["SI"] = run_si_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=5e-5,
        si_lambda=0.5,  # 这里也可以尝试再调大一些
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("LP")
    results_all["LP"] = run_lp_pytorch(
        data_gen=SplitMnistGenerator(config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=num_tasks,
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        alpha=0.1,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    plot_file = os.path.join(config['results_path'], "split_all9_comparison.png")
    plot_results(plot_file, results_all, num_tasks)

    from utils import plot_split_mnist_figure4

    plot_split_mnist_figure4(results_all, 
                            tasks_labels=[
                                "Task 1 (0 or 1)",
                                "Task 2 (2 or 3)",
                                "Task 3 (4 or 5)",
                                "Task 4 (6 or 7)",
                                "Task 5 (8 or 9)",
                                "Average"
                            ],
                            out_file="my_splitmnist_figure4.png")

    return results_all


if __name__ == "__main__":
    print("\n========== RUN SPLIT EXAMPLE (ALL 9 METHODS) =============")
    run_split_example()
