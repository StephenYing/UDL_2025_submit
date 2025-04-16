import os
import numpy as np
import torch

from dataset import PermutedMnistGenerator, SplitMnistGenerator
from utils import (rand_from_batch, k_center, k_means_coreset,plot_results)
from baseline import (run_vcl_pytorch,run_vcl_pytorch_laplace,run_ewc_pytorch,run_si_pytorch,run_lp_pytorch)


def run_permuted_example():
    config = {
        'hidden_sizes': [100, 100],
        'batch_size': 256,
        'num_epochs_vanilla': 10,
        'num_epochs_mfvi': 10,
        'learning_rate': 1e-3,
        'single_head': True,
        'num_tasks': 10,
        'data_path': 'data/mnist.pkl.gz',
        'results_path': 'results_permuted_example',
        'seed': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'laplace_b': 0.5,
    }
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed(config['seed'])

    data_gen = PermutedMnistGenerator(
        max_iter=config['num_tasks'],
        data_path=config['data_path']
    )
    input_dim, output_dim = data_gen.get_dims()

    results_all = {}

    def kmeans_wrapper(xc, yc, xt, yt, csz):
        return k_means_coreset(xc, yc, xt, yt, csz, n_init=5, max_iter=100)

    print("VCL (Gaussian, no coreset)")
    results_all["VCL_nocoreset"] = run_vcl_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
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
    results_all["VCL+Rand200"] = run_vcl_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=rand_from_batch,
        coreset_size=200,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print(" VCL-KMeans (Gaussian)")
    results_all["VCL+KMeans200"] = run_vcl_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=kmeans_wrapper,
        coreset_size=200,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("VCL-Laplace (no coreset)")
    results_all["Laplace_nocoreset"] = run_vcl_pytorch_laplace(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
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
    results_all["Laplace+Rand200"] = run_vcl_pytorch_laplace(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=rand_from_batch,
        coreset_size=200,
        single_head=config['single_head'],
        device=config['device'],
        prior_b=config['laplace_b'],
        results_path=config['results_path']
    )

    print("VCL-Laplace-KMeans")
    results_all["Laplace+KMeans200"] = run_vcl_pytorch_laplace(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        num_epochs_vanilla=config['num_epochs_vanilla'],
        num_epochs_mfvi=config['num_epochs_mfvi'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        coreset_method=kmeans_wrapper,
        coreset_size=200,
        single_head=config['single_head'],
        device=config['device'],
        prior_b=config['laplace_b'],
        results_path=config['results_path']
    )

    print("EWC")
    results_all["EWC"] = run_ewc_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        ewc_lambda=500,  
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("SI")
    results_all["SI"] = run_si_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=5e-5,
        si_lambda=500, 
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    print("LP")
    results_all["LP"] = run_lp_pytorch(
        data_gen=PermutedMnistGenerator(config['num_tasks'], config['data_path']),
        input_dim=input_dim,
        hidden_sizes=config['hidden_sizes'],
        output_dim=output_dim,
        num_tasks=config['num_tasks'],
        epochs_per_task=10,
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        alpha=500.0,
        single_head=config['single_head'],
        device=config['device'],
        results_path=config['results_path']
    )

    plot_file = os.path.join(config['results_path'], "permuted_all9_comparison.png")
    plot_results(plot_file, results_all, config['num_tasks'])
    return results_all



if __name__ == "__main__":
    print("RUN PERMUTED EXAMPLE (ALL 9 METHODS)")
    run_permuted_example()
