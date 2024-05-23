from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import pickle
import os
import random
import models
import datasets
import helpers
from training import TrainerStochasticGradient, TrainerFullGradient, TrainerGDLineSearch
from tasks_icml2024 import get_task_params

NEGLIGIBLE_LOSS = 0 # training terminates after validation loss achieves this value
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999

def get_folder_name(params):
    file_name = "GOTU/results/"
    for val in params.values():
       file_name = file_name + str(val) + '_'
    file_name = file_name[:-1]
    return file_name

def print_results(results, print_every=5000, print_coeffs=True, print_coeffs_last=True):
    
    n_evals = len(results['epochs'])
    verbose_ids = list(range(0, n_evals, print_every))
    if (n_evals - 1) not in verbose_ids:
        verbose_ids.append(n_evals - 1)

    for i in verbose_ids:
        print(f"Epoch: {results['epochs'][i]:4}, Train Loss: {results['train_losses'][i]:0.6}, Valid Loss: {results['valid_losses'][i]:0.6}, Test Loss: {results['test_losses'][i]:0.6}")
        if print_coeffs:
            print('Coefficients:')
            with np.printoptions(precision=4, suppress=True):
                for mono, coeff in zip(results['monomials'], results['coefficients'][i]):
                    print(f'{mono}: {coeff:0.4f}', end = ' ')
                print()
            if results['conditional_variances'] is not None:
              print(f"Cond variance: {results['conditional_variances'][i]:.6f}", end = ' ')
            if results['squared_distances_to_mdi'] is not None:
              print(f"Dist to MDI squared: {results['squared_distances_to_mdi'][i]:.6f}", end = ' ')
            print()
    
    if (not print_coeffs) and print_coeffs_last:
        i = verbose_ids[-1]
        print('Coefficients:')
        with np.printoptions(precision=4, suppress=True):
            for mono, coeff in zip(results['monomials'], results['coefficients'][i]):
                print(f'{mono}: {coeff:0.4f}', end = ' ')
            print()
    
    print(f"monomials basis: {results['monomials_basis']}")
    

def check_args_validity(args):
    if args.distr == 'unif_discrete':
        assert args.support_size % 2 == 1, "For discrete uniform distribution, please indicade an odd support size"

    if args.arch == 'rf':
        assert (args.opt == 'sgd') or (args.training_method == 'GD_line_search')
        assert args.num_features is not None
        assert args.activation is not None
    elif args.arch == 'mlp':
       assert args.n_layers is not None
       assert args.layer_width is not None
       assert args.activation is not None
    elif args.arch == 'transformer':
       # assert args.opt == 'adamw'
       assert args.distr != 'gauss', 'Transformer must receive discrete inputs'
    
    if args.activation == 'poly':
        assert args.deg is not None

    if args.opt == 'sgd':
        assert args.weight_decay is None
        assert args.beta1 is None
        assert args.beta2 is None
    elif args.opt == 'adamw':
        if args.weight_decay is None: args.weight_decay = DEFAULT_WEIGHT_DECAY
        if args.beta1 is None: args.beta1 = DEFAULT_BETA1
        if args.beta2 is None: args.beta2 = DEFAULT_BETA2     
        
    if args.support_size is not None:
        assert args.distr == 'unif_discrete', "support size can only be indicted for discrete uniform distribution"
    
    if args.num_features is not None:
        assert args.arch == 'rf', "-num-features can only be used with Random Features model"
    if (args.n_layers is not None) or (args.layer_width is not None):
        assert args.arch == 'mlp', "-n-layers or -layer-width can only be used with MLP model"
    
    if args.training_method in ['stochastic_grad', 'full_grad']:
        assert args.opt is not None
        assert args.batches_per_epoch is not None
    
    if args.opt is not None:
        assert args.lr is not None

    if args.training_method == 'GD_line_search':
        assert args.train_size is not None
        assert args.batches_per_epoch is None
    
    if args.train_size is not None:
        assert args.training_method == 'GD_line_search'

    assert args.compute_int % args.verbose_int == 0


if __name__ == '__main__':
    parser = ArgumentParser(description="Training script for testing the generalization of different models on unseen domain",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required runtime params
    parser.add_argument('-task', required=True, type=str, help='name of the task')
    parser.add_argument('-distr', required=True, type=str, help='gauss, unif_discrete or boolean')
    parser.add_argument('-arch', required=True, type=str, help='rf, mlp or transformer')
    parser.add_argument('-dimension', required=True, type=int, help='dimension of data')
    parser.add_argument('-epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-seed', required=True, type=int, help='random seed')
    
    # Other runtime params
    parser.add_argument('-training-method', type=str, default='stochastic_grad', help='stochastic_grad, full_grad, or GD_line_search')
    parser.add_argument('-opt', type=str, help='SGD or AdamW')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('--small-features', action='store_true', help='whether to use small features regime')
    parser.add_argument('-epsilon', type=float, help='std of params initialization, if small feature regime is used')
    parser.add_argument('-num-features', type=int, help='number of random features (for random features model only)')
    parser.add_argument('-activation', type=str, help='activation function')
    parser.add_argument('-deg', type=int, help='degree of polynomial activation (if activation == poly)')
    parser.add_argument('-weight-decay', type=float, help='only used with AdamW optimizer')
    parser.add_argument('-beta1', type=float, help='only used with AdamW optimizer')
    parser.add_argument('-beta2', type=float, help='only used with AdamW optimizer')
    parser.add_argument('-support-size', type=int, help='only used with -distr == unif_discrete')
    parser.add_argument('-batch-size', default=256, type=int, help='batch size')
    parser.add_argument('-batches-per-epoch', type=int, help='number of bathces in one epoch')
    parser.add_argument('-train-size', type=int, help='train dataset size (only when using GD with line search method)')
    parser.add_argument('-valid-size', default=2**17, type=int, help='validation dataset size')
    parser.add_argument('-test-size', default=2**17, type=int, help='test dataset size')
    parser.add_argument('-test-batch-size', type=int, default=8192, help='batch size for test samples')
    parser.add_argument('-verbose-int', default=1, type=int, help="the interval between prints. Set to 0 to disable output")
    parser.add_argument('-compute-int', default=1, type=int, help="the interval between computations of monomials and losses")
    parser.add_argument('-n-layers', type=int, help='number of layers (for MLP model only)')
    parser.add_argument('-layer-width', type=int, help='width of the layer (for MLP model only)')
    parser.add_argument('-device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--no-save', action='store_true', help="do not save training stats")
    
    args = parser.parse_args()
    
    if args.opt is not None:
        args.opt = args.opt.lower()
    args.arch = args.arch.lower()
    check_args_validity(args)
    
    hyperparams = vars(args)
    task_params = get_task_params(args.task, args.dimension)
    print(hyperparams)
    folder = get_folder_name(hyperparams)
    
    training_logs_file = os.path.join(folder, 'training_logs.pkl')
    print('Training logs file:')
    print(training_logs_file)

    try:
        with open(training_logs_file, 'rb') as file:
            results = pickle.load(file)
        print('Results loaded successfully')
        print_results(results)
    except:
        target_function = task_params['target']
        seen_condition = task_params['seen_cond']
        distribution = hyperparams['distr']
        dimension = hyperparams['dimension']
        batch_size = hyperparams['batch_size']
        valid_size = hyperparams['valid_size']
        test_size = hyperparams['test_size']
        support_size = hyperparams['support_size']
        device = hyperparams['device']
        monomials = task_params['monomials']
        if hyperparams['distr'] == 'unif_discrete':
            sparsity_const = sum(task_params['active_vars'])
            monomials = helpers.generate_mask(sparsity_const, support_size)

        np.random.seed(hyperparams['seed'])
        torch.manual_seed(hyperparams['seed'])
        random.seed(hyperparams['seed'])

        if (distribution == 'boolean') and (dimension <= 20):
            assert (hyperparams['test_size'] == 0) and (hyperparams['valid_size'] == 0)
            assert (hyperparams['train_size'] is None) or (hyperparams['train_size'] == 0)
            assert hyperparams['batches_per_epoch'] == 0
            # generating datasets explicitly
            test_X = datasets.generate_all_binaries(dimension, device)
            test_y = target_function(test_X)
            valid_X, valid_y = datasets.get_seen_samples(test_X, test_y, seen_condition)
            assert len(valid_X) % batch_size == 0, f"length of validation dataset = {len(valid_X)} is not divisible by batch size = {batch_size}"
            if hyperparams['training_method'] in ['stochastic_grad', 'full_grad']:
              train_batch_generator, batches_per_epoch_exact = datasets.get_train_batch_generator(valid_X, valid_y, batch_size)
              hyperparams['batches_per_epoch'] = batches_per_epoch_exact
              print(f"new value of batches_per_epoch: {hyperparams['batches_per_epoch']}")
              print(f'Test size: {test_X.shape[0]}')
              print(f'Valid size: {valid_X.shape[0]}')
            elif hyperparams['training_method'] == 'GD_line_search':
              train_X, train_y = valid_X, valid_y
            else:
              raise ValueError('Illegal value of training_method')
        else:
            # generating datasets by subsampling
            test_X = datasets.get_test_data(distribution, test_size, dimension, device, support_size=support_size)
            test_y = target_function(test_X)
            valid_X, valid_y = datasets.get_random_batch(distribution, valid_size, dimension, target_function, device, 
                                                         seen_condition, support_size=support_size)
            if hyperparams['training_method'] in ['stochastic_grad', 'full_grad']:
              train_batch_generator = lambda : datasets.get_random_batch(distribution, batch_size, dimension, target_function, device, 
                                                                        seen_condition, support_size=support_size)
            elif hyperparams['training_method'] == 'GD_line_search':
              train_X, train_y = datasets.get_random_batch(distribution, hyperparams['train_size'], dimension, target_function, device, 
                                                         seen_condition, support_size=support_size)
        
        print(f'valid_X.shape: {valid_X.shape}')
        sampled_ids = random.sample(list(torch.arange(valid_X.shape[0])), min(10, valid_X.shape[0]))
        print('examples from valid_X:')
        print(valid_X[sampled_ids,:4])
        
        model = models.build_model(hyperparams['arch'], dimension, **hyperparams).to(device)
        
        print(f'monomials.shape: {monomials.shape}. Monomials:')
        print(monomials)

        if hyperparams['training_method'] == 'stochastic_grad':
            trainer = TrainerStochasticGradient(model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
                                                task_params['active_vars'], task_params['mdi_coeffs'], **hyperparams)
        elif hyperparams['training_method'] == 'full_grad':
            trainer = TrainerFullGradient(model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
                                                task_params['active_vars'], task_params['mdi_coeffs'], **hyperparams)
        elif hyperparams['training_method'] == 'GD_line_search':
            trainer = TrainerGDLineSearch(model, train_X, train_y, valid_X, valid_y, test_X, test_y, monomials, 
                                                task_params['active_vars'], task_params['mdi_coeffs'], **hyperparams)

        training_logs = trainer.train(hyperparams['epochs'], hyperparams['batches_per_epoch'], 
                                      hyperparams['compute_int'], hyperparams['verbose_int'])

        if not hyperparams['no_save']:
          saved_data = {'run_params': vars(args)}
          saved_data.update(training_logs)
          os.makedirs(os.path.dirname(training_logs_file), exist_ok=True)
          with open(training_logs_file, 'wb') as file:
              pickle.dump(saved_data, file)
