import torch
import numpy as np
from datasets import generate_all_binaries

def get_task_params(task, dimension):

    if task == 'const_fn':
        output = {
            'target': lambda X: torch.ones(X.shape[0], 1, device=X.device),
            'seen_cond': 'x1eq1',
            'active_vars': np.array([False] * dimension),
            'monomials': np.array([[0,0],
                                    [1,0],
                                    [2,0]]),
            'mdi_coeffs': [1., 0., 0.], # in monomials basis specified above
        }
    elif task == 'linear':
        output = {
            'target': lambda X: (X[:,1])[:,None],
            'seen_cond': 'x1eq1',
            'active_vars': np.array([True] * 2 + [False] * (dimension - 2)),
            'monomials': np.array([[0,0],
                                    [1,0],
                                    [0,1],
                                    [1,1]]),
            'mdi_coeffs': [0., 0., 1., 0.], # minimun degree-profile interpolator in this case
        }
    elif task == '2prod':
        output = {
            'target': lambda X: (X[:,0] * X[:,1])[:,None],
            'seen_cond': 'x1eq1orx2eq1',
            'active_vars': np.array([True] * 2 + [False] * (dimension - 2)),
            'monomials': np.array([[0,0],
                                    [1,0],
                                    [0,1],
                                    [1,1]]),
            'mdi_coeffs': [-1., 1., 1., 0.],
        }
    elif task == '3prod':
        output = {
            'target': lambda X: (X[:,0] * X[:,1] * X[:,2])[:,None],
            'seen_cond': 'x1eq1orx2eq1orx3eq1',
            'active_vars': np.array([True] * 3 + [False] * (dimension - 3)),
            'monomials': np.array([[0,0,0],
                                    [1,0,0],
                                    [0,1,0],
                                    [0,0,1],
                                    [1,1,0],
                                    [1,0,1],
                                    [0,1,1],
                                    [1,1,1]]),
            'mdi_coeffs': [1., -1., -1., -1., 1., 1., 1., 0.],
        }
    elif task == '2geometric':
        output = {
            'target': lambda X: (1 + X[:,1] + X[:,1]**2)[:,None],
            'seen_cond': 'x1eq1',
            'active_vars': np.array([True] * 2 + [False] * (dimension - 2)),
            'monomials': np.array([[2,0],
                                   [0,2],
                                   [1,1],
                                   [1,0],
                                   [0,1],
                                   [0,0]]),
            'mdi_coeffs': None,
        }
    else:
        raise ValueError("Unsupported value of 'task' parameter")

    return output