import numpy as np
import torch
from scipy.special import eval_hermitenorm, factorial
import math
from torch.utils.data import TensorDataset, DataLoader

def prepare_projection_unif_discrete(monomials, test_X, support_size):
  k, p = monomials.shape
  window = round((support_size-1)/2)
  moments = np.zeros((2*support_size - 1,))
  moments[0] = 1
  for i in range(1,support_size):
    moments[2*i] = 2/support_size * sum([t**(2*i) for t in range(1,window+1)])

  gram_matrix = np.empty((k,k))
  for i in range(k):
    for j in range(k):
      degrees_sum = monomials[i] + monomials[j]
      gram_matrix[i,j] = np.prod(moments[degrees_sum])
  gram_matrix_inv = np.linalg.inv(gram_matrix)

  monomials_evaluated = np.where(monomials[:, None, :] == 0, 1.0, test_X[None, :, :p]**monomials[:, None, :]).prod(axis = -1) # k * n

  return gram_matrix_inv, monomials_evaluated

def prepare_projection_gauss(monomials, test_X, **params):
   
   # eval_hermitenorm evaluates monic probabilists Hermite polynomial
    k, p = monomials.shape
    test_X = test_X[:,:p]
    monomials_evaluated = eval_hermitenorm(monomials[:, None, :], test_X)
    normalization_factor = np.sqrt(factorial(monomials))
    monomials_evaluated = monomials_evaluated / normalization_factor[:, None, :]
    monomials_evaluated = monomials_evaluated.prod(axis=-1)

    # Hermite basis is othogonal w.r.t. gaussian distribution
    gram_matrix_inv = None

    return gram_matrix_inv, monomials_evaluated

def prepare_projection_boolean(monomials, test_X, **params):
   
    k, p = monomials.shape
    monomials_evaluated = np.where(monomials[:, None, :] == 0, 1.0, test_X[None, :, :p]**monomials[:, None, :]).prod(axis = -1) # k * n

    # Fourier-Walsh basis is othogonal w.r.t. uniform boolean distribution
    gram_matrix_inv = None

    return gram_matrix_inv, monomials_evaluated

class BasisCoefficientsCalculator:

   def  __init__(self, monomials, test_X, distribution, **params):

      if distribution == 'boolean':
         self.gram_matrix_inv, self.monomials_evaluated = prepare_projection_boolean(monomials, test_X, **params)
      elif distribution == 'gauss':
         self.gram_matrix_inv, self.monomials_evaluated = prepare_projection_gauss(monomials, test_X, **params)
      elif distribution == 'unif_discrete':
         self.gram_matrix_inv, self.monomials_evaluated = prepare_projection_unif_discrete(monomials, test_X, **params)
      else:
         raise ValueError(f"Illegal argument of 'distribution' parameter: got {distribution}")

   def evaluate(self, y_pred):
      coeffs = (self.monomials_evaluated @ y_pred.squeeze()) / y_pred.shape[0]
      if self.gram_matrix_inv is not None:
         # the case when basis is not orthonormal
         coeffs = self.gram_matrix_inv @ coeffs
      return coeffs

def get_monomials_basis(distribution):
    if distribution == 'gauss': basis = 'hermite'
    elif distribution == 'unif_discrete': basis = 'simple_polynomials'
    elif distribution == 'boolean': basis = 'fourier-walsh'
    else: raise ValueError("Illegal argument of 'distribution' parameter")
   
    return basis

def get_distribution_type(distribution):
  if distribution in ['unif_discrete', 'boolean']:
     distribution_type = 'discrete'
  elif distribution == 'gauss':
     distribution_type = 'continuous'
  else:
     raise ValueError(f"Illegal argument of 'distribution': got {distribution}")
  return distribution_type

def generate_mask(sparsity_const, support_size):
  
  if sparsity_const == 0:
      monomials = np.array([[0], [1]]).astype('int')
  else:
      degrees = np.arange(support_size)
      monomials = np.array(np.meshgrid(*([degrees] * sparsity_const))).T.reshape(-1, sparsity_const)

      sums = np.sum(monomials, axis=1)
      monomials = monomials[np.argsort(sums)]
  
  return monomials

class ConditionalVarianceCalculatorDiscrete:

   def __init__(self, test_X, active_vars):
      self.principal_input_groups = []
      self.group_fractions = []
      if not any(active_vars):
         self.principal_input_groups.append(torch.arange(test_X.shape[0]))
         self.group_fractions.append(1)
      else:
         active_vars = torch.from_numpy(active_vars)
         principal_inputs = torch.unique(test_X[:,active_vars], dim=0)
         for p_input in principal_inputs:
            self.principal_input_groups.append(torch.where((test_X[:,active_vars] == p_input).all(axis=1))[0])
            self.group_fractions.append(len(self.principal_input_groups[-1]) / test_X.shape[0])
      assert math.isclose(sum(self.group_fractions), 1), f'sum(self.group_fractions) must be equal 1, but recieved {sum(self.group_fractions)}'
   
   def evaluate(self, y_pred):
      cond_variance = 0
      for ids, fraction in zip(self.principal_input_groups, self.group_fractions):
         cond_variance += fraction * torch.var(y_pred[ids])
      
      return cond_variance.item()

class ConditionalVarianceCalculatorContinuous:

   def __init__(self, active_vars, num_principal_inputs, num_dummys, batch_size, device):
      self.num_principal_inputs = num_principal_inputs
      self.num_dummys = num_dummys

      active_vars = torch.from_numpy(active_vars)
      dimension = len(active_vars)
      num_active_vars = active_vars.sum()
      X = torch.zeros(num_principal_inputs, dimension)
      X[:, active_vars] = torch.randn(num_principal_inputs, num_active_vars)
      X = X.contiguous()
      X = X[:, None, :].expand(num_principal_inputs, num_dummys, dimension).clone()
      X[:, :, ~active_vars] = torch.randn(num_principal_inputs, num_dummys, dimension-num_active_vars)
      X = X.contiguous()
      X = X.view(num_principal_inputs * num_dummys, dimension)
      X = X.to(device)
      dataset = TensorDataset(X)
      self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
   
   def evaluate(self, model):
      y_pred = torch.vstack([model(x) for x, in self.dataloader])
      y_pred = y_pred.view(self.num_principal_inputs, self.num_dummys)
      var_estimations = torch.var(y_pred, dim=1)
      return var_estimations.mean().item()


def calculate_squared_distance_to_mdi(y_pred, model_coeffs, mdi_coeffs):
   model_norm_squared = (y_pred**2).mean()
   projection_norm_squared = (model_coeffs**2).sum()
   orth_projection_norm_squared = model_norm_squared - projection_norm_squared
   projection_diff_squared = ((model_coeffs - mdi_coeffs)**2).sum()
   diff_squared = projection_diff_squared + orth_projection_norm_squared
   return diff_squared.item()