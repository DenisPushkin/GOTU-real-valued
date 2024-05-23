import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
import time
import math
import helpers
from abc import ABC, abstractmethod

NEGLIGIBLE_LOSS = 0 # training terminates after validation loss achieves this value

class TrainerBase(ABC):
  
  def __init__(self, model, valid_X, valid_y, test_X, test_y, monomials, 
          active_vars, mdi_coeffs, **params):
    
    self.model = model
    self.monomials = monomials
    self.mdi_coeffs = mdi_coeffs
    self.test_y = test_y
    self.monomials_basis = helpers.get_monomials_basis(params['distr'])
    
    # initialize training stats
    self.epoch_logs = []
    self.train_losses = []
    self.valid_losses = []
    self.test_losses = []

    self.coefficients = []
    self.basis_coeffs_calculator = helpers.BasisCoefficientsCalculator(self.monomials, test_X.cpu().detach().numpy(), params['distr'], support_size=params['support_size'])

    # for discrete uniform distribution, we use non-orthogonal basis
    # thus, calculating the distance to mdi becomes more complicated, so we don't do this
    self.is_eval_dist_to_mdi = (self.mdi_coeffs is not None) and (params['distr'] != 'unif_discrete')
    self.squared_distances_to_mdi = [] if self.is_eval_dist_to_mdi else None

    # variance of target function conditioned on the active variables (the one that participate in target function)
    self.is_eval_cond_variance = not all(active_vars)
    self.conditional_variances = [] if self.is_eval_cond_variance else None
    if self.is_eval_cond_variance:
      self.distribution_type = helpers.get_distribution_type(params['distr'])
      if self.distribution_type == 'discrete':
        self.cond_variance_calculator = helpers.ConditionalVarianceCalculatorDiscrete(test_X.cpu(), active_vars)
      else:
        # distribution_type == 'continuous'
        self.cond_variance_calculator = helpers.ConditionalVarianceCalculatorContinuous(active_vars, 
                num_principal_inputs=2000, num_dummys=50, batch_size=params['test_batch_size'], device=params['device'])
    
    self.valid_dl = DataLoader(TensorDataset(valid_X, valid_y), batch_size=params['test_batch_size'], shuffle=False)
    self.test_dl = DataLoader(TensorDataset(test_X, test_y), batch_size=params['test_batch_size'], shuffle=False)
    
    self.loss_func = nn.MSELoss()

  def print_epoch_logs(self, start_time):
      print(f"Epoch: {self.epoch_logs[-1]:4}, Train Loss: {self.train_losses[-1]:0.6f}, Valid Loss: {self.valid_losses[-1]:0.6f}, Test Loss: {self.test_losses[-1]:0.6f}, Elapsed Time:", time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))
      print('Coeffs:', end=' ')
      for mono, coeff in zip(self.monomials, self.coefficients[-1]):
         print(f'{mono}: {coeff: .4f},', end=' ')
      print()
      if self.is_eval_cond_variance:
         print(f'Cond variance: {self.conditional_variances[-1]:.6f}', end = ' ')
      if self.is_eval_dist_to_mdi:
         print(f'Dist to MDI squared: {self.squared_distances_to_mdi[-1]:.6f}', end = ' ')
      print()

  def model_evaluation(self, epoch, train_loss, is_print, start_time):
    self.model.eval()
    with torch.no_grad():
      test_pred = torch.vstack([self.model(xb) for xb, _ in self.test_dl])
      if self.monomials is not None:
        coeffs = self.basis_coeffs_calculator.evaluate(test_pred.cpu().detach().numpy())
        self.coefficients.append(coeffs)
      
      if self.is_eval_cond_variance:
        if self.distribution_type == 'discrete':
           self.conditional_variances.append(self.cond_variance_calculator.evaluate(test_pred))
        else:
           # distribution_type == 'continuous'
           self.conditional_variances.append(self.cond_variance_calculator.evaluate(self.model))
      
      if self.is_eval_dist_to_mdi:
         self.squared_distances_to_mdi.append(helpers.calculate_squared_distance_to_mdi(test_pred, coeffs, self.mdi_coeffs))
      
      test_loss = self.loss_func(test_pred, self.test_y)
      valid_loss = sum([self.loss_func(self.model(xb), yb).item() for xb, yb in self.valid_dl]) / len(self.valid_dl)
      if train_loss is None:
        train_loss = valid_loss
      
      self.train_losses.append(train_loss)
      self.valid_losses.append(valid_loss)
      self.test_losses.append(test_loss)
      self.epoch_logs.append(epoch)

      if is_print:
        self.print_epoch_logs(start_time)
  
  @abstractmethod
  def perform_one_epoch_training(self, batches_per_epoch):
      pass

  def train(self, epochs, batches_per_epoch, compute_int, verbose_int):
    start_time = time.time()
    is_print = (verbose_int != 0)
    self.model_evaluation(0, None, is_print, start_time)

    for epoch in range(1, epochs+1):
      train_loss = self.perform_one_epoch_training(batches_per_epoch)

      if math.isnan(train_loss) or math.isinf(train_loss):
         print("Breaking the training: train loss exploded")
         break

      if (epoch % compute_int == 0) or (epoch == epochs):
         is_print = (verbose_int != 0) and (epoch % verbose_int == 0 or epoch == epochs)
         self.model_evaluation(epoch, train_loss, is_print, start_time)
         if (self.valid_losses[-1] < NEGLIGIBLE_LOSS):
            print(f'Achieved sufficient validation loss after {epoch} epochs')
            break
    
    training_logs = {
     'epochs': self.epoch_logs,
     'train_losses': self.train_losses,
     'valid_losses': self.valid_losses,
     'test_losses': self.test_losses,
     'monomials_basis': self.monomials_basis,
     'monomials': self.monomials,
     'coefficients': self.coefficients,
     'conditional_variances': self.conditional_variances,
     'squared_distances_to_mdi': self.squared_distances_to_mdi,
    }
    return training_logs


class TrainerWithStandardOptimizer(TrainerBase):
   
   def __init__(self, model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params):
      super().__init__(model, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params)
      self.train_batch_generator = train_batch_generator
      if params['opt'].lower() == 'sgd':
         print('Using SGD')
         self.opt = optim.SGD(self.model.parameters(), lr=params['lr'], momentum=0.0, weight_decay=0.0)
      elif params['opt'].lower() == 'adamw':
         print(f"Using AdamW with lr = {params['lr']} and weight decay = {params['weight_decay']}")
         self.opt = optim.AdamW(self.model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']), 
                           weight_decay=params['weight_decay'])
      else:
         raise ValueError('Illegal argument of -opt')


class TrainerStochasticGradient(TrainerWithStandardOptimizer):
  
   def __init__(self, model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params):
      super().__init__(model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params)
   
   def perform_one_epoch_training(self, batches_per_epoch):
      self.model.train()
      train_loss = 0
      for _ in range(batches_per_epoch):
         # generating training data on fly
         xb, yb = self.train_batch_generator()
         pred = self.model(xb)
         loss = self.loss_func(pred, yb)
         train_loss += loss.item()
         self.opt.zero_grad()
         loss.backward()
         self.opt.step()
      train_loss /= batches_per_epoch
      return train_loss


class TrainerFullGradient(TrainerWithStandardOptimizer):
  
   def __init__(self, model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params):
      super().__init__(model, train_batch_generator, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params)
   
   def perform_one_epoch_training(self, batches_per_epoch):
      self.model.train()
      self.opt.zero_grad()
      train_loss = 0
      for _ in range(batches_per_epoch):
         # generating training data on fly
         xb, yb = self.train_batch_generator()
         pred = self.model(xb)
         loss = (1 / batches_per_epoch) * self.loss_func(pred, yb)
         train_loss += loss.item()
         loss.backward()
      self.opt.step()
      return train_loss


class TrainerGDLineSearch(TrainerBase):
   
   NUM_TOL = 1e-7 # numerical tolerance
   
   def __init__(self, model, train_X, train_y, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params):
      super().__init__(model, valid_X, valid_y, test_X, test_y, monomials, 
               active_vars, mdi_coeffs, **params)
      # shuffling does not make sense, since we always calculate gradient on whole dataset
      self.train_dl = DataLoader(TensorDataset(train_X, train_y), batch_size=params['batch_size'], shuffle=False)
      self.Lipschitz_const = 1.0
   
   def print_epoch_logs(self, start_time):
      print(f"Epoch: {self.epoch_logs[-1]:4}, Train Loss: {self.train_losses[-1]:0.6f}, Valid Loss: {self.valid_losses[-1]:0.6f}, lr = {1/self.Lipschitz_const:0.6f}, Elapsed Time:", time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))
      print('Coeffs:', end=' ')
      for mono, coeff in zip(self.monomials, self.coefficients[-1]):
         print(f'{mono}: {coeff: .4f},', end=' ')
      print()
      if self.is_eval_cond_variance:
         print(f'Cond variance: {self.conditional_variances[-1]:.6f}', end = ' ')
      if self.is_eval_dist_to_mdi:
         print(f'Dist to MDI squared: {self.squared_distances_to_mdi[-1]:.6f}', end = ' ')
      print()
   
   def calculate_training_loss(self):
      self.model.eval()
      with torch.no_grad():
         train_loss = sum([self.loss_func(self.model(xb), yb).item() for xb, yb in self.train_dl]) / len(self.train_dl)
      return train_loss

   def perform_one_epoch_training(self, batches_per_epoch=None):
      self.model.train()
      self.model.zero_grad()
      train_loss = 0.0
      for xb, yb in self.train_dl:
         pred = self.model(xb)
         loss = 1/len(self.train_dl) * self.loss_func(pred, yb)
         loss.backward()
         train_loss += loss.item()

      trainable_parameters = [param for param in self.model.parameters() if param.requires_grad]
      prev_weights = [param.data.clone() for param in trainable_parameters]
      weights_grad = [param.grad.clone() for param in trainable_parameters]

      grad_norm_squared = sum([torch.sum(torch.square(grad)).item() for grad in weights_grad])
      for param, weights, grad in zip(trainable_parameters, prev_weights, weights_grad):
         param.data = weights - 1 / self.Lipschitz_const * grad
      while (self.calculate_training_loss() > train_loss - 1 / (2*self.Lipschitz_const) * grad_norm_squared + self.NUM_TOL):
         self.Lipschitz_const *= 2
         for param, weights, grad in zip(trainable_parameters, prev_weights, weights_grad):
            param.data = weights - 1 / self.Lipschitz_const * grad

      self.Lipschitz_const /= 2

      return train_loss