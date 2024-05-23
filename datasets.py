import torch
from torch.utils.data import TensorDataset, DataLoader
import itertools

class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            # Restart the iterator once it runs out of samples
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

def generate_all_binaries(d, device):
    """
    Generate all binary sequences of length d where bits are +1 and -1
    :param d: dimension
    :return: the output is a numpy array of size 2^d * d
    """
    return torch.tensor([list(seq) for seq in itertools.product([-1, 1], repeat=d)], dtype=torch.float32, device=device)

def get_random_batch(distribution, batch_size, dimension, true_model, device, seen_condition, **params):
    X = get_test_data(distribution, batch_size, dimension, device, **params)
    if seen_condition == 'x1eq1':
        X[:,0] = 1.0
    elif seen_condition == 'x1eqx2':
        X[:,0] = X[:,1]
    elif seen_condition == 'x1eq1orx2eq1':
        mask = torch.randint(0,2,(batch_size,), device=device)
        X[:,0] = torch.where(mask == 0, 1, X[:,0])
        X[:,1] = torch.where(mask == 1, 1, X[:,1])
    elif seen_condition == 'x1eq1orx2eq1orx3eq1':
        mask = torch.randint(0,3,(batch_size,), device=device)
        X[:,0] = torch.where(mask == 0, 1, X[:,0])
        X[:,1] = torch.where(mask == 1, 1, X[:,1])
        X[:,2] = torch.where(mask == 2, 1, X[:,2])
    elif seen_condition == 'x1eq1orx2eq1orx3eq1orx4eq1':
        mask = torch.randint(0,4,(batch_size,), device=device)
        X[:,0] = torch.where(mask == 0, 1, X[:,0])
        X[:,1] = torch.where(mask == 1, 1, X[:,1])
        X[:,2] = torch.where(mask == 2, 1, X[:,2])
        X[:,3] = torch.where(mask == 3, 1, X[:,3])
    elif seen_condition == 'x1eq1orx2eq1orx3eq1orx4eq1orx5eq1':
        mask = torch.randint(0,5,(batch_size,), device=device)
        X[:,0] = torch.where(mask == 0, 1, X[:,0])
        X[:,1] = torch.where(mask == 1, 1, X[:,1])
        X[:,2] = torch.where(mask == 2, 1, X[:,2])
        X[:,3] = torch.where(mask == 3, 1, X[:,3])
        X[:,4] = torch.where(mask == 4, 1, X[:,4])
    elif seen_condition == 'x1+x2+x3=x4':
        X[:,3] = X[:,0] + X[:,1] + X[:,2]
    else:
        raise Exception("Unexpected value of 'seen_condition'")
    y = true_model(X)
    return X, y

def get_seen_samples(data, targets, seen_condition):
    if seen_condition == 'x1eq1':
        seen_cond_logical = lambda X: (X[:,0] == 1)
    elif seen_condition == 'x1eqx2':
        seen_cond_logical = lambda X: (X[:,0] == X[:,1])
    elif seen_condition == 'x1eq1orx2eq1':
        seen_cond_logical = lambda X: ((X[:,0] == 1) | (X[:,1] == 1))
    elif seen_condition == 'x1eq1orx2eq1orx3eq1':
        seen_cond_logical = lambda X: ((X[:,0] == 1) | (X[:,1] == 1) | (X[:,2] == 1))
    elif seen_condition == 'x1eq1orx2eq1orx3eq1orx4eq1':
        seen_cond_logical = lambda X: ((X[:,0] == 1) | (X[:,1] == 1) | (X[:,2] == 1) | (X[:,3] == 1))
    elif seen_condition == 'x1eq1orx2eq1orx3eq1orx4eq1orx5eq1':
        seen_cond_logical = lambda X: ((X[:,0] == 1) | (X[:,1] == 1) | (X[:,2] == 1) | (X[:,3] == 1)| (X[:,4] == 1))
    elif seen_condition == 'x1+x2+x3=x4':
        seen_cond_logical = lambda X: (X[:,0] + X[:,1] + X[:,2] == X[:,3])
    else:
        raise Exception("Unexpected value of 'seen_condition'")
    mask = seen_cond_logical(data)
    data = data[mask]
    targets = targets[mask]
    return data, targets

def get_train_batch_generator(data, targets, batch_size):
    permutation = torch.randperm(data.shape[0])
    data = data[permutation]
    targets = targets[permutation]
    dataset = TensorDataset(data, targets)
    # check that all batches will be balanced
    assert len(dataset) % batch_size == 0, f'len(dataset) = {len(dataset)} is not divisible by batch_size = {batch_size}'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_iterator = DataLoaderIterator(dataloader)
    train_batch_generator = lambda : next(dataloader_iterator)
    
    batches_per_epoch =  len(dataloader)
    return train_batch_generator, batches_per_epoch


def get_test_data(distribution, num_samples, dimension, device, **params):
    if distribution == 'gauss':
        data = torch.randn(num_samples, dimension, device=device)
    elif distribution == 'unif_discrete':
        window = round((params['support_size'] - 1) / 2)
        data = torch.randint(-window, window+1, (num_samples, dimension), device=device).float()
    elif distribution == 'boolean':
        data = torch.randint(0, 2, (num_samples, dimension), device=device)
        data[data == 0] = -1
        data = data.float()
    else:
      raise ValueError("Illegal argument of 'distribution' parameter")
   
    return data