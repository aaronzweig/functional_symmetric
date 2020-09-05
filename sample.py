import numpy as np
import torch

def generate_narrow_data(N, batch_size, input_dim, objective):
    x = np.random.uniform(low = -1, high = 1, size = (batch_size, N, input_dim))
    y = objective(x)
    
    #bias term
    x[:,:,-1] = 1
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return (x,y)

def generate_wide_data(N, batch_size, input_dim, objective):
    x = np.zeros((batch_size, N, input_dim))
    for i in range(input_dim):
        
        a = np.random.uniform(low = -1, high = 1, size = batch_size)
        b = np.random.uniform(low = -1, high = 1, size = batch_size)
        a, b = np.minimum(a,b), np.maximum(a,b)

        x_fill = np.random.uniform(low = np.tile(a, (N,1)), high = np.tile(b, (N,1)))
        x[:,:,i] = x_fill.T
    
    y = objective(x)
    
    #bias term
    x[:,:,-1] = 1
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return (x,y)

def generate_data(N, batch_size, input_dim, objective, narrow):
    if narrow:
        return generate_narrow_data(N, batch_size, input_dim, objective)
    else:
        return generate_wide_data(N, batch_size, input_dim, objective)
    