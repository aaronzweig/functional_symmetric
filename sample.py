import numpy as np
import torch

def generate_corrupted_data(N, batch_size, input_dim, objective, bias_first = False, eps = 0.2):
    
    N_fake = int(eps * N)
    N_real = N - N_fake
    
    m_fake = np.random.normal(size = (batch_size, 1, input_dim))
    m_real = np.random.normal(size = (batch_size, 1, input_dim))
    
    x_fake = np.random.normal(size = (batch_size, N_fake, input_dim)) + m_fake
    x_real = np.random.normal(size = (batch_size, N_real, input_dim)) + m_real
    
    x = np.concatenate([x_real, x_fake], axis = 1)
    
    bias = np.ones((batch_size, N, 1))
    if not bias_first:
        y = objective(x)
        x = np.concatenate([x, bias], axis = 2)
    else:
        x = np.concatenate([x, bias], axis = 2)
        y = objective(x)
    
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return (x,y)
    
    
def generate_narrow_data(N, batch_size, input_dim, objective, bias_first = False):
    x = np.random.uniform(low = -1, high = 1, size = (batch_size, N, input_dim))
    
    bias = np.ones((batch_size, N, 1))
    if not bias_first:
        y = objective(x)
        x = np.concatenate([x, bias], axis = 2)
    else:
        x = np.concatenate([x, bias], axis = 2)
        y = objective(x)
    
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return (x,y)

def generate_wide_data(N, batch_size, input_dim, objective, bias_first = False):
    x = np.random.uniform(low = -3, high = 3, size = (batch_size, N, input_dim))
    
    bias = np.ones((batch_size, N, 1))
    if not bias_first:
        y = objective(x)
        x = np.concatenate([x, bias], axis = 2)
    else:
        x = np.concatenate([x, bias], axis = 2)
        y = objective(x)
    
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return (x,y)

# def generate_wide_data(N, batch_size, input_dim, objective, bias_first = False):
#     x = np.zeros((batch_size, N, input_dim))
#     for i in range(input_dim):
        
#         a = np.random.uniform(low = -1, high = 1, size = batch_size)
#         b = np.random.uniform(low = -1, high = 1, size = batch_size)
#         a, b = np.minimum(a,b), np.maximum(a,b)

#         x_fill = np.random.uniform(low = np.tile(a, (N,1)), high = np.tile(b, (N,1)))
#         x[:,:,i] = x_fill.T
    
#     bias = np.ones((batch_size, N, 1))
#     if not bias_first:
#         y = objective(x)
#         x = np.concatenate([x, bias], axis = 2)
#     else:
#         x = np.concatenate([x, bias], axis = 2)
#         y = objective(x)
    
#     x = torch.from_numpy(x).float()
#     y = torch.from_numpy(y).float()
#     return (x,y)

def generate_data(N, batch_size, input_dim, objective, narrow, bias_first = False):    
    if narrow:
        return generate_narrow_data(N, batch_size, input_dim, objective, bias_first)
    else:
        return generate_wide_data(N, batch_size, input_dim, objective, bias_first)
    