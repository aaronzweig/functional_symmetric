import numpy as np
from numpy.linalg import norm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Symmetric, DeepSets, KNN, KK
from sample import generate_data, generate_narrow_data


def get_objective_func(name, input_dim):
    if name == "mean":
        func = lambda x: np.mean(norm(x, axis = 2), axis = 1, keepdims = True)
        
    elif name == "median":
        func = lambda x: np.median(norm(x, axis = 2), axis = 1, keepdims = True)
        
    elif name == "maximum":
        func = lambda x: np.max(norm(x, axis = 2), axis = 1, keepdims = True)
        
    elif name == "softmax":
        lamb = 0.1
        softmax = lambda x: lamb * np.log(np.mean(np.exp(norm(x, axis = 2) / lamb), axis = 1, keepdims = True))
        
    elif name == "neuron":
        teacher = Symmetric(input_dim, 10, 1)
        torch.nn.init.uniform_(teacher.phi.fc.weight, a = -0.2, b = 1.0)
        teacher.eval()

        def neuron(x):
            x = torch.from_numpy(x).float()
            y = teacher(x)
            return y.data.numpy().reshape(-1, 1)

        x, y = generate_narrow_data(3, 15, input_dim, neuron)
        print("Sample outputs of neuron")
        print("Note: if all 0s or no 0s, or values uniformly close to 0, this neuron may be degenerate on the domain")
        print(y.data.numpy().flatten())
        func = neuron
        
    elif name == "smooth_neuron":
        smooth_teacher = Symmetric(input_dim, 20, 1)
        torch.nn.init.uniform_(teacher.rho.fc1.weight,a = -0.2, b = 1.0)
        smooth_teacher.eval()

        def smooth_neuron(x):
            x = torch.from_numpy(x).float()
            y = smooth_teacher(x)
            return y.data.numpy().reshape(-1, 1)

        x, y = generate_narrow_data(3, 15, input_dim, smooth_neuron)
        print("Sample outputs of neuron")
        print("Note: if all 0s or no 0s, or values uniformly close to 0, this neuron may be degenerate on the domain")
        print(y.data.numpy().flatten())
        func = smooth_neuron
        
    func.__name__ = name
    return func
