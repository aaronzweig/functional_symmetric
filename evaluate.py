import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import copy
from sample import generate_data
from train import train

from model import Symmetric, DeepSets, KNN, KK
from sample import generate_data, generate_narrow_data
from train import train

def generalization_error(N_list, batch_size, input_dim, model, objective, narrow, bias_first):
    errors = []
    for N in N_list:
        x,y = generate_data(N, batch_size, input_dim, objective, narrow, bias_first)
        outputs = model(x)
        error = nn.MSELoss()(outputs, y).item()
        errors.append(error)
    return np.array(errors)

def cross_validate(model, x, y, iterations, lambs, verbose):
    models = []
    for lamb in lambs:
        model_copy = copy.deepcopy(model)
        losses = train(model_copy, x, y, iterations, lamb)
        models.append(model_copy)
        if verbose and lamb == 0:
            print("check for overfitting power of", model.__name__)
            print("lowest loss:", np.min(np.array(losses)))
    return models