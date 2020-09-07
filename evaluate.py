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

def generalization_error(N_list, batch_size, model, objective, narrow):
    errors = []
    for N in N_list:
        input_dim = model.input_dim
        x,y = generate_data(N, batch_size, input_dim, objective, narrow)
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

def compare_models(N_max, hidden_dim, iterations, batch_size, input_dim, objective, narrow, verbose = True, log_plot = False, scaleup = False):
    
    c = 1 if not scaleup else 2

    f1 = Symmetric(input_dim, c * hidden_dim, hidden_dim)
    f2 = KNN(input_dim, c * hidden_dim, hidden_dim)
    f3 = KK(input_dim, c * hidden_dim, hidden_dim)

    f1.__name__ = "S1"
    f2.__name__ = "S2"
    f3.__name__ = "S3"

    models = [f1, f2, f3]
    
    lambs = [0., 1e-6, 1e-4, 1e-2]
    N_list = np.arange(2, N_max + 16)

    for model in models:
        x, y = generate_data(N_max, batch_size, input_dim, objective, narrow)
        cv_models = cross_validate(model, x, y, iterations, lambs, verbose)
        
        validation_errors = np.zeros_like(lambs)
        for i, cv_model in enumerate(cv_models):
            validation_errors[i] = generalization_error([N_max], 1000, cv_model, objective, narrow)[0]
        
        i = np.argmin(validation_errors)
        lamb = lambs[i]
            
        runs = 10
        run_errors = np.zeros((runs, len(N_list)))
        for i in range(runs):
            x, y = generate_data(N_max, batch_size, input_dim, objective, narrow)
            model_copy = copy.deepcopy(model)
            model_copy.reinit()
            train(model_copy, x, y, iterations, lamb)
            errors = generalization_error(N_list, 1000, model_copy, objective, narrow)
            run_errors[i] = np.array(errors)
        
        mean_error = np.mean(run_errors, axis = 0)
        std_error = np.std(run_errors, axis = 0)
        if verbose:
            print("performance of ", model.__name__, " on ", objective.__name__)
            print("CV lambda =", lamb)
            print("mean:", mean_error)
            print("std:", std_error)
            
            
        narrow_str = "Narrow" if narrow else "Wide"
        scaleup_str = "scaleup" if scaleup else ""
        save_str = model.__name__ + "_" + objective.__name__ + "_" + narrow_str + "_" + str(input_dim) + scaleup_str
            
        np.save(save_str + "_mean", mean_error)
        np.save(save_str + "_std", std_error)
        
        if log_plot:
            plt.semilogy(N_list, mean_error, label = model.__name__)
        else:
            plt.plot(N_list, mean_error, label = model.__name__)
        plt.fill_between(N_list, mean_error - std_error, mean_error + std_error, alpha = 0.2)

    
    plt.legend()
    plt.ylim([1e-5, 1e-1]) 

    plt.xlabel("N")
    plt.ylabel("Mean Square Error")
    narrow_str = "Narrow" if narrow else "Wide"
    plt.title(narrow_str + " generalization for " + objective.__name__)
    scaleup_str = "scaleup" if scaleup else ""
#     plt.savefig("plots_high_dim/" + objective.__name__ + "_" + narrow_str + "_" + str(input_dim) + scaleup_str)
    plt.show()
    plt.close()