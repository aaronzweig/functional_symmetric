import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import itertools
import copy

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from model import Symmetric, DeepSets, KNN, KK
import argparse

class Overkill(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, output_dim = 1):
        super(Overkill, self).__init__()
        
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.input_dim = input_dim + 1 #Explicit bias term to simplify path norm
        self.output_dim = output_dim
        
        self.rho = None
        self.phi = None
        self.reinit()
    
    def reinit(self):
        self.phi = nn.Sequential(
            nn.Linear(self.input_dim, self.h1),
            nn.ReLU(),
            nn.Linear(self.h1, self.h1),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(self.h1, self.h2),
#             nn.BatchNorm1d(self.h2),
            nn.ReLU(),
            nn.Linear(self.h2, self.h3),
#             nn.BatchNorm1d(self.h3),
            nn.ReLU(),
            nn.Linear(self.h3, self.output_dim)
        )
    
    def forward(self, x):        
        batch_size, input_set_dim, input_dim = x.shape
        
        x = x.view(-1, input_dim)
        z = self.phi(x)
        z = z.view(batch_size, input_set_dim, -1)
        z = torch.mean(z, 1)
        return self.rho(z)
    
    def regularize(self, lamb):
        return 0.

def train(model, dataloader, iterations, lamb = 0.1, lr = 0.003):
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    losses = []
    for i in range(iterations):
        print("iter", i)
        iter_losses = []
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss += model.regularize(lamb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            iter_losses.append(loss.item())
        print("iter_loss: ", np.mean(np.array(iter_losses)))
    
    return losses

def test(model, dataloader):
    model.eval()
    
    correct = 0.
    total = 0.
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim = 1)
        correct += preds.eq(y).sum()
        total += y.shape[0]
    
    return 1 - (correct / total)

def cross_validate(model, dataloader, iterations, lambs, verbose):
    models = []
    for lamb in lambs:
        model_copy = copy.deepcopy(model)
        losses = train(model_copy, dataloader, iterations, lamb)
        models.append(model_copy)
        if verbose:
            print(losses[::100])
    return models

def compare_models(hidden_dim, iterations, input_dim = 3, verbose = False):
        
    f1 = Symmetric(input_dim, hidden_dim, hidden_dim, 10)
    f2 = KNN(input_dim, hidden_dim, hidden_dim, 10)
    f3 = KK(input_dim, hidden_dim, hidden_dim, 10)

    f1.__name__ = "S1"
    f2.__name__ = "S2"
    f3.__name__ = "S3"

    models = [f1, f2, f3]
    
    lambs = [0., 1e-6, 1e-4, 1e-2]

    for model in models:
        print("model", model.__name__)
        cv_models = cross_validate(model, train_loader, iterations, lambs, verbose)
        
        validation_errors = np.zeros_like(lambs)
        for i, cv_model in enumerate(cv_models):
            validation_errors[i] = test(cv_model, train_loader)
        
        i = np.argmin(validation_errors)
        lamb = lambs[i]
            
        runs = 3
        run_errors = np.zeros(runs)
        for i in range(runs):
            print("run", i)
            model_copy = copy.deepcopy(model)
            model_copy.reinit()
            train(model_copy, train_loader, iterations, lamb)
            run_errors[i] = test(model_copy, test_loader)
        
        mean_error = np.mean(run_errors)
        std_error = np.std(run_errors)
        
        print("mean: {}, std: {}".format(mean_error, std_error))
        
#         if log_plot:
#             plt.semilogy(N_list, mean_error, label = model.__name__)
#         else:
#             plt.plot(N_list, mean_error, label = model.__name__)
#         plt.fill_between(N_list, mean_error - std_error, mean_error + std_error, alpha = 0.2)

    
#     plt.legend()
#     plt.ylim([1e-5, 1e-1]) 
#     plt.xlabel("N")
#     plt.ylabel("Mean Square Error")
#     narrow_str = "Narrow" if narrow else "Wide"
#     plt.title(narrow_str + " generalization for " + objective.__name__)
#     scale_str = "" if not scaleup else "scaled"
#     plt.savefig("plots_high_dim/" + objective.__name__ + "_" + narrow_str + "_" + str(input_dim) + scale_str)
# #     plt.show()
#     plt.close()

class PointCloud(object):

    def __init__(self, cloud_size):
        self.cloud_size = cloud_size

    def __call__(self, image):

        flat = image.flatten()
        flat = (flat > 0.5).float() * flat
        
        vertex_count = torch.nonzero(flat).shape[0]
        
        size = min(self.cloud_size, vertex_count)
        
        args = torch.argsort(flat)[-size:].int()
        args = args[torch.randperm(size)]
        if size < self.cloud_size:
            repeat = self.cloud_size // size + 1
            args = args.repeat(repeat)[:self.cloud_size]
        
        
        rows = (args / 28.).int()
        cols = torch.fmod(args, 28)
        
        image = torch.zeros(self.cloud_size, 4)
        
        rows = rows.float()
        rows -= torch.mean(rows)
        rows /= torch.std(rows)
        cols = cols.float()
        cols -= torch.mean(cols)
        cols /= torch.std(cols)

        image[:,0] = rows
        image[:,1] = cols
        image[:,2] = flat[args.long()]
        image[:,3] = 1 #bias term

        return image


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='symmetric')

    parser.add_argument('--h1', type=int, default=100, help='')
    parser.add_argument('--h2', type=int, default=500, help='')
    parser.add_argument('--h3', type=int, default=500, help='')
    parser.add_argument('--iterations', type=int, default=50, help='')
    parser.add_argument('--lr', type=float, default=0.003, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--lamb', type=float, default=0.0, help='')
    parser.add_argument('--model', default='overkill', help='')

    parser.add_argument('--run_id', type=int, default=1, help='')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cloud_size = 200
    valid_size = 0.1
    train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   PointCloud(cloud_size)
                                               ]))

    test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   PointCloud(cloud_size)
                                               ]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)



    input_dim = 3
    if args.model == 'overkill':
        model = Overkill(input_dim, args.h1, args.h2, args.h3, 10).to(device)
    elif args.model == 's1':
        model = Symmetric(input_dim, args.h1, args.h2, 10).to(device)
    elif args.model == 's2':
        model = KNN(input_dim, args.h1, args.h2, 10).to(device)
    elif args.model == 's3':
        model = KK(input_dim, args.h1, args.h2, 10).to(device)
            
        train(model, train_loader, args.iterations, lamb = args.lamb, lr = args.lr)
    
        print(args)
            
        error = test(model, train_loader)
        print("train error: ", error)

        error = test(model, val_loader)
        print("val error: ", error)
            
        error = test(model, test_loader)
        print("test error: ", error)
    
