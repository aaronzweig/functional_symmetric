import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LittleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, squared):
        super(LittleBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias = False)
        self.squared = squared

    def forward(self, x):
        z = F.relu(self.fc(x))
        return z ** 2 if self.squared else z
    
class Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Block, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias = False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias = False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
#F1: NN + NN
class Symmetric(nn.Module):
    def __init__(self, input_dim, hidden_dim_phi, hidden_dim_rho, output_dim = 1, squared = False):
        super(Symmetric, self).__init__()
        
        self.hidden_dim_phi = hidden_dim_phi
        self.hidden_dim_rho = hidden_dim_rho
        self.input_dim = input_dim + 1 #Explicit bias term to simplify path norm
        self.output_dim = output_dim
        self.squared = squared
        
        self.rho = None
        self.phi = None
        self.reinit()
    
    def reinit(self):
        self.rho = Block(self.hidden_dim_phi, self.hidden_dim_rho, self.output_dim)
        self.phi = LittleBlock(self.input_dim, self.hidden_dim_phi, self.squared)
    
    def forward(self, x):        
        batch_size, input_set_dim, input_dim = x.shape
        
        x = x.view(-1, input_dim)
        z = self.phi(x)
        z = z.view(batch_size, input_set_dim, -1)
        z = torch.mean(z, 1)
        return self.rho(z)
    
    def regularize(self, lamb):
        reg_loss = 0.
        W1 = self.phi.fc.weight
        W2 = self.rho.fc1.weight
        w = self.rho.fc2.weight
        
        W1 = torch.norm(W1, dim = 1, keepdim = True)
        if  self.squared:
            W1 = W1 ** 2
        W2 = torch.abs(W2)
        w = torch.abs(w)
        
        reg_loss = torch.matmul(w, torch.matmul(W2, W1))
        reg_loss = torch.sum(w).item()
        
        return lamb * reg_loss
    
class DeepSets(Symmetric):
    def __init__(self, input_dim, hidden_dim_phi, hidden_dim_rho):
        super(DeepSets, self).__init__(input_dim, hidden_dim_phi, hidden_dim_rho)

    def forward(self, x):        
        batch_size, input_set_dim, input_dim = x.shape
        
        x = x.view(-1, input_dim)
        z = self.phi(x)
        z = z.view(batch_size, input_set_dim, -1)
        z = torch.sum(z, 1)
        return self.rho(z)
    
#F2: K + NN
class KNN(Symmetric):
    def __init__(self, input_dim, hidden_dim_phi, hidden_dim_rho, output_dim = 1, squared = False):
        super(KNN, self).__init__(input_dim, hidden_dim_phi, hidden_dim_rho, output_dim, squared)

    def reinit(self):
        super(KNN, self).reinit()
        
        self.phi.fc.weight.requires_grad = False        
        self.phi.fc.weight.div_(torch.norm(self.phi.fc.weight, dim = 1, keepdim = True))
        
    def regularize(self, lamb):
        reg_loss = 0.

        W2 = self.rho.fc1.weight
        w = self.rho.fc2.weight
        
        W2 = torch.norm(W2, dim = 1, keepdim = True)
        w = torch.abs(w)
        
        reg_loss = torch.matmul(w, W2)
        reg_loss = torch.sum(w).item()
        
        return lamb * reg_loss
    
#F3: K + K
class KK(KNN):
    def __init__(self, input_dim, hidden_dim_phi, hidden_dim_rho, output_dim = 1, squared = False):
        super(KK, self).__init__(input_dim, hidden_dim_phi, hidden_dim_rho, output_dim, squared)

    def reinit(self):
        super(KK, self).reinit()
        
        self.rho.fc1.weight.requires_grad = False        
        self.rho.fc1.weight.div_(torch.norm(self.rho.fc1.weight, dim = 1, keepdim = True))

        
    def regularize(self, lamb):
        reg_loss = 0.
        
        w = self.rho.fc2.weight
        w = torch.norm(w, dim = 1, keepdim = True)

        reg_loss = torch.sum(w).item()

        return lamb * reg_loss
    