import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, x, y, iterations, lamb = 0.1, lr=0.0005):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(iterations):
        outputs = model(x)

        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss += model.regularize(lamb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    model.eval()
    return losses