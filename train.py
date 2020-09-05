import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, x, y, iterations, lamb = 0.1):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    indices = np.array_split(np.arange(x.shape[0]), x.shape[0]/20)

    losses = []
    for i in range(iterations):
        index = indices[np.random.randint(len(indices))]
        outputs = model(x[index])

        optimizer.zero_grad()
        loss = criterion(outputs, y[index])
        loss += model.regularize(lamb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    model.eval()
    return losses