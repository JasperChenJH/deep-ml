import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    loss = torch.nn.MSELoss()
    X = features
    y = labels
    W = initial_weights.clone().detach().float().requires_grad_(True)
    b = torch.tensor(initial_bias,requires_grad=True,dtype=torch.float)
    mean_mse = []
    sigmoid = nn.Sigmoid()
    optimizer = torch.optim.SGD([W,b],learning_rate)
    for epoch in range(epochs):
        y_hat = sigmoid(X@W+b)
        l = loss(y_hat,y)
        mean_mse.append(round(l.item(),4))
        optimizer.zero_grad()
        # W.grad = None
        # b.grad = None
        l.backward()
        optimizer.step()
        # with torch.no_grad():
        #     W-=learning_rate*W.grad
        #     b-=learning_rate*b.grad
    updated_weights = [round(v, 4) for v in W.detach().tolist()]
    updated_bias = round(b.item(), 4)
    return updated_weights,updated_bias,mean_mse