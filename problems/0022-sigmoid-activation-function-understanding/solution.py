import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z = torch.tensor(z,dtype=torch.float64)
    return 1/(1+torch.exp(-z).item())
print(sigmoid(0))
