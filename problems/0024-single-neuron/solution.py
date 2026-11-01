import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[
	list[float], float]:

	features = torch.tensor(features,dtype=torch.float32)
	labels = torch.tensor(labels,dtype=torch.float32)
	weights = torch.tensor(weights,dtype=torch.float32)
	out = features @ weights + bias
	res1 = torch.sigmoid(out)
	res2 = F.mse_loss(res1,labels)
	return torch.round(res1,decimals=4).tolist(),torch.round(res2,decimals=4).tolist()