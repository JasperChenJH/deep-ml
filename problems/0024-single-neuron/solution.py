import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	out = features @ weights + bias
	probabilities = sigmoid(out)
	mse = np.mean((probabilities - labels) ** 2)
	return np.round(probabilities,4).tolist() , np.round(mse,4)

def sigmoid(x):
	return 1/(1+np.exp(-x))


