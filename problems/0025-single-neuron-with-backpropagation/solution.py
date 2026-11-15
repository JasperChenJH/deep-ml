import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int) -> tuple[np.ndarray, float, list[float]]:

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    w = np.asarray(initial_weights, dtype=float).copy()
    b = float(initial_bias)

    mse_values = []
    m = len(y)

    for epoch in range(epochs):
        # 1. 前向传播
        z = X @ w + b
        y_pred = 1 / (1 + np.exp(-z))

        # 2. 计算 MSE（更新前）
        mse = np.mean((y_pred - y) ** 2)
        mse_values.append(round(float(mse), 4))

        # 3. 反向传播
        delta = 2 * (y_pred - y) * y_pred * (1 - y_pred)

        dw = X.T @ delta / m
        db = np.mean(delta)

        # 4. 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

    updated_weights = np.round(w, 4)
    updated_bias = round(float(b), 4)

    return updated_weights, updated_bias, mse_values