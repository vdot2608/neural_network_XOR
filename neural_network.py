
import numpy as np

# 1. Обучающие данные: два входа и правильный ответ
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# 2. Создаём нейросеть: 2 входа, 4 скрытых нейрона, 1 выход
np.random.seed(42)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 3. Обучение
learning_rate = 0.01

for epoch in range(100000):
    # Прямой проход: получаем предсказания
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)

    z2 = a1 @ W2 + b2
    prediction = sigmoid(z2)

    # Ошибка предсказания
    error = prediction - y

    # Обратное распространение ошибки
    dz2 = error / len(X)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Обновляем веса и смещения
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 20000 == 0:
        # Заново проверяем сеть с текущими весами
        z1_check = X @ W1 + b1
        a1_check = np.tanh(z1_check)
        prediction_check = sigmoid(a1_check @ W2 + b2)

        # Считаем ошибку
        loss = -np.mean(
            y * np.log(prediction_check + 1e-9)
            + (1 - y) * np.log(1 - prediction_check + 1e-9)
        )

        print(f"\nЭпоха {epoch + 1}: ошибка = {loss:.4f}")

        for inputs, probability in zip(X, prediction_check):
            print(
                f"{inputs.astype(int)} -> "
                f"{probability[0]:.4f}"
            )

# 4. Проверяем, чему научилась сеть
print("\nРезультаты обучения:")

hidden = np.tanh(X @ W1 + b1)
result = sigmoid(hidden @ W2 + b2)

for inputs, probability, answer in zip(X, result, y):
    print(
        f"{inputs.astype(int)} -> "
        f"{probability[0]:.3f} "
        f"(ответ сети: {int(probability[0] >= 0.5)}, "
        f"правильный ответ: {int(answer[0])})"
    )
