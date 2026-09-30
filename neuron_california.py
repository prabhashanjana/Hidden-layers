import math
import random
import pandas as pd


def import_data():
    df = pd.read_csv("housing.csv")
    median_price = df["median_house_value"].median()

    rows = [
        ((lon, lat), 1 if price > median_price else 0)
        for lon, lat, price in zip(
            df["longitude"], df["latitude"], df["median_house_value"]
        )
    ]

    # fixed seed -> same split every run
    random.Random(42).shuffle(rows)
    split = int(0.8 * len(rows))
    train_raw, test_raw = rows[:split], rows[split:]

    # normalization: min/max from TRAIN only, then apply to both
    lons = [x[0] for x, _ in train_raw]
    lats = [x[1] for x, _ in train_raw]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    def normalize(lon, lat):
        return ((lon - lon_min) / (lon_max - lon_min),
                (lat - lat_min) / (lat_max - lat_min))

    train = [(normalize(*x), y) for x, y in train_raw]
    test = [(normalize(*x), y) for x, y in test_raw]

    return train, test


def weighted_sum(inputs, weights, bias):
    return sum(i * w for i, w in zip(inputs, weights)) + bias


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def loss(predicted, actual):
    # binary cross-entropy
    eps = 1e-12
    return -(actual * math.log(predicted + eps)
             + (1 - actual) * math.log(1 - predicted + eps))


def gradient(prediction, actual, x):
    # for cross-entropy + sigmoid the p*(1-p) term cancels
    return (prediction - actual) * x


def predict(weights, bias, inputs):
    return sigmoid(weighted_sum(inputs, weights, bias))


def train(weights, bias, train_data, learning_rate=0.1, epochs=10):
    for epoch in range(1, epochs + 1):
        total_loss = 0

        for inputs, actual in train_data:
            prediction = predict(weights, bias, inputs)
            total_loss += loss(prediction, actual)

            # update weights
            for i in range(len(weights)):
                weights[i] -= learning_rate * \
                    gradient(prediction, actual, inputs[i])

            # update bias (its "input" is always 1)
            bias -= learning_rate * gradient(prediction, actual, 1)

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.6f} | "
              f"w1={weights[0]:.4f}, w2={weights[1]:.4f}, bias={bias:.4f}")

    return weights, bias


def evaluate(weights, bias, data):
    correct = 0
    for inputs, actual in data:
        prediction = predict(weights, bias, inputs)
        predicted_class = 1 if prediction >= 0.5 else 0
        if predicted_class == actual:
            correct += 1
    return correct / len(data)


def main():
    train_data, test_data = import_data()
    print(f"train rows: {len(train_data)}, test rows: {len(test_data)}")

    weights = [0.0, 0.0]
    bias = 0.0
    weights, bias = train(weights, bias, train_data)

    print(f"Train accuracy: {evaluate(weights, bias, train_data) * 100:.1f}%")
    print(f"Test accuracy : {evaluate(weights, bias, test_data) * 100:.1f}%")


if __name__ == "__main__":
    main()
