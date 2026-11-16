import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Part A(a): Number of observations and features
print("Dataset shape:", X.shape)

# Part A(b): Number of observations in each class
for label, name in enumerate(data.target_names):
    count = np.sum(y == label)
    print(f"Class {label} ({name}): {count}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Part A(c): Fit the scaler ONLY on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Standardization completed.")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Part B: Train the neural network
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Predict labels for training and testing data
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

# Evaluate the model
print("\n--- Part B: Neural Network Results ---")
print(
    f"Training accuracy: "
    f"{accuracy_score(y_train, y_train_pred):.6f}"
)
print(
    f"Testing accuracy: "
    f"{accuracy_score(y_test, y_test_pred):.6f}"
)
print(
    f"Testing precision: "
    f"{precision_score(y_test, y_test_pred, pos_label=1):.6f}"
)
print(
    f"Testing recall: "
    f"{recall_score(y_test, y_test_pred, pos_label=1):.6f}"
)
print(
    f"Testing F1-score: "
    f"{f1_score(y_test, y_test_pred, pos_label=1):.6f}"
)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))
import pandas as pd
from pathlib import Path

# Part C: Compare five network architectures
architectures = [
    (10,),
    (50,),
    (100,),
    (50, 50),
    (100, 50)
]

results = []

for architecture in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    results.append({
        "Architecture": str(architecture),
        "Training Accuracy": accuracy_score(y_train, train_pred),
        "Testing Accuracy": accuracy_score(y_test, test_pred),
        "F1-score": f1_score(y_test, test_pred, pos_label=1)
    })

# Display the comparison table
results_df = pd.DataFrame(results)

print("\n--- Part C: Architecture Comparison ---")
print(results_df.to_string(index=False, float_format="%.6f"))

# Save the table beside this Python file
output_path = Path(__file__).resolve().parent / "architecture_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nTable saved to: {output_path}")
import matplotlib.pyplot as plt

# Train each architecture and plot its training loss
for architecture in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    plt.plot(
        range(1, len(model.loss_curve_) + 1),
        model.loss_curve_,
        label=str(architecture)
    )

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss by Network Architecture")
plt.legend(title="Hidden Layers")
plt.tight_layout()

# Save beside the Python script
image_path = Path(__file__).resolve().parent / "training_loss.png"
plt.savefig(image_path, dpi=300)

print(f"Figure saved to: {image_path}")
plt.show()