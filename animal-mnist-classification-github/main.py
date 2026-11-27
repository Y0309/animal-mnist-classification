"""
Animal-MNIST: Logistic Regression, MLP, and CNN comparison
----------------------------------------------------------
This script combines the LR/MLP code from main.py and the CNN code
from the teammate's notebook into one reproducible experiment.

Run from the project folder:
    python main.py

Required packages:
    pip install animal-mnist numpy matplotlib pandas scikit-learn torch
"""

import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from animal_mnist.dataset_Loader import (
    load_animal_mnist_dataset,
    load_animal_mnist_splits,
)


# ============================================================
# 1. Configuration and reproducibility
# ============================================================

SEED = 42
MLP_EPOCHS = 200
CNN_EPOCHS = 30
CNN_PATIENCE = 6
CNN_BATCH_SIZE = 64
LEARNING_RATE = 0.001

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ANIMAL_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Dog",
    "Elephant", "Giraffe", "Horse", "Sheep", "Zebra",
]
CLASS_LABELS = list(range(len(ANIMAL_NAMES)))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use deterministic algorithms where supported.
# This may make training slower, but improves reproducibility.
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("ANIMAL-MNIST: LR vs MLP vs CNN")
print("=" * 65)
print("Using device:", device)


# ============================================================
# 2. Load the dataset and official train/validation/test splits
# ============================================================

X, y = load_animal_mnist_dataset()
train_idx, val_idx, test_idx = load_animal_mnist_splits()

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("\nDataset shapes")
print("Full dataset:", X.shape)
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Classes:", np.unique(y))


# ============================================================
# 3. Save example images from the ten classes
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for class_id, ax in enumerate(axes.flat):
    image_index = np.where(y == class_id)[0][0]
    ax.imshow(X[image_index], cmap="gray")
    ax.set_title(ANIMAL_NAMES[class_id])
    ax.axis("off")

fig.suptitle("Examples from the Animal-MNIST Dataset")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "animal_mnist_examples.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 4. Prepare flattened and normalized images for LR and MLP
# ============================================================

def flatten_and_normalize(images):
    """Convert each 28x28 image into 784 normalized pixel features."""
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0


X_train_flat = flatten_and_normalize(X_train)
X_val_flat = flatten_and_normalize(X_val)
X_test_flat = flatten_and_normalize(X_test)

print("\nFlattened input shapes")
print("Training:", X_train_flat.shape)
print("Validation:", X_val_flat.shape)
print("Testing:", X_test_flat.shape)


# ============================================================
# 5. Shared evaluation function
# ============================================================

all_test_predictions = {}
all_test_metrics = {}


def calculate_metrics(model_name, true_labels, predictions):
    """Calculate accuracy and macro-averaged precision, recall, and F1."""
    metrics = {
        "Accuracy": accuracy_score(true_labels, predictions),
        "Precision": precision_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "Macro-F1": f1_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
    }

    all_test_predictions[model_name] = np.asarray(predictions)
    all_test_metrics[model_name] = metrics

    print(f"\n{model_name} test results")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f} ({value:.2%})")

    return metrics


# ============================================================
# 6. Logistic Regression
# ============================================================

print("\n" + "=" * 65)
print("Training Logistic Regression")
print("=" * 65)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_flat, y_train)

# Validation evaluation (development only)
lr_val_predictions = lr_model.predict(X_val_flat)
print(
    "LR validation accuracy:",
    f"{accuracy_score(y_val, lr_val_predictions):.4f}"
)

# Final test evaluation
lr_test_predictions = lr_model.predict(X_test_flat)
calculate_metrics("Logistic Regression", y_test, lr_test_predictions)


# ============================================================
# 7. Multilayer Perceptron (MLP)
# ============================================================

print("\n" + "=" * 65)
print("Training Multilayer Perceptron")
print("=" * 65)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


# Reset the seed before model initialization for reproducibility.
torch.manual_seed(SEED)

mlp_model = MLP().to(device)

X_train_mlp_tensor = torch.tensor(X_train_flat, dtype=torch.float32).to(device)
y_train_mlp_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_mlp_tensor = torch.tensor(X_val_flat, dtype=torch.float32).to(device)
X_test_mlp_tensor = torch.tensor(X_test_flat, dtype=torch.float32).to(device)

criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = torch.optim.Adam(
    mlp_model.parameters(),
    lr=LEARNING_RATE,
)

mlp_train_losses = []

for epoch in range(MLP_EPOCHS):
    mlp_model.train()

    outputs = mlp_model(X_train_mlp_tensor)
    loss = criterion_mlp(outputs, y_train_mlp_tensor)

    optimizer_mlp.zero_grad()
    loss.backward()
    optimizer_mlp.step()

    mlp_train_losses.append(loss.item())

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"MLP Epoch [{epoch + 1}/{MLP_EPOCHS}], "
            f"Loss: {loss.item():.4f}"
        )

# Validation evaluation (development only)
mlp_model.eval()
with torch.no_grad():
    val_outputs = mlp_model(X_val_mlp_tensor)
    mlp_val_predictions = val_outputs.argmax(dim=1).cpu().numpy()

print(
    "MLP validation accuracy:",
    f"{accuracy_score(y_val, mlp_val_predictions):.4f}"
)

# Final test evaluation
with torch.no_grad():
    test_outputs = mlp_model(X_test_mlp_tensor)
    mlp_test_predictions = test_outputs.argmax(dim=1).cpu().numpy()

calculate_metrics("MLP", y_test, mlp_test_predictions)

# Save MLP training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, MLP_EPOCHS + 1), mlp_train_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("MLP Training Loss")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mlp_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 8. Convolutional Neural Network (CNN)
# ============================================================

print("\n" + "=" * 65)
print("Training Convolutional Neural Network")
print("=" * 65)


def prepare_cnn_images(images):
    """Normalize images and add the single grayscale channel dimension."""
    images = images.astype(np.float32) / 255.0
    images = images.reshape(-1, 1, 28, 28)
    return torch.tensor(images, dtype=torch.float32)


X_train_cnn_tensor = prepare_cnn_images(X_train)
X_val_cnn_tensor = prepare_cnn_images(X_val)
X_test_cnn_tensor = prepare_cnn_images(X_test)

y_train_cnn_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_cnn_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_cnn_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_cnn_tensor, y_train_cnn_tensor),
    batch_size=CNN_BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val_cnn_tensor, y_val_cnn_tensor),
    batch_size=CNN_BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    TensorDataset(X_test_cnn_tensor, y_test_cnn_tensor),
    batch_size=CNN_BATCH_SIZE,
    shuffle=False,
)


class AnimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Reset the seed before CNN initialization.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

cnn_model = AnimalCNN().to(device)

criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.Adam(
    cnn_model.parameters(),
    lr=LEARNING_RATE,
)

best_val_accuracy = -1.0
best_weights = None
epochs_without_improvement = 0

cnn_train_losses = []
cnn_val_accuracies = []

for epoch in range(CNN_EPOCHS):
    cnn_model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_cnn.zero_grad()

        outputs = cnn_model(images)
        loss = criterion_cnn(outputs, labels)

        loss.backward()
        optimizer_cnn.step()

        total_loss += loss.item() * images.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    # Evaluate on validation data without updating model weights.
    cnn_model.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = cnn_model(images)
            predictions = outputs.argmax(dim=1)

            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_accuracy = accuracy_score(val_labels, val_predictions)

    cnn_train_losses.append(avg_train_loss)
    cnn_val_accuracies.append(val_accuracy)

    print(
        f"CNN Epoch [{epoch + 1}/{CNN_EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_weights = copy.deepcopy(cnn_model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= CNN_PATIENCE:
        print("CNN early stopping triggered.")
        break

if best_weights is None:
    raise RuntimeError("CNN training did not produce a best model checkpoint.")

# Restore the checkpoint with the highest validation accuracy.
cnn_model.load_state_dict(best_weights)
print(f"\nBest CNN validation accuracy: {best_val_accuracy:.2%}")

# Save CNN training curves.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cnn_train_losses) + 1), cnn_train_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("CNN Training Loss")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cnn_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cnn_val_accuracies) + 1), cnn_val_accuracies)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy")
ax.set_title("CNN Validation Accuracy")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cnn_validation_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Final CNN test evaluation.
cnn_model.eval()
cnn_test_predictions = []
cnn_test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = cnn_model(images)
        predictions = outputs.argmax(dim=1)

        cnn_test_predictions.extend(predictions.cpu().numpy())
        cnn_test_labels.extend(labels.numpy())

calculate_metrics("CNN", cnn_test_labels, cnn_test_predictions)


# ============================================================
# 9. Combined results table
# ============================================================

results_df = pd.DataFrame(all_test_metrics).T
results_df.index.name = "Model"

print("\n" + "=" * 65)
print("FINAL TEST SET COMPARISON")
print("=" * 65)
print((results_df * 100).round(2).to_string())

results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv")


# ============================================================
# 10. Confusion matrices and classification reports
# ============================================================

for model_name, predictions in all_test_predictions.items():
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=ANIMAL_NAMES,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    print(f"\n{model_name} classification report")
    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=ANIMAL_NAMES,
        zero_division=0,
    )
    print(report)

    report_dict = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=ANIMAL_NAMES,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report_dict).T.to_csv(
        OUTPUT_DIR / f"{safe_name}_classification_report.csv"
    )


# ============================================================
# 11. Test accuracy comparison chart
# ============================================================

accuracy_values = {
    model_name: metrics["Accuracy"] * 100
    for model_name, metrics in all_test_metrics.items()
}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(list(accuracy_values.keys()), list(accuracy_values.values()))

ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test Accuracy: Logistic Regression vs MLP vs CNN")
ax.set_ylim(0, 100)
ax.bar_label(bars, fmt="%.2f%%", padding=3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 12. Per-class accuracy comparison
# ============================================================

per_class_accuracy = {}

for model_name, predictions in all_test_predictions.items():
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS,
    )
    class_totals = cm.sum(axis=1)
    correct = np.diag(cm)

    per_class_accuracy[model_name] = np.divide(
        correct,
        class_totals,
        out=np.zeros_like(correct, dtype=float),
        where=class_totals != 0,
    ) * 100

per_class_df = pd.DataFrame(
    per_class_accuracy,
    index=ANIMAL_NAMES,
)
per_class_df.index.name = "Animal class"

print("\nPer-class test accuracy (%)")
print(per_class_df.round(2).to_string())
per_class_df.to_csv(OUTPUT_DIR / "per_class_accuracy.csv")

fig, ax = plt.subplots(figsize=(12, 6))
per_class_df.plot(kind="bar", ax=ax)
ax.set_ylabel("Test accuracy (%)")
ax.set_xlabel("Animal class")
ax.set_title("Per-Class Test Accuracy by Model")
ax.set_ylim(0, 100)
ax.legend(title="Model")
plt.xticks(rotation=35, ha="right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "per_class_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 13. Save exact test predictions for later analysis
# ============================================================

predictions_df = pd.DataFrame({
    "true_label": y_test,
    "true_animal": [ANIMAL_NAMES[int(label)] for label in y_test],
    "lr_prediction": all_test_predictions["Logistic Regression"],
    "mlp_prediction": all_test_predictions["MLP"],
    "cnn_prediction": all_test_predictions["CNN"],
})

predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print(f"All tables, reports, predictions, and figures are saved in: {OUTPUT_DIR.resolve()}")
print("=" * 65)
