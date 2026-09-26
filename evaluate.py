import os
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.captcha_cnn import CaptchaCNN
from utils.labels import IDX_TO_CHAR, NUM_CLASSES


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "data/train"
MODEL_PATH = "models/captcha_cnn.pth"

IMAGE_SIZE = 32
BATCH_SIZE = 256

NUM_WORKERS = 4

TEST_SPLIT = 0.20
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}\n"
        f"Please prepare the dataset first."
    )


full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=test_transform
)

print(f"Total images: {len(full_dataset)}")
print(f"Number of classes found: {len(full_dataset.classes)}")

print("\nDataset classes:")
print(full_dataset.classes)


# ============================================================
# CHECK NUMBER OF CLASSES
# ============================================================

if len(full_dataset.classes) != NUM_CLASSES:
    print("\nWARNING:")
    print(
        f"Expected {NUM_CLASSES} classes, "
        f"but dataset contains {len(full_dataset.classes)} classes."
    )

    print(
        "\nThe class ordering used by ImageFolder must "
        "match the ordering used during training."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

test_size = int(
    len(full_dataset) * TEST_SPLIT
)

train_size = (
    len(full_dataset) - test_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

_, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=generator
)

print(f"\nTest images: {len(test_dataset)}")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = CaptchaCNN(
    num_classes=NUM_CLASSES
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Train the model first using:\n"
        f"python train.py"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Supports the checkpoint format from train.py
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    # Also supports directly saved state_dict
    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)
model.eval()

print(f"Device: {DEVICE}")
print("Model loaded successfully.")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

total_loss = 0.0
total_correct = 0
total_samples = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        predictions = outputs.argmax(
            dim=1
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# OVERALL METRICS
# ============================================================

test_loss = (
    total_loss / total_samples
)

test_accuracy = (
    total_correct / total_samples
)


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Correct       : {total_correct}/{total_samples}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


# ImageFolder's class ordering
class_names = full_dataset.classes

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("GENERATING CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

fig_size = max(
    12,
    len(class_names) * 0.25
)

fig, ax = plt.subplots(
    figsize=(fig_size, fig_size)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    cmap="Blues",
    colorbar=True
)

plt.title(
    "69-Class Character Recognition Confusion Matrix"
)

plt.tight_layout()

os.makedirs(
    "results",
    exist_ok=True
)

confusion_path = (
    "results/confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to: "
    f"{confusion_path}"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

cm = np.asarray(cm)

for class_index, class_name in enumerate(
    class_names
):

    total = cm[class_index].sum()

    if total == 0:
        accuracy = 0.0

    else:
        accuracy = (
            cm[class_index, class_index]
            / total
        )

    print(
        f"{class_name:>3} : "
        f"{accuracy * 100:6.2f}% "
        f"({cm[class_index, class_index]}/{total})"
    )


# ============================================================
# RANDOM SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM TEST PREDICTIONS")
print("=" * 60)


# Select 16 random samples
num_samples = min(
    16,
    len(test_dataset)
)

random.seed(RANDOM_SEED)

random_indices = random.sample(
    range(len(test_dataset)),
    num_samples
)


# Create a figure
fig, axes = plt.subplots(
    4,
    4,
    figsize=(12, 12)
)

axes = axes.flatten()


for plot_index, dataset_index in enumerate(
    random_indices
):

    image, label = test_dataset[
        dataset_index
    ]

    input_tensor = image.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(dim=1)
        )

    predicted_index = prediction.item()

    true_character = class_names[
        label
    ]

    predicted_character = class_names[
        predicted_index
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # Undo normalization for display
    display_image = (
        image.squeeze(0).numpy() * 0.5
    ) + 0.5

    display_image = np.clip(
        display_image,
        0,
        1
    )

    axes[plot_index].imshow(
        display_image,
        cmap="gray"
    )

    axes[plot_index].axis("off")

    if label == predicted_index:

        title = (
            f"True: {true_character}\n"
            f"Pred: {predicted_character}\n"
            f"Conf: {confidence_value:.1f}%"
        )

    else:

        title = (
            f"True: {true_character}\n"
            f"Pred: {predicted_character}\n"
            f"Conf: {confidence_value:.1f}%"
        )

    axes[plot_index].set_title(
        title,
        fontsize=10
    )


# Hide unused axes
for i in range(
    num_samples,
    len(axes)
):

    axes[i].axis("off")


plt.tight_layout()

prediction_path = (
    "results/random_predictions.png"
)

plt.savefig(
    prediction_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Random predictions saved to: "
    f"{prediction_path}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Results directory: results/"
)

print(
    "  ├── confusion_matrix.png"
)

print(
    "  └── random_predictions.png"
)
