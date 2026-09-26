import os
import copy
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from tqdm import tqdm

from models.captcha_cnn import CaptchaCNN
from utils.labels import NUM_CLASSES


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "data/train"

IMAGE_SIZE = 32
BATCH_SIZE = 256
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/captcha_cnn.pth"


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomAffine(
        degrees=15,
        translate=(0.08, 0.08),
        scale=(0.85, 1.15),
        shear=10
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


val_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Validation should not use augmentation
val_dataset.dataset.transform = val_transform


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = CaptchaCNN(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_model = None


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

        progress.set_postfix(
            loss=loss.item()
        )

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)


    print(
        f"\nEpoch {epoch + 1}"
        f"\nTrain Loss: {train_loss:.4f}"
        f"\nTrain Accuracy: {train_accuracy * 100:.2f}%"
        f"\nVal Loss: {val_loss:.4f}"
        f"\nVal Accuracy: {val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model,
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE
            },
            MODEL_PATH
        )

        print("Saved best model.")


print("\nTraining completed.")
print("Best model:", MODEL_PATH)