import torch

from PIL import Image
from torchvision import transforms

from models.captcha_cnn import CaptchaCNN
from utils.labels import IDX_TO_CHAR, NUM_CLASSES


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/captcha_cnn.pth"


transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

model = CaptchaCNN(
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# ============================================================
# PREDICT
# ============================================================

def predict_character(image_path):

    image = Image.open(
        image_path
    ).convert("L")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        logits = model(image)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(dim=1)
        )

    character = IDX_TO_CHAR[
        prediction.item()
    ]

    return character, confidence.item()


if __name__ == "__main__":

    image_path = "test_character.png"

    character, confidence = predict_character(
        image_path
    )

    print(
        f"Prediction: {character}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )