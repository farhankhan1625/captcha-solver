import torch

from PIL import Image
from torchvision import transforms

from models.captcha_cnn import CaptchaCNN
from utils.labels import IDX_TO_CHAR, NUM_CLASSES
from utils.segmentation import segment_characters


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


MODEL_PATH = "models/captcha_cnn.pth"


transform = transforms.Compose([

    transforms.ToPILImage(),

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
# SOLVE CAPTCHA
# ============================================================

def solve_captcha(image_path):

    characters = segment_characters(
        image_path
    )

    result = ""

    confidences = []

    for character in characters:

        tensor = transform(
            character
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

        predicted_character = IDX_TO_CHAR[
            prediction.item()
        ]

        result += predicted_character

        confidences.append(
            confidence.item()
        )

    return result, confidences


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_path = "captcha.png"

    result, confidences = solve_captcha(
        image_path
    )

    print("\n==============================")
    print("CAPTCHA SOLVER")
    print("==============================")

    print(
        "Prediction:",
        result
    )

    print(
        "Character confidences:"
    )

    for i, confidence in enumerate(
        confidences
    ):

        print(
            f"Character {i + 1}: "
            f"{confidence * 100:.2f}%"
        )