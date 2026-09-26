import cv2
import numpy as np


def segment_characters(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive threshold
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )[1]

    # Morphological cleanup
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    height, width = gray.shape

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Ignore tiny noise
        if w < 5 or h < 10:
            continue

        # Ignore objects that are too large
        if w > width * 0.5:
            continue

        if h > height * 0.95:
            continue

        boxes.append(
            (x, y, w, h)
        )

    # Left-to-right ordering
    boxes.sort(
        key=lambda box: box[0]
    )

    characters = []

    for x, y, w, h in boxes:

        # Padding
        padding = 4

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            width,
            x + w + padding
        )

        y2 = min(
            height,
            y + h + padding
        )

        crop = gray[
            y1:y2,
            x1:x2
        ]

        characters.append(crop)

    return characters