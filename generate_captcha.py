import random
import string
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


CHARACTERS = (
    string.digits +
    string.ascii_uppercase +
    string.ascii_lowercase
)


WIDTH = 250
HEIGHT = 70

OUTPUT_DIR = "data/captcha"


def random_color():

    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


def generate_captcha():

    text = "".join(
        random.choice(CHARACTERS)
        for _ in range(5)
    )

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        random_color()
    )

    draw = ImageDraw.Draw(
        image
    )

    # You should put several .ttf fonts
    # in a fonts/ directory.
    font_path = random.choice([
        "fonts/DejaVuSans.ttf",
        "fonts/DejaVuSans-Bold.ttf"
    ])

    font = ImageFont.truetype(
        font_path,
        40
    )

    x = 10

    for char in text:

        y = random.randint(
            5,
            20
        )

        draw.text(
            (x, y),
            char,
            font=font,
            fill=random_color()
        )

        x += 45

    # Noise lines
    for _ in range(5):

        draw.line(
            [
                (
                    random.randint(0, WIDTH),
                    random.randint(0, HEIGHT)
                ),
                (
                    random.randint(0, WIDTH),
                    random.randint(0, HEIGHT)
                )
            ],
            fill=random_color(),
            width=1
        )

    # Noise dots
    for _ in range(300):

        x = random.randint(
            0,
            WIDTH - 1
        )

        y = random.randint(
            0,
            HEIGHT - 1
        )

        draw.point(
            (x, y),
            fill=random_color()
        )

    return image, text


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for i in range(50000):

        image, text = generate_captcha()

        filename = os.path.join(
            OUTPUT_DIR,
            f"{text}_{i}.png"
        )

        image.save(filename)

    print(
        "Generated 50,000 CAPTCHA images."
    )


if __name__ == "__main__":
    main()