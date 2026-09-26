CLASSES = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("abcdefghijklmnopqrstuvwxyz") +
    ["😊", "😂", "😍", "😎", "😢", "😡", "👍"]
)

CHAR_TO_IDX = {
    char: idx
    for idx, char in enumerate(CLASSES)
}

IDX_TO_CHAR = {
    idx: char
    for idx, char in enumerate(CLASSES)
}

NUM_CLASSES = len(CLASSES)

print("Number of classes:", NUM_CLASSES)