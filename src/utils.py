from cv2 import cv2
from numpy.typing import NDArray


def load(filename: str) -> NDArray:
    return cv2.imread(filename)


def save(filename: str, image: NDArray) -> None:
    cv2.imwrite(f"results/{filename}.jpg", image)
