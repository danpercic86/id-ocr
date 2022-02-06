# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot
from functools import singledispatch

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


@singledispatch
def _get_image(src) -> NDArray:
    return NotImplementedError(f"Cannot format value of type {type(src)}")


@_get_image.register
def __get_image(src: str) -> NDArray:
    return plt.imread(f'results/{src}.jpg')


@_get_image.register(np.ndarray)
def __get_image(src: NDArray) -> NDArray:
    return src


def show(src: str | NDArray) -> None:
    dpi = matplotlib.rcParams['figure.dpi']

    image = _get_image(src)

    height, width = image.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figure_size = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_axes([0, 0, 1, 1])

    # Display the image.
    ax.imshow(image, cmap='gray')
    plt.show()
