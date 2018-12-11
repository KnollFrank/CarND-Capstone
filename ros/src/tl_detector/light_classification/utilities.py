import errno
import os

import numpy as np
from PIL import Image


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def PILImage2numpyImage(PILImage):
    (im_width, im_height) = PILImage.size
    return np.array(PILImage.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def numpyImage2PILImage(numpyImage):
    return Image.fromarray(numpyImage)


def loadNumpyImage(imagePath):
    return PILImage2numpyImage(loadPILImage(imagePath))


def loadPILImage(imagePath):
    return Image.open(imagePath)


def resizePILImage(PILImage, width, height):
    return PILImage.resize((width, height), Image.ANTIALIAS)
