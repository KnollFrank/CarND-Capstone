import numpy as np
from PIL import Image

def PILImage2numpyImage(PILImage):
    (im_width, im_height) = PILImage.size
    return np.array(PILImage.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def numpyImage2PILImage(numpyImage):
    return Image.fromarray(numpyImage)
