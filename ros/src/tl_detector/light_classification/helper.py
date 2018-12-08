import numpy as np


def PILImage2numpyImage(PILImage):
    (im_width, im_height) = PILImage.size
    return np.array(PILImage.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
