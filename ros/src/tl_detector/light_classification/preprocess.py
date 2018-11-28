import cv2
import numpy as np
from keras.layers import Cropping2D
from scipy import ndimage
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import os

factor = 1.0 / 4.0

def resize(image):
    return cv2.resize(image, dsize=None, fx=factor, fy=factor)


def preprocess(image):
    return resize(image)


# TODO: rename to get_input_shape_channels_last() or get_input_shape_height_widht_channels()?
def get_input_shape():
    width = 800
    height = 600
    return (int(height * factor), int(width * factor), 3)


def get_images_and_measurements():
    def read_and_preprocess(image_file):
        return preprocess(ndimage.imread(image_file))

    def get_images_and_categories(category):
        images = list(map(read_and_preprocess, map(lambda fileName: "images/" + category + "/" + fileName, os.listdir("images/" + category))))
        categories = [category] * len(images)
        return images, categories

    images_red, categories_red = get_images_and_categories('red')
    images_green, categories_green = get_images_and_categories('green')
    images_yellow, categories_yellow = get_images_and_categories('yellow')
    images_unknown, categories_unknown = get_images_and_categories('unknown')
    images_no_traffic_light, categories_no_traffic_light = get_images_and_categories('no_traffic_light')
    Y = []
    Y.extend(categories_red)
    Y.extend(categories_green)
    Y.extend(categories_yellow)
    Y.extend(categories_unknown)
    Y.extend(categories_no_traffic_light)
    encoder = LabelEncoder()
    encoder.fit(Y)
    np.save('classes.npy', encoder.classes_)
    encoded_Y = encoder.transform(Y)
    images = []
    images.extend(images_red)
    images.extend(images_green)
    images.extend(images_yellow)
    images.extend(images_unknown)
    images.extend(images_no_traffic_light)
    return images, np_utils.to_categorical(encoded_Y, 5)


def flip_image(image):
    return np.fliplr(image)


def flip_measurement(measurement):
    return -measurement


def flip_images(images):
    return map(flip_image, images)


def flip_measurements(measurements):
    return map(flip_measurement, measurements)


def get_steering_left(steering_center):
    return steering_center + 0.2


def get_steering_right(steering_center):
    return steering_center - 0.2
