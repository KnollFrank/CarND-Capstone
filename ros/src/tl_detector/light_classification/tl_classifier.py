import rospy
from styx_msgs.msg import TrafficLight
from keras import __version__ as keras_version
from keras.models import load_model
import h5py
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing import image

factor = 1.0 / 4.0


def resize(image):
    img_height, img_width = 150, 150
    return cv2.resize(image, dsize=(img_width, img_height))


def preprocess(image):
    return resize(image)


class TLClassifier(object):
    def __init__(self):
        self.model = load_model('light_classification/model.h5')
        # see: https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
        self.graph = tf.get_default_graph()
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load('light_classification/classes.npy')

    def asTrafficLightState(self, stateStr):
        if stateStr == "red":
            return TrafficLight.RED
        if stateStr == "yellow":
            return TrafficLight.YELLOW
        if stateStr == "green":
            return TrafficLight.GREEN
        if stateStr == "unknown":
            return TrafficLight.UNKNOWN
        if stateStr == "no_traffic_light":
            return TrafficLight.UNKNOWN

    def channels_last_2_channels_first(self, image):
        return np.moveaxis(image, -1, 0)

    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(img_height, img_width))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        labels = {0: 'green', 1: 'no_traffic_light', 2: 'red', 3: 'unknown', 4: 'yellow'}
        image_array = preprocess(np.asarray(image))
        # image_array = self.channels_last_2_channels_first(image_array)
        with self.graph.as_default():
            ynew = self.model.predict(image_array[None, :, :, :])
        label = labels[np.argmax(ynew)]
        rospy.loginfo("label: %s", label)
        return self.asTrafficLightState(label)
