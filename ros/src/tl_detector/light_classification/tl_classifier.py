import rospy
from styx_msgs.msg import TrafficLight
from keras import __version__ as keras_version
from keras.models import load_model
import h5py
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

factor = 1.0 / 4.0

def resize(image):
    return cv2.resize(image, dsize=None, fx=factor, fy=factor)


def preprocess(image):
    return resize(image)


class TLClassifier(object):
    def __init__(self):
        self.model = load_model('light_classification/model_squeezenet.h5')
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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_array = preprocess(np.asarray(image))
        image_array = self.channels_last_2_channels_first(image_array)
        with self.graph.as_default():
            ynew = self.model.predict(image_array[None, :, :, :])
        label = self.encoder.inverse_transform([np.argmax(ynew)])[0]
        rospy.loginfo("label: %s", label)
        return self.asTrafficLightState(label)
