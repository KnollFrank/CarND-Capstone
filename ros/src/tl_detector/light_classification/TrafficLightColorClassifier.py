import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

from TrafficLightColor import TrafficLightColor
from utilities import numpyImage2PILImage


class TrafficLightColorClassifier:

    def __init__(self, modelFile):
        self.model = load_model(modelFile)
        # see: https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
        self.graph = tf.get_default_graph()

    def detectTrafficLightColor(self, numpyImage):
        PILImage = self.resize(numpyImage, width=50, height=120)
        with self.graph.as_default():
            predictions = self.model.predict(self.asCNNInput(PILImage))
        return self.getMostLikelyTrafficLightColor(predictions)

    def resize(self, numpyImage, width, height):
        return numpyImage2PILImage(numpyImage).resize((width, height), Image.ANTIALIAS)

    def asCNNInput(self, PILImage):
        return np.expand_dims(image.img_to_array(PILImage), axis=0)

    def getMostLikelyTrafficLightColor(self, predictions):
        labels = {0: TrafficLightColor.GREEN, 1: TrafficLightColor.RED, 3: TrafficLightColor.YELLOW}
        return labels[np.argmax(predictions)]
