from unittest import TestCase

from TrafficLightClassifier import TrafficLightClassifier, TrafficLight
from PIL import Image
import numpy as np

from TrafficLightProvider import TrafficLightProvider


class TrafficLightClassifierTest(TestCase):

    def test_classifyTrafficLights_RED_RED_RED(self):
        self.shouldClassifyTrafficLights(imageFile='../images/red/img_0001_red.jpg',
                                         trafficLights=[TrafficLight.RED, TrafficLight.RED, TrafficLight.RED])

    def test_classifyTrafficLights_GREEN_GREEN(self):
        self.shouldClassifyTrafficLights(imageFile='../images/green/img_0175_green.jpg',
                                         trafficLights=[TrafficLight.GREEN, TrafficLight.GREEN])

    def shouldClassifyTrafficLights(self, imageFile, trafficLights):
        # Given
        classifier = TrafficLightClassifier('../../model.h5', TrafficLightProvider('../../data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'))
        image = self.load_image_into_numpy_array(Image.open(imageFile))

        # When
        trafficLightsActual = classifier.classifyTrafficLights(image)

        # Then
        self.assertListEqual(trafficLights, trafficLightsActual)

    # TODO: DRY with TrafficLightExtractor.load_image_into_numpy_array()
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
