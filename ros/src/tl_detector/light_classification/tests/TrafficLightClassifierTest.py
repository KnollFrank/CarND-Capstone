from unittest import TestCase

from PIL import Image

from TrafficLightClassifier import TrafficLightClassifier, TrafficLight
from TrafficLightDetector import TrafficLightDetector
from helper import PILImage2numpyImage


class TrafficLightClassifierTest(TestCase):

    def test_classifyTrafficLights_RED_RED_RED(self):
        self.shouldClassifyTrafficLights(imageFile='../images/red/img_0001_red.jpg',
                                         trafficLights=[TrafficLight.RED, TrafficLight.RED, TrafficLight.RED])

    def test_classifyTrafficLights_GREEN_GREEN(self):
        self.shouldClassifyTrafficLights(imageFile='../images/green/img_0175_green.jpg',
                                         trafficLights=[TrafficLight.GREEN, TrafficLight.GREEN])

    def shouldClassifyTrafficLights(self, imageFile, trafficLights):
        # Given
        classifier = TrafficLightClassifier('../../model.h5', TrafficLightDetector('../../data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'))
        image = PILImage2numpyImage(Image.open(imageFile))

        # When
        trafficLightsActual = classifier.classifyTrafficLights(image)

        # Then
        self.assertListEqual(trafficLights, trafficLightsActual)
