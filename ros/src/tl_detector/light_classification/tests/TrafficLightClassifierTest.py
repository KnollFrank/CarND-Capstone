from unittest import TestCase

from PIL import Image

from TrafficLightClassifier import TrafficLightClassifier, TrafficLightColor
from TrafficLightDetector import TrafficLightDetector
from TrafficLightExtractorTest import get_script_path
from utilities import PILImage2numpyImage


class TrafficLightClassifierTest(TestCase):

    def test_classifyTrafficLights_RED_RED_RED(self):
        self.shouldClassifyTrafficLights(imageFile=get_script_path() + '/images/red/img_0001_red.jpg',
                                         trafficLightColors=[TrafficLightColor.RED, TrafficLightColor.RED,
                                                             TrafficLightColor.RED])

    def test_classifyTrafficLights_GREEN_GREEN(self):
        self.shouldClassifyTrafficLights(imageFile=get_script_path() + '/images/green/img_0175_green.jpg',
                                         trafficLightColors=[TrafficLightColor.GREEN, TrafficLightColor.GREEN])

    def shouldClassifyTrafficLights(self, imageFile, trafficLightColors):
        # Given
        classifier = TrafficLightClassifier(get_script_path() + '/../model.h5', TrafficLightDetector(
            get_script_path() + '/../data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'))
        image = PILImage2numpyImage(Image.open(imageFile))

        # When
        trafficLightColorsActual = classifier.classifyTrafficLights(image)

        # Then
        self.assertListEqual(trafficLightColors, trafficLightColorsActual)
