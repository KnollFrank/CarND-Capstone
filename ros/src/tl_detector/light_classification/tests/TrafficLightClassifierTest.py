from unittest import TestCase

from PIL import Image

from TrafficLightClassifier import TrafficLightClassifier
from TrafficLightColor import TrafficLightColor
from TrafficLightColorClassifier import TrafficLightColorClassifier
from TrafficLightColorClassifierFactory import IMG_WIDTH, IMG_HEIGHT, TRAFFIC_LIGHT_COLOR_CLASSIFIER_FILE
from TrafficLightDetector import TrafficLightDetector
from TrafficLightExtractor import TRAFFIC_LIGHT_DETECTOR_NAME
from TrafficLightExtractorTest import get_script_path
from TrafficLightHavingMinScoreDetector import TrafficLightHavingMinScoreDetector
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

        classifier = TrafficLightClassifier(
            TrafficLightHavingMinScoreDetector(
                TrafficLightDetector(
                    get_script_path() + '/../data/' + TRAFFIC_LIGHT_DETECTOR_NAME + '/frozen_inference_graph.pb')),
            TrafficLightColorClassifier(get_script_path() + '/../' + TRAFFIC_LIGHT_COLOR_CLASSIFIER_FILE,
                                        img_height=IMG_HEIGHT, img_width=IMG_WIDTH))
        image = PILImage2numpyImage(Image.open(imageFile))

        # When
        trafficLightColorsActual = classifier.classifyTrafficLights(image)

        # Then
        print(trafficLightColors)
        print(trafficLightColorsActual)
        self.assertListEqual(trafficLightColors, trafficLightColorsActual)
