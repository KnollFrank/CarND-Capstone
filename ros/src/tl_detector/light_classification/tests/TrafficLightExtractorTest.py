import os
import os.path
import shutil
import tempfile
from unittest import TestCase

from TrafficLightDetector import TrafficLightDetector
from TrafficLightExtractor import TrafficLightExtractor


def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

class TrafficLightExtractorTest(TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_traffic_lights(self):
        # GIVEN
        trafficLightExtractor = TrafficLightExtractor(
            TrafficLightDetector(
                get_script_path() + '/../data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'))

        # WHEN
        trafficLightExtractor.extractAndSaveTrafficLights(srcDir=get_script_path() + '/images', dstDir=self.test_dir)

        # THEN
        # check red
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_2.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_3.jpg')))

        # check yellow
        self.assertEqual(self.getNumberOfFilesContainedIn(self.test_dir + '/yellow'), 0)

        # check green
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0175_green_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0175_green_2.jpg')))

        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_2.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_3.jpg')))

    def getNumberOfFilesContainedIn(self, directory):
        return len(self.getFilesContainedIn(directory))

    def getFilesContainedIn(self, directory):
        return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
