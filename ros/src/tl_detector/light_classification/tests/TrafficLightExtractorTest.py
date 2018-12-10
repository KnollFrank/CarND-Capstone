import glob
import os
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
                get_script_path() + '/../data/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'),
            minScore=0.5)

        # WHEN
        trafficLightExtractor.extractAndSaveTrafficLights(srcDir=get_script_path() + '/images', dstDir=self.test_dir)

        # THEN
        # check red
        self.assertEqual(self.getNumberOfFilesMatching(self.test_dir + '/red/img_0001_red_*.jpg'), 3)

        # check yellow
        self.assertEqual(self.getNumberOfFilesContainedIn(self.test_dir + '/yellow'), 0)

        # check green
        self.assertEqual(self.getNumberOfFilesMatching(self.test_dir + '/green/img_0175_green_*.jpg'), 2)
        self.assertEqual(self.getNumberOfFilesMatching(self.test_dir + '/green/img_0475_green_*.jpg'), 3)

    def getNumberOfFilesMatching(self, pattern):
        return len([file for file in glob.glob(pattern) if os.path.isfile(file)])

    def getNumberOfFilesContainedIn(self, directory):
        return len(self.getFilesContainedIn(directory))

    def getFilesContainedIn(self, directory):
        return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
