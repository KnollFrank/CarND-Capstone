from unittest import TestCase
import os.path
from TrafficLightExtractor import TrafficLightExtractor
import shutil, tempfile


class TrafficLightExtractorTest(TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_traffic_lights(self):
        trafficLightExtractor = TrafficLightExtractor()
        trafficLightExtractor.extractTrafficLights(srcDir='../images', dstDir=self.test_dir)
        self.assertEqual(self.getNumberOfFilesContainedIn(self.test_dir + '/green'), 5)
        self.assertEqual(self.getNumberOfFilesContainedIn(self.test_dir + '/red'), 3)

    def getNumberOfFilesContainedIn(self, directory):
        return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
