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
        # self.assertTrue(os.path.isfile(self.test_dir + '/green/1.jpg'))
        DIR = self.test_dir + '/green'
        self.assertEqual(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]), 3)
