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
        # GIVEN
        trafficLightExtractor = TrafficLightExtractor()

        # WHEN
        trafficLightExtractor.extractTrafficLights(srcDir='../images', dstDir=self.test_dir)

        # THEN
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0175_green_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0175_green_2.jpg')))

        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_2.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/green', 'img_0475_green_3.jpg')))

        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_1.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_2.jpg')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir + '/red', 'img_0001_red_3.jpg')))
