from unittest import TestCase
import os.path
from TrafficLightExtractor import TrafficLightExtractor
import shutil, tempfile

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


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
