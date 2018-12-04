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
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_extract_traffic_lights(self):
        trafficLightExtractor = TrafficLightExtractor()
        trafficLightExtractor.extractTrafficLights(srcDir='../images', dstDir=self.test_dir)
        self.assertTrue(os.path.isfile('../images/green/img_0175_green.jpg'))
