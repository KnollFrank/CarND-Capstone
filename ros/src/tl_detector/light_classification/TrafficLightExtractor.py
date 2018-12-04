from utilities import mkdir

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

sys.path.append('/home/frankknoll/udacity/SDCND/models/research')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


class TrafficLightExtractor:

    def __init__(self):
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation from ' + str(StrictVersion(tf.__version__)) + ' to v1.9.* or later!')

        self.PATH_TO_FROZEN_GRAPH = '/home/frankknoll/udacity/SDCND/models/research/object_detection/rfcn_resnet101_coco_2018_01_28//frozen_inference_graph.pb'

    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def extractTrafficLights(self, srcDir, dstDir):
        self.load_model()
        mkdir(dstDir + '/green')
