#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.TrafficLightClassifier import TrafficLightClassifier
from light_classification.TrafficLightColorClassifier import TrafficLightColorClassifier
from light_classification.TrafficLightColorClassifierFactory import IMG_WIDTH, IMG_HEIGHT, TRAFFIC_LIGHT_COLOR_CLASSIFIER_FILE
from light_classification.TrafficLightExtractor import TRAFFIC_LIGHT_DETECTOR_NAME
from light_classification.TrafficLightColor import TrafficLightColor
from light_classification.TrafficLightDetector import TrafficLightDetector
from light_classification.TrafficLightHavingMinScoreDetector import TrafficLightHavingMinScoreDetector
import tf
import cv2
import yaml
from scipy.spatial import KDTree
from data_collector import trafficLightStateAsString
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.classify_only_every_n_th_camera_image = 4
        self.count = 0
        self.prevTrafficLight = None

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoint_tree = None

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TrafficLightClassifier(
            TrafficLightHavingMinScoreDetector(
                TrafficLightDetector('light_classification/data/' + TRAFFIC_LIGHT_DETECTOR_NAME + '/frozen_inference_graph.pb')),
            TrafficLightColorClassifier('light_classification/' + TRAFFIC_LIGHT_COLOR_CLASSIFIER_FILE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH))
        self.listener = tf.TransformListener()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # TODO: DRY with waypoint_updater/waypoint_updater.py:
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the light state
        # return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.count = (self.count + 1) % self.classify_only_every_n_th_camera_image
        if self.count != 0 and self.prevTrafficLight is not None:
            rospy.loginfo("previous traffic light color: %s", trafficLightStateAsString(self.prevTrafficLight))
            return self.prevTrafficLight

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        trafficLightColors = self.light_classifier.classifyTrafficLights(cv_image)
        trafficLight = self.asTrafficLight(trafficLightColors[0]) if len(trafficLightColors) > 0 else TrafficLight.UNKNOWN
        self.prevTrafficLight = trafficLight
        rospy.loginfo("traffic light color: %s", trafficLightStateAsString(trafficLight))
        return trafficLight


    def asTrafficLight(self, trafficLightColor):
        trafficLightByTrafficLightColor = {
            TrafficLightColor.RED: TrafficLight.RED,
            TrafficLightColor.YELLOW: TrafficLight.YELLOW,
            TrafficLightColor.GREEN:  TrafficLight.GREEN
        }
        return trafficLightByTrafficLightColor[trafficLightColor]


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light, car_wp_idx, line_wp_idx = self.get_closest_light()
        if closest_light:
            state = self.get_light_state(closest_light) if self.isCarCloseToTrafficLight(car_wp_idx, line_wp_idx) else TrafficLight.UNKNOWN
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

    def isCarCloseToTrafficLight(self, car_wp_idx, line_wp_idx):
        return self.distance(self.waypoints.waypoints, car_wp_idx, line_wp_idx) <= 50

    # TODO: DRY with WaypointUpdater.distance()
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # TODO: DRY with DataCollector.get_closest_light()
    # List of positions that correspond to the line to stop in front of for a given intersection
    def get_closest_light(self):
        closest_light = None
        line_wp_idx = None
        car_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        return closest_light, car_wp_idx, line_wp_idx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
