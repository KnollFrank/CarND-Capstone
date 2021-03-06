#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math

from light_classification.utilities import mkdir

STATE_COUNT_THRESHOLD = 3

def trafficLightStateAsString(state):
    strByTrafficLight = {
        TrafficLight.RED: "red",
        TrafficLight.YELLOW: "yellow",
        TrafficLight.GREEN: "green",
        TrafficLight.UNKNOWN: "unknown"
    }
    return strByTrafficLight[state]

class DataCollector(object):

    def __init__(self):
        rospy.init_node('collect_data')

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoint_tree = None
        self.img_counter = 0

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        active = rospy.get_param('~active', 'no')
        if active == 'no':
            return

        self.image_dir = "light_classification/data/simulator_images"
        mkdir(self.image_dir)
        mkdir(self.image_dir + "/red")
        mkdir(self.image_dir + "/green")
        mkdir(self.image_dir + "/yellow")
        mkdir(self.image_dir + "/unknown")

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
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
        self.has_image = True
        self.camera_image = msg
        self.process_traffic_lights()

    def get_closest_waypoint(self, x, y):
        return self.waypoint_tree.query([x, y], 1)[1]

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

    def saveCameraImage(self, fileNameSuffix):
        self.img_counter += 1
        fileName = "{0}/{2}/img_{1:04d}_{2}.jpg".format(self.image_dir, self.img_counter, fileNameSuffix)
        cv2.imwrite(fileName, self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8"))

    def process_traffic_lights(self):
        closest_light, car_wp_idx, line_wp_idx = self.get_closest_light()
        if closest_light:
            dist = self.distance(self.waypoints.waypoints, car_wp_idx, line_wp_idx)
            rospy.loginfo("dist(car, light): %d", dist)
            # rospy.loginfo("state: %d", closest_light.state)
            if dist <= 50:
                self.saveCameraImage(trafficLightStateAsString(closest_light.state))

    # TODO: DRY with WaypointUpdater.distance()
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        DataCollector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
