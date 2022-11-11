from tkinter.messagebox import NO
import torch
import pickle
import argparse
import sys
import numpy as np
import rospy
from pathlib import Path as Pt
from flightevo.bencher import Bencher
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from ruamel.yaml import YAML
from flightevo.utils import AgileQuadState
from avoid_msgs.msg import TaskState


class DodgeTestNode:
    def __init__(self, env_cfg):
        self._state = None
        self.cv_bridge = CvBridge()
        self.depth_sub_ = rospy.Subscriber(
            "/depth", Image, self.depthCallback,
            queue_size=1, tcp_nodelay=True)
        self.odom_sub_ = rospy.Subscriber(
            "/hummingbird/ground_truth/odometry", Odometry, self.stateCallback,
            queue_size=1, tcp_nodelay=True)
        self.cmd_pub_ = rospy.Publisher(
            "/hummingbird/autopilot/velocity_command", TwistStamped,
            queue_size=1)
        self.target_sub_ = rospy.Subscriber(
            "/hummingbird/goal_point", Path, self.target_callback,
            queue_size=1)
        self._target = None
        with open("best_eval.pickle", "rb") as f:
            WEIGHTS = pickle.load(f)
        with open(env_cfg) as f:
            config = YAML().load(f)

        self._dodger = Bencher(
            resolution_width=config["dodger"]["resolution_width"],
            resolution_height=config["dodger"]["resolution_height"],
            speed_x=config["dodger"]["speed_x"],
            speed_y=config["dodger"]["speed_y"],
            speed_z=config["dodger"]["speed_z"],
            gamma=config["dodger"]["gamma"],
            acc=config["dodger"]["acc"],
            margin=config["dodger"]["margin"],
            bounds=config['environment']['world_box'][2:],
            creep_z=config["dodger"]["creep_z"],
            creep_yaw=config["dodger"]["creep_yaw"],
        )
        self._dodger.load(WEIGHTS)

    def target_callback(self, data):
        self._target = np.zeros(3, dtype=np.float32)
        self._target[0] = data.poses[0].pose.position.x
        self._target[1] = data.poses[0].pose.position.y
        self._target[2] = data.poses[0].pose.position.z
        self._dodger.set_target(self._target)

    def depthCallback(self, data):
        if self._state is None:
            return
        cv_image = self.cv_bridge.imgmsg_to_cv2(
            data, desired_encoding='passthrough')
        command = self._dodger.compute_command_vision_based(
            self._state, cv_image)
        msg = TwistStamped()
        msg.header.stamp = rospy.Time(command.t)
        msg.twist.linear.x = command.velocity[0]
        msg.twist.linear.y = command.velocity[1]
        msg.twist.linear.z = command.velocity[2]
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = command.yawrate
        self.cmd_pub_.publish(msg)

    def stateCallback(self, data):
        self._state = AgileQuadState(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cfg/env.yaml")
    args = parser.parse_args()
    rospy.init_node('DodgeTestNode', anonymous=True)
    test = DodgeTestNode(args.env)
    rospy.spin()
