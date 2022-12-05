import pickle
import argparse
import numpy as np
import rospy

from pathlib import Path
from flightevo.bencher import Bencher
from nav_msgs.msg import Path as PathMsg, Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from ruamel.yaml import YAML
from flightevo.utils import AgileQuadState
from avoid_msgs.msg import TaskState


class BenchRunner:
    def __init__(self, env_cfg, genome_pickle):
        with open(Path(genome_pickle), "rb") as f:
            genome = pickle.load(f)
        with open(Path(env_cfg)) as f:
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
        self._dodger.load(genome)
        self._target = np.zeros(3, dtype=np.float32)
        self._dodger.set_target(self._target)
        self.cv_bridge = CvBridge()
        self._state = None
        self._active = False

    def run(self):
        rospy.Subscriber(
            "/hummingbird/ground_truth/odometry",
            Odometry, self.state_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/depth", Image, self.img_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/hummingbird/goal_point", PathMsg, self.target_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/hummingbird/task_state", TaskState, self.task_callback,
            queue_size=1, tcp_nodelay=True)

        self._cmd_pub = rospy.Publisher(
            "/hummingbird/autopilot/velocity_command", TwistStamped,
            queue_size=1)
        rospy.spin()

    def state_callback(self, msg):
        self._state = AgileQuadState(msg)

    def img_callback(self, msg):
        if not self._active:
            return
        cv_image = self._cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')
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
        self._cmd_pub.publish(msg)

    def task_callback(self, msg):
        if (
            msg.Mission_state == TaskState.PREPARING or
            msg.Mission_state == TaskState.UNITYSETTING or
            msg.Mission_state == TaskState.GAZEBOSETTING
        ):
            self._active = False
        else:
            self._active = True

    def target_callback(self, data):
        self._target[0] = data.poses[0].pose.position.x
        self._target[1] = data.poses[0].pose.position.y
        self._target[2] = data.poses[0].pose.position.z
        self._dodger.set_target(self._target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="env.yaml")
    parser.add_argument("--agent", default="agent.pickle")
    args = parser.parse_args()
    rospy.init_node('bencher', anonymous=True)
    n = BenchRunner(args.env, args.weights)
    n.run()
