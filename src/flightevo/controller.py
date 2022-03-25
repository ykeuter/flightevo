import rospy
import numpy as np
from flightros.msg import Cmd, State
from flightros.srv import ResetCtl
from sensor_msgs.msg import Image

from .mlp import Mlp
from . import utils


class Controller:
    def __init__(self):
        self._pub = None
        self.net = None

    def run(self):
        self._pub = rospy.Publisher('cmd', Cmd, queue_size=1)
        rospy.Subscriber("rgb", Image, self.img_callback)
        rospy.Subscriber("state", State, self.state_callback)
        rospy.Service("reset_ctl", ResetCtl, self.reset_callback)
        rospy.spin()

    def img_callback(self, msg):
        rospy.loginfo(
            (
                "height ({}): {}\n" +
                "width ({}): {}\n" +
                "encoding ({}): {}\n" +
                "is_bigendian ({}): {}\n" +
                "step ({}): {}\n" +
                "data ({}): {}\n"
            ).format(
                type(msg.height), msg.height,
                type(msg.width), msg.width,
                type(msg.encoding), msg.encoding,
                type(msg.is_bigendian), msg.is_bigendian,
                type(msg.step), msg.step,
                type(msg.data), max(msg.data)
            )
        )
        self._pub.publish(Cmd(0, [3, 3, 3, 3]))

    def state_callback(self, msg):
        rospy.loginfo(
            (
                "\n" +
                "position: {:.2f}, {:.2f}, {:.2f}\n" +
                "velocity: {:.2f}, {:.2f}, {:.2f}\n" +
                "orientation: {:.2f}, {:.2f}, {:.2f}, {:.2f}\n" +
                "angular_velocity: {:.2f}, {:.2f}, {:.2f}\n"
            ).format(
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                msg.pose.orientation.w, msg.pose.orientation.x,
                msg.pose.orientation.y, msg.pose.orientation.z,
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
            )
        )
        if self.net:
            euler_x, euler_y, euler_z = utils.quaternion_to_euler(
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w
            )
            x = np.array([
                msg.pose.position.x, -msg.pose.position.x,
                msg.pose.position.y, -msg.pose.position.y,
                msg.pose.position.z,
                euler_x, -euler_x, euler_y, -euler_y, euler_z,
                msg.twist.linear.x, -msg.twist.linear.x,
                msg.twist.linear.y, -msg.twist.linear.y,
                msg.twist.linear.z,
                msg.twist.angular.x, -msg.twist.angular.x,
                msg.twist.angular.y, -msg.twist.angular.y,
                msg.twist.angular.z,
            ])
            self._pub.publish(Cmd(msg.time, self.net.activate(x)))

    def reset_callback(self, msg):
        self.net = Mlp.from_msg(msg.mlp)
