import rospy
from flightros.msg import Cmd, State
from sensor_msgs.msg import Image
import math

class Controller:
    def __init__(self):
        self._pub = None
        
    def run(self):
        self._pub = rospy.Publisher('cmd', Cmd, queue_size=1)
        rospy.Subscriber("rgb", Image, self.img_callback)
        rospy.Subscriber("state", State, self.state_callback)
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
                msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
            )
        )
        self._pub.publish(Cmd(rospy.Time(0), [3, 3, 3, 3]))

    def reset_callback(self, msg):
        pass

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw
