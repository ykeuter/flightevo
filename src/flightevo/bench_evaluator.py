import rospy
import neat
import numpy as np
import argparse
import pickle

from pathlib import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path as PathMsg
from ruamel.yaml import YAML
from geometry_msgs.msg import TwistStamped
from avoid_msgs.msg import TaskState
from itertools import cycle

from flightevo.utils import AgileQuadState
from flightevo.bencher import Bencher


class BenchEvaluator:
    def __init__(self, genomes, fn, env_cfg):
        self._filename = fn
        self._genomes = genomes
        self._generator = cycle(self._genomes.items())
        self._current_name = None
        self._current_genome = None
        self._env_cfg = env_cfg
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
        self._target = np.zeros(3, dtype=np.float32)
        self._dodger.set_target(self._target)
        self._cv_bridge = CvBridge()
        self._state = None
        self._crashed = False
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
            "/hummingbird/collision", Bool,
            self.obstacle_callback, queue_size=1, tcp_nodelay=True)
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
        if not self._active:
            return
        if self._crashed:
            return
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z])

        d = np.linalg.norm(pos - self._target)
        self._current_genome.fitness = max(self._current_genome.fitness,
                                           100 - d)

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

    def obstacle_callback(self, msg):
        if self._active and not self._crashed and msg.data:
            print("crashed")
            self._crashed = True

    def task_callback(self, msg):
        if (
            msg.Mission_state == TaskState.PREPARING or
            msg.Mission_state == TaskState.UNITYSETTING or
            msg.Mission_state == TaskState.GAZEBOSETTING
        ):
            self._active = False
        elif not self._active:
            if self._current_genome is not None:
                print(self._current_genome.fitness)
                with open(self._filename, "a") as f:
                    f.write("{},{},{}\n".format(self._current_level,
                                                self._current_name,
                                                self._current_genome.fitness))
            self._current_name, self._current_genome = next(self._generator)
            self._current_genome.fitness = 0
            self._dodger.load(self._current_genome)
            self._active = True
            self._crashed = False

    def target_callback(self, msg):
        self._target[0] = msg.poses[0].pose.position.x
        self._target[1] = msg.poses[0].pose.position.y
        self._target[2] = msg.poses[0].pose.position.z
        self._dodger.set_target(self._target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval-stats.csv")
    parser.add_argument("--env", default="env.yaml")
    parser.add_argument(
        # "--checkpoint", default="logs/paper/checkpoint-257-medium")
        "--checkpoint", default="")
    parser.add_argument(
        # "--agent", default="logs/winner/member-4-winner.pickle")
        "--agent", default="")
    args = parser.parse_args()
    if args.checkpoint:
        pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
        cp = Path(args.checkpoint).stem
        parent = Path(args.checkpoint).parent.name
        genomes = {
            "{}-{}-{}".format(parent, cp, i): v
            for i, v in enumerate(pop.population.values())
        }
    if args.agent:
        p = Path(args.agent)
        with open(p, "rb") as f:
            genomes = {p.stem: pickle.load(f)}
    rospy.init_node('evaluator', anonymous=False)
    e = BenchEvaluator(genomes, Path(args.out), args.env)
    e.run()
