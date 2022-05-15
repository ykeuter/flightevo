import rospy
import time
import neat
import numpy as np
import argparse
import random
import string
import shutil
import pickle
import roslaunch
from threading import Lock
from pathlib import Path
from dodgeros_msgs.msg import QuadState, Command
from envsim_msgs.msg import ObstacleArray
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Bool
from cv_bridge import CvBridge
from ruamel.yaml import YAML
from geometry_msgs.msg import TwistStamped
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter
from neat.function_reporter import FunctionReporter
from itertools import repeat, cycle
from threading import Thread

from flightevo.utils import replace_config, reset_stagnation
from flightevo.dodger import Dodger, AgileQuadState
from flightevo.genome import Genome


class Selector:
    def __init__(self, log_dir, env_cfg, checkpoint, size):
        self._size = size
        self._log_dir = Path(log_dir)
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
        self._genomes = list(pop.population.values())
        self._generator = iter(self._genomes)
        self._current_genome = None
        with open(Path(env_cfg)) as f:
            config = YAML().load(f)
        self._dodger = Dodger(
            resolution_width=config["dodger"]["resolution_width"],
            resolution_height=config["dodger"]["resolution_height"],
            speed_x=config["dodger"]["speed_x"],
            speed_y=config["dodger"]["speed_y"],
            speed_z=config["dodger"]["speed_z"],
            gamma=config["dodger"]["gamma"],
            bounds=config['environment']['world_box'][2:],
        )
        self._xmax = int(config['environment']['target'])
        self._timeout = config['environment']['timeout']
        self._bounding_box = np.reshape(np.array(
            config['environment']['world_box'], dtype=float), (3, 2))
        self._cv_bridge = CvBridge()
        self._state = None
        self._start_time = None
        self._lock = Lock()
        self._crashed = False
        self._active = False
        self._rluuid = None
        self._roslaunch = None
        self._current_level = config["environment"]["env_folder"]

    def run(self):
        self._rluuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self._rluuid)
        self._launch()
        rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/state", QuadState, self.state_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/unity/depth", Image, self.img_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/groundtruth/obstacles", ObstacleArray,
            self.obstacle_callback, queue_size=1, tcp_nodelay=True)
        self._cmd_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/velocity_command", TwistStamped,
            queue_size=1)
        self._off_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/off", Empty, queue_size=1)
        self._reset_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/reset_sim", Empty, queue_size=1)
        self._enable_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/enable", Bool, queue_size=1)
        self._start_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/start", Empty, queue_size=1)
        self._reset()
        rospy.spin()

    def _save(self):
        s = sorted(self._genomes, reverse=True, key=lambda x: x.fitness)
        for i, g in enumerate(s[:self._size]):
            fn = self._log_dir / "member-{}.pickle".format(i)
            with open(fn, "wb") as f:
                pickle.dump(g, f)

    def _launch(self):
        fn = "/home/ykeuter/flightevo/cfg/simulator.launch"
        args = ["env:={}".format(self._current_level)]
        self._roslaunch = roslaunch.parent.ROSLaunchParent(
            self._rluuid, [(fn, args)])
        self._roslaunch.start()
        time.sleep(10.)

    def _reset(self):
        self._active = False
        Thread(target=self._reset_and_wait).start()

    def _reset_and_wait(self):
        try:
            self._current_genome = next(self._generator)
        except StopIteration:
            self._save()
            rospy.signal_shutdown("All evaluated!")
            raise
        self._current_genome.fitness = 0
        with self._lock:
            self._dodger.load(self._current_genome)
        self._start_time = None
        self._state = None
        # make sure dodger has no more cached actions
        time.sleep(.1)
        self._off_pub.publish()
        # make sure off signal is processed
        time.sleep(.1)
        self._reset_pub.publish()
        self._enable_pub.publish(True)
        # make sure reset is processed
        time.sleep(.1)
        self._start_pub.publish()
        # make sure drone took off
        time.sleep(.5)
        self._crashed = False
        self._active = True

    def state_callback(self, msg):
        if not self._active:
            return
        if self._crashed:
            return self._reset()
        if self._start_time is None:
            self._start_time = msg.t
        if msg.t - self._start_time > self._timeout:
            return self._reset()
        pos = np.array([msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z])
        vel = np.array([msg.velocity.linear.x,
                        msg.velocity.linear.y,
                        msg.velocity.linear.z])
        if (
            (pos <= self._bounding_box[:, 0]) |
            (pos >= self._bounding_box[:, 1])
        ).any():
            return self._reset()
        self._current_genome.fitness = msg.pose.position.x
        self._state = AgileQuadState(t=msg.t, pos=pos, vel=vel)

    def img_callback(self, msg):
        if not self._active:
            return
        # store state in case of reset
        s = self._state
        if s is None:
            return
        cv_image = self._cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')
        with self._lock:
            command = self._dodger.compute_command_vision_based(s, cv_image)
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
        if not self._active:
            return
        o = msg.obstacles[0]
        d = np.linalg.norm(np.array(
            [o.position.x, o.position.y, o.position.z]))
        if d - o.scale < 0:
            self._crashed = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs/f062czs2_eval")
    parser.add_argument("--env", default="logs/f062czs2_eval/env.yaml")
    parser.add_argument(
        "--checkpoint", default="logs/f062czs2_eval/checkpoint-48")
    parser.add_argument("--size", default=20)
    args = parser.parse_args()
    rospy.init_node('selector', anonymous=False)
    e = Selector(args.dir, args.env, args.checkpoint, args.size)
    e.run()
