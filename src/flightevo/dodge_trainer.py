import rospy
import time
import neat
import numpy as np
import argparse
import random
import string
import shutil
import pickle
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

from flightevo.utils import replace_config, reset_stagnation, repeat_yield
from flightevo.dodger import Dodger, AgileQuadState


class DodgeTrainer:
    def __init__(self, env_cfg, neat_cfg, log_dir, winner_pickle, checkpoint):
        self._neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_cfg,
        )
        if winner_pickle:
            with open(winner_pickle, "rb") as f:
                w = pickle.load(f)
            self._generator = repeat_yield(w)
        else:
            Path(log_dir).mkdir()
            shutil.copy2(env_cfg, log_dir)
            shutil.copy2(neat_cfg, log_dir)
            if checkpoint:
                pop = neat.Checkpointer.restore_checkpoint(checkpoint)
                pop = replace_config(pop, self._neat_config)
                reset_stagnation(pop)
            else:
                pop = neat.Population(self._neat_config)

            pop.add_reporter(neat.Checkpointer(
                1, None, str(Path(log_dir) / "checkpoint-")
            ))
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(CsvReporter(Path(log_dir)))
            pop.add_reporter(WinnerReporter(Path(log_dir)))
            self._generator = iter(pop)
        self._current_genome = None
        with open(env_cfg) as f:
            config = YAML().load(f)
        self._dodger = Dodger(config["inputs"]["resolution_width"],
                              config["inputs"]["resolution_height"])
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

    def run(self):
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

    def _reset(self):
        self._current_genome = next(self._generator)
        self._current_genome.fitness = 0
        with self._lock:
            self._dodger.load(self._current_genome, self._neat_config)
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
        time.sleep(.1)
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
        if (
            (pos < self._bounding_box[:, 0]) | (pos > self._bounding_box[:, 1])
        ).any():
            return self._reset()
        self._current_genome.fitness = msg.pose.position.x
        self._state = AgileQuadState(t=msg.t)

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
    parser.add_argument("--winner", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neat", default="cfg/neat.cfg")
    parser.add_argument("--env", default="cfg/env.yaml")
    parser.add_argument("--log", default="logs/" + "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    ))
    args = parser.parse_args()
    rospy.init_node('dodge_trainer', anonymous=False)
    t = DodgeTrainer(args.env, args.neat, args.log,
                     args.winner, args.checkpoint)
    t.run()