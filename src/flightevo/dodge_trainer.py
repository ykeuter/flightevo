import rospy
import neat
import numpy as np
import argparse
import random
import string
import shutil
import pickle
from pathlib import Path
from dodgeros_msgs.msg import QuadState, Command
from envsim_msgs.msg import ObstacleArray
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from ruamel.yaml import YAML
from geometry_msgs.msg import TwistStamped
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter

from flightevo.utils import replace_config, reset_stagnation
from flightevo.dodger import Dodger, AgileQuadState


class DodgeTrainer:
    def __init__(self, env_cfg, neat_cfg, log_dir, winner_pickle, checkpoint):
        if winner_pickle:
            with open(winner_pickle, "rb") as f:
                w = pickle.load(f)
            self._generator = self._yield(w)
        else:
            Path(log_dir).mkdir()
            shutil.copy2(env_cfg, log_dir)
            shutil.copy2(neat_cfg, log_dir)
            self._neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                neat_cfg,
            )
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
        self._xmax = int(config['target'])
        self._timeout = config['timeout']
        self._bounding_box = np.reshape(np.array(
            config['bounding_box'], dtype=float), (3, 2))
        self._cv_bridge = CvBridge()
        self._state = None
        self._start_time = None

    def _yield(self, x):
        while True:
            yield x

    def run(self):
        rospy.Subscriber("state", QuadState, self.state_callback,
                         queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("depth", Image, self.img_callback,
                         queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("obstacles", ObstacleArray, self.obstacle_callback,
                         queue_size=1, tcp_nodelay=True)
        self._cmd_pub = rospy.Publisher("velocity_command", TwistStamped,
                                        queue_size=1)
        self._reset_pub = rospy.Publisher("reset_sim", Empty,
                                          queue_size=1)
        self._reset()
        rospy.spin()

    def _reset(self):
        self._current_genome = next(self._generator)
        self._current_genome.fitness = 0
        self._start_time = None
        self._state = None
        self._reset_pub.publish()
        rospy.sleep(.5)

    def state_callback(self, msg):
        if self._current_start_time is None:
            self._current_start_time = msg.t
        if msg.t - self._current_start_time > self._timeout:
            self._reset()
            return
        pos = np.array([msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z])
        if (pos < self._bounding_box[:, 0] | pos > self._bounding_box[:, 1]):
            self._reset()
            return
        self._current_genome.fitness = msg.pose.position.x
        self._state = AgileQuadState(t=msg.t)

    def img_callback(self, msg):
        if self._state is None:
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
        self._cmd_pub(msg)

    def obstacle_callback(self, msg):
        o = msg.obstacles[0]
        d = np.linalg.norm(np.array(
            [o.position.x, o.position.y, o.position.z]))
        if d - o.scale < 0:
            self._reset()


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
    rospy.init_node('trainer', anonymous=True)
    t = DodgeTrainer(args.env, args.neat, args.log,
                     args.winner, args.checkpoint)
    t.run()
