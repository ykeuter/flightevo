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
from pathlib import Path as Pt
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from ruamel.yaml import YAML
from geometry_msgs.msg import TwistStamped
from avoid_msgs.msg import TaskState
from neat.csv_reporter import CsvReporter
from neat.winner_reporter import WinnerReporter
from neat.function_reporter import FunctionReporter
from itertools import repeat
from threading import Thread

from flightevo.utils import replace_config, reset_stagnation, AgileQuadState
from flightevo.bencher import Bencher
from flightevo.genome import Genome

class BenchTrainer:
    def __init__(
        self, env_cfg, neat_cfg, log_dir, winner_pickle, checkpoint, seed=None
    ):
        self._neat_config = neat.Config(
            Genome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_cfg,
        )
        self._env_cfg = env_cfg
        if winner_pickle:
            with open(winner_pickle, "rb") as f:
                w = pickle.load(f)
            self._generator = repeat(w)
        else:
            Pt(log_dir).mkdir()
            shutil.copy2(env_cfg, log_dir)
            shutil.copy2(neat_cfg, log_dir)
            if checkpoint:
                pop = neat.Checkpointer.restore_checkpoint(checkpoint)
                pop = replace_config(pop, self._neat_config)
                reset_stagnation(pop)
            else:
                pop = neat.Population(self._neat_config)

            pop.add_reporter(neat.Checkpointer(
                1, None, str(Pt(log_dir) / "checkpoint-")
            ))
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(CsvReporter(Pt(log_dir)))
            self._winner_reporter = WinnerReporter(Pt(log_dir))
            pop.add_reporter(self._winner_reporter)
            pop.add_reporter(FunctionReporter(self._level_up))
            self._generator = iter(pop)
            self._population = pop
        self._current_genome = None
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
        self._target = None
        self._timeout = config['environment']['timeout']
        self._bounding_box = np.reshape(np.array(
            config['environment']['world_box'], dtype=np.float32), (3, 2))
        self._cv_bridge = CvBridge()
        self._state = None
        self._start_time = None
        self._lock = Lock()
        self._crashed = False
        self._active = False
        self._rluuid = None
        self._roslaunch = None
        if "env_folder" in config["environment"]:
            self._levels = repeat(config["environment"]["env_folder"])
        else:
            r = config["environment"]["env_range"]
            rng = random.Random(seed)
            self._levels = (
                "environment_{}".format(i)
                for i in rng.sample(range(r[0], r[1]), r[1] - r[0])
            )
        self._current_level = None
    
    def run(self):
        rospy.Subscriber(
            "/hummingbird/ground_truth/odometry", Odometry, self.state_callback,
            queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/depth", Image, self.img_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/hummingbird/collision", Bool, self.obstacle_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            "/hummingbird/goal_point", Path, self.target_callback, queue_size=1)
        rospy.Subscriber(
            "/hummingbird/task_state", TaskState, self.task_state_callback, queue_size=1, tcp_nodelay=True)

        self._cmd_pub = rospy.Publisher(
            "/hummingbird/autopilot/velocity_command", TwistStamped, queue_size=1)

        self._rluuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self._rluuid)
        # self._launch()
        rospy.spin()

    def state_callback(self, msg):
        if not self._active:
            return
        if self._crashed:
            print("crashed")
        if self._start_time is None:
            self._start_time = msg.header.stamp
        if msg.t - self._start_time > self._timeout:
            print("timeout")
        #     return self._reset()
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z])
        if (
            (pos <= self._bounding_box[:, 0]) |
            (pos >= self._bounding_box[:, 1])
        ).any():
            print("oob")
        #     return self._reset()
        self._current_genome.fitness = msg.pose.pose.position.x
        # if msg.pose.position.x >= self._xmax:
        if np.linalg.norm(pos - self._target) <= 1.:
            print("success")
            # return self._reset()
        self._state = AgileQuadState(msg)

    def img_callback(self, msg):
        if not self._active:
            return
        s = self._state
        if s is None:
            return
        t = self._target
        if t is None:
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
        self._crashed = msg.data

    def task_state_callback(self, msg):
        if (msg.Mission_state is not TaskState.PREPARING and
            msg.Mission_state is not TaskState.UNITYSETTING and
            msg.Mission_state is not TaskState.GAZEBOSETTING):
            self._active = True


    def target_callback(self, msg):
        self._target[0] = msg.poses[0].pose.position.x
        self._target[1] = msg.poses[0].pose.position.y
        self._target[2] = msg.poses[0].pose.position.z
        self._dodger.set_target(self._target)

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
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rospy.init_node('dodge_trainer', anonymous=False)
    t = BenchTrainer(args.env, args.neat, args.log,
                     args.winner, args.checkpoint, args.seed)
    t.run()