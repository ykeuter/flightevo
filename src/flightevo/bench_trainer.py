import rospy
import neat
import numpy as np
import argparse
import random
import string
import shutil
import pickle
from pathlib import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path as PathMsg
from ruamel.yaml import YAML
from geometry_msgs.msg import TwistStamped
from avoid_msgs.msg import TaskState
from neat.csv_reporter import CsvReporter
from itertools import repeat

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
        self._target = np.zeros(3, dtype=np.float32)
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
            print("crashed")
            return
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z])

        d = np.linalg.norm(pos - self._target)
        self._current_genome.fitness = max(self._current_genome.fitness,
                                           100 - d)
        if d <= 1.:
            print("success")

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
        if not self._crashed and msg.data:
            self._crashed = True

    def task_callback(self, msg):
        if (
            msg.Mission_state == TaskState.PREPARING or
            msg.Mission_state == TaskState.UNITYSETTING or
            msg.Mission_state == TaskState.GAZEBOSETTING
        ):
            self._active = False
        elif not self._active:
            self._current_genome = next(self._generator)
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
    parser.add_argument("--winner", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neat", default="neat.cfg")
    parser.add_argument("--env", default="env.yaml")
    parser.add_argument("--log", default="".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    ))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rospy.init_node('dodge_trainer', anonymous=False)
    t = BenchTrainer(args.env, args.neat, args.log,
                     args.winner, args.checkpoint, args.seed)
    t.run()
