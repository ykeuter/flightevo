from re import L
import rospy
import neat
import numpy as np
from flightros.msg import State
from flightros.srv import ResetSim, ResetSimRequest, ResetCtl
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist

from . import utils
from .mlp import Mlp


class Trainer:
    MAX_T = 5.0
    GOAL_HEIGHT = 5.

    POS_COEFF = -0.002
    ORI_COEFF = -0.002
    LIN_VEL_COEFF = -0.0002
    ANG_VEL_COEFF = -0.0002
    # ACT_COEFF = -0.0002

    def __init__(self):
        self._reset_sim = None
        self._reset_ctl = None
        self._is_resetting = False
        self._population = None
        self._generator = None
        self._current_agent = None
        self._prev_state_t = 0
        self._device = "cpu"
        self._coords = self._get_coords()

    def run(self, cfg):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            cfg,
        )
        self._population = neat.Population(config)
        self._generator = iter(self._population)
        # rospy.wait_for_service('reset_sim')
        # self._reset_sim = rospy.ServiceProxy('reset_sim', ResetSim)
        rospy.wait_for_service('reset_ctl')
        self._reset_ctl = rospy.ServiceProxy('reset_ctl', ResetCtl)
        rospy.Subscriber("state", State, self.state_callback)

        self._current_agent = next(self._generator)
        tic = rospy.get_time()
        for _ in range(100):
            self._reset_ctl(self._get_weights(self._current_agent))
        toc = rospy.get_time()
        rospy.loginfo("secs elapsed: {}".format(toc - tic))

        # rospy.spin()

    def _reset(self):
        self._current_agent = next(self._generator)
        self._current_agent.fitness = 0
        try:
            self._reset_ctl(self._get_weights(self._current_agent))
            self._reset_sim(self._get_random_state())
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def _get_random_state(self):
        rng = np.random.default_rng()
        p = rng.uniform(-1, 1, 3)
        p[2] += self.GOAL_HEIGHT
        q = rng.uniform(-1, 1, 4)
        q /= np.linalg.norm(q)
        v = rng.uniform(-1, 1, 3)
        a = rng.uniform(-1, 1, 3)

        return ResetSimRequest(
            Pose(Point(*p), Quaternion(*q)), Twist(Vector3(*v), Vector3(*a))
        )

    def _get_weights(self, genome):
        net = Mlp.from_cppn(
            genome, self._population.config, self._coords, self._device)
        return net.to_msg()

    def state_callback(self, msg):
        t = msg.time.to_sec()
        if t < self.MAX_T:
            self._is_resetting = False
        if self._is_resetting:
            return
        if self._current_agent:
            r = self._get_reward(msg)
            self._current_agent.fitness += (
                r * (min(t, self.MAX_T) - self._prev_state_t))
            self._prev_state_t = t
        rospy.loginfo("\ntime: {:.2f}\n".format(msg.time.to_sec()))
        if t > self.MAX_T:
            self._is_resetting = True
            self._reset()

    def _get_reward(self, s):
        r = 0

        d = np.array([
            s.pose.position.x, s.pose.position.y,
            s.pose.position.z - self.GOAL_HEIGHT
        ])
        r += np.inner(d, d) * self.POS_COEFF

        d = np.array(utils.quaternion_to_euler(
            s.pose.orientation.w,
            s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z
        ))
        r += np.inner(d, d) * self.ORI_COEFF

        d = np.array(
            [s.twist.linear.x, s.potwistse.linear.y, s.twist.linear.z])
        r += np.inner(d, d) * self.LIN_VEL_COEFF

        d = np.array([s.twist.angular.x, s.twist.angular.y, s.twist.angular.z])
        r += np.inner(d, d) * self.ANG_VEL_COEFF

        return r

    def _get_coords(self):
        # inputs
        cross = [
            (1, 0, 0),  # x positive
            (-1, 0, 0),  # x negative
            (0, 1, 0),  # y positive
            (0, -1, 0),  # y negative
            (0, 0, 0),  # z
        ]
        positions = [(x - 2, y + 2, z) for x, y, z in cross]
        euler = [(x - 2, y - 2, z) for x, y, z in cross]
        linear = [(x + 2, y + 2, z) for x, y, z in cross]
        angular = [(x + 2, y - 2, z) for x, y, z in cross]
        inputs = positions + euler + linear + angular

        # layers
        layer1 = [
            (x - 3.5, y - 3.5, z)
            for x in range(8)
            for y in range(8)
            for z in [1, 2]
        ]
        layer2 = [(x, y, z + 2) for x, y, z in layer1]

        # outputs (not sure about motor config)
        outputs = [(2, 2, 5), (-2, 2, 5), (-2, -2, 5), (2, -2, 5), ]

        return [inputs, layer1, layer2, outputs]
