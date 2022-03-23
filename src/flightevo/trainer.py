import rospy
import os
import neat
import numpy as np
from flightros.msg import State
from flightros.srv import ResetSim, ResetSimRequest, ResetCtl, ResetCtlRequest
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist

from . import utils


class Trainer:
    MAX_T = 5.0
    POS_COEFF = -0.002
    ORI_COEFF = -0.002
    LIN_VEL_COEFF = -0.0002
    ANG_VEL_COEFF = -0.0002
    # ACT_COEFF = -0.0002
    GOAL_STATE = np.array([.0, .0, 5., .0, .0, .0, .0, .0, .0, .0, .0, .0])

    def __init__(self):
        self._reset_sim = None
        self._reset_ctl = None
        self._is_resetting = False
        self._population = None
        self._generator = None
        self._current_reward = 0
        self._current_agent = None
        self._prev_state_t = 0

    def run(self):
        config_path = os.path.join(os.path.dirname(__file__), "nsssssssat.cfg")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        self._population = neat.Population(config)
        self._generator = iter(self._population)
        rospy.wait_for_service('reset_sim')
        self._reset_sim = rospy.ServiceProxy('reset_sim', ResetSim)
        self._reset_ctl = rospy.ServiceProxy('reset_ctl', ResetCtl)
        rospy.Subscriber("state", State, self.state_callback)
        rospy.spin()

    def _reset(self):
        self._current_agent.fitness = self._current_reward
        self._current_agent = next(self._generator)
        self._current_reward = 0
        try:
            self._reset_ctl(self._get_weights(self._current_agent))
            self._reset_sim(self._get_random_state())
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def _get_random_state(self):
        return ResetSimRequest(
            Pose(Point(0, 0, 7), Quaternion(1, 0, 0, 0)),
            Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        )

    def _get_weights(self, genome):
        return ResetCtlRequest(data=[1, 2, 3])

    def state_callback(self, msg):
        t = msg.time.to_sec()
        if t < self.MAX_T:
            self._is_resetting = False
        if self._is_resetting:
            return
        r = self._get_reward(msg)
        self._current_reward += r * (min(t, self.MAX_T) - self._prev_state_t)
        self._prev_state_t = t
        rospy.loginfo("\ntime: {:.2f}\n".format(msg.time.to_sec()))
        if t > self.MAX_T:
            self._is_resetting = True
            self._reset()

    def _get_reward(self, s):
        r = 0

        v = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z])
        d = v - self.GOAL_STATE[:3]
        r += np.inner(d, d) * self.POS_COEFF

        v = np.array(utils.quaternion_to_euler(
            s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z,
            s.pose.orientation.w
        ))
        d = v - self.GOAL_STATE[3:6]
        r += np.inner(d, d) * self.ORI_COEFF

        v = np.array(
            [s.twist.linear.x, s.potwistse.linear.y, s.twist.linear.z])
        d = v - self.GOAL_STATE[6:9]
        r += np.inner(d, d) * self.LIN_VEL_COEFF

        v = np.array([s.twist.angular.x, s.twist.angular.y, s.twist.angular.z])
        d = v - self.GOAL_STATE[9:]
        r += np.inner(d, d) * self.ANG_VEL_COEFF

        return r
