import rospy
from flightros.msg import State
from flightros.srv import ResetSim, ResetSimRequest, ResetCtl, ResetCtlRequest
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Twist
import os
import neat
from .mlp import Mlp


class Trainer:
    EPOCH_LENGTH = 5.0

    def __init__(self):
        self._reset_sim = None
        self._reset_ctl = None
        self._is_resetting = False
        self._population = None
        self._generator = None
        self._current_reward = 0
        self._current_agent = None
        self._prev_t = 0

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
        # reset controller
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
        if t < self.EPOCH_LENGTH:
            self._is_resetting = False
        if self._is_resetting:
            return
        # update reward
        rospy.loginfo("\ntime: {:.2f}\n".format(msg.time.to_sec()))
        if t > self.EPOCH_LENGTH:
            self._is_resetting = True
            self._reset()
