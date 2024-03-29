import math
from collections import namedtuple
# from flightros.msg import Genome, Node, Connection
from pytorch_neat.activations import tanh_activation
from pytorch_neat.aggregations import sum_aggregation
import neat
import argparse
from pathlib import Path
import numpy as np
from dodgeros_msgs.msg import QuadState


class AgileCommandMode(object):
    """Defines the command type."""
    # Control individual rotor thrusts.
    SRT = 0
    # Specify collective mass-normalized thrust and bodyrates.
    CTBR = 1
    # Command linear velocities. The linear velocity is expressed in world frame.
    LINVEL = 2

    def __new__(cls, value):
        """Add ability to create CommandMode constants from a value."""
        if value == cls.SRT:
            return cls.SRT
        if value == cls.CTBR:
            return cls.CTBR
        if value == cls.LINVEL:
            return cls.LINVEL

        raise ValueError('No known conversion for `%r` into a command mode' % value)


class AgileCommand:
    def __init__(self, mode):
        self.mode = AgileCommandMode(mode)
        self.t = 0.0

        # SRT functionality
        self.rotor_thrusts = [0.0, 0.0, 0.0, 0.0]

        # CTBR functionality
        self.collective_thrust = 0.0
        self.bodyrates = [0.0, 0.0, 0.0]

        # LINVEL functionality
        self.velocity = [0.0, 0.0, 0.0]
        self.yawrate = 0.0


class AgileQuadState:
    def __init__(self, quad_state):
        self.t = quad_state.header.stamp.to_sec()
        if isinstance(quad_state, QuadState):
            pose = quad_state.pose
            twist = quad_state.velocity
        else:
            pose = quad_state.pose.pose
            twist = quad_state.twist.twist

        self.pos = np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z], dtype=np.float32)
        self.att = np.array([pose.orientation.w,
                             pose.orientation.x,
                             pose.orientation.y,
                             pose.orientation.z], dtype=np.float32)
        self.vel = np.array([twist.linear.x,
                             twist.linear.y,
                             twist.linear.z], dtype=np.float32)
        self.omega = np.array([twist.angular.x,
                               twist.angular.y,
                               twist.angular.z], dtype=np.float32)

    def __repr__(self):
        repr_str = "AgileQuadState:\n" \
                   + " t:     [%.2f]\n" % self.t \
                   + " pos:   [%.2f, %.2f, %.2f]\n" % (self.pos[0], self.pos[1], self.pos[2]) \
                   + " att:   [%.2f, %.2f, %.2f, %.2f]\n" % (self.att[0], self.att[1], self.att[2], self.att[3]) \
                   + " vel:   [%.2f, %.2f, %.2f]\n" % (self.vel[0], self.vel[1], self.vel[2]) \
                   + " omega: [%.2f, %.2f, %.2f]" % (self.omega[0], self.omega[1], self.omega[2])
        return repr_str


def quaternion_to_euler(w, x, y, z):
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


def replace_config(pop, config):
    config.genome_config.node_indexer = pop.config.genome_config.node_indexer
    return neat.Population(
        config, (pop.population, pop.species, pop.generation))


def reset_stagnation(pop):
    for s in pop.species.species.values():
        s.last_improved = pop.generation
        s.fitness_history = []


def genome_to_msg(genome):
    msg = Genome(
        nodes=[Node(key=k, bias=n.bias) for k, n in genome.nodes.items()],
        connections=[
            Connection(
                from_node=i, to_node=o, weight=c.weight, enabled=c.enabled)
            for (i, o), c in genome.connections.items()
        ]
    )
    return msg


def msg_to_genome(msg):
    Node = namedtuple(
        "Node",
        ["bias", "response", "activation", "aggregation"],
        defaults=[1., tanh_activation, sum_aggregation],
    )
    nodes = {
        n.key: Node(n.bias)
        for n in msg.nodes
    }
    Connection = namedtuple("Connection", ["weight", "enabled"])
    connections = {
        (c.from_node, c.to_node): Connection(c.weight, c.enabled)
        for c in msg.connections
    }
    Genome = namedtuple("Genome", ["nodes", "connections"])
    return Genome(nodes, connections)


def deactivate_inputs(population, input_keys, value):
    cfg = population.config.genome_config
    output_keys = cfg.output_keys
    first_genome = next(iter(population.population.values()))
    for i in input_keys:
        k0 = cfg.get_new_node_key(first_genome.nodes)
        for g in population.population.values():
            add_selector(g, value, i, k0, cfg)
            for o in output_keys:
                k1 = cfg.get_new_node_key(first_genome.nodes)
                k2 = cfg.get_new_node_key(first_genome.nodes)
                negate_output(g, o, k0, k1, k2, cfg)
    population.species.speciate(
        population.config, population.population, population.generation)


def add_selector(genome, value, input_key, selector_key, cfg):
    n = genome.create_node(cfg, selector_key)
    n.bias = -value
    n.aggregation = "sum"
    n.activation = "tri"
    genome.nodes[selector_key] = n
    genome.add_connection(cfg, input_key, selector_key, 1.0, True)


def negate_output(
    genome, output_key, selector_key, replacement_key, agg_key, cfg
):
    # update original output node
    replacement_node = genome.nodes[output_key]
    replacement_node.key = replacement_key
    genome.nodes[replacement_key] = replacement_node
    # update connections
    conns = [c for (i, o), c in genome.connections.items() if o == output_key]
    for c in conns:
        del genome.connections[c.key]
        c.key = (c.key[0], replacement_key)
        genome.connections[c.key] = c
    # create aggregation node
    n = genome.create_node(cfg, agg_key)
    n.bias = .0
    n.aggregation = "prod"
    n.activation = "identity"
    genome.nodes[agg_key] = n
    # create new output node
    n = genome.create_node(cfg, output_key)
    n.bias = .0
    n.aggregation = "sum"
    n.activation = "identity"
    genome.nodes[output_key] = n
    # add connections
    genome.add_connection(cfg, selector_key, agg_key, 1.0, True)
    genome.add_connection(cfg, replacement_key, agg_key, -1.0, True)
    genome.add_connection(cfg, agg_key, output_key, 1.0, True)
    genome.add_connection(cfg, replacement_key, output_key, 1.0, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/qsnsqeyy/checkpoint-54")
    parser.add_argument("--inputs", type=int, nargs="+", default=[-3, -6])
    parser.add_argument("--value", type=float, default=-6)
    args = parser.parse_args()

    pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    deactivate_inputs(pop, args.inputs, args.value)
    prefix = str(Path(args.checkpoint).parent / "transformed-neat-checkpoint-")
    cp = neat.Checkpointer(filename_prefix=prefix)
    cp.save_checkpoint(pop.config, pop.population, pop.species, pop.generation)
