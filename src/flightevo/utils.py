import math
from collections import namedtuple
from flightros.msg import Genome, Node, Connection
from pytorch_neat.activations import tanh_activation
from pytorch_neat.aggregations import sum_aggregation


def quaternion_to_euler(x, y, z, w):
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
