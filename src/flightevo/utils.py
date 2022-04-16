import math
from collections import namedtuple
# from flightros.msg import Genome, Node, Connection
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


def deactivate_inputs(population, input_keys, output_keys, value):
    cfg = population.config.genome_config
    first_genome = next(iter(population.population.values()))
    for i in input_keys:
        k0 = cfg.get_new_key(first_genome.nodes)
        for g in population.population.values():
            add_selector(g, value, i, k0, cfg)
            for o in output_keys:
                k1 = cfg.get_new_key(first_genome.nodes)
                k2 = cfg.get_new_key(first_genome.nodes)
                deactivate_inputs(g, o, k0, k1, k2, cfg)
    population.species.speciate(
        population.config, population.population, population.generation)


def add_selector(genome, value, input_key, selector_key, cfg):
    n = genome.create_node(cfg, selector_key)
    n.bias = -value
    n.aggregation = sum_aggregation
    n.activation = tri_activation
    genome.nodes[selector_key] = n
    genome.add_connection(cfg, input_key, selector_key, 1.0, True)


def deactivate_input(
    genome, output_key, selector_key, replacement_key, agg_key, cfg
):
    # update original output node
    replacement_node = genome.nodes[output_key]
    replacement_node.key = replacement_key
    genome.nodes[replacement_key] = replacement_node
    # update connections
    conns = [c for (i, o), c in genome.connections if o == output_key]
    for c in conns:
        del genome.connections[c.key]
        c.key = (c.key[0], replacement_key)
        genome.connections[c.key] = c
    # create aggregation node
    n = genome.create_node(cfg, agg_key)
    n.bias = .0
    n.aggregation = prod_aggregation
    n.activation = identity_activation
    genome.nodes[agg_key] = n
    # create new output node
    n = genome.create_node(cfg, output_key)
    n.bias = .0
    n.aggregation = sum_aggregation
    n.activation = identity_activation
    genome.nodes[output_key] = n
    # add connections
    genome.add_connection(cfg, selector_key, agg_key, 1.0, True)
    genome.add_connection(cfg, replacement_key, agg_key, -1.0, True)
    genome.add_connection(cfg, agg_key, output_key, 1.0, True)
    genome.add_connection(cfg, replacement_key, output_key, 1.0, True)
