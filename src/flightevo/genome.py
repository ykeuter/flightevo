from operator import imod
from neat import DefaultGenome
from neat.graphs import creates_cycle
from random import choice


class Genome(DefaultGenome):
    def mutate_add_node(self, config):
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        key = (in_node, out_node)

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        cg1 = self.create_connection(config, in_node, new_node_id)
        self.connections[cg1.key] = cg1
        cg2 = self.create_connection(config, new_node_id, out_node)
        self.connections[cg2.key] = cg2
