from neat.config import DefaultClassConfig, ConfigParameter
from random import random, choice
from neat.genes import BaseGene
from neat.attributes import FloatAttribute
from itertools import count


class ZoomedGaussGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('zoom'),
        FloatAttribute('scale'),
        FloatAttribute('weight'),
    ]


class GaussGene(BaseGene):
    _gene_attributes = [
        FloatAttribute('scale'),
        FloatAttribute('weight'),
    ]


class Config(DefaultClassConfig):
    def __init__(self, param_dict):
        self.node_indexer = count()
        param_list = [
            ConfigParameter('node_add_prob', float),
            ConfigParameter('node_delete_prob', float),
        ]
        param_list += ZoomedGaussGene.get_config_params()
        super().__init__(param_dict, param_list)


class Genome:
    @classmethod
    def parse_config(cls, param_dict):
        return Config(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.write_config(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.center_nodes = {}
        self.vertical_nodes = {}
        self.horizontal_nodes = {}

        # Fitness results.
        self.fitness = None

    def _add_node(self, config):
        node_id = next(config.node_indexer)
        r = random()
        if r < .33:
            n = ZoomedGaussGene(node_id)
            self.vertical_nodes[node_id] = n
        elif r < .66:
            n = ZoomedGaussGene(node_id)
            self.horizontal_nodes[node_id] = n
        else:
            n = GaussGene(node_id)
            self.center_nodes[node_id] = n
        n.init_attributes(config)

    def _delete_node(self):
        r = random()
        if r < .33:
            d = self.vertical_nodes
        elif r < .66:
            d = self.horizontal_nodes
        else:
            d = self.center_nodes
        if d:
            key = choice(list(d.keys()))
            del d[key]

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        self._add_node(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        for name in ("vertical_nodes", "horizontal_nodes", "center_nodes"):
            nodes0 = getattr(self, name)
            nodes1 = getattr(genome1, name)
            nodes2 = getattr(genome2, name)
            keys1 = set(nodes1)
            keys2 = set(nodes2)
            for k in (keys1 & keys2):
                nodes0[k] = nodes1[k].crossover(nodes2[k])
            for k in (keys1 ^ keys2):
                if random() < .5:
                    n = nodes1.get(k, nodes2.get(k))
                    nodes0[k] = n.copy()

    def mutate(self, config):
        """ Mutates this genome. """
        # Mutate node genes.
        for ng in (list(self.vertical_nodes.values()) +
                   list(self.horizontal_nodes.values()) +
                   list(self.center_nodes.values())):
            ng.mutate(config)

        if random() < config.node_add_prob:
            self._add_node(config)
        if random() < config.node_delete_prob:
            self._delete_node()

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This
        distance value is used to compute genome compatibility for speciation.
        """
        nodes_self = set(list(self.horizontal_nodes) +
                         list(self.vertical_nodes) +
                         list(self.center_nodes))
        nodes_other = set(list(other.horizontal_nodes) +
                          list(other.vertical_nodes) +
                          list(other.center_nodes))
        inter_ = len(nodes_self & nodes_other)
        avg_ = (len(nodes_self) + len(nodes_other)) / 2
        if avg_ == 0.:
            return 0.
        return 1 - inter_ / avg_

    def size(self):
        return (len(self.vertical_nodes) +
                len(self.center_nodes) +
                len(self.horizontal_nodes))
