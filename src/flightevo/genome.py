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
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def _add_node(self, config):
        node_id = next(config.node_indexer)
        if random() < .5:
            n = ZoomedGaussGene(node_id)
            self.nodes[node_id] = n
        else:
            n = GaussGene(node_id)
            self.center_nodes[node_id] = n
        n.init_attributes(config)

    def _delete_node(self):
        if random() < .5:
            d = self.nodes
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
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit node genes
        for key, cg1 in parent1.nodes.items():
            cg2 = parent2.nodes.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.nodes[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = cg1.crossover(cg2)
        for key, cg1 in parent1.center_nodes.items():
            cg2 = parent2.center_nodes.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.center_nodes[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.center_nodes[key] = cg1.crossover(cg2)

    def mutate(self, config):
        """ Mutates this genome. """
        # Mutate node genes.
        for cg in self.nodes.values():
            cg.mutate(config)
        for cg in self.center_nodes.values():
            cg.mutate(config)

        if random() < config.node_add_prob:
            self._add_node(config)
        if random() < config.node_delete_prob:
            self._delete_node(config)

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This
        distance value is used to compute genome compatibility for speciation.
        """
        nodes_self = set(list(self.nodes) + list(self.center_nodes))
        nodes_other = set(list(other.nodes) + list(other.center_nodes))
        inter_ = len(nodes_self & nodes_other)
        avg_ = (len(nodes_self) + len(nodes_other)) / 2
        return 1 - inter_ / avg_

    def size(self):
        return len(self.nodes) + len(self.center_nodes)
