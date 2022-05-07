import torch


class Cppn:
    def __init__(self, genome):
        self._nodes = list(genome.nodes.values())
        self._vertical_nodes = list(genome.vertical_nodes.values())
        self._center_nodes = list(genome.center_nodes.values())
        self._vertical_bias = genome.vertical_bias
        self._horizontal_bias = genome.horizontal_bias
        self._center_bias = genome.center_bias

    def get_weights(self, x_in, y_in, x_out, y_out):
        output = torch.zeros_like(x_in)
        for n in self._vertical_nodes:
            zx_out = x_out * n.zoom
            zy_out = y_out * n.zoom
            dx = x_in - zx_out
            dy = y_in - zy_out
            d2 = dx * dx + dy * dy
            output += torch.exp(-d2 / n.scale / n.scale) * n.weight
        idx = ((x_out == 0) & (y_out != 0))
        for n in self._vertical_nodes:
            zx_out = x_out * n.zoom
            zy_out = y_out * n.zoom
            dx = x_in - zx_out
            dy = y_in - zy_out
            d2 = dx * dx + dy * dy
            output += torch.exp(-d2 / n.scale / n.scale) * n.weight * idx
        d2 = x_in * x_in + y_in * y_in
        idx = ((x_out == 0) & (y_out == 0))
        for n in self._center_nodes:
            output += torch.exp(-d2 / n.scale / n.scale) * n.weight * idx
        return output

    def get_biases(self, x_in, y_in, x_out, y_out):
        output = torch.zeros_like(x_in)
        idx = ((x_out == 0) & (y_out != 0))
        output[idx] = self._vertical_bias.bias
        idx = ((x_out != 0) & (y_out == 0))
        output[idx] = self._horizontal_bias.bias
        idx = ((x_out == 0) & (y_out == 0))
        output[idx] = self._center_bias.bias
        return output
