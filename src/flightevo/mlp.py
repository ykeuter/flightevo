import torch
from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import tanh_activation


class Mlp:
    def __init__(self, weights, biases, device):
        self.device = device
        self.weights = weights
        self.biases = biases
        self.activation = tanh_activation

    def activate(self, inputs):
        with torch.no_grad():
            x = torch.tensor(
                inputs, dtype=torch.float32, device=self.device).unsqueeze(1)
            for w, b in zip(self.weights, self.biases):
                x = self.activation(w.mm(x) + b)
        return x.squeeze(1)

    def to_msg(self):
        pass

    @staticmethod
    def from_msg(msg, device):
        pass

    @staticmethod
    def from_cppn(
        genome,
        config,
        coords,
        device="cpu",
    ):
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"],
            ["weight", "bias"],
        )
        coords = [
            torch.tensor(c, dtype=torch.float32, device=device)
            for c in coords
        ]
        w, b = Mlp._apply_cppn(nodes[0], nodes[1], coords, device)
        return Mlp(w, b, device)

    @staticmethod
    def _get_coord_inputs(in_coords, out_coords):
        n_in = len(in_coords)
        n_out = len(out_coords)

        x_out = out_coords[:, 0].unsqueeze(1).expand(n_out, n_in)
        y_out = out_coords[:, 1].unsqueeze(1).expand(n_out, n_in)
        z_out = out_coords[:, 2].unsqueeze(1).expand(n_out, n_in)
        x_in = in_coords[:, 0].unsqueeze(0).expand(n_out, n_in)
        y_in = in_coords[:, 1].unsqueeze(0).expand(n_out, n_in)
        z_in = in_coords[:, 2].unsqueeze(0).expand(n_out, n_in)

        return (x_out, y_out, z_out), (x_in, y_in, z_in)

    @staticmethod
    def _apply_cppn(weight_node, bias_node, coords, device):
        weights = []
        biases = []
        with torch.no_grad():
            bias_coords = torch.zeros(
                (1, 2), dtype=torch.float32, device=device)
            for in_coords, out_coords in zip(coords[:-1], coords[1:]):
                (x_out, y_out, z_out), (x_in, y_in, z_in) = \
                    Mlp._get_coord_inputs(in_coords, out_coords)
                weights.append(weight_node(
                    x_in=x_in, y_in=y_in, z_in=z_in,
                    x_out=x_out, y_out=y_out, z_out=z_out,
                ))
                (x_out, y_out, z_out), (x_in, y_in, z_in) = \
                    Mlp._get_coord_inputs(bias_coords, out_coords)
                biases.append(bias_node(
                    x_in=x_in, y_in=y_in, z_in=z_in,
                    x_out=x_out, y_out=y_out, z_out=z_out,
                ))
        return weights, biases
