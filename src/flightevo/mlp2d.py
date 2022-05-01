import torch
import math
# import flightros
from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import sigmoid_activation, identity_activation
from pytorch_neat.aggregations import sum_aggregation
from std_msgs.msg import (
    Float32MultiArray, MultiArrayDimension, MultiArrayLayout)
from flightevo.cppn import Cppn


class Mlp2D:
    def __init__(self, weights, biases,
                 device="cuda", activation=sigmoid_activation):
        self.device = device
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def activate(self, inputs):
        with torch.no_grad():
            x = torch.as_tensor(
                inputs, dtype=torch.float32, device=self.device).unsqueeze(1)
            for w, b in zip(self.weights, self.biases):
                x = self.activation(w.mm(x) + b)
        return x.squeeze(1)

    def to_msg(self):
        return flightros.msg.Mlp(
            weights=[
                Float32MultiArray(
                    data=w.cpu().numpy().reshape(-1),
                    layout=MultiArrayLayout(
                        dim=[MultiArrayDimension(size=s) for s in w.size()]
                    )
                )
                for w in self.weights
            ],
            biases=[
                Float32MultiArray(
                    data=b.cpu().numpy().reshape(-1),
                    layout=MultiArrayLayout(
                        dim=[MultiArrayDimension(size=s) for s in b.size()]
                    )
                )
                for b in self.biases
            ]
        )

    @staticmethod
    def from_msg(msg, device):
        weights = []
        biases = []
        for w, b in zip(msg.weights, msg.biases):
            shape = [d.size for d in w.layout.dim]
            weights.append(torch.tensor(w.data.reshape(shape), device=device))
            shape = [d.size for d in b.layout.dim]
            biases.append(torch.tensor(b.data.reshape(shape), device=device))
        return Mlp2D(weights, biases, device)

    @staticmethod
    def from_cppn(
        genome,
        config,
        coords,
        device="cpu",
    ):
        # nodes = create_cppn(
        #     genome,
        #     config,
        #     ["x_in", "y_in", "x_out", "y_out", ],
        #     # ["weight", "bias"],
        #     ["weight", ],
        #     output_activation=identity_activation,
        #     output_aggregation=sum_aggregation,
        # )
        # cppn  = nodes[0]
        cppn = Cppn(genome)
        coords = [
            torch.tensor(c, dtype=torch.float32, device=device)
            for c in coords
        ]
        w, b = Mlp2D._apply_cppn(cppn, None, coords, device)
        torch.cuda.empty_cache()
        return Mlp2D(w, b, device, identity_activation)

    @staticmethod
    def _get_coord_inputs(in_coords, out_coords):
        n_in = len(in_coords)
        n_out = len(out_coords)

        x_out = out_coords[:, 0].unsqueeze(1).expand(n_out, n_in)
        y_out = out_coords[:, 1].unsqueeze(1).expand(n_out, n_in)
        x_in = in_coords[:, 0].unsqueeze(0).expand(n_out, n_in)
        y_in = in_coords[:, 1].unsqueeze(0).expand(n_out, n_in)

        return (x_out, y_out), (x_in, y_in)

    @staticmethod
    def _apply_cppn(weight_node, bias_node, coords, device):
        weights = []
        biases = []
        with torch.no_grad():
            bias_coords = torch.zeros(
                (1, 2), dtype=torch.float32, device=device)
            for in_coords, out_coords in zip(coords[:-1], coords[1:]):
                (x_out, y_out, ), (x_in, y_in, ) = \
                    Mlp2D._get_coord_inputs(in_coords, out_coords)
                weights.append(Mlp2D._apply_node(
                    weight_node, x_in, y_in, x_out, y_out, ))
                if bias_node is not None:
                    (x_out, y_out), (x_in, y_in, ) = \
                        Mlp2D._get_coord_inputs(bias_coords, out_coords)
                    biases.append(Mlp2D._apply_node(
                        bias_node, x_in, y_in, x_out, y_out,))
                else:
                    biases.append(0)
        return weights, biases

    @staticmethod
    def _apply_node(node, x_in, y_in, x_out, y_out, ):
        s = x_in.size()[0]
        bs = s
        # print(bs)
        while True:
            try:
                return torch.cat([
                    node(
                        x_in=x_in[i:i + bs, :],
                        y_in=y_in[i:i + bs, :],
                        x_out=x_out[i:i + bs, :],
                        y_out=y_out[i:i + bs, :],
                    )
                    for i in range(0, s, bs)
                ])
            except Exception as e:
                if bs <= 1:
                    raise e
                else:
                    bs = math.ceil(bs / 2)
                    # print(bs)
