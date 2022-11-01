# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from typing import no_type_check_decorator
import torch

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs


class HyperLinearNet:
    def __init__(
        self,
        nodes,
        input_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):

        self.nodes = nodes

        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(
            input_coords, dtype=torch.float32, device=device
        )

        self.n_outputs = len(nodes)

        self.weight_threshold = weight_threshold
        self.weight_max = weight_max

        self.activation = activation
        self.cppn_activation = cppn_activation

        self.batch_size = batch_size
        self.device = device
        self.reset()

    def get_init_weights(self, in_coords, nodes):
        dummy_out = torch.tensor(
            [[0, 0]], dtype=torch.float32, device=self.device
        )
        _, (x_in, y_in) = get_coord_inputs(in_coords, dummy_out)

        weights = self.cppn_activation(
            torch.cat([
                node(x_in=x_in, y_in=y_in) for node in nodes
            ])
        )
        clamp_weights_(weights, self.weight_threshold, self.weight_max)

        return weights

    def reset(self):
        with torch.no_grad():
            self.input_to_output = (
                self.get_init_weights(
                    self.input_coords, self.nodes
                )
                .unsqueeze(0)
                .expand(self.batch_size, self.n_outputs, self.n_inputs)
            )

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """
        if inputs.ndim < 2:
            no_batch = True
            inputs = inputs.reshape(1, -1)
        with torch.no_grad():
            inputs = torch.tensor(
                inputs, dtype=torch.float32, device=self.device
            ).unsqueeze(2)

            outputs = self.activation(self.input_to_output.matmul(inputs))

        if no_batch:
            return outputs.squeeze(2).squeeze(0).numpy()

        return outputs.squeeze(2).numpy()

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        n_outputs,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cpu",
    ):

        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in"],
            ["out_" + str(i) for i in range(n_outputs)],
            output_activation=output_activation,
        )

        return HyperLinearNet(
            nodes,
            input_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
            device=device,
        )
