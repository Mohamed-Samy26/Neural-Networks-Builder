from typing import List


class LayerInfo:
    def __init__(
        self,
        has_bias: bool,
        neuron_count: int,
        layer_type: ["input", "output", "hidden"] = "hidden",
        activation: ["sigmoid", "tanh"] = "sigmoid",
    ):
        self.layer_type = layer_type
        self.has_bias = has_bias
        self.neuron_count = neuron_count
        self.activation = activation