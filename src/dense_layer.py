import math
import torch
from model_util import is_const

class DenseLayer(torch.nn.Module):
    def __init__(self, size_in, size_out, weights_input=None, bias_input=None):
        """ 
            Creates a dense layer with the given input and output size. 
            If weights_input and bias_input are given, the layer is initialized with these values. 
            Otherwise, the layer is initialized randomly.
        """
        super().__init__()

        weights = torch.Tensor(size_in, size_out)
        self.weights = torch.nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = torch.nn.Parameter(bias)

        if weights_input is not None:
            assert weights_input.shape == weights.shape, "weights_input.shape: " + str(weights_input.shape) + ", weights.shape: " + str(weights.shape)
            self.weights.data = weights_input
        else:
            torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
        if bias_input is not None:
            assert bias_input.shape == bias.shape, "bias_input.shape: " + str(bias_input.shape) + ", bias.shape: " + str(bias.shape)
            self.bias.data = bias_input
        else:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, possibly_affine_input):
        """ Multiplies the weight matrix with the input and adds the bias afterwards to the possibly affine input."""
        assert not torch.isnan(self.weights).any(), "weights contains nan"
        
        device = self.weights.device

        # If the input is not a range but a constant vector, we multiply the weights only with the base.
        if not type(possibly_affine_input) is tuple:
            central_value = possibly_affine_input
            central_value = central_value.to(device)
            central_value = self._apply_layer(central_value)
            return central_value

        if is_const(possibly_affine_input):
            central_value, _, _ = possibly_affine_input
            central_value = central_value.to(device)
            central_value = self._apply_layer(central_value)
            return central_value, None, None

        central_value, partial_deviations, err = possibly_affine_input

        central_value, partial_deviations, err = central_value.to(device), partial_deviations.to(device), err.to(device)

        central_value = self._apply_layer(central_value)
        partial_deviations = torch.vmap(self._apply_weights)(partial_deviations)
        err = self._apply_weights(err, with_abs=True)
        return central_value, partial_deviations, err
    
    def check_for_nan_weights(self):
        return torch.isnan(self.weights).any()

    def add_to_bias(self, value_to_add_to_bias):
        """ Adds add_to_bias to the bias of the layer."""
        self.bias.data += value_to_add_to_bias

    def print_max_min_weights(self):
        """ Prints the maximal and minimal values of the weights of the layer."""
        print("max weights: ", self.weights.max().item(), "min weights: ", self.weights.min().item())

    def interpolate_weights_with_other_layer(self, layer_other, interpolation_factor):
        state_dict = self.state_dict()
        state_dict_other = layer_other.state_dict()
        weigths_self = state_dict["weights"]
        weights_other = state_dict_other["weights"]
        
        result_weights = torch.lerp(weigths_self, weights_other, interpolation_factor)

        bias_other = state_dict_other["bias"]
        bias_self = state_dict["bias"]

        result_bias = torch.lerp(bias_self, bias_other, interpolation_factor)

        self.weights.data = result_weights
        self.bias.data = result_bias


    def _apply_weights(self, input, with_abs=False):
        """ Multiplies the weight matrix with the input. If with_abs=True, the absolute value of the weights is used."""
        myA = torch.abs(self.weights) if with_abs else self.weights
        out = torch.matmul(input, myA)
        return out

    def _apply_layer(self, input):
        """ Multiplies the weight matrix with the input and adds the bias afterwards."""
        out = self._apply_weights(input)
        if self.bias is not None:
            out += self.bias
        return out
    