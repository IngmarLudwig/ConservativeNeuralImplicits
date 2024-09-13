import torch.nn as nn
from model_util import is_const


class SqueezeLayer(nn.Module):
    """ Squeezes (removes all dimensions of size 1) a possibly_affine_input if applied. """

    def __init__(self):
        super().__init__()
        
    def forward(self, possibly_affine_input):
        if not type(possibly_affine_input) is tuple:
            central_value = possibly_affine_input
            central_value = central_value.squeeze()
            return central_value
        central_value, partial_deviations, err = possibly_affine_input
        central_value = central_value.squeeze()
        if is_const(possibly_affine_input):
            return central_value, None, None
        partial_deviations = partial_deviations.squeeze()
        err = err.squeeze()
        return central_value, partial_deviations, err
    
    def print_max_min_weights(self):
        pass

    def check_for_nan_weights(self):
        False