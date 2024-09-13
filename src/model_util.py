import torch

# Affine data is represented as a tuple input=(base,aff,err). Base is a normal shape (d,) primal vector value, affine is a (v,d)
# array of affine coefficients (may be v=0), err is a centered interval error shape (d,), which must be nonnegative.
# For constant values, aff == err == None. If is_const(input) == False, then it is guaranteed that aff and err are non-None.
def is_const(input):
    base, aff, err = input
    if err is not None: return False
    return aff is None or aff.shape[0] == 0


def calculate_range(affine_input):
    """ Calculates the lower and upper bounds of the affine input by adding up the maximal values of the partial_deviations and the errors and adding and subtracting them from the central_value."""
    central_value, _, _ = affine_input
    max_deviation = calculate_maximal_deviation(affine_input)
    return central_value-max_deviation, central_value+max_deviation


def calculate_maximal_deviation(affine_input):
    """ Calculates the maximal deviation (width of the approximation) of the affine input by adding up the maximal values of the partial_deviations and the errors."""
    if is_const(affine_input):
        return torch.tensor([0.])
    
    central_value, partial_deviations, err = affine_input

    if len(central_value.shape) == 0:
        # A single value, not a batch is processed.
        axis = 0
    elif len(central_value.shape) == 1:
        # A batch after squeeze_last is processed.
        axis = 1
    elif len(central_value.shape) == 2 and len(partial_deviations.shape) == 1:
        # A batch is processed and the affine coefficients (v,d) has v =0
        axis = 0
    elif len(central_value.shape) == 2 and (len(partial_deviations.shape) == 3 or len(partial_deviations.shape) == 2):
        # A batch is processed and the affine coefficients (v,d) has v !=0.
        axis = 1
    else:
        raise ValueError("Invalid affine shape. base has shape {}, aff has shape {}".format(central_value.shape, partial_deviations.shape))

    rad = torch.sum(torch.abs(partial_deviations), axis=axis)

    if err is not None:
        rad = torch.add(rad, err)
    return rad
        