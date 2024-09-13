import torch
from model_util import is_const, calculate_range


class ReluLayer(torch.nn.Module):
    """ 
        Applies the relu function to the input, if it is a single value. 
        If it is an affine input, it applies the linearized relu approximation ( a line of best fit for all values in the interval)
    """

    def __init__(self):
        super().__init__()

    def forward(self, possibly_affine_input):
        relu = torch.nn.ReLU()
        if not type(possibly_affine_input) is tuple:
            central_value = possibly_affine_input
            central_value = relu(central_value)
            return central_value

        central_value, partial_deviations, err = possibly_affine_input

        # If the input is not a range but a constant, we apply relu only to the central_value.
        if is_const(possibly_affine_input):
            relu = torch.nn.ReLU()
            return relu(central_value), partial_deviations, err

        lower, upper = calculate_range(possibly_affine_input)

        # Compute the linearized approximation
        relu = torch.nn.ReLU()
        
        # The lambda value is the slope of the linearized approximation. 
        # If lower and upper are both positive, lambda is 1, since then relu(upper) = upper and relu(lower) = lower if upper >= 0 and lower >= 0.
        # If lower and upper are both negative, lambda is 0, since then relu(upper) = relu(lower) = 0 if upper < 0 and lower < 0.
        # If lower is negative and upper is positive, lambda is the slope of the line between relu(lower)= 0 and relu(upper) = upper, which is a value between 0 and 1.
        # Add 1e-8 to the denominator to avoid division by zero. Although nan_to_num should handle this, it can still cause gradient problems.
        current_lambda = (relu(upper) - relu(lower)) / (upper - lower + 1e-8)
        current_lambda = torch.where(lower>=0, 1., current_lambda)
        current_lambda = torch.where(upper<0, 0., current_lambda)

        # handle numerical badness
        current_lambda = torch.nan_to_num(current_lambda, nan=0.0)
        current_lambda = torch.clamp(current_lambda, 0.0, 1.0)

        # here, lambda/beta are necessarily positive, which makes this simpler        
        # beta is the difference between the means of relu(x) from lower to upper and lambda*x from lower to upper.
        # If lower and upper are both positive, beta is 0, since then relu(lower) = lower and  lambda*lower = lower (since current_lambda = 1).
        # If lower and upper are both negative, beta is 0, since then relu(lower) = 0 and lambda*lower = 0 (since current_lambda = 0).
        # If lower is negative and upper is positive, beta is the difference between the means of relu(x) from lower to upper and lambda*x from lower to upper. Therefore, by adding beta to lambda*x, we get the same mean as relu(x).
        beta = (relu(lower) - current_lambda * lower) / 2
        
        # here, delta is equal to beta. if the intervall is completely positive or completely negative, delta is 0. 
        # In the other cases, the maxima error or the approximation (y = lambda*x + beta) is beta. 
        # The maximal error occurs at the start, x=0 and the end point, so for example error = y_correct - y_approx = lambda*0 + beta - relu(0) = beta - 0 = beta.
        delta = beta
        
        # apply_linear_approx applies:
        # central_value = lambda * central_value + beta
        # for all partial deviations, applies lambda * partial_deviation
        # err = lambda * err
        # also appends abs(delta) to each partial deviation
        central_value, partial_deviations, err = apply_linear_approx(possibly_affine_input, current_lambda, beta, delta)

        return central_value, partial_deviations, err

    def print_max_min_weights(self):
        pass

    def check_for_nan_weights(self):
        return False
    

def apply_linear_approx(affine_input, current_lambda, beta, delta):
    """ Applies the linear approximation lambda * central_value + beta to the affine input and adds the approximation error delta as a new partial deviation."""    
    central_value, partial_deviations, err = affine_input
    
    assert partial_deviations is not None, "partial_deviations must not be None"
    
    # calculate the new central value by applying the linear approximation lambda * central_value + beta to the old central value.
    central_value = current_lambda * central_value + beta

    # calculate the new partial deviations by applying the linear approximation lambda * central_value to the old partial deviations.
    partial_deviations = partial_deviations * current_lambda.unsqueeze(1)
        
    # Add a new partial deviation for the approximation error given by delta. The new partial deviation is abs(delta). Delta should be positive (by definition).
    # Delta (in the case of 3D input) gets spread over three new entries, one for each dimension, e.g. delta = (1,2,3) becomes [[1,0,0],[0,2,0],[0,0,3]].
    delta = torch.abs(delta)
    additional_partial_deviations = torch.diag_embed(delta)
    
    partial_deviations = torch.cat((partial_deviations, additional_partial_deviations), dim=1)
    
    # Calculate the new error by applying the linear approximation lambda * error to the old error.
    err = current_lambda * err

    return central_value, partial_deviations, err
