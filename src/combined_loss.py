import torch
from torch import nn as nn


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs_inside, outputs_outside, current_lambda=1.0):
        # Calculate loss for the data representing the inside of the object.
        # In the case of points output_inside are the outputs of the NN for the points inside the object.
        # In the case of general boxes output_inside are the upper bounds of the outputs of the NN for general boxes.
        # We can use the log and the sigmoid function here, altough they are not affine, 
        # since the loss is not part of the evaluation and therefore does not influence the correctness of the final output.
        # The minus one makes the losses bigger zero for net outputs >= 0
        inside_loss = calculate_loss_from_input(-1.0*outputs_inside)
        
        # calculate losses for the data representing the outside of the object.
        outside_loss = calculate_loss_from_input(outputs_outside)

        # Consider size difference between inside and outside dataloader
        inside_size = outputs_inside.size(0)
        outside_size = outputs_outside.size(0)
        
        # apply current_lambda to affine_loss and sum
        combined_loss = current_lambda*(outside_size/inside_size)*inside_loss + outside_loss
                
        return combined_loss, inside_loss, outside_loss
    
def calculate_loss_from_input(input):
    logsigmoid = nn.LogSigmoid()
    losses = logsigmoid(input)
    loss = torch.sum(losses)
    loss = -1.0 * loss
    return loss