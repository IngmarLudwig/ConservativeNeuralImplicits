import time
import torch
import torch.bin
from util import get_device
from constants import SIGN_OUTSIDE, SIGN_INSIDE, SIGN_UNKNOWN


def test_with_dataloader_points(model, point_dataloader, outside_point_dataloader=True):
    start_time = time.time()

    model.eval()

    device = get_device()
    model = model.to(device)

    points = point_dataloader.positions
    num_points = len(points)
    points = points.to(device)
        
    with torch.no_grad():
        output = model(points)
    del points


    if outside_point_dataloader:
        true_estimations = torch.sum(output > 0.0).item()
    else:
        true_estimations = torch.sum(output <= 0.0).item()
    del output


    percent_outside_right = true_estimations / num_points * 100
    
    test_time = time.time() - start_time

    return percent_outside_right, test_time


def test_with_parallelepipeds(model, affine_dataloader, test_batch_size=32_768):
    start_time = time.time()

    model.eval()
    device = get_device()
    model = model.to(device)

    boxes_sign_outside = 0
    boxes_sign_unknown = 0

    old_batch_size = affine_dataloader.batch_size
    affine_dataloader.batch_size = test_batch_size

    with torch.no_grad():
        for centers, face_vecs in affine_dataloader:
            centers, face_vecs = centers.to(device), face_vecs.to(device)
            output = model.classify_general_box(centers=centers, vecs=face_vecs)
            del centers, face_vecs
            
            # Using bincount is much faster than using comparison and sum
            counts = torch.bincount(output.to(torch.int32))
            del output
            boxes_sign_outside += counts[SIGN_OUTSIDE].item()
            boxes_sign_unknown += counts[SIGN_UNKNOWN].item()

    falsely_classified = boxes_sign_outside + boxes_sign_unknown
    
    affine_dataloader.batch_size = old_batch_size

    test_time = time.time() - start_time
    return falsely_classified, test_time


def sanity_check_with_validation_points(model, validation_points):
    """
        Test if the model classifies all random points inside the shape as inside.
        This is used as a sanity check for the model, which should be able to classify all points inside the shape as inside.
    """
    # Create number_of_points random points in the defined space
    random_points = validation_points.positions
 
    # get the ground truth
    are_inside_object = validation_points.labels == -1.
    
    # get only the points inside the object
    random_points = random_points[are_inside_object]

    # classify the points with the model
    are_inside_model = model.are_points_inside(random_points).cpu() <= 0.
    
    # check if all points are classified as inside, which should be the case
    passed = torch.sum(are_inside_model).item() == len(random_points)
    return passed
        

def sanity_check_with_random_points(model, tet_mesh, number_of_points, lower_point, upper_point):
    # Create number_of_points random points in the defined space
    random_points = torch.rand(number_of_points, 3)* (upper_point - lower_point) + lower_point
 
    # get the ground truth
    are_inside_object = tet_mesh.are_points_inside(random_points).cpu() <= 0.
    
    # get only the points inside the object
    random_points = random_points[are_inside_object]

    # classify the points with the model
    are_inside_model = model.are_points_inside(random_points).cpu() <= 0.
    
    # check if all points are classified as inside, which should be the case
    passed = torch.sum(are_inside_model).item() == len(random_points)
    return passed
        
def test_accuracy_with_random_points(model, tet_mesh, number_of_points):
    """
        Test the accuracy of the model by testing the classification of random points and comparing it to the ground truth (as defined by the tet mesh).
    """
    percent_correct, inside_model_div_inside_shape = _test_with_random_points(model, tet_mesh, number_of_points)
    return percent_correct, inside_model_div_inside_shape


def test_accuracy_with_validation_points(model, validation_points):
    # CAUTION: This function only makes sense if there are no false negatives!

    # Create number_of_points random points in the defined space
    random_points = validation_points.positions
 
    # get the ground truth
    are_inside_object = validation_points.labels <= 0.
    num_points_inside_shape = are_inside_object.sum().item()

    # classify the points with the model
    are_inside_model = model.are_points_inside(random_points).cpu() <= 0.
    num_points_inside_model = are_inside_model.sum().item()

    # This only works because no false negatives are possible
    inside_model_div_inside_shape = num_points_inside_model / num_points_inside_shape

    num_correct = torch.sum(are_inside_object == are_inside_model).item()
    percent_correct = num_correct / len(random_points) * 100

    return percent_correct, inside_model_div_inside_shape


def determine_FPR_FNR(model, validation_points):
    # Create number_of_points random points in the defined space
    positions = validation_points.positions
 
    # get the ground truth
    are_inside_object = validation_points.labels == -1.

    num_points_inside_shape = torch.sum(are_inside_object).item()
    print("num_points_inside_shape", num_points_inside_shape)
    
    # classify the points with the model
    are_inside_model = model.are_points_inside(positions).cpu() <= 0.
    
    # false positives
    false_positives = torch.sum(are_inside_model & ~are_inside_object).item()
    print("false_positives", false_positives)
    false_positive_rate = false_positives / num_points_inside_shape
    print("false_positives / num_points_inside_shape", false_positive_rate)
    
    # false negatives
    false_negatives = torch.sum(~are_inside_model & are_inside_object).item()
    print("false_negatives", false_negatives)
    false_negative_rate = false_negatives / num_points_inside_shape
    print("false_negatives / num_points_inside_shape", false_negative_rate)

    return false_positive_rate, false_negative_rate


def _test_with_random_points(model, tet_mesh, number_of_points, test_only_inside=False):
    # Create number_of_points random points in the defined space
    random_points = torch.rand(number_of_points, 3)* (tet_mesh.upper_point - tet_mesh.lower_point) + tet_mesh.lower_point
 
    # get the ground truth
    are_inside_object = tet_mesh.are_points_inside(random_points).cpu() <= 0.
    num_points_inside_shape = are_inside_object.sum().item()

    if test_only_inside:
        random_points = random_points[are_inside_object]
        are_inside_object = are_inside_object[are_inside_object]

    # classify the points with the model
    are_inside_model = model.are_points_inside(random_points).cpu() <= 0.
    num_points_inside_model = are_inside_model.sum().item()

    inside_model_div_inside_shape = num_points_inside_model / num_points_inside_shape


    num_correct = torch.sum(are_inside_object == are_inside_model).item()
    percent_correct = num_correct / len(random_points) * 100

    return percent_correct, inside_model_div_inside_shape
