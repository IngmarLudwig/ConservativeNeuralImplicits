import torch
from util import ensure_tensor_and_batched, ensure_tensor_and_batched_for_all, get_device, get_parallelepiped_corners, assert_same_length

def are_points_inside_one_of_the_boxes(points,  lower_points, upper_points, batch_size_spec=None, epsilon=0):
    """ 
        Checks if the given points are inside one of the given boxes that are given by the lower and upper points of the boxes.
        If a point is in ONE of the given boxes (including boarders), -1 is returned for it, 1. else.
    """
    # check and prepare input
    points = ensure_tensor_and_batched(points)
    lower_points, upper_points = ensure_tensor_and_batched_for_all([lower_points, upper_points])
    
    # get device
    device = get_device()
    
    # move points to device
    points = points.to(device)
    
    # Start with the assumption that all points are outside of all parallelepipeds.
    is_inside = torch.full((len(points),), 1., dtype=torch.float32, device=device)
    
    # Iterate over all boxes and check for each points, whether it is outside all of the given boxes.
    for lp, up in zip(lower_points, upper_points):
        lp = lp.to(device)
        up = up.to(device)
        are_points_inside_this_box = (torch.all(points >= lp, dim=1) & torch.all(points <= up, dim=1))
        is_inside = torch.where(are_points_inside_this_box, -1., is_inside)
    
    return is_inside.cpu()


def are_points_inside_one_of_the_tetrahedra(points,  p_1s, p_2s, p_3s, p_4s, batch_size_spec=None, epsilon=0):
    """ 
        Checks if the given points are inside one of the given parallelepipeds that are spanned by the given bases and vectors v_1, v_2 and v_3.
        If a point is in ONE of the given parallelepipeds (including boarders), -1 is returned for it, 1. else.
    """
    # check and prepare input
    points = ensure_tensor_and_batched(points)
    p_1s, p_2s, p_3s, p_4s = ensure_tensor_and_batched_for_all([p_1s, p_2s, p_3s, p_4s])
    
    # get device
    device = get_device()
    
    # Define the function that will be called in the batched calculation. 
    
    # Start with the assumption that all points are outside of all parallelepipeds.
    is_inside = torch.full((len(points),), 1., dtype=torch.float32, device=device)
    def _check_inside_tetrahedra_batch_local(points, batched_input_tuple, nonbatched_input_tuple):
        # extract the input
        p_1s, p_2s, p_3s, p_4s = batched_input_tuple
        epsilon = nonbatched_input_tuple[0]
        
        # check for each points, whether it is outside all of the given parallelepipeds 
        are_points_outside_everywhere = _check_inside_tetrahedra_batch(points, p_1s, p_2s, p_3s, p_4s, epsilon)
        
        # Using this information, we set all points that are outside everywhere to outside(1) and the others to inside(-1).
        nonlocal is_inside
        is_inside = torch.where(are_points_outside_everywhere == False, -1., is_inside)

    # define input tuples
    batched_input_tuple = (p_1s, p_2s, p_3s, p_4s)
    nonbatched_input_tuple = [epsilon]
    
    # calculate the result
    _calculate_with_batches(_check_inside_tetrahedra_batch_local, points, batched_input_tuple, nonbatched_input_tuple, device, batch_size_spec)
    return is_inside.cpu()


def are_points_inside_one_of_the_parallelepipeds(points, bases, v_1s, v_2s, v_3s, batch_size_spec=None, epsilon=0):
    """ 
        Checks if the given points are inside one of the given parallelepipeds that are spanned by the given bases and vectors v_1, v_2 and v_3.
        If a point is in ONE of the given parallelepipeds (including boarders), -1 is returned for it, 1. else.
    """
    # check and prepare input
    points = ensure_tensor_and_batched(points)
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched_for_all([bases, v_1s, v_2s, v_3s])
    
    # get device
    device = get_device()

    # Define the function that will be called in the batched calculation. 
    
    # Start with the assumption that all points are outside of all parallelepipeds.
    is_inside = torch.full((len(points),), 1., dtype=torch.float32, device=device)
    def _check_inside_parallelepipeds_batch_local(points, batched_input_tuple, nonbatched_input_tuple):
        # extract the input
        bases, v_1s, v_2s, v_3s = batched_input_tuple
        epsilon = nonbatched_input_tuple[0]
        
        # check for each points, whether it is outside all of the given parallelepipeds
        are_points_outside_everywhere = _check_inside_parallelepipeds_batch(points, bases, v_1s, v_2s, v_3s, epsilon)

        # Using this information, we set all points that are outside everywhere to outside(1) and the others to inside(-1).
        nonlocal is_inside
        is_inside = torch.where(are_points_outside_everywhere == False, -1., is_inside)
    
   
    # define input tuples
    batched_input_tuple = (bases, v_1s, v_2s, v_3s)
    nonbatched_input_tuple = [epsilon]
    
    # calculate the result
    _calculate_with_batches(_check_inside_parallelepipeds_batch_local, points, batched_input_tuple, nonbatched_input_tuple, device, batch_size_spec)
    return is_inside.cpu()


def check_which_points_are_inside_which_parallelepiped(points, bases, v_1s, v_2s, v_3s, epsilon=0, batch_size_spec=None, device=None):
    # check and prepare input
    points = ensure_tensor_and_batched(points)
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched_for_all([bases, v_1s, v_2s, v_3s])

    # get device, if not given
    if device is None:
        device = get_device()

    # Define the function that will be called in the batched calculation. 
    inside_all_of_the_planes_per_pe = []
    def _check_which_points_are_inside_which_parallelepiped_batch_local(points, batched_input_tuple, nonbatched_input_tuple):
        bases, v_1s, v_2s, v_3s = batched_input_tuple
        epsilon = nonbatched_input_tuple[0]
        result = _check_which_points_are_inside_which_parallelepiped_batch(points, bases, v_1s, v_2s, v_3s, epsilon)
        result = ensure_tensor_and_batched(result)
        nonlocal inside_all_of_the_planes_per_pe
        inside_all_of_the_planes_per_pe.append(result)
        
    # define input tuples
    batched_input_tuple = (bases, v_1s, v_2s, v_3s)
    nonbatched_input_tuple = [epsilon]
    
    # calculate the result
    _calculate_with_batches(_check_which_points_are_inside_which_parallelepiped_batch_local, points, batched_input_tuple, nonbatched_input_tuple, device, batch_size_spec)
    
    # concatenate the resulting list of tensors to one tensor
    inside_all_of_the_planes_per_pe = torch.cat(inside_all_of_the_planes_per_pe, dim=0)
    return inside_all_of_the_planes_per_pe.cpu()


def _calculate_with_batches(function, points, batched_input_tuple, nonbatched_input_tuple, device, batch_size_spec=None):
    # prepare input
    points = points.to(device)
    assert_same_length(batched_input_tuple)
    
    # Batch the input to avoid memory errors
    number_of_tets = len(batched_input_tuple[0])
    batch_size = number_of_tets
    if batch_size_spec is not None:
        batch_size = batch_size_spec
    elif device == "cuda" or device == "mps":
        # On GPU, we need to use smaller batch sizes to avoid memory errors. Depending on the system, smaller batch sizes might be necessary.
        batch_size = 256
    
    number_of_elements = len(batched_input_tuple[0])
    num_full_batches = number_of_elements // batch_size
    for i in range(num_full_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batched = _get_batches(batched_input_tuple, start=start, end=end, device=device)
        function(points, batched, nonbatched_input_tuple)

    # Handle the last batch.
    if number_of_elements % batch_size != 0:
        start = num_full_batches * batch_size
        end = number_of_elements
        batched = _get_batches(batched_input_tuple, start=start, end=end, device=device)
        function(points, batched, nonbatched_input_tuple)


def _get_batches(batched_input_tuple, start, end, device):
    batches = []
    for element in batched_input_tuple:
        batches.append(element[start:end, :].to(device))
    return batches


def _check_inside_tetrahedra_batch(points, p_1s, p_2s, p_3s, p_4s, epsilon=0):
    # Workaround for the case of a single point: Append one point and remove it at the end.
    is_single_point = False
    if len(points) == 1:
        points = torch.cat((points, points), dim=0)
        is_single_point = True

    # Now we check for all points, whether each point seperately is inside one of the tetrahedra.
    # To decide, whether a point is inside a tetrahedron, we check whether it is on the inside side of all the planes of the tetrahedron.
    # A point is on the inside side of a plane defined by three of the points of the tetrahedra, 
    # if it is on the same side as the remaining point of the tetrahedron for all of its points.
    # The result is a boolean tensor of shape (num_tetrahedra, num_points) where True means that the point is on the outside of the plane.       
    outside_for_p1_p2_p3_planes = ~_is_same_side_of_plane(p_1s, p_2s, p_3s, p_4s, points, epsilon)
    outside_for_p2_p3_p4_planes = ~_is_same_side_of_plane(p_2s, p_3s, p_4s, p_1s, points, epsilon)
    outside_for_p3_p4_p1_planes = ~_is_same_side_of_plane(p_3s, p_4s, p_1s, p_2s, points, epsilon)
    outside_for_p4_p1_p2_planes = ~_is_same_side_of_plane(p_4s, p_1s, p_2s, p_3s, points, epsilon)

    # Now we create a boolean tensor of shape (num_tetrahedra, num_points) where True means that a point is outside of the parallelepiped.
    outside_one_of_the_planes = outside_for_p1_p2_p3_planes | outside_for_p2_p3_p4_planes | outside_for_p3_p4_p1_planes | outside_for_p4_p1_p2_planes

    # To decide, whether a point is inside of one of the tetrahedron, we check whether it is on the inside side of all planes of the parallelepiped in at least one case.
    # This is the case, if the outside_one_of_the_planes tensor is False in at least one row.
    # However, the special case of a single parallelepiped needs to be handled separately, since the all function must not be applied in the case of a single tetrahedron 
    # (it would & the True and False values of all points instead of all tetrahedron ).

    if len(outside_one_of_the_planes.shape) == 1 and points.shape[0]>1:
        are_points_outside_everywhere = outside_one_of_the_planes
    else:
        are_points_outside_everywhere = torch.all(outside_one_of_the_planes, dim=0)

    # Remove the additional point, if it was added (see above).
    if is_single_point:
        are_points_outside_everywhere = are_points_outside_everywhere[0:-1]
    return are_points_outside_everywhere


def _check_inside_parallelepipeds_batch(points, bases_batch, v_1s_batch, v_2s_batch, v_3s_batch, epsilon=0):
    # Workaround for the case of a single point: Append one point and remove it at the end.
    is_single_point = False
    if len(points) == 1:
        points = torch.cat((points, points), dim=0)
        is_single_point = True

    # Find all points of the parallelepipeds.
    front_lower_left, front_lower_right, front_upper_left, front_upper_right, back_lower_left, back_lower_right, back_upper_left, back_upper_right = get_parallelepiped_corners(bases_batch, v_1s_batch, v_2s_batch, v_3s_batch)

    # Now we check for all points, whether each point seperately is inside one of the parallelepipeds.
    # To decide, whether a point is inside a parallelepiped, we check whether it is on the inside side of all the planes of the parallelepiped.
    # A point is on the inside side of a plane defined by three of the points of the parallelepipeds, 
    # if it is on the same side as the remaining point of the parallelepiped for all of its points.
    # The result is a boolean tensor of shape (num_parallelepipeds, num_points) where True means that the point is on the outside of the plane.       
    #                                                  plane_p1s,        plane_p2s,         plane_p3s,        opposite_points,  points
    outside_for_bottom_plane = ~_is_same_side_of_plane(front_lower_left, front_lower_right, back_lower_left,  front_upper_left,  points, epsilon)
    outside_for_top_planes   = ~_is_same_side_of_plane(front_upper_left, front_upper_right, back_upper_left,  front_lower_right, points, epsilon)
    outside_for_front_planes = ~_is_same_side_of_plane(front_lower_left, front_lower_right, front_upper_left, back_lower_left,   points, epsilon)
    outside_for_back_planes  = ~_is_same_side_of_plane(back_lower_left,  back_lower_right,  back_upper_left,  front_lower_right, points, epsilon)
    outside_for_left_planes  = ~_is_same_side_of_plane(front_lower_left, back_lower_left,   front_upper_left, front_upper_right, points, epsilon)
    outside_for_right_planes = ~_is_same_side_of_plane(front_lower_right, back_lower_right,  front_upper_right, back_upper_left, points, epsilon)

    # Now we create a boolean tensor of shape (num_parallelepipeds, num_points) where True means that a point is outside of the parallelepiped.
    outside_one_of_the_planes = outside_for_bottom_plane | outside_for_top_planes | outside_for_front_planes | outside_for_back_planes | outside_for_left_planes | outside_for_right_planes

    # To decide, whether a point is inside of one of the parallelepiped, we check whether it is on the inside side of all planes of the parallelepiped in at least one case.
    # This is the case, if the outside_one_of_the_planes tensor is False in at least one row.
    # However, the special case of a single parallelepiped needs to be handled separately, since the all function must not be applied in the case of a single parallelepiped 
    # (it would & the True and False values of all points instead of all parallelepiped ).

    if len(outside_one_of_the_planes.shape) == 1 and points.shape[0]>1:
        are_points_outside_everywhere = outside_one_of_the_planes
    else:
        are_points_outside_everywhere = torch.all(outside_one_of_the_planes, dim=0)

    # Remove the additional point, if it was added (see above).
    if is_single_point:
        are_points_outside_everywhere = are_points_outside_everywhere[0:-1]
    return are_points_outside_everywhere


def _check_which_points_are_inside_which_parallelepiped_batch(points, bases, v_1s, v_2s, v_3s, epsilon=0):
    """ Returns an integer mask for each parallelepiped that indicates which points are inside it. """
    # Find all points of the parallelepipeds.
    front_lower_left, front_lower_right, front_upper_left, front_upper_right, back_lower_left, back_lower_right, back_upper_left, back_upper_right = get_parallelepiped_corners(bases, v_1s, v_2s, v_3s)

    # Now we check for all points, whether each point seperately is inside one of the parallelepipeds.
    # To decide, whether a point is inside a parallelepiped, we check whether it is on the inside side of all the planes of the parallelepiped.
    # A point is on the inside side of a plane defined by three of the points of the parallelepipeds, 
    # if it is on the same side as the remaining point of the parallelepiped for all of its points.
    # The result is a boolean tensor of shape (num_parallelepipeds, num_points) where True means that the point is on the outside of the plane.       
    #                                                  plane_p1s,        plane_p2s,         plane_p3s,         opposite_points,   points
    inside_for_bottom_plane = _is_same_side_of_plane(front_lower_left,  front_lower_right, back_lower_left,   front_upper_left,  points, epsilon)
    inside_for_top_planes   = _is_same_side_of_plane(front_upper_left,  front_upper_right, back_upper_left,   front_lower_right, points, epsilon)
    inside_for_front_planes = _is_same_side_of_plane(front_lower_left,  front_lower_right, front_upper_left,  back_lower_left,   points, epsilon)
    inside_for_back_planes  = _is_same_side_of_plane(back_lower_left,   back_lower_right,  back_upper_left,   front_lower_right, points, epsilon)
    inside_for_left_planes  = _is_same_side_of_plane(front_lower_left,  back_lower_left,   front_upper_left,  front_upper_right, points, epsilon)
    inside_for_right_planes = _is_same_side_of_plane(front_lower_right, back_lower_right,  front_upper_right, back_upper_left,   points, epsilon)

    # Now we create a boolean tensor of shape (num_parallelepipeds, num_points) where True means that a point is outside of the parallelepiped.
    inside_all_of_the_planes_per_pe = inside_for_bottom_plane & inside_for_top_planes & inside_for_front_planes & inside_for_back_planes & inside_for_left_planes & inside_for_right_planes
    return inside_all_of_the_planes_per_pe


def _is_same_side_of_plane(p_1s, p_2s, p_3s, ps_remaining, points, epsilon=0):
    """ 
        Checks whether the points ps_remaining are on the same side of the planes defined by p_1s, p_2s and p_3s as the points in points.
        The result is a boolean tensor of shape (num_tetrahedra, num_points) where True means that the point is on the outside of the plane.
        Caution: Boarder cases might be handled incorrectly due to floating point errors.
    """
    num_points, dimensions = points.shape
    num_tetrahedra, _ = p_1s.shape

    normal = torch.cross(p_2s-p_1s, p_3s-p_1s, dim=1)

    # First we check on which side of the planes defined by p_1s, p_2s and p_3s the points ps_remaining are.
    # To perform the dot product for multiple tetrahedra at the same time, we need to bring the tensors into the correct shape.
    normal_view = normal.view(num_tetrahedra, 1, dimensions)
    ps_remaining_minus_p1 = (ps_remaining - p_1s).unsqueeze(-1)
    ps_remaining_minus_p1_view = ps_remaining_minus_p1.view(num_tetrahedra, dimensions, 1)
    dot_ps_remaining = torch.matmul(normal_view, ps_remaining_minus_p1_view).squeeze()

    # We need all possible combinations points - plane_bases for all points and all parallelepipeds. 
    # We therefore expand the tensors. 
    # The points is repeated num_tetrahedra times. 
    # The bases are modified to repeat each base num_points times in dim 1.
    # E.G. for bases = [[1, 2, 3], [4, 5, 6]] and points_in_question = [[7, 8, 9], [10, 11, 12], [13, 14, 15]] we get:
    # bases = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6], [4, 5, 6]]]
    # points_in_question = [[[7, 8, 9], [10, 11, 12], [13, 14, 15]], [[7, 8, 9], [10, 11, 12], [13, 14, 15]]]
    # we use expand to save memory compared to repeat, since the memory is not actually copied.

    points = points.expand(num_tetrahedra, num_points, dimensions)
    p_1s = p_1s.unsqueeze(1).expand(num_tetrahedra, num_points, dimensions)
    points_minus_p_1s = points - p_1s

    # Now we calculate the dot product of the normal and the point minus the base for all points.
    # Since we need to do this in a batched fashion, we need to bring the tensors into the correct shape.
    # To get pytorch to calculate the dot product between the two points, the last two dimensions of the tensors need to be extended to (3, 1) and (1, 3).
    # To be able to perform a batch multiplication on a (now 4-dimensional) tensor, we need to unsqueeze the tensors in the correct dimensions.
    normal_view = normal.view(num_tetrahedra, 1, 1, dimensions)
    px_view = points_minus_p_1s.view(num_tetrahedra, num_points, dimensions, 1)
    dot_points = torch.matmul(normal_view, px_view).squeeze()

    # If there are multiple points, we need to expand the dot_ps_remaining to the same shape as dot_points.
    if len(points) > 1:
        dot_ps_remaining = dot_ps_remaining.unsqueeze(1).expand(num_tetrahedra, num_points)

    # now we check whether the ps_remaining are on the same side of the plane as the points by comparing the sign of the dot products.
    # we use an epsilon to handle numerical errors. Emphasis is to never let a point be considered outside, if it is actually inside.
    result = dot_points*dot_ps_remaining >= (0 - epsilon)
    return result
