import os
import torch


####### Device handling #######

def get_device():
    """ Returns the device that is used for computation, depending on the availability of cuda and mps."""
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


####### Formatting #######

def make_bold(string):
    """ Returns the given string in bold for output in the terminal."""
    return '\033[1m' + string + '\033[0m'


####### Tensor handling #######

def ensure_is_tensor(tensor):
    """ Ensures that the given tensor is a torch.Tensor."""
    if not type(tensor) == torch.Tensor:
        tensor = torch.tensor(tensor)
    return tensor


def ensure_tensor_is_batch(tensor):
    """ Ensures that the given tensor is a batch, i.e. has at least 2 dimensions."""
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def ensure_tensor_is_batch_of_2d_tensors(tensor):
    """ Ensures that the given tensor is a batch, i.e. has at least 2 dimensions."""
    if len(tensor.shape) <= 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def ensure_tensor_and_batched(input):
    """ Ensures that the given input is a torch.Tensor and a batch, i.e. has at least 2 dimensions."""
    input = ensure_is_tensor(input)
    input = ensure_tensor_is_batch(input)
    return input


def ensure_tensor_and_batched_for_all(input_list):
    """ Ensures that all given inputs are torch.Tensors and batches, i.e. have at least 2 dimensions."""
    for i in range(len(input_list)):
        input_list[i] = ensure_tensor_and_batched(input_list[i])
    return input_list


def assert_same_length(input_list):
    """ Asserts that all given tensors have the same length."""
    length = len(input_list[0])
    for t in input_list:
        assert len(t) == length, "All input tensors must have the same length. Got: {}".format([len(t) for t in input_list])

def prepare_bounds_for_torch(upper_point, lower_point, mesh):
        assert upper_point is not None and lower_point is not None or upper_point is None and lower_point is None, "Either both or none of upper_point and lower_point must be given"
        
        # In the case no definition volume was provided, use the bounding box of the mesh
        if upper_point is None and lower_point is None:
            upper_point = mesh.upper_point
            lower_point = mesh.lower_point

        upper_point = ensure_is_tensor(upper_point)
        lower_point = ensure_is_tensor(lower_point)

        return upper_point, lower_point


####### Grids #######

def get_3d_grid_points(step_size, skip_last=False):
    """ 
        Returns points on a 3d grid with the given step size. 
        The grid is centered around the origin. 
        It ranges from -1 to 1 in each dimension.
        In the case of skip_last=True, the last point in each dimension is not included. 
        This is useful if cubes are to be constructed using lower_points = grid_points(step_size, True) and upper_points = grid_points + step_size.
    """
    steps = int(2 / step_size) + 1
    xs = torch.linspace(start=-1, end=1, steps=steps, dtype=torch.float)
    ys = torch.linspace(start=-1, end=1, steps=steps, dtype=torch.float)
    zs = torch.linspace(start=-1, end=1, steps=steps, dtype=torch.float)
    if skip_last:
        xs = xs[:-1]
        ys = ys[:-1]
        zs = zs[:-1]
    grid = torch.meshgrid(xs, ys, zs, indexing='ij')
    xs, ys, zs = torch.flatten(grid[0]), torch.flatten(grid[1]), torch.flatten(grid[2])
    return torch.stack((xs, ys, zs), dim=1)


####### Corners #######

def get_cube_corners(lower_points, upper_points):
    """ Returns the coordinates of the corners of the cube that is spanned by the given boxes."""
    lower_points = ensure_tensor_and_batched(lower_points)
    upper_points = ensure_tensor_and_batched(upper_points)
    base, v_1, v_2, v_3 = get_parallelepiped_coords_from_boxes(lower_points, upper_points)
    return get_parallelepiped_corners(base, v_1, v_2, v_3)


def get_parallelepiped_corners(base, v_1, v_2, v_3):
    """ Returns the coordinates of the corners of the parallelepiped that is spanned by the given base and vectors v_1, v_2 and v_3."""
    # For construction of the points the parallelepiped is assumed to be a box and the view is assumed to be in positive y-direction.
    # The assumed box has the vectors base = (0, 0, 0), v_1 = (1, 0, 0), v_2 = (0, 1, 0) and v_3 = (0, 0, 1).
    front_lower_left  = base
    front_lower_right = base + v_1
    front_upper_left  = base + v_3
    front_upper_right = base + v_1 + v_3
    back_lower_left   = base + v_2
    back_lower_right  = base + v_2 + v_1
    back_upper_left   = base + v_2 + v_3
    back_upper_right  = base + v_2 + v_1 + v_3
    return front_lower_left, front_lower_right, front_upper_left, front_upper_right, back_lower_left, back_lower_right, back_upper_left, back_upper_right


####### Limits #######

def get_booleans_for_boxes_within_limits(limits, lower_points, upper_points):
    """ Checks if the given boxes are within the limits given by limits. Returns a boolean tensor of the same length as lower_points and upper_points."""
    lower_points, upper_points = ensure_tensor_and_batched(lower_points), ensure_tensor_and_batched(upper_points)
    base, v_1, v_2, v_3 = get_parallelepiped_coords_from_boxes(lower_points, upper_points)
    return get_booleans_for_parallelepipeds_within_limits(limits, base, v_1, v_2, v_3)


def get_booleans_for_parallelepipeds_within_limits(limits, bases, v_1s, v_2s, v_3s):
    """ Checks if the given parallelepipeds are within the limits given by limits. Returns a boolean tensor of the same length as bases, v_1s, v_2s and v_3s."""
    pe_points = get_parallelepiped_corners(bases, v_1s, v_2s, v_3s)
    
    is_one_point_bigger_than_lower_limit = torch.zeros(len(bases), dtype=torch.bool)
    is_one_point_smaller_than_upper_limit = torch.zeros(len(bases), dtype=torch.bool)
    for p in pe_points:
        # Check if the point is bigger than the lower limit.
        bigger_lower_limit = torch.ones(len(bases), dtype=torch.bool)
        bigger_lower_limit  &= p[:, 0] >= limits.x_limits[0]
        bigger_lower_limit  &= p[:, 1] >= limits.y_limits[0]
        bigger_lower_limit  &= p[:, 2] >= limits.z_limits[0]
        is_one_point_bigger_than_lower_limit |= bigger_lower_limit
        
        # Check if the point is smaller than the upper limit.
        smaller_upper_limit = torch.ones(len(bases), dtype=torch.bool)
        smaller_upper_limit  &= p[:, 0] <= limits.x_limits[1]
        smaller_upper_limit  &= p[:, 1] <= limits.y_limits[1]
        smaller_upper_limit  &= p[:, 2] <= limits.z_limits[1]
        is_one_point_smaller_than_upper_limit |= smaller_upper_limit
        
    return is_one_point_bigger_than_lower_limit & is_one_point_smaller_than_upper_limit


def apply_limits_to_points(points, limits):
    """ Checks if the given points are within the limits given by limits. """
    if limits is not None:
        within_limits = torch.ones(len(points), dtype=torch.bool, device=points.device)
        within_limits &= _check_limits_one_dim(points, limits.x_limits, axis=0)
        within_limits &= _check_limits_one_dim(points, limits.y_limits, axis=1)
        within_limits &= _check_limits_one_dim(points, limits.z_limits, axis=2)
        points = points[within_limits]
    return points


def _check_limits_one_dim(points, limit_tuple, axis):
    """ Checks if the given points are within the limits given by limit_tuple on the axis given by axis. """
    assert len(limit_tuple) == 2, "limit_tuples must be tuples of length 2. Got: {}".format(limit_tuple)
    within_limits  = points[:, axis] <= limit_tuple[1]
    within_limits &= points[:, axis] >= limit_tuple[0]
    return within_limits


####### Representation Transformations #######

def get_parallelepiped_coords_from_boxes(lower_points, upper_points):
    """ Returns the given boxes in parallelepiped coordinates."""
    lower_points = ensure_tensor_and_batched(lower_points)
    upper_points = ensure_tensor_and_batched(upper_points)

    bases = lower_points
    v_1s = torch.stack([upper_points[:, 0] - lower_points[:, 0], torch.zeros_like(lower_points[:, 0]), torch.zeros_like(lower_points[:, 0])], dim=1)
    v_2s = torch.stack([torch.zeros_like(lower_points[:, 0]), upper_points[:, 1] - lower_points[:, 1], torch.zeros_like(lower_points[:, 0])], dim=1)
    v_3s = torch.stack([torch.zeros_like(lower_points[:, 0]), torch.zeros_like(lower_points[:, 0]), upper_points[:, 2] - lower_points[:, 2]], dim=1)

    return bases, v_1s, v_2s, v_3s


def get_parallelepiped_coords_in_affine_form_from_base_form(bases, v_1s, v_2s, v_3s):
    """ 
        Returns the affine coordinates of the parallelepiped that is spanned by the given base and vectors v_1, v_2 and v_3.
        While base, v_1, v_2 and v_3 define the parallelepiped by one corner and three vectors giving three of the edges,
        the affine coordinates define the parallelepiped by its center and three vectors pointing to the middle of three faces.
    """
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched(bases), ensure_tensor_and_batched(v_1s), ensure_tensor_and_batched(v_2s), ensure_tensor_and_batched(v_3s)
    centers = bases + (v_1s + v_2s + v_3s)/2
    face_vectors_1 = v_1s / 2
    face_vectors_2 = v_2s / 2
    face_vectors_3 = v_3s / 2
    box_vecs = torch.stack((face_vectors_1, face_vectors_2, face_vectors_3), dim=1)
    return centers, box_vecs


def get_parallelepiped_coords_in_base_form_from_affine_form(centers, box_vecs):
    """ 
        Returns the bases and vectors v_1, v_2 and v_3 that define the parallelepiped that is spanned by the given affine coordinates.
        While base, v_1, v_2 and v_3 define the parallelepiped by one corner and three vectors giving three of the edges,
        the affine coordinates define the parallelepiped by its center and three vectors pointing to the middle of three faces.
    """
    centers, box_vecs = ensure_tensor_and_batched(centers), ensure_tensor_is_batch_of_2d_tensors(box_vecs)
    bases = centers - box_vecs.sum(dim=1)
    
    v_1s = box_vecs[:, 0, :] * 2
    v_2s = box_vecs[:, 1, :] * 2
    v_3s = box_vecs[:, 2, :] * 2
    return bases, v_1s, v_2s, v_3s


####### Path Creation #######

def get_points_path(prefix, mesh, number_of_points, lower_point, upper_point, scale_factor, outside=None, type=None, axis=None, axis_value=None):
    name = mesh.get_name()

    lower_point_str = "_".join([str(x) for x in lower_point.tolist()])
    upper_point_str = "_".join([str(x) for x in upper_point.tolist()])
    bounding_box_str = "bb_"+lower_point_str + "_" + upper_point_str
    
    inside_outside_str = ""
    if outside is not None:
        inside_outside_str = "outside" if outside else "inside"
        inside_outside_str = "_" + inside_outside_str

    if type is None:
        type = ""
    else:
        type += "_"

    if axis is None:
        axis = ""
    else:
        axis = "_axis_"+axis
    
    if axis_value is None:
        axis_value = ""
    else:
        axis_value = "_axis_value_" + str(axis_value)

    file_name = prefix+ "_" + type + str(number_of_points) + "_" + name + "_" + bounding_box_str + inside_outside_str + axis + axis_value + "_scale_" +str(scale_factor)+ ".pt"
    path = os.path.join("..", "point_cache", file_name)
    
    return path


####### Exceptions #######

class GetOutOfLoop(Exception):
    pass
