import dataclasses
import numpy as np
import numpy
import torch
from matplotlib import pyplot as plt
from constants import SIGN_OUTSIDE, SIGN_INSIDE, SIGN_UNKNOWN
from util import get_device, ensure_tensor_and_batched, apply_limits_to_points, get_parallelepiped_corners
from util import get_booleans_for_parallelepipeds_within_limits, get_parallelepiped_coords_from_boxes, get_3d_grid_points
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclasses.dataclass
class Limits:
    """ A class that represents the limits of a 3D space."""
    x_limits: tuple = (-1, 1)
    y_limits: tuple = (-1, 1)
    z_limits: tuple = (-1, 1)


def create_plt_axes(upper_point=None, lower_point=None):
    """ Creates a matplotlib.pyplot axis with 3D projection."""
    fig = plt.figure(figsize=(10, 10))
    plt_ax = fig.add_subplot(111, projection='3d')
    plt_ax.set_xlim(xmin=-1, xmax=1)
    plt_ax.set_ylim(ymin=-1, ymax=1)
    plt_ax.set_zlim(zmin=-1, zmax=1)

    if upper_point is not None and lower_point is not None:
        plt_ax.set_xlim(xmin=lower_point[0], xmax=upper_point[0])
        plt_ax.set_ylim(ymin=lower_point[1], ymax=upper_point[1])
        plt_ax.set_zlim(zmin=lower_point[2], zmax=upper_point[2])
        print(f"Setting limits to {lower_point} and {upper_point}")

    plt_ax.set_xlabel('x')
    plt_ax.set_ylabel('y')
    plt_ax.set_zlabel('z')
    return plt_ax


def create_plt_axes_2d():
    fig = plt.figure()
    plt_ax = fig.add_subplot(111)
    plt_ax.set_xlim(xmin=-1, xmax=1)
    plt_ax.set_ylim(ymin=-1, ymax=1)
    return plt_ax


def add_cubes_to_plt_ax_with_color(lower_points, upper_points, plt_axes, color, current_lambda=0.75):
    """ Adds a cube to the given matplotlib.pyplot axes plt_axes. The color of the cube is fixed."""
    lower_points, upper_points = ensure_tensor_and_batched(lower_points), ensure_tensor_and_batched(upper_points)
    bases, v_1s, v_2s, v_3s = get_parallelepiped_coords_from_boxes(lower_points, upper_points)
    add_parallelepipeds_to_plt_ax_with_color(plt_axes, bases, v_1s, v_2s, v_3s, color, current_lambda=current_lambda)


def add_parallelepipeds_to_plt_ax_with_color_array(plt_axes, bases, v_1s, v_2s, v_3s, colors, current_lambda=0.75, linewidth=0.1):
    """ Adds parallelepipeds to the given matplotlib.pyplot axes plt_axes. The color of the cube is fixed."""
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched(bases), ensure_tensor_and_batched(v_1s), ensure_tensor_and_batched(v_2s), ensure_tensor_and_batched(v_3s)
    _add_parallelepipeds_to_plt_axis(colors, plt_axes, bases, v_1s, v_2s, v_3s, current_lambda, linewidth=linewidth)
   
    
def add_parallelepipeds_to_plt_ax_with_color(plt_axes, bases, v_1s, v_2s, v_3s, color="black", current_lambda=0.75, linewidth=0.1):
    """ Adds parallelepipeds to the given matplotlib.pyplot axes plt_axes. The color of the cube is fixed."""
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched(bases), ensure_tensor_and_batched(v_1s), ensure_tensor_and_batched(v_2s), ensure_tensor_and_batched(v_3s)
    colors = numpy.full(bases.shape[0], color, dtype=object)
    _add_parallelepipeds_to_plt_axis(colors, plt_axes, bases, v_1s, v_2s, v_3s, current_lambda, linewidth=linewidth)
    
    
def add_tetrahedra_to_plt_ax_with_color(plt_axes, p_1s, p_2s, p_3s, p_4s, color="black"):
    p_1s, p_2s, p_3s, p_4s = ensure_tensor_and_batched(p_1s), ensure_tensor_and_batched(p_2s), ensure_tensor_and_batched(p_3s), ensure_tensor_and_batched(p_4s)
    for p_1, p_2, p_3, p_4 in zip(p_1s, p_2s, p_3s, p_4s):
        for edge in  [[p_1, p_2], [p_1, p_3], [p_2, p_3], [p_1, p_4], [p_2, p_4], [p_3, p_4],]:
            start_point = edge[0]
            end_point   = edge[1] 
            plt_axes.plot3D([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=color, zorder=10)


def add_cubes_to_plt_ax_with_decision_maker(lower_points, upper_points, plt_axes, decision_maker, limits=None, show_inside_cubes=True, show_outside_cubes=True, show_unknown_cubes=True):
    """ Adds a cube to the given matplotlib.pyplot axes plt_axes. The color of the cube is determined using the decision function."""
    lower_points, upper_points = ensure_tensor_and_batched(lower_points), ensure_tensor_and_batched(upper_points)
    bases, v_1s, v_2s, v_3s = get_parallelepiped_coords_from_boxes(lower_points, upper_points)
    add_parallelepipeds_to_plt_ax_with_decision_maker(bases, v_1s, v_2s, v_3s, plt_axes, decision_maker, limits, show_inside_cubes, show_outside_cubes, show_unknown_cubes)


def add_parallelepipeds_to_plt_ax_with_decision_maker(bases, v_1s, v_2s, v_3s, plt_axes, decision_maker, limits=None, show_inside_pes=True, show_outside_pes=True, show_unknown_pes=True, current_lambda=0.75, linewidth=0.1):
    """ Adds parallelepipeds to the given matplotlib.pyplot axes plt_axes. The color of the cube is determined using the decision function."""
    bases, v_1s, v_2s, v_3s = ensure_tensor_and_batched(bases), ensure_tensor_and_batched(v_1s), ensure_tensor_and_batched(v_2s), ensure_tensor_and_batched(v_3s)
    
    bases, v_1s, v_2s, v_3s = _apply_limits_to_parallelepipeds(bases, v_1s, v_2s, v_3s, limits)
    
    pe_classification = decision_maker.classify_parallelepiped(bases, v_1s, v_2s, v_3s)
    inside_pes  = pe_classification == SIGN_INSIDE
    outside_pes = pe_classification == SIGN_OUTSIDE
    unknown_pes = pe_classification == SIGN_UNKNOWN
    
    colors = _get_colors_from_classification(pe_classification)

    # move to cpu to use numpys ability to work with strings for the colors
    bases, v_1s, v_2s, v_3s, inside_pes, outside_pes, unknown_pes = bases.cpu(), v_1s.cpu(), v_2s.cpu(), v_3s.cpu(), inside_pes.cpu(), outside_pes.cpu(), unknown_pes.cpu()
    
    if not show_inside_pes:
        inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors = _remove_parallelepipeds(inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors, inside_pes)
    if not show_outside_pes:
        inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors = _remove_parallelepipeds(inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors, outside_pes)
    if not show_unknown_pes:
        inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors = _remove_parallelepipeds(inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors, unknown_pes)

    _add_parallelepipeds_to_plt_axis(colors, plt_axes, bases, v_1s, v_2s, v_3s, current_lambda, linewidth=linewidth)


def render_decision_maker_with_points(decision_maker, step_size, plt_axes, limits=None, color='b'):
    """ Renders the are_points_inside function of the given decision maker using the given matplotlib.pyplot axes plt_axes with points. """
    render_function_with_points(decision_maker.are_points_inside, step_size, plt_axes, limits, color)


def render_function_with_points(function, step_size, plt_axes, limits=None, color='b'):
    """ Renders a function of  using the given matplotlib.pyplot axes plt_axes with points """
    inside_points = _get_points_that_are_inside_the_shape(function=function, step_size=step_size)
    inside_points = apply_limits_to_points(inside_points, limits)
    positions = inside_points.cpu().numpy()
    plt_axes.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], zorder=1, s=1.0, c=color)


def _remove_parallelepipeds(inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors, bools_to_remove):
    """ Removes parallelepipeds from the given lists. The parallelepipeds to remove are determined by the given boolean tensor (True means remove).)"""
    bases = bases[~bools_to_remove]
    v_1s = v_1s[~bools_to_remove]
    v_2s = v_2s[~bools_to_remove]
    v_3s = v_3s[~bools_to_remove]
    colors = colors[~bools_to_remove]
    outside_pes = outside_pes[~bools_to_remove]
    unknown_pes = unknown_pes[~bools_to_remove]
    inside_pes = inside_pes[~bools_to_remove]
    return inside_pes, outside_pes, unknown_pes, bases, v_1s, v_2s, v_3s, colors


def _apply_limits_to_parallelepipeds(bases, v_1s, v_2s, v_3s, limits):
    """ Applies the given limits to the given parallelepipeds. Returns the parallelepipeds that are within the limits."""
    if limits is not None:
        within_limits = get_booleans_for_parallelepipeds_within_limits(limits, bases, v_1s, v_2s, v_3s)
        bases = bases[within_limits]
        v_1s = v_1s[within_limits]
        v_2s = v_2s[within_limits]
        v_3s = v_3s[within_limits]
    return bases, v_1s, v_2s, v_3s


def _get_colors_from_classification(classification):
    """ Returns a numpy array with the colors for the given classifications with len(colors) = len(classifications). Green for inside, red for outside, blue for unknown."""
    classification = classification.cpu().numpy()
    colors = numpy.full_like(classification, "black", dtype=object)
    colors = np.where(classification == SIGN_INSIDE, "green", colors)
    colors = np.where(classification == SIGN_OUTSIDE, "red", colors)
    colors = np.where(classification == SIGN_UNKNOWN, "blue", colors)
    if len(colors.shape) == 0:
        colors = np.expand_dims(colors, axis=0)
    return colors


def _get_points_that_are_inside_the_shape(function, step_size):
    """ Returns a tensor with points on an integer grid that are inside the shape defined by the given decision maker."""
    positions = get_3d_grid_points(step_size=step_size)
    positions = positions.to(get_device())
    are_points_inside = function(positions)
    are_points_inside = are_points_inside <= 0
    positions = positions[are_points_inside]
    return positions


def _add_parallelepipeds_to_plt_axis(pe_colors, plt_axes, bases, v_1s, v_2s, v_3s, current_lambda=0.75, linewidth=0.1):
    """ 
        Adds parallelepipeds to the given matplotlib.pyplot axes plt_axes. The color of the cube is given in the colors array. 
        See _get_colors_from_classification or add_parallelepipeds_to_plt_ax_with_color for more information.
    """
    # get the corners of the parallelepiped
    front_lower_left, front_lower_right, front_upper_left, front_upper_right, back_lower_left, back_lower_right, back_upper_left, back_upper_right = get_parallelepiped_corners(bases, v_1s, v_2s, v_3s)

    # with the corners, create the faces of the parallelepiped
    front_faces = torch.stack([ front_lower_left,  front_lower_right, front_upper_right, front_upper_left  ])
    back_faces  = torch.stack([ back_lower_left,   back_lower_right,  back_upper_right,  back_upper_left   ])
    left_faces  = torch.stack([ front_lower_left,  back_lower_left,   back_upper_left,   front_upper_left  ])
    right_faces = torch.stack([ front_lower_right, back_lower_right,  back_upper_right,  front_upper_right ])
    lower_faces = torch.stack([ front_lower_left,  front_lower_right, back_lower_right,  back_lower_left   ])
    upper_faces = torch.stack([ front_upper_left,  front_upper_right, back_upper_right,  back_upper_left   ])

    # bring the faces into the right shape: instead of the faces ordered by their relative position in the pe, we want a list of all faces one ater the other
    faces = torch.stack([front_faces, back_faces, left_faces, right_faces, lower_faces, upper_faces])
    faces = faces.permute(0, 2, 1, 3)
    faces = faces.reshape(-1, 4, 3)

    # create the colors for the faces. The color is given for the pe in this function but needs to be given for each face in the facecolors parameter of Poly3DCollection
    faces_colors = np.stack([pe_colors, pe_colors, pe_colors, pe_colors, pe_colors, pe_colors]).reshape(-1)

    poly3d = Poly3DCollection(faces, facecolors=faces_colors, linewidths=linewidth, edgecolors='silver', current_lambda=current_lambda)
    plt_axes.add_collection3d(poly3d)    
