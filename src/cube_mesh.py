import numpy as np
import torch
from inside_checks import are_points_inside_one_of_the_boxes
from util import get_device, ensure_tensor_and_batched, get_parallelepiped_coords_from_boxes
from visualization import add_cubes_to_plt_ax_with_color
import binvox_rw


class CubeMesh():
    def __init__(self, binvox_file):
        self.binvox_file = binvox_file
        self.voxel_map = _get_voxel_map(binvox_file)
        self.lower_points, self.upper_points = _read_binvox_file(self.voxel_map)
        self.bases, self.v_1s, self.v_2s, self.v_3s = get_parallelepiped_coords_from_boxes(self.lower_points, self.upper_points)

        # axis-aligned bounding box
        self.lower_point, _ = self.lower_points.min(axis=0)
        self.upper_point, _ = self.upper_points.max(axis=0)

        
    def are_points_inside(self, points, batch_size_spec=None, epsilon=0):
        points = ensure_tensor_and_batched(points)
        device = get_device()

        # To improve performance, we first check if the points are inside the bounding box of the mesh
        lower_point, _ = self.lower_points.min(axis=0)
        upper_point, _ = self.upper_points.max(axis=0)

        are_outside_x = (points[:, 0] < lower_point[0]) | (points[:, 0] > upper_point[0])
        are_outside_y = (points[:, 1] < lower_point[1]) | (points[:, 1] > upper_point[1])
        are_outside_z = (points[:, 2] < lower_point[2]) | (points[:, 2] > upper_point[2])
        are_outside = are_outside_x | are_outside_y | are_outside_z
        
        inside_points = points[~are_outside]
        
        # For the remaining points, we check if they are inside the cubes
        are_inside_cubes = are_points_inside_one_of_the_boxes(inside_points, self.lower_points, self.upper_points, batch_size_spec, epsilon)        
        are_inside_cubes = are_inside_cubes.to(device)
        
        # Then we create the return values, positive values are outside
        are_inside = torch.ones(len(points), dtype=torch.float32).to(device)
        are_inside[~are_outside] = are_inside_cubes  
        return are_inside.cpu()
    
    def render(self, plt_axes, color='black'):
        add_cubes_to_plt_ax_with_color(lower_points=self.lower_points, upper_points=self.upper_points, plt_axes=plt_axes, color=color)

    def render_bounding_box(self, plt_axes, color='black', current_lambda=0.1):
        add_cubes_to_plt_ax_with_color(self.lower_point, self.upper_point, plt_axes, color, current_lambda=current_lambda)

    def get_name(self):
        return self.binvox_file.split("/")[-1].split(".")[0]


def load_binvox(fn):
    """Load a binvox file and return a 3D numpy array."""
    with open(fn, 'rb') as fin:
        out = binvox_rw.read_as_3d_array(fin)
        return np.array(out.data)

def _get_voxel_map(binvox_file):
    voxel_map = np.where(load_binvox(binvox_file) == True, 1.0, 0.0)
    voxel_map = torch.tensor(voxel_map, dtype=torch.float32)
    return voxel_map
        
def _read_binvox_file(voxel_map):
    lower_points = []
    upper_points = []

    step_size_x = 2/voxel_map.shape[0]
    step_size_y = 2/voxel_map.shape[1]
    step_size_z = 2/voxel_map.shape[2]

    x = -1
    for i in range (0, voxel_map.shape[0]):
        y = -1
        for j in range (0, voxel_map.shape[1]):
            z = -1
            for k in range (0, voxel_map.shape[2]):
                if voxel_map[i][j][k] == 1:
                    lower_points.append([x, y, z])
                    upper_points.append([x + step_size_x, y + step_size_y, z + step_size_z])
                z += step_size_z
            y += step_size_y
        x += step_size_x 
        
    assert len(upper_points) == len(lower_points) == voxel_map.sum()
    
    return torch.tensor(lower_points), torch.tensor(upper_points)
