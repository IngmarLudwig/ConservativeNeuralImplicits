import numpy as np
import torch
from inside_checks import are_points_inside_one_of_the_tetrahedra
from util import get_device, ensure_tensor_and_batched, ensure_is_tensor
from visualization import add_tetrahedra_to_plt_ax_with_color, add_cubes_to_plt_ax_with_color


class TetrahedralMesh():
    def __init__(self, vtk_file, scale_factor=1.0):
        self.vtk_file = vtk_file

        vertices, tetrahedra = _read_vtk_file(vtk_file)
        vertices = _scale_to_range_minus_1_to_1(vertices, scale_factor)
        
        self.vertices     = torch.tensor(vertices,     dtype=torch.float32)
        self.tetrahedra   = torch.tensor(tetrahedra,   dtype=torch.int64  )

        # axis-aligned bounding box
        self.lower_point, _ = self.vertices.min(axis=0)
        self.upper_point, _ = self.vertices.max(axis=0)
        
    def are_points_inside(self, points, batch_size_spec=None, epsilon=0):
        points = ensure_tensor_and_batched(points)
        device = get_device()

        # To improve performance, we first check if the points are inside the bounding box of the mesh
        lower_point, _ = self.vertices.min(axis=0)
        upper_point, _ = self.vertices.max(axis=0)
        are_outside_x = (points[:, 0] < lower_point[0]) | (points[:, 0] > upper_point[0])
        are_outside_y = (points[:, 1] < lower_point[1]) | (points[:, 1] > upper_point[1])
        are_outside_z = (points[:, 2] < lower_point[2]) | (points[:, 2] > upper_point[2])
        are_outside = are_outside_x | are_outside_y | are_outside_z
        
        inside_points = points[~are_outside]
        
        # For the remaining points, we check if they are inside the tetrahedra
        tet_points = self.vertices[self.tetrahedra]
        p_1s = tet_points[:, 0]
        p_2s = tet_points[:, 1]
        p_3s = tet_points[:, 2]
        p_4s = tet_points[:, 3]
        are_inside_tets = are_points_inside_one_of_the_tetrahedra(inside_points, p_1s, p_2s, p_3s, p_4s, batch_size_spec, epsilon)        
        are_inside_tets = are_inside_tets.to(device)
        
        # Then we create the return values, positive values are outside
        are_inside = torch.ones(len(points), dtype=torch.float32).to(device)
        are_inside[~are_outside] = are_inside_tets  
        return are_inside.cpu()
    
    def render_tets(self, plt_axes, color='black'):
        p1s = self.vertices[self.tetrahedra[:, 0]]
        p2s = self.vertices[self.tetrahedra[:, 1]]
        p3s = self.vertices[self.tetrahedra[:, 2]]
        p4s = self.vertices[self.tetrahedra[:, 3]]
        add_tetrahedra_to_plt_ax_with_color(plt_axes, p1s, p2s, p3s, p4s, color)

    def render_shape(self, plt_axes, color='black'):
        faces = _extract_surface(self.tetrahedra)
        plt_axes.plot_trisurf(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2], triangles=faces, color=color)

    def render_bounding_box(self, plt_axes, color='black', current_lambda=0.1):
        add_cubes_to_plt_ax_with_color(self.lower_point, self.upper_point, plt_axes, color, current_lambda=current_lambda)

    def render_with_points(self, plt_axes, lower_point, upper_point, num_points=1_000_000, color='black'):
        assert upper_point is not None and lower_point is not None or upper_point is None and lower_point is None, "Either both or none of upper_point and lower_point must be given"
        lower_point = ensure_is_tensor(lower_point)
        upper_point = ensure_is_tensor(upper_point)

        random_points = torch.rand((num_points, 3), dtype=torch.float32) * (upper_point - lower_point) + lower_point
        are_inside = self.are_points_inside(random_points)
        inside_points = random_points[are_inside <= 0]
        plt_axes.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], color=color)


    def get_name(self):
        return self.vtk_file.split("/")[-1].split(".")[0]


def _scale_to_range_minus_1_to_1(points, scale_factor):
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    centers = (max_point + min_point) / 2.0
    scale = np.max(max_point - min_point) / 2.0
    points = (points - centers) / scale
    points = points * scale_factor
    return points


def _read_vtk_file(vtk_file):
    vertices = []
    tetrahedra = []
    with open(vtk_file, 'r') as f:
        lines = f.readlines()

        point_mode = False
        cell_mode = False
        for line in lines:
            if line.startswith('POINTS'):
                point_mode = True
                continue
            if line.startswith('CELLS'):
                point_mode = False
                cell_mode = True
                continue
            if line.startswith('CELL_TYPES'):                  
                cell_mode = False
                continue
            
            if point_mode:
                    vertices.append([float(x) for x in line.split()])

            if cell_mode:
                    tetrahedra.append([int(x) for x in line.split()[1:]])
        vertices = np.array(vertices)
        tetrahedra = np.array(tetrahedra)
    return vertices, tetrahedra


# Thanks to Futurologist on StackOverflow https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
def _extract_surface(tets):
    def list_faces(tets):
        tets.sort(axis=1)
        num_tets, vertices_per_tet= tets.shape
        faces = np.empty((4*num_tets, 3) , dtype=int)
        i = 0
        for j in range(4):
            faces[i:i+num_tets,0:j] = tets[:,0:j]
            faces[i:i+num_tets,j:3] = tets[:,j+1:4]
            i=i+num_tets
        return faces
        
    def extract_unique_triangles(faces):
        _, indxs, count  = np.unique(faces, axis=0, return_index=True, return_counts=True)
        return faces[indxs[count==1]]

    faces = list_faces(tets)
    faces = extract_unique_triangles(faces)
    faces = _add_inverted_faces(faces)
    return faces

    
def _add_inverted_faces(faces):
    inverted_faces = np.array([[face[1], face[0], face[2]] for face in faces])
    return np.concatenate((faces, inverted_faces))
