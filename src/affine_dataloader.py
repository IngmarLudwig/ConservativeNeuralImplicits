import torch
import numpy as np
from visualization import add_parallelepipeds_to_plt_ax_with_color_array
from util import get_parallelepiped_coords_in_affine_form_from_base_form, get_booleans_for_parallelepipeds_within_limits, get_parallelepiped_coords_in_base_form_from_affine_form, get_parallelepiped_coords_from_boxes


class AffineDataloader():
    """ Dataloader for indicator functions or tetrahedral meshes. 
        The dataloader returns batches of parallelepipeds, where each parallelepiped is defined by a center and three vector pointing to the faces of the parallelepiped.
        All returned parallelepipeds are either inside or partly inside the shape.
    """
    def __init__(self, mesh, batch_size,  shuffle=True, num_initial_splits_for_voxel=0, split_all=0):
        """ 
            Creates a dataloader for the given indicator function or tetrahedral mesh. The dataloader returns batches of boxes, where each box is defined by its lower and upper point. 
        """
        # Save the parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mesh = mesh
        
        # Get the parallelepipeds from the tet mesh or cube mesh
        if type(mesh).__name__ == "TetrahedralMesh":
            bases, v_1s, v_2s, v_3s = get_filling_parallelepipeds_for_multiple_tetrahedra(mesh.vertices, mesh.tetrahedra)
        elif type(mesh).__name__ == "CubeMesh":
            bases, v_1s, v_2s, v_3s = get_parallelepiped_coords_from_boxes(mesh.lower_points, mesh.upper_points)
        else:
            raise ValueError("The mesh must be a TetrahedralMesh or CubeMesh, but got {}".format(type(mesh)))
        
        self.centers, self.vecs = get_parallelepiped_coords_in_affine_form_from_base_form(bases, v_1s, v_2s, v_3s)
        
        # Will be reseted after the initial splits
        self.color_ints = torch.zeros(len(self.centers))
        self.color_counter = 0

        # For voxel input, it is useful to split boarder boxes to reduce the approximation error from Affine Arithmetic
        for _ in range(num_initial_splits_for_voxel):
            are_at_boarder = torch.zeros(len(self.centers), dtype=torch.bool)
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        if i == 0 and j == 0 and k == 0:
                            continue
                        new_centers = self.centers + self.vecs[:, 0, :]*i*2. + self.vecs[:, 1, :]*j*2. + self.vecs[:, 2, :]*k*2.
                        new_centers_inside = mesh.are_points_inside(new_centers)
                        
                        # if are_points_inside returns 1, the point is inside the shape, if it returns 1, the point is outside the shape    
                        are_at_boarder[new_centers_inside > 0] = 1.
            indices = torch.nonzero(are_at_boarder).squeeze()
            self._split_parallelepipeds(indices)
            
        for _ in range(split_all):
            all_indices = torch.arange(len(self.centers))
            self._split_parallelepipeds(all_indices)
        
        # Initialize the color counter for displaying the parallelepipeds in different colors depending on how often they were split (after initial splits)
        self.color_ints = torch.zeros(len(self.centers))
        self.color_counter = 0

    def __len__(self):
        """ Returns the number of batches in the dataloader."""
        add = 0
        if len(self.centers) % self.batch_size != 0:
            add = 1
        return len(self.centers)//self.batch_size + add

    def get_num_full_batches(self):
        """ Returns the number of full batches in the dataloader."""
        return len(self.centers)//self.batch_size

    def get_num_samples(self):
        """ Returns the number of samples (parallelepipeds) in the dataloader."""
        return len(self.centers)

    def __getitem__(self, idx):
        """ Returns the parallelepipeds for the given batch index in affine form. If shuffle is True, the returned parallelepipeds changes after a call of shuffle (e.g. in __iter__).)"""
        return self.centers[idx], self.vecs[idx]

    def __iter__(self):
        """ Returns an iterator over all batches in the dataloader. If shuffle is True, the order of the batches is random."""
        if self.shuffle:
            self._shuffle_data()

        # Yield all full batches
        num_full_batches = self.get_num_full_batches()
        for i in range(num_full_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            yield self.centers[start:end, :], self.vecs[start:end, :, :]

        # Yield the last, incomplete batch if there is one
        if len(self.centers) % self.batch_size != 0:
            end = num_full_batches * self.batch_size
            yield self.centers[end:, :], self.vecs[end:, :, :]

    def render(self, plt_axes, limits=None, colors_list=["k", "b", "g", "r", "c", "m", "y"], linewidth=0.1):
        """ Renders the parallelepipeds in the dataloader into the given matplotlib.pyplot axes plt_ax."""
        bases_to_show, v_1s_to_show, v_2s_to_show, v_3s_to_show  = get_parallelepiped_coords_in_base_form_from_affine_form(self.centers, self.vecs)
        color_ints = self.color_ints
        if limits is not None:
            within_limits = get_booleans_for_parallelepipeds_within_limits(limits, bases_to_show, v_1s_to_show, v_2s_to_show, v_3s_to_show)
            bases_to_show = bases_to_show[within_limits]
            v_1s_to_show = v_1s_to_show[within_limits]
            v_2s_to_show = v_2s_to_show[within_limits]
            v_3s_to_show = v_3s_to_show[within_limits]
            color_ints = self.color_ints[within_limits]
        
        # set color for each parallelepiped depending on how often it was split
        colors = np.full(bases_to_show.shape[0], "silver", dtype=object)
        for i in range(len(colors_list)):
            mask = color_ints == i
            colors[mask] = colors_list[i]
        mask = color_ints >= len(colors_list)
        colors[mask] = colors_list[-1]
        
        add_parallelepipeds_to_plt_ax_with_color_array(plt_axes, bases_to_show, v_1s_to_show, v_2s_to_show, v_3s_to_show, colors, linewidth=linewidth)
    
    def _split_parallelepipeds(self, indices):
        # get the parallelepipeds to split
        indices = indices.cpu()
        centers = self.centers[indices]
        vecs = self.vecs[indices]
        
        # transform the parallelepipeds to base form
        bases, v_1s, v_2s, v_3s = get_parallelepiped_coords_in_base_form_from_affine_form(centers, vecs)
        
        # split the parallelepipeds
        # half the size of the original parallelepipeds vertices because each parallelepiped is split into 8 parallelepipeds
        v_1s = v_1s / 2
        v_2s = v_2s / 2
        v_3s = v_3s / 2
        
        p_1s = bases + v_1s
        p_2s = bases + v_2s
        p_3s = bases + v_3s
        p_4s = p_1s + v_2s
        p_5s = p_1s + v_3s
        p_6s = p_2s + v_3s
        p_7s = p_6s + p_5s - p_3s
        
        # Define subparallelepipeds
        bases_split = torch.cat([bases, p_1s, p_2s, p_3s, p_4s, p_5s, p_6s, p_7s])
        v_1s_split  = torch.cat([v_1s,  v_1s, v_1s, v_1s, v_1s, v_1s, v_1s, v_1s])
        v_2s_split  = torch.cat([v_2s,  v_2s, v_2s, v_2s, v_2s, v_2s, v_2s, v_2s])
        v_3s_split  = torch.cat([v_3s,  v_3s, v_3s, v_3s, v_3s, v_3s, v_3s, v_3s])
        
        # transform the subparallelepipeds to affine form
        centers, vecs =  get_parallelepiped_coords_in_affine_form_from_base_form(bases_split, v_1s_split, v_2s_split, v_3s_split)
        
        # remove the original parallelepipeds and add the subparallelepipeds
        mask = torch.ones(len(self.centers), dtype=torch.bool)
        mask[indices] = False
        self.centers= self.centers[mask]
        self.vecs = self.vecs[mask]
        self.color_ints = self.color_ints[mask]
        self.color_counter += 1
        
        # add the subparallelepipeds
        self.centers = torch.cat((self.centers, centers))
        self.vecs = torch.cat((self.vecs, vecs))
        self.color_ints = torch.cat((self.color_ints, torch.ones(len(centers), dtype=torch.int) * self.color_counter))
    
    def _shuffle_data(self):
        """ Shuffles the data in the dataloader."""
        number_of_rows = len(self.centers)
        random_permutation_of_indices = torch.randperm(number_of_rows)
        
        self.centers    = self.centers   [random_permutation_of_indices]
        self.vecs       = self.vecs      [random_permutation_of_indices]
        self.color_ints = self.color_ints[random_permutation_of_indices]

class Parallelepiped():
    def __init__(self, base, v_1, v_2, v_3):
        self.base = base
        self.v_1 = v_1
        self.v_2 = v_2
        self.v_3 = v_3
        self.p_1 = self.base + self.v_1
        self.p_2 = self.base + self.v_2
        self.p_3 = self.base + self.v_3
        self.p_4 = self.p_1 + self.v_2
        

def get_filling_parallelepipeds_for_multiple_tetrahedra(vertices, tetrahedra):
    # get filling parallelepipeds for each tetrahedron
    bases, v_1s, v_2s, v_3s = [], [], [], []
    for tetrahedron in tetrahedra:
        tet_vertices = vertices[tetrahedron]
        tet_bases, tet_v_1s, tet_v_2s, tet_v_3s= get_filling_parallelepipeds_for_one_tetrahedron(tet_vertices)
        bases.append(tet_bases)
        v_1s.append(tet_v_1s)
        v_2s.append(tet_v_2s)
        v_3s.append(tet_v_3s)
    
    # concatenate the results to tensors
    dtype = torch.float32
    bases = torch.tensor(np.concatenate(bases), dtype=dtype)
    v_1s  = torch.tensor(np.concatenate(v_1s),  dtype=dtype)
    v_2s  = torch.tensor(np.concatenate(v_2s),  dtype=dtype)
    v_3s  = torch.tensor(np.concatenate(v_3s),  dtype=dtype)
    return bases, v_1s, v_2s, v_3s


def get_filling_parallelepipeds_for_one_tetrahedron(vertices):
    # get points of the parallelepiped
    p_1 = vertices[0]
    p_2 = vertices[1]
    p_3 = vertices[2]
    p_4 = vertices[3]
    
    # add corner parallelepipeds
    pe_1 = Parallelepiped(p_1, (p_2-p_1)/3, (p_3-p_1)/3, (p_4-p_1)/3)
    pe_2 = Parallelepiped(p_2, (p_1-p_2)/3, (p_3-p_2)/3, (p_4-p_2)/3)
    pe_3 = Parallelepiped(p_3, (p_1-p_3)/3, (p_2-p_3)/3, (p_4-p_3)/3)
    pe_4 = Parallelepiped(p_4, (p_1-p_4)/3, (p_2-p_4)/3, (p_3-p_4)/3)
    
    # Add parallelepipeds at the edges

    # 1. parallelepiped at edge p_1-p_2
    base = pe_1.base + pe_1.v_1
    v_1 = pe_1.v_1
    v_2 = (pe_3.base + pe_3.v_1 - base)/2
    v_3 = (pe_4.base + pe_4.v_1 - base)/2
    pe_1_2 = Parallelepiped(base, v_1, v_2, v_3)

    # 2. parallelepiped at edge p_2-p_3
    base = pe_2.base + pe_2.v_2
    v_1 = pe_2.v_2
    v_2 = (pe_1.base + pe_1.v_1 - base)/2
    v_3 = (pe_4.base + pe_4.v_2 - base)/2
    pe_1_3 = Parallelepiped(base, v_1, v_2, v_3)

    # 3. parallelepiped at edge p_3-p_4
    base = pe_3.base + pe_3.v_3
    v_1 = pe_3.v_3
    v_2 = (pe_2.base + pe_2.v_2 - base)/2
    v_3 = (pe_1.base + pe_1.v_2 - base)/2
    pe_1_4 = Parallelepiped(base, v_1, v_2, v_3)

    #4. parallelepiped at edge p_1-p_4
    base = pe_4.base + pe_4.v_1
    v_1 = pe_4.v_1
    v_2 = (pe_3.base + pe_3.v_3 - base)/2
    v_3 = (pe_2.base + pe_2.v_3 - base)/2
    pe_1_5 = Parallelepiped(base, v_1, v_2, v_3)

    # 5. parallelepiped at edge p_2-p_4
    base = pe_2.base + pe_2.v_3
    v_1 = pe_2.v_3
    v_2 = (pe_1.base + pe_1.v_1 - base)/2
    v_3 = (pe_3.base + pe_3.v_2 - base)/2
    pe_2_4 = Parallelepiped(base, v_1, v_2, v_3)

    # 6. parallelepiped at edge p_1-p_3
    base = pe_1.base + pe_1.v_2
    v_1 = pe_1.v_2
    v_2 = (pe_4.base + pe_4.v_1 - base)/2
    v_3 = (pe_2.base + pe_2.v_1 - base)/2
    pe_3_1 = Parallelepiped(base, v_1, v_2, v_3)
    
    bases = np.stack([pe_1.base, pe_2.base, pe_3.base, pe_4.base, pe_1_2.base, pe_1_3.base, pe_1_4.base, pe_1_5.base, pe_2_4.base, pe_3_1.base])
    v_1s  = np.stack([pe_1.v_1,  pe_2.v_1,  pe_3.v_1,  pe_4.v_1,  pe_1_2.v_1,  pe_1_3.v_1,  pe_1_4.v_1,  pe_1_5.v_1,  pe_2_4.v_1,  pe_3_1.v_1])
    v_2s  = np.stack([pe_1.v_2,  pe_2.v_2,  pe_3.v_2,  pe_4.v_2,  pe_1_2.v_2,  pe_1_3.v_2,  pe_1_4.v_2,  pe_1_5.v_2,  pe_2_4.v_2,  pe_3_1.v_2])
    v_3s  = np.stack([pe_1.v_3,  pe_2.v_3,  pe_3.v_3,  pe_4.v_3,  pe_1_2.v_3,  pe_1_3.v_3,  pe_1_4.v_3,  pe_1_5.v_3,  pe_2_4.v_3,  pe_3_1.v_3])
    return bases, v_1s, v_2s, v_3s
