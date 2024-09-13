import os
import torch
from util import apply_limits_to_points, get_points_path, prepare_bounds_for_torch


class PointDataloader():
    def __init__(self, mesh, target_number_of_points, type, batch_size=1, shuffle=True, batch_size_spec=None, outside=True, buffer=False, upper_point=None, lower_point=None, scale_factor=1.0):
        self.mesh = mesh
        self.shuffle = shuffle
        self.type = type
        self.outside = outside
        self.scale_factor = scale_factor

        self.upper_point, self.lower_point = prepare_bounds_for_torch(upper_point, lower_point, mesh)
        self.positions = self.get_positions(target_number_of_points, batch_size_spec, buffer)

        self.batch_size = batch_size

    def __len__(self):
        return len(self.positions) // self.batch_size
    
    def get_num_samples(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx]

    def __iter__(self):
        if self.shuffle:
            self._shuffle_data()

        # Only full batches are returned
        for i in range(len(self)):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            yield self.positions[start:end, :]
            
    def change_num_batches(self, number_of_batches):
        self.batch_size = len(self.positions) // number_of_batches

    def render(self, plt_axes, limits=None):
        # Sorting is not necessary for 3D data
        positions = apply_limits_to_points(self.positions, limits)
        plt_axes.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1., c='g')

    def _shuffle_data(self):
        number_of_rows = self.positions.size()[0]
        random_permutation_of_indices = torch.randperm(number_of_rows)
        self.positions = self.positions[random_permutation_of_indices]


    def get_positions(self, target_number_of_points, batch_size_spec, buffer):
        if buffer:
            # Check if points are in cache
            path = get_points_path(prefix= "points", mesh=self.mesh, lower_point=self.lower_point, upper_point=self.upper_point, type=self.type, number_of_points=target_number_of_points, outside=self.outside, scale_factor=self.scale_factor)
            print("Trying to load points from cache. File:", path)
            if os.path.exists(path):
                print("Loading points from cache. File:", path)
                data = torch.load(path)
                positions = data["positions"]
                return positions
            
        print("Generating points")
        
        # Generate desired number_of_points points that are outside the tet mesh
        pos_list = []
        found_points = 0
        num_points_per_step = 100000
        while found_points < target_number_of_points:
            positions = torch.rand(num_points_per_step, 3) * (self.upper_point - self.lower_point) + self.lower_point
            with torch.no_grad():
                label = self.mesh.are_points_inside(positions, batch_size_spec=batch_size_spec, epsilon=0.0)
            
            if self.outside:
                positions = positions[(label.cpu() > 0.0), :]
            else:
                positions = positions[(label.cpu() <= 0.0), :]
            pos_list.append(positions)
            found_points += len(positions)
            print("Found", found_points, "points of ", target_number_of_points, "total.", end="\r")
        print("")
        positions = torch.cat(pos_list, dim=0)
        positions  = positions.cpu()

        if buffer:
            torch.save({"positions": positions}, path)

        return positions