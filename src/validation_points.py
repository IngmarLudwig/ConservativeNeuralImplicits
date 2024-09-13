import os
import torch
from util import apply_limits_to_points, prepare_bounds_for_torch, get_points_path


class ValidationPoints():
    def __init__(self, mesh, number_of_points, scale_factor, batch_size_spec=None, buffer=False, lower_point=None, upper_point=None):
        self.upper_point, self.lower_point = prepare_bounds_for_torch(upper_point, lower_point, mesh)
        self.scale_factor = scale_factor

        self.mesh = mesh

        self.positions, self.labels = self.get_positions(number_of_points, batch_size_spec, buffer)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.labels[idx]
            
    def render(self, plt_axes, limits=None):
        are_inside = self.labels < 1.
        positions_inside = self.positions[are_inside]
        positions_outside = self.positions[~are_inside]

        positions_inside = apply_limits_to_points(positions_inside, limits)
        positions_outside = apply_limits_to_points(positions_outside, limits)

        plt_axes.scatter(positions_inside[:, 0], positions_inside[:, 1], positions_inside[:, 2], s=1., c='r', label='inside')
        plt_axes.scatter(positions_outside[:, 0], positions_outside[:, 1], positions_outside[:, 2], s=1., c='b', label='outside')

        plt_axes.legend()

    def get_positions(self, number_of_points, batch_size_spec, buffer):
        if buffer:
            # Check if points are cache
            path = get_points_path(prefix= "points_validation", mesh=self.mesh, lower_point=self.lower_point, upper_point=self.upper_point, number_of_points=number_of_points, scale_factor=self.scale_factor)            
            print("Trying to load points from cache. File:", path)
            if os.path.exists(path):
                print("Found points in cache. Loading.")
                data = torch.load(path)
                positions = data["positions"]
                label = data["label"]
                return positions, label
            print("No points in cache found.")
        
        print("Generating validation points")
        
        # Generate desired number_of_points points that are outside the mesh
        pos_list = []
        label_list = []
        found_points = 0
        num_points_per_step = 100000
        while found_points < number_of_points:
            positions = torch.rand(num_points_per_step, 3) * (self.upper_point - self.lower_point) + self.lower_point
            with torch.no_grad():
                label = self.mesh.are_points_inside(positions, batch_size_spec=batch_size_spec, epsilon=0.0)
            pos_list.append(positions)
            label_list.append(label)
            found_points += len(positions)
            print("Found", found_points, "points of ", number_of_points, "total.", end="\r")
        print("")
        positions = torch.cat(pos_list, dim=0)
        label = torch.cat(label_list, dim=0)
        positions  = positions.cpu()
        label = label.cpu()
        
        if buffer:
            torch.save({"positions": positions, "label": label}, path)
        
        return positions, label
    