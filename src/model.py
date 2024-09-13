from copy import deepcopy
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from util import get_device, ensure_tensor_and_batched, ensure_is_tensor, ensure_tensor_is_batch_of_2d_tensors, get_parallelepiped_coords_in_affine_form_from_base_form
from util import get_parallelepiped_coords_from_boxes, apply_limits_to_points, get_points_path
from constants import SIGN_OUTSIDE, SIGN_INSIDE, SIGN_UNKNOWN
from relu_layer import ReluLayer
from squeeze_layer import SqueezeLayer
from dense_layer import DenseLayer
from model_util import calculate_range


class Model(torch.nn.Module):


    ############## Setup Model ##############

    def __init__(self, lower_point, upper_point, path=None, layer_width=None, num_hidden_layers=None, affine_dataloader=None):
        super().__init__()
        assert upper_point is not None and lower_point is not None or upper_point is None and lower_point is None, "Either both or none of upper_point and lower_point must be given"
        lower_point = ensure_is_tensor(lower_point)
        upper_point = ensure_is_tensor(upper_point)
        self.upper_point = upper_point
        self.lower_point = lower_point

        is_random_init = path is None and layer_width is not None and num_hidden_layers is not None
        is_load_from_file = path is not None and layer_width is None and num_hidden_layers is None
        if is_load_from_file:
            layers = self._load_layer_from_npz_file(path)
            for layer in layers:
                assert not layer.check_for_nan_weights(), "weights in the provided file contain nan"
        elif is_random_init:
            num_layers_pytorch_notation = num_hidden_layers - 1
            layers = self._create_random_initialized_layers(layer_width, num_layers_pytorch_notation)
        else:
            raise ValueError("Either path or layer_width and num_middle_layers must be given.")
        
        self.layers = torch.nn.ModuleList(layers)
            
        if affine_dataloader is not None:
            lower_limit, upper_limit = self.calculte_model_limits_with_affine_dataloader(affine_dataloader)
            if upper_limit > 0:
                # use epsilon to prevent numerical errors
                epsilon = 2e-5
                last_layer = layers[-2]
                last_layer.add_to_bias(-1.*upper_limit- epsilon)

    def _create_random_initialized_layers(self, layer_width, num_middle_layers):
        """ Creates the random initialized layers for the model."""
        layers = [DenseLayer(size_in=3, size_out=layer_width), ReluLayer()]
        for _ in range(num_middle_layers):
            layers.extend([DenseLayer(size_in=layer_width, size_out=layer_width), ReluLayer()])
        layers.extend([DenseLayer(size_in=layer_width, size_out=1), SqueezeLayer()])
        return layers


    ############## Model Limits Functions ##############
    
    def calculte_model_limits(self):
        """ Calculate the maximal and minimal values of the model for the given input."""
        self.eval()
        with torch.no_grad():
            lower_point = torch.tensor([-1., -1., -1.])
            upper_point = torch.tensor([1., 1., 1.])
            lower, upper = self.classify_box(lower_point, upper_point, soft=True)
        return lower.item(), upper.item()


    def calculte_model_limits_with_affine_dataloader(self, affine_dataloader):
        """ Calculate the maximal and minimal values of the model for the given input."""
        device = get_device()
        self.to(device)
        
        self.eval()
        
        with torch.no_grad():
            lower_limit = float('inf')
            upper_limit = float('-inf')
            for centers, face_vecs in affine_dataloader:
                centers, face_vecs = centers.to(device), face_vecs.to(device)
                may_lower, may_upper = self.classify_general_box(centers, face_vecs, soft=True)
                
                min_val = may_lower.min().item()
                max_val = may_upper.max().item()
                
                if min_val < lower_limit:
                    lower_limit = min_val
                    
                if max_val > upper_limit:
                    upper_limit = max_val
        return lower_limit, upper_limit



    ############## Forward Functions ##############

    def forward(self, possibly_affine_input):
        """ Compute the output of the model. Negative values indicate inside, positive values indicate outside. Accepts affine inputs (base,aff,err) or constant inputs (base).)"""
        for layer in self.layers:
            possibly_affine_input = layer.forward(possibly_affine_input)
        return possibly_affine_input
    
    def are_points_inside(self, points):
        """ 
            Determine if the points are inside the shape. Evaluation version of forward (without gradient calculation). 
            Negative values indicate inside, positive values indicate outside.
        """
        points = ensure_tensor_and_batched(points)
        self.eval()
        with torch.no_grad():
            output = self.forward(points)
        return output

    def classify_box(self, box_lower, box_upper, soft=False):
        """ 
            Determines whether the whole volume given by the box can be classified as inside or outside. If soft=False, it reports one of SIGN_UNKNOWN, SIGN_INSIDE, SIGN_OUTSIDE.
            If the soft=True, the function returns the lower and upper bounds of the function within the box. 
        """
        box_lower, box_upper = ensure_tensor_and_batched(box_lower), ensure_tensor_and_batched(box_upper)
        base, v1, v2, v3 = get_parallelepiped_coords_from_boxes(box_lower, box_upper)
        return self.classify_parallelepiped(base, v1, v2, v3, soft=soft)

    def classify_parallelepiped(self, base, v_1, v_2, v_3, soft=False):
        """ 
            Determines whether the whole volume given by the parallelepiped can be classified as inside or outside. If soft=False, it reports one of SIGN_UNKNOWN, SIGN_INSIDE, SIGN_OUTSIDE.
            If the soft flag is set, the function returns the lower and upper bounds of the function within the box.
        """
        base, v_1, v_2, v_3 = ensure_tensor_and_batched(base), ensure_tensor_and_batched(v_1), ensure_tensor_and_batched(v_2), ensure_tensor_and_batched(v_3)
        centers, box_vecs = get_parallelepiped_coords_in_affine_form_from_base_form(base, v_1, v_2, v_3)
        return self.classify_general_box(centers, box_vecs, soft=soft)

    def classify_general_box(self, centers, vecs, soft=False):
        """
            Classify the whole volume given by a general box (a parallelepiped given in the form: center, vectors pointing to the center of the faces from the center).
            If soft=False, it reports one of SIGN_UNKNOWN, SIGN_INSIDE, SIGN_OUTSIDE.
            If the soft flag is set, the function returns the lower and upper bounds of the function within the box.
        """
        centers = ensure_tensor_and_batched(centers)
        vecs = ensure_is_tensor(vecs)
        vecs = ensure_tensor_is_batch_of_2d_tensors(vecs)

        num_boxes = vecs.shape[0]
        assert centers.shape == (num_boxes, 3) or centers.shape == (3,),  "bad centers shape. Is: " + str(centers.shape)
        assert vecs.shape == (num_boxes, 3, 3)or vecs.shape == (3,3), "bad vecs shape. Is: " + str(vecs.shape)

        # evaluate the function
        affine_input = self.affine_input_from_general_box(centers, vecs)
        output = self.forward(affine_input)
    
        # compute relevant bounds
        may_lower, may_upper = calculate_range(output)
        
        if soft:
            return may_lower, may_upper

        # determine the type of the region
        output_type = torch.full_like(may_lower, SIGN_UNKNOWN)
        output_type = torch.where(may_lower >  0., SIGN_OUTSIDE, output_type)
        output_type = torch.where(may_upper <= 0., SIGN_INSIDE,  output_type)

        return output_type
    
    # Construct affine inputs for the coordinates in k-dimensional box, which is not necessarily axis-aligned
    #  - center is the center of the box
    #  - vecs is a (V,D) array of vectors which point from the center of the box to its
    #    faces. These will correspond to each of the affine symbols, with the direction
    #    of the vector becoming the positive orientation for the symbol.
    # (this function is nearly a no-op, but giving it this name makes it easier to
    #  reason about)
    @staticmethod
    def affine_input_from_general_box(centers, vecs):
        base = centers
        aff = vecs
        err = torch.zeros_like(centers)
        return base, aff, err
    

    ############## Info Functions ##############

    def check_for_nan_weights(self):
        for layer in self.layers:
            assert not layer.check_for_nan_weights(), "weights in the provided file contain nan"
        print("No nan weights found.")

    def get_device(self):
        """ Returns the device of the model."""
        return next(self.parameters()).device

    def get_number_of_parameters(self):
        """ Returns the number of learnable parameters of the model."""
        return sum(p.numel() for p in self.parameters())
    
    def print_max_min_weights(self):
        for layer in self.layers:
            layer.print_max_min_weights()
    

    ############## Saving and Loading Functions ##############
    
    def save_as_npz_file(self, path):
        """ Save the model as an npz file (compressed numpy)."""
        layer_dict = {}
        cnt = 0
        for layer_tuple in enumerate(self.layers):
            _, layer = layer_tuple
            if type(layer) is DenseLayer:
                layer_dict, cnt = self._add_dense_layer_to_layer_dict(layer, layer_dict, cnt)
            elif type(layer) is ReluLayer:
                layer_dict, cnt = self._add_relu_to_layer_dict(layer_dict, cnt)
            elif type(layer) is SqueezeLayer:
                layer_dict, cnt = self._add_squeeze_last_to_layer_dict(layer_dict, cnt)
        np.savez_compressed(path, **layer_dict)

    @staticmethod
    def _add_dense_layer_to_layer_dict(dense_layer, layer_dict, cnt):
        weights = dense_layer.weights
        weights = weights.detach().cpu().numpy()
        key = f"{cnt:04d}.dense.A"
        layer_dict[key] = weights

        bias = dense_layer.bias
        bias = bias.squeeze()
        bias = bias.detach().cpu().numpy()
        key = f"{cnt:04d}.dense.b"
        if len(bias.shape) == 0:
            bias = np.array([bias])
        layer_dict[key] = bias

        cnt += 1
        return layer_dict, cnt

    @staticmethod
    def _add_relu_to_layer_dict(layer_dict, cnt):
        key = f"{cnt:04d}.relu._"
        empty_value = np.array([])
        layer_dict[key] = empty_value
        cnt += 1
        return layer_dict, cnt

    @staticmethod
    def _add_squeeze_last_to_layer_dict(layer_dict, cnt):
        key = f"{cnt:04d}.squeeze_last._"
        empty_value = np.array([])
        layer_dict[key] = empty_value
        cnt += 1
        return layer_dict, cnt

    def _load_layer_from_npz_file(self, path):
        """ Loads the layers from an npz file (compressed numpy)."""
        self.path = path
        layer_info_dict = {}
        with np.load(path) as data:
            for key, val in data.items():
                layer_info_dict[key] = torch.tensor(val)
        layers = []
        for i_op in range(self._get_number_of_layers(layer_info_dict)):
            # weights_and_biases is a dict with keys "A" and "b" if dense layer, empty otherwise. The name gives the layer type.
            name, weights_and_biases = self._get_layer_data(layer_info_dict, i_op)
            if name == "dense":
                in_width = weights_and_biases["A"].shape[0]
                out_width = weights_and_biases["A"].shape[1]
                layer = DenseLayer(size_in=in_width, size_out=out_width, weights_input=weights_and_biases["A"], bias_input=weights_and_biases["b"])
            elif name == "relu":
                layer = ReluLayer()
            elif name == "squeeze_last":
                layer = SqueezeLayer()
            else:
                raise ValueError(f"Unrecognized layer name {name}")
            layers.append(layer)
        return layers
    
    @staticmethod
    def _get_number_of_layers(layer_info_dict):
        number_of_layers = 0
        for key in layer_info_dict:
            # In the layer info dict the key is layernumber.layertype.parametername, e.g. 0000.dense.A. A is weights, b is bias.
            layer_number = int(key.split(".")[0])
            number_of_layers = max(number_of_layers, layer_number + 1)
        return number_of_layers

    @staticmethod
    def _get_layer_data(layer_info_dict, layer_number):
        layer_number_with_zeros_str = f"{layer_number:04d}"
        name = ""
        args = {}
        for key in layer_info_dict:
            if key.startswith(layer_number_with_zeros_str):
                tokens = key.split(".")
                name = tokens[1]
                # Dense layers have two arguments, A and b. These are given in different params.
                # Other layers have no arguments.
                if name == "dense":
                    argname = tokens[2]
                    args[argname] = layer_info_dict[key]
        return name, args

    
    ############## Rendering Functions ##############
    
    def render_with_points(self, plt_axes, num_points, limits=None, fixed_x=None, fixed_y=None, fixed_z=None, set_view_axis=None, show_all=False, value_limits=None, show_outside_only=False):
        # create random points
        random_points = torch.rand(num_points, 3) * (self.upper_point - self.lower_point) + self.lower_point
        
        # set axis to fixed value if given
        if fixed_x is not None:
            random_points[:, 0] = fixed_x
        if fixed_y is not None:
            random_points[:, 1] = fixed_y
        if fixed_z is not None:
            random_points[:, 2] = fixed_z

        if set_view_axis is not None:
            if set_view_axis == "x":
                plt_axes.view_init(elev=0, azim=0)
            elif set_view_axis == "y":
                plt_axes.view_init(elev=0, azim=90)
            elif set_view_axis == "z":
                plt_axes.view_init(elev=90, azim=0)
        
        # move to device
        device = get_device()
        random_points = random_points.to(device)
        
        # apply limits
        random_points = apply_limits_to_points(random_points, limits)
        
        # classify points
        classifications = self.are_points_inside(random_points)
        
        # remove points that are classified as outside
        if not show_all and not show_outside_only:
            random_points   = random_points  [classifications <= 0]
            classifications = classifications[classifications <= 0]

        if show_outside_only:
            random_points   = random_points  [classifications > 0]
            classifications = classifications[classifications > 0]
            print("number of points classified as outside:", len(random_points))

        if value_limits is not None:
            classifications = torch.clamp(classifications, value_limits[0], value_limits[1])

        random_points = random_points.cpu()
        classifications = classifications.cpu()

        if len(random_points) <= 0:
            print("No points to render.")
            return float('nan'), float('nan')

        # render using classifications for color with heatmap
        im = plt_axes.scatter(random_points[:, 0], random_points[:, 1], random_points[:,2], c=classifications, s=1, cmap=matplotlib.cm.jet)
        # add colorbar
        plt.colorbar(im, fraction=0.025, pad=0.04)

        return classifications.max().item(), classifications.min().item()
                
    def render_difference_to_tet_mesh(self, num_points, tet_mesh, plt_axes, fixed_x=None, fixed_y=None, fixed_z=None, set_view_axis=None):
        # create random points
        random_points = torch.rand(num_points, 3) * (self.upper_point - self.lower_point) + self.lower_point
        
        # set axis to fixed value if given
        if fixed_x is not None:
            random_points[:, 0] = fixed_x
        if fixed_y is not None:
            random_points[:, 1] = fixed_y
        if fixed_z is not None:
            random_points[:, 2] = fixed_z
            
        # move to device
        device = get_device()
        random_points = random_points.to(device)

        # get Points, that are classified as inside the model and their classification values
        model_classification = self.are_points_inside(random_points)
        random_points = random_points[model_classification <= 0]
        model_classification = model_classification[model_classification <= 0]

        # of these, get Points, that are classified as outside by the tet_mesh and their model classification values
        tet_mesh_classification = tet_mesh.are_points_inside(random_points)
        random_points = random_points[tet_mesh_classification > 0]
        model_classification = model_classification[tet_mesh_classification > 0]

        # scatter with model classification values
        random_points = random_points.cpu()
        model_classification = model_classification.cpu()
        im = plt_axes.scatter(random_points[:, 0], random_points[:, 1], random_points[:,2], c=model_classification, s=0.1, cmap=matplotlib.cm.jet)

        # add colorbar
        plt.colorbar(im,fraction=0.025, pad=0.04)

        if set_view_axis is not None:
            if set_view_axis == "x":
                plt_axes.view_init(elev=0, azim=0)
            elif set_view_axis == "y":
                plt_axes.view_init(elev=0, azim=90)
            elif set_view_axis == "z":
                plt_axes.view_init(elev=90, azim=0)
                
    def show_difference_to_mesh_2d(self, plt_axes, num_points, axis, mesh_to_show_difference_to, scale_factor, axis_value=0.0,  buffer=False, legend=True):
        positions = None
        label = None
        
        if buffer:
            # Check if points are cache
            path = get_points_path(prefix= "axis_points", mesh=mesh_to_show_difference_to, lower_point=self.lower_point, upper_point=self.upper_point, number_of_points=num_points, axis=axis, axis_value=axis_value, scale_factor=scale_factor)       
            print("Trying to load points from cache. File:", path)
            if os.path.exists(path):
                print("Found points in cache. Loading.")
                data = torch.load(path)
                positions = data["positions"]
                label = data["label"]
        
        if positions is None or label is None:
            print("Generating points")
            
            # create random points
            positions = torch.rand(num_points, 3) * (self.upper_point - self.lower_point) + self.lower_point
            
            # set axis to fixed value if given
            if axis=="x":
                positions[:, 0] = axis_value
            if axis=="y":
                positions[:, 1] = axis_value
            if axis=="z":
                positions[:, 2] = axis_value
            
            label = mesh_to_show_difference_to.are_points_inside(positions)

            if buffer:
                print("Saving points to cache.")
                torch.save({"positions": positions, "label": label}, path)
        
        # classify points
        device = get_device()
        positions = positions.to(device)
        classifications = self.are_points_inside(positions)

        positions = positions.cpu()
        classifications = classifications.cpu()
        label = label.cpu()

        inside_positions = positions[label <= 0]
        inside_classifications = classifications[label <= 0]
        false_negatives  = inside_positions[inside_classifications > 0] 


        outside_positions = positions[label > 0]
        outside_classifications = classifications[label > 0] 
        false_positives = outside_positions[outside_classifications <= 0]

        there_are_false_positives = len(false_positives) > 0
        there_are_false_negatives = len(false_negatives) > 0

        if axis=="x":
            plt_axes.set_xlabel("y")
            plt_axes.set_ylabel("z")
            
            x_values_false_positives = false_positives[:, 1]
            y_values_false_positives = false_positives[:, 2]

            x_values_false_negatives = false_negatives[:, 1]
            y_values_false_negatives = false_negatives[:, 2]
        if axis=="y":
            plt_axes.set_xlabel("x")
            plt_axes.set_ylabel("z")
            
            x_values_false_positives = false_positives[:, 0]
            y_values_false_positives = false_positives[:, 2]
            
            x_values_false_negatives = false_negatives[:, 0]
            y_values_false_negatives = false_negatives[:, 2]

        if axis=="z":
            plt_axes.set_xlabel("x")
            plt_axes.set_ylabel("y")

            x_values_false_positives = false_positives[:, 0]
            y_values_false_positives = false_positives[:, 1]

            x_values_false_negatives = false_negatives[:, 0]
            y_values_false_negatives = false_negatives[:, 1]


        if not there_are_false_negatives and not there_are_false_positives:
            print("No false positives or false negatives found.")
            print("No points to render.")
            return

        if not there_are_false_positives:
            print("No false positives found.")
        else:
            print("Number of false positives:", len(false_positives))
            plt_axes.scatter(x_values_false_positives, y_values_false_positives, c='b', s=0.01, label='false positives')
        
        if not there_are_false_negatives:
            print("No false negatives found.")
        else:
            print("Number of false negatives:", len(false_negatives))
            plt_axes.scatter(x_values_false_negatives, y_values_false_negatives, c='r', s=0.01, label='false negatives')

        if legend:
            plt_axes.legend()
    
############## Linear Weight Interpolation Functions ##############

def interpolate_models(correct_model, false_model, affine_dataloader, depth):
    best_model = correct_model

    interpolation_factor = 0.5
    found = False
    for _ in range(depth):
        res_model = _interpolate_models_with_factor(correct_model, false_model, interpolation_factor)
        _, upper_limit = res_model.calculte_model_limits_with_affine_dataloader(affine_dataloader)
        if upper_limit > 0:
            interpolation_factor /= 2
        else:
            interpolation_factor += interpolation_factor / 2
            best_model = res_model
            found = True
    return best_model, found


def _interpolate_models_with_factor(correct_model, false_model, interpolation_factor):
    res_model = deepcopy(correct_model)
    correct_model_device = correct_model.get_device()
    res_model.to(correct_model_device)
    false_model.to(correct_model_device)
    for layer, layer_other in zip(res_model.layers, false_model.layers):
        if type(layer) is DenseLayer:
            layer.interpolate_weights_with_other_layer(layer_other, interpolation_factor)
    return res_model
