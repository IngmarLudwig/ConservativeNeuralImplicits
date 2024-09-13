import time
import os
import torch
import matplotlib.pyplot as plt
from src.combined_loss import CombinedLoss
from src.evaluation import test_with_dataloader_points, test_with_parallelepipeds
from src.util import get_device, make_bold, GetOutOfLoop
from src.model import Model, interpolate_models


class PointTrainer():
    def __init__(self, point_dataloader_outside_train, point_dataloader_inside_train, point_dataloader_outside_test, point_dataloader_inside_test, mesh):
        self.train_dataloader_outside = point_dataloader_outside_train
        self.train_dataloader_inside = point_dataloader_inside_train
        self.test_dataloader_outside = point_dataloader_outside_test
        self.test_dataloader_inside = point_dataloader_inside_test
        self.mesh = mesh
        
        self.loss_fn = CombinedLoss()
        self.device = get_device()
        self.total_losses = []
        self.accuracies = []

    def train(self, model, best_model_path, lrs, num_epochs, max_gradient=None):
        total_time = time.time()

        # Prepare the model for training.
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.) # lr is set in the loop

        epoch = 0
        best_found_acc = 0
        self.total_losses = []
        self.accuracies = []

        for epoch_num, lr in zip(num_epochs, lrs):
            # Set learning rate.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            for _ in range(epoch_num):
                start_time_epoch = time.time()
                epoch += 1
                total_loss = 0
                
                start_time_train = time.time()
                total_loss += self._train_one_epoch_points(model, max_gradient, optimizer)
                self.total_losses.append(total_loss)     
                train_time = time.time() - start_time_train
                
                start_time_test = time.time()
                accuracy_outside, _ = test_with_dataloader_points(model, self.test_dataloader_outside)
                accuracy_inside, _ = test_with_dataloader_points(model, self.test_dataloader_inside, outside_point_dataloader=False)
                accuracy_mean = (accuracy_outside + accuracy_inside) / 2
                test_time = time.time() - start_time_test
                self.accuracies.append(accuracy_mean)

                new_best_found = "False"
                if accuracy_mean > best_found_acc:
                    best_found_acc = accuracy_mean
                    model.save_as_npz_file(best_model_path)
                    new_best_found = make_bold("True")
                epoch_time = time.time() - start_time_epoch
                print(f"Epoch: {epoch}, Loss: {total_loss:.2f}, LR: {lr:.4f}, Accuracy Outside: {accuracy_outside:.2f}, Accuracy Inside: {accuracy_inside:.2f}, Accuracy Mean: {accuracy_mean:.2f}, Time: {epoch_time:.2f}s,, Train Time: {train_time:.2f}s Test Time: {test_time:.2f}s,  New Best: {new_best_found}")

        print("Total Time: ", (time.time() - total_time) / 60, "min")

        return model
    
    def _train_one_epoch_points(self, model, max_gradient, optimizer):
        model.train()

        epoch_loss = 0
        for data_outside, data_inside in zip(self.train_dataloader_outside, self.train_dataloader_inside):
            optimizer.zero_grad()
                    
            data_outside = data_outside.to(self.device)
            res_outside = model.forward(data_outside)
            del data_outside

            data_inside = data_inside.to(self.device)
            res_inside = model.forward(data_inside)
            del data_inside

            combined_loss, _, _ = self.loss_fn(outputs_inside=res_inside, outputs_outside=res_outside)
            del res_inside, res_outside
                    
            combined_loss.backward()

            # Clip the gradients to avoid exploding gradients.
            if max_gradient is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), max_gradient)

            optimizer.step()
            epoch_loss += combined_loss.item()
        return epoch_loss

    def plot_training(self, start_loss_epoch=None, start_acc_epoch=None):
        best_epoch = self.accuracies.index(max(self.accuracies))
        print("Best epoch:", best_epoch)

        plt.plot(self.total_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        min_loss = min(self.total_losses)
        if start_loss_epoch is not None:
            plt.ylim(min_loss, self.total_losses[start_loss_epoch])
        
        plt.scatter(best_epoch, self.total_losses[best_epoch], color='red')
        plt.show()

        plt.plot(self.accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracies')
        
        max_acc = max(self.accuracies)
        if start_acc_epoch is not None:
            plt.ylim(self.accuracies[start_acc_epoch], max_acc)
        
        plt.scatter(best_epoch, self.accuracies[best_epoch], color='red')
        
        plt.show()


class OptimizationTrainer:
    def __init__(self, box_dataloader, train_dataloader_outside, test_dataloader_outside, mesh):
        self.box_dataloader = box_dataloader
        self.train_dataloader_outside = train_dataloader_outside
        self.test_dataloader_outside = test_dataloader_outside
        self.mesh = mesh

        self.loss_fn = CombinedLoss()
        self.device = get_device()

        self.total_losses = []
        self.outside_accuracies = []
        self.lambdas = []
        self.current_lambda = 1.0
    
    def train(self, model, best_model_path, lrs, num_epochs, lambda_start, lambda_increase_factor, max_lambda_increases, max_gradient=None, lambda_reduction_factor=None, reduce_lambda_after_epochs=None):
        total_time = time.time()

        assert lambda_reduction_factor is None or reduce_lambda_after_epochs is not None, "If lambda_reduction_factor is set, reduce_lambda_after_epochs must be set as well."
        
        model.train()
        model.to(self.device)

        # lr is set in the loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)

        self.check_ready_for_affine_training(model)

        # init saved models
        last_model_path = self.init_model_saving(model, best_model_path)
        
        best_percent_outside_correct, test_time = test_with_dataloader_points(model, self.train_dataloader_outside)
        print("Initial outside accuracy:", best_percent_outside_correct, ". Time: {:.2f} s,".format(test_time))

        self.current_lambda = lambda_start
        lambda_increase_counter = 0
        epoch = 0

        self.total_losses = []
        self.outside_accuracies = []
        self.lambdas = []

        for lr, epochs in zip(lrs, num_epochs):
            # set learn rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # train the model the given number of epochs with the given learn rates
            for _ in range(epochs):
                epoch += 1
                    
                # We train the model one epoch. However, if the model enteres a bad state, we will load the last model 
                # and increase lambda and try again, until the training results in a model that is usable.
                while True:
                    train_start_time = time.time()

                    saved = False

                    affine_loss, points_loss, combined_loss = _train_one_epoch_affine(model, max_gradient, optimizer, self.box_dataloader, self.train_dataloader_outside, self.loss_fn, self.current_lambda, self.device)

                    train_time = time.time() - train_start_time

                    # test
                    falsely_classified, test_time_affine = test_with_parallelepipeds(model, self.box_dataloader)
                    percent_outside_right, test_time_points = test_with_dataloader_points(model, self.test_dataloader_outside)
                    
                    # decide
                    if falsely_classified == 0:
                        model.save_as_npz_file(last_model_path)
                        lambda_increase_counter = 0
                        break
                    else:
                        # If the model has falsely classified boxes we load the last model and increase lambda.
                        print("Loading last model")
                        model = Model(self.mesh.lower_point, self.mesh.upper_point, path=last_model_path, affine_dataloader=self.box_dataloader)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        self.current_lambda = self.current_lambda * lambda_increase_factor
                        lambda_increase_counter += 1
                        print("Increasing lambda to {:.4f}".format(self.current_lambda), "Lambda increase counter:", lambda_increase_counter)
                        if lambda_increase_counter > max_lambda_increases:
                            print("Stopping training.")
                            raise GetOutOfLoop("Lambda increased too often:", max_lambda_increases," Times. Stopping training.")
                
                # save the model if it is better then the previous best model
                if percent_outside_right > best_percent_outside_correct:
                    best_percent_outside_correct = percent_outside_right
                    model.save_as_npz_file(best_model_path)
                    saved = True
                    
                self.print_update_console(epoch, lr, saved, affine_loss, points_loss, combined_loss, train_time, test_time_affine, percent_outside_right, test_time_points)
                self.outside_accuracies.append(percent_outside_right)
                self.total_losses.append(combined_loss)
                self.lambdas.append(self.current_lambda)

                if reduce_lambda_after_epochs is not None and epoch % reduce_lambda_after_epochs == 0:
                    old_lambda = self.current_lambda
                    self.current_lambda /= lambda_reduction_factor
                    print("Reducing lambda from {:.4f} to {:.4f}".format(old_lambda, self.current_lambda))
        
        model = Model(self.mesh.lower_point, self.mesh.upper_point, path=last_model_path)
        os.remove(last_model_path)
        
        print("Total Time: ", (time.time() - total_time) / 60, "min")
        return model

    def check_ready_for_affine_training(self, model):
        # Model must not have false negatives at the beginning of the training
        falsely_classified, _ = test_with_parallelepipeds(model, self.box_dataloader)
        assert falsely_classified == 0, "Model must not have false negatives at the beginning of the training. Found: " + str(falsely_classified) + " false negatives."

        # It is ok for the point dataloader to be larger then the affine dataloader, but not the other way around, because all parallelepipeds must be trained, but not neccessary all random points.
        assert len(self.box_dataloader) <= len(self.train_dataloader_outside), "Affine Dataloader must not be smaller then point dataloader " + str(len(self.box_dataloader)) + " and " + str(len(self.train_dataloader_outside))

    def init_model_saving(self, model, best_model_path):
        last_model_path = "last_" + best_model_path
        model.save_as_npz_file(best_model_path)
        model.save_as_npz_file(last_model_path)
        return last_model_path
    
    def print_update_console(self, epoch, lr, saved, affine_loss, points_loss, combined_loss, train_time, test_time_affine, percent_outside_right, test_time_points):
        saved_string = "Not saved"
        if saved:
            saved_string = make_bold("Saved")
        print("epoch: {:4d}".format(epoch), 
                      "train: {:.2f} s,".format(train_time), 
                      "test: {:.2f} s,".format(test_time_affine + test_time_points),  
                      "affine loss: {:.1f}".format(affine_loss),
                      "points loss: {:.1f}".format(points_loss),
                      "non-intersections correct: {:.4f}%".format(percent_outside_right),
                      "Learn Rate: {:.8f}".format(lr), 
                      "lambda: {:.4f}".format(self.current_lambda),
                      "saved:", saved_string)

    def plot_training(self, start_loss_epoch=None, start_acc_epoch=None):
        best_epoch = self.outside_accuracies.index(max(self.outside_accuracies))
        print("Best epoch:", best_epoch)

        plt.plot(self.total_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        min_loss = min(self.total_losses)
        if start_loss_epoch is not None:
            plt.ylim(min_loss, self.total_losses[start_loss_epoch])
        
        plt.scatter(best_epoch, self.total_losses[best_epoch], color='red')
        
        plt.show()

        plt.plot(self.outside_accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Outside Accuracy')
        plt.title('Outside Accuracies')
        max_acc = max(self.outside_accuracies)
        if start_acc_epoch is not None:
            plt.ylim(self.outside_accuracies[start_acc_epoch], max_acc)
        plt.scatter(best_epoch, self.outside_accuracies[best_epoch], color='red')
        plt.show()

        plt.plot(self.lambdas)
        plt.xlabel('Epoch')
        plt.ylabel('Lambda')
        plt.title('Lambdas')
        plt.scatter(best_epoch, self.lambdas[best_epoch], color='red')
        plt.show()


class LambdaReductionTrainer:
    def __init__(self, box_dataloader, train_dataloader_outside, test_dataloader_outside, mesh):
        self.box_dataloader = box_dataloader
        self.train_dataloader_outside = train_dataloader_outside
        self.test_dataloader_outside = test_dataloader_outside
        self.mesh = mesh

        self.total_losses = []
        self.outside_accuracies = []
        self.current_lambda = 1.0
    
    def train(self, model, best_model_path, lrs, num_epochs, lambda_start, max_gradient=None):
        total_time = time.time()

        device = get_device()
        model.train()
        model.to(device)

        last_model_path = "last_" + best_model_path

        # lr is set in the loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)
        loss_fn = CombinedLoss()

        # Model must not have false negatives at the beginning of the training
        falsely_classified, _ = test_with_parallelepipeds(model, self.box_dataloader)
        assert falsely_classified == 0, "Model must not have false negatives at the beginning of the training. Found: " + str(falsely_classified) + " false negatives."

        # It is ok for the point dataloader to be larger then the affine dataloader, but not the other way around, because all parallelepipeds must be trained, but not neccessary all random points.
        assert len(self.box_dataloader) <= len(self.train_dataloader_outside), "Affine Dataloader must not be smaller then point dataloader " + str(len(self.box_dataloader)) + " and " + str(len(self.train_dataloader_outside))

        # init saved models
        model.save_as_npz_file(best_model_path)
        model.save_as_npz_file(last_model_path)
        
        best_percent_outside_correct, _ = test_with_dataloader_points(model, self.train_dataloader_outside)
        print("Initial outside accuracy:", best_percent_outside_correct)

        self.current_lambda = lambda_start
        epoch = 0

        self.total_losses = []
        self.outside_accuracies = []

        for lr, epochs in zip(lrs, num_epochs):
            # set learn rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # train the model the given number of epochs with the given learn rates
            for _ in range(epochs):
                train_start_time = time.time()
                
                epoch += 1

                saved = False
                total_affine_loss = 0
                total_points_loss = 0
                total_combined_loss = 0
                
                for (centers, face_vecs), data_points in zip(self.box_dataloader, self.train_dataloader_outside):
                    optimizer.zero_grad()

                    centers, face_vecs = centers.to(device), face_vecs.to(device)
                    _, may_upper = model.classify_general_box(centers, face_vecs, soft=True)
                    
                    data_points = data_points.to(device)
                    points_output = model(data_points)

                    combined_loss, affine_loss, points_loss = loss_fn(outputs_inside=may_upper, outputs_outside=points_output, current_lambda=self.current_lambda)
                    
                    combined_loss.backward()

                    # Clip gradient to avoid exploding gradients
                    if max_gradient is not None:
                        torch.nn.utils.clip_grad_value_(model.parameters(), max_gradient)

                    optimizer.step()

                    total_affine_loss += affine_loss.item()
                    total_points_loss += points_loss.item()
                    total_combined_loss += combined_loss.item()

                train_time = time.time() - train_start_time

                # test
                falsely_classified, test_time_affine = test_with_parallelepipeds(model, self.box_dataloader)
                percent_outside_right, test_time_points = test_with_dataloader_points(model, self.test_dataloader_outside)
                
                # decide
                if not falsely_classified == 0:
                    model = Model(self.mesh.lower_point, self.mesh.upper_point, path=last_model_path, affine_dataloader=self.box_dataloader)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    print("epoch:", epoch, "Model has classified {} boxes as outside or unknown.".format(falsely_classified), "Loading last model.")
                else:
                    model.save_as_npz_file(last_model_path)

                    # save the model if it is better then the previous best model
                    if percent_outside_right > best_percent_outside_correct:
                        best_percent_outside_correct = percent_outside_right
                        model.save_as_npz_file(best_model_path)
                        saved = True
                        
                    saved_string = "Not saved"
                    if saved:
                        saved_string = make_bold("Saved")
                    print("epoch: {:4d}".format(epoch), 
                          "train: {:.2f} s,".format(train_time), 
                          "test: {:.2f} s,".format(test_time_affine + test_time_points),  
                          "affine loss: {:.1f}".format(total_affine_loss),
                          "points loss: {:.1f}".format(total_points_loss),
                          "non-intersections correct: {:.4f}%".format(percent_outside_right),
                          "Learn Rate: {:.8f}".format(lr), 
                          "lambda: {:.4f}".format(self.current_lambda),
                          "saved:", saved_string)
                    self.total_losses.append(total_combined_loss)
                    self.outside_accuracies.append(percent_outside_right)

        model = Model(self.mesh.lower_point, self.mesh.upper_point, path=last_model_path)
        
        # remove last model
        os.remove(last_model_path)
        
        print("Total Time: ", (time.time() - total_time) / 60, "min")
        return model
        
    def plot_training(self, start_loss_epoch=None, start_acc_epoch=None):
        plt.plot(self.total_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        min_loss = min(self.total_losses)
        if start_loss_epoch is not None:
            plt.ylim(min_loss, self.total_losses[start_loss_epoch])
        plt.show()

        plt.plot(self.outside_accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Outside Accuracy')
        plt.title('Outside Accuracies')
        max_acc = max(self.outside_accuracies)
        if start_acc_epoch is not None:
            plt.ylim(self.outside_accuracies[start_acc_epoch], max_acc)
        plt.show()


def _train_one_epoch_affine(model, max_gradient, optimizer, box_dataloader, train_dataloader_outside, loss_fn, current_lambda, device):
    total_affine_loss = 0
    total_points_loss = 0
    total_combined_loss = 0            

    for (centers, face_vecs), data_points in zip(box_dataloader, train_dataloader_outside):
        optimizer.zero_grad()
        
        centers, face_vecs = centers.to(device), face_vecs.to(device)
        _, may_upper = model.classify_general_box(centers, face_vecs, soft=True)
        del centers, face_vecs
                    
        data_points = data_points.to(device)
        points_output = model(data_points)
        del data_points

        combined_loss, affine_loss, points_loss = loss_fn(outputs_inside=may_upper, outputs_outside=points_output, current_lambda=current_lambda)
        del may_upper, points_output
                    

        combined_loss.backward()

        # Clip gradient to avoid exploding gradients
        if max_gradient is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_gradient)

        optimizer.step()

        total_affine_loss += affine_loss.item()
        total_points_loss += points_loss.item()
        total_combined_loss += combined_loss.item()

    return total_affine_loss, total_points_loss, total_combined_loss
