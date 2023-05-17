import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime, localtime
import torch
from tqdm import tqdm
from utils import get_3d_angle_torch
import csv


class Trainer:
    """
    Trainer class, used to train the neural networks.
    """

    def __init__(
        self,
        training_loader,
        validation_loader,
        loss_function_classification,
        device,
        model,
        optimizer,
        ROOT_DIR_PATH,
        MODEL_INFO_FILE_NAME,
        FEATURE_EXTRACTION_FILE_NAME,
        patience,
        training_name,
        training_session_dir_name,
        ratio_loss_doa,
    ):
        """
        Constructs all the necessary attributes for the Trainer class.
        Args:
            training_loader (Dataloader): Dataloader that returns audio signals classification labels and doa labels of
                the training dataset.
            validation_loader (Dataloader): Dataloader that returns audio signals classification labels and doa labels
                of the validation dataset.
            loss_function_classification (Function): Loss function used to compute the training and validation losses
                for classification.
            device (string): Device on which to perform the computations.
            model (Module): Neural network to be trained.
            optimizer (Optimizer): Optimizer used to update the weights during training.
            ROOT_DIR_PATH (string): Directory of project root. Will save the training session file in the subfolder : training_sessions
            MODEL_INFO_FILE_NAME (string): Filename of the model to load/save.
        """
        # Training related objects and variables
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function_classification = loss_function_classification
        self.device = device
        self.model = model
        self.optimizer = optimizer

        self.min_validation_loss = np.inf
        self.patience = patience

        self.ratio_loss_doa = ratio_loss_doa
        self.last_saved_model_index = 0

        # Path and names
        self.ROOT_DIR_PATH = ROOT_DIR_PATH
        self.MODEL_INFO_FILE_NAME = MODEL_INFO_FILE_NAME
        self.FEATURE_EXTRACTION_FILE_NAME = FEATURE_EXTRACTION_FILE_NAME
        self.training_name = training_name
        self.training_dir = training_session_dir_name

        # Graph buffers

        self.validation_losses = []
        self.validation_accuracy = []
        self.validation_doa_losses = []
        self.validation_prediction_losses = []
        self.validation_angle_error = []

        self.training_losses = []
        self.training_accuracy = []
        self.training_doa_losses = []
        self.training_prediction_losses = []
        self.training_angle_error = []

    def train_loop(self, nb_epochs):
        """
        Main method of the class, used to train the model.
        Args:
            nb_epochs (int): Number of epoch for which to train the model.
        """
        steps_without_change = 0
        for epoch in range(nb_epochs):

            # forward pass for training and validation
            (
                training_loss,
                training_accuracy,
                training_metric_doa,
                training_doa_loss,
                training_prediction_loss,
            ) = self.calculate_training_loss()
            (
                validation_loss,
                validation_accuracy,
                validation_metric_doa,
                validation_doa_loss,
                validation_prediction_loss,
            ) = self.calculate_validation_loss()

            training_angle_error = training_metric_doa * 180 / torch.pi
            validation_angle_error = validation_metric_doa * 180 / torch.pi

            self.save_epoch_raw(epoch,
                                training_accuracy*100,training_loss,training_doa_loss,training_prediction_loss,training_angle_error,
                                validation_accuracy*100,validation_loss,validation_doa_loss,validation_prediction_loss,validation_angle_error)

            epoch_stats = (
                f"Epoch {epoch:0>4d} | "
                f"{self.min_validation_loss:.6f}--->"
                f"{validation_loss:.6f} | "
                f"Train accuracy {training_accuracy*100:.2f}% - "
                f"Validation accuracy {validation_accuracy*100:.2f}% | "
                f"DoA train angle error {training_angle_error:.6f} - "
                f"DoA validation angle error {validation_angle_error:.6f}"
            )

            if self.min_validation_loss > validation_loss:
                epoch_stats = epoch_stats + (
                    "  | Min validation loss decreased("
                    f"{self.min_validation_loss:.6f}--->"
                    f"{validation_loss:.6f}): Saved the model"
                )
                self.min_validation_loss = validation_loss
                # save a copy in the Root folder and also in the training dir folder
                self.save_model(self.ROOT_DIR_PATH)
                self.save_model(os.path.join(self.ROOT_DIR_PATH, self.training_dir))
                steps_without_change = 0
                self.last_saved_model_index = epoch
            else:
                steps_without_change += 1
                if steps_without_change > self.patience:
                    print(f"Training stop at epoch number {epoch} , patience set at {self.patience}")
                    break

            print(epoch_stats)

        min_training_loss = min(self.training_losses)
        time_of_completion = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        figure_title = (
            f"{self.training_name}_"
            f"{time_of_completion}_"
            f'min_validation_loss={"{:10.3f}".format(self.min_validation_loss).replace(".","_")}|'
            f'min_training_loss={"{:10.3f}".format(min_training_loss).replace(".","_")}'
        )
        self.update_plot(figure_title)

        print(figure_title)
        plt.close(None)

    def calculate_training_loss(self):
        """
        Compute the training loss for the current epoch.
        Returns:
            float: The total training loss.
            float: The training accuracy.
            float: The doa training loss.
        """
        self.model.train()
        accuracy = 0
        training_loss = 0.0
        total_loss_doa = 0.0
        prediction_loss = 0.0
        total_loss_doa_ratio = 0.0
        for signals, labels, labels_doa in tqdm(self.training_loader, "training", leave=False):
            signals, labels, labels_doa = (
                signals.to(self.device),
                labels.to(self.device),
                labels_doa.to(self.device),
            )
            self.optimizer.zero_grad()

            predictions_class, predictions_doa, _ = self.model(signals)
            loss_class = self.loss_function_classification(predictions_class, labels)

            loss_doa = get_3d_angle_torch(predictions_doa, labels_doa)
            total_loss = loss_class + self.ratio_loss_doa * loss_doa

            total_loss.backward()
            self.optimizer.step()

            training_loss += total_loss
            total_loss_doa += loss_doa
            total_loss_doa_ratio += self.ratio_loss_doa * loss_doa
            prediction_loss += loss_class

            predictions = torch.softmax(predictions_class, dim=1)
            pred_idx = torch.argmax(predictions, dim=-1)
            accuracy += torch.sum(pred_idx == labels) / signals.shape[0]

        return (
            training_loss.item() / len(self.training_loader),
            accuracy.item() / len(self.training_loader),
            total_loss_doa.item() / len(self.training_loader),
            total_loss_doa_ratio.item() / len(self.training_loader),
            prediction_loss.item() / len(self.training_loader),
        )

    def calculate_validation_loss(self):
        """
        Compute the validation loss for the current epoch.
        Returns:
            float: The total validation loss.
            float: The validation accuracy.
            float: The doa validation loss.
            float: The doa validation loss with the doa loss ratio.
            float: The class prediction validation loss.
        """
        with torch.no_grad():
            self.model.eval()
            self.model.feature_extraction.set_bn(True)
            accuracy = 0
            validation_loss = 0.0
            total_loss_doa = 0.0
            prediction_loss = 0.0
            total_loss_doa_ratio = 0.0
            for signals, labels, labels_doa in tqdm(self.validation_loader, "validation", leave=False):
                signals, labels, labels_doa = (
                    signals.to(self.device),
                    labels.to(self.device),
                    labels_doa.to(self.device),
                )

                predictions_class, predictions_doa, _ = self.model(signals)
                loss_class = self.loss_function_classification(predictions_class, labels)

                loss_doa = get_3d_angle_torch(predictions_doa, labels_doa)
                total_loss = loss_class + self.ratio_loss_doa * loss_doa

                validation_loss += total_loss
                total_loss_doa += loss_doa
                total_loss_doa_ratio += self.ratio_loss_doa * loss_doa
                prediction_loss += loss_class

                predictions = torch.softmax(predictions_class, dim=1)
                pred_idx = torch.argmax(predictions, dim=-1)
                accuracy += torch.sum(pred_idx == labels) / signals.shape[0]

            return (
                validation_loss.item() / len(self.validation_loader),
                accuracy.item() / len(self.validation_loader),
                total_loss_doa.item() / len(self.validation_loader),
                total_loss_doa_ratio.item() / len(self.validation_loader),
                prediction_loss.item() / len(self.validation_loader),
            )

    def save_epoch_raw(self,epoch,training_accuracy,training_loss,training_doa_loss,training_prediction_loss,training_angle_error,
                       validation_accuracy,validation_loss,validation_doa_loss,validation_prediction_loss,validation_angle_error):
        """Logs the epoch raw data and add them to their respective array so it can be used when saving the final graph

        Args:
            epoch (int): The epoch number
            training_accuracy (float): the training accuracy
            training_loss (float): the total training loss
            training_doa_loss (float): the training loss for the doa
            training_prediction_loss (float): the training loss for the keyword prediction
            training_angle_error (float): the training angle error in degree
            validation_accuracy (float): the validation accuracy
            validation_loss (float): the total validation loss
            validation_doa_loss (float): the validation loss for the doa
            validation_prediction_loss (float): the validation loss for the keyword prediction
            validation_angle_error (float): the validation angle error in degree
        """
        self.training_accuracy.append(training_accuracy)
        self.training_losses.append(training_loss)
        self.training_doa_losses.append(training_doa_loss)
        self.training_prediction_losses.append(training_prediction_loss)
        self.training_angle_error.append(training_angle_error)

        self.validation_losses.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)
        self.validation_doa_losses.append(validation_doa_loss)
        self.validation_prediction_losses.append(validation_prediction_loss)
        self.validation_angle_error.append(validation_angle_error)
        
        with open(os.path.join(self.ROOT_DIR_PATH, self.training_dir,"raw_data.csv"),"a") as f:
            write =csv.writer(f)
            if epoch ==0: 
                write.writerow([
                            "Epoch",
                            "Training loss","Training accuracy","Training_doa","Training_prediction","Training angle error",
                            "Validation loss","Validation accuracy","Validation doa","TValidation prediction","Validation angle error"])
            write.writerow([
                    epoch,
                    training_loss,training_accuracy,training_doa_loss,training_prediction_loss,training_angle_error,
                    validation_loss,validation_accuracy,validation_doa_loss,validation_prediction_loss,validation_angle_error])
            

    def update_plot(self, figure_title):
        """
        Create the plot for the entire training data
        """
        plt.clf()
        fig, ax = plt.subplots()

        ax.plot(
            range(len(self.training_losses)),
            self.training_losses,
            color="red",
            marker=".",
            label="tr loss",
        )
        ax.plot(
            range(len(self.training_doa_losses)),
            self.training_doa_losses,
            color="orange",
            marker=".",
            label="tr doa loss",
        )
        ax.plot(
            range(len(self.training_prediction_losses)),
            self.training_prediction_losses,
            color="cyan",
            marker=".",
            label="tr pred loss",
        )

        ax.plot(
            range(len(self.validation_losses)),
            self.validation_losses,
            color="blue",
            label="va loss",
        )
        ax.plot(
            range(len(self.validation_doa_losses)),
            self.validation_doa_losses,
            color="green",
            label="va doa loss",
        )
        ax.plot(
            range(len(self.validation_prediction_losses)),
            self.validation_prediction_losses,
            color="yellow",
            label="va pred loss",
        )

        ax2 = ax.twinx()
        ax2.plot(
            range(len(self.validation_accuracy)),
            self.validation_accuracy,
            color="black",
            marker="*",
            label="va accuracy",
        )
        ax2.plot(
            range(len(self.training_accuracy)),
            self.training_accuracy,
            color="red",
            marker="*",
            label="tr accuracy",
        )

        ax2.set_ylabel("accuracy %", color="black", fontsize=10)
        ax.set_ylabel("losses", color="red", fontsize=10)
        ax.set_xlabel("epochs", fontsize=10)
        ax.legend(loc="upper center", prop={"size": 6}, bbox_to_anchor=(0.3, 1))
        ax2.legend(loc="upper center", prop={"size": 6}, bbox_to_anchor=(0.3, 0.78))

        ax.axvline(self.last_saved_model_index)

        fig.savefig(
            os.path.join(self.ROOT_DIR_PATH, self.training_dir, figure_title),
            bbox_inches="tight",
            dpi=200,
        )

    def save_model(self, path):
        """Saves a checkpoint, which contains the model weights and the necessary information to continue the training at
            some other time.
        Args:
            model(nn.Module) : The model that will be saved of the disk
            path(str) : The path where the file will be saved
        """
        save = {
            "model_state_dict": self.model.feature_extraction.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "min_validation_loss": self.min_validation_loss,
        }
        torch.save(save, os.path.join(path, self.FEATURE_EXTRACTION_FILE_NAME))

        save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "min_validation_loss": self.min_validation_loss,
        }
        torch.save(save, os.path.join(path, self.MODEL_INFO_FILE_NAME))

    @staticmethod
    def load_best_model(model, directory, device, model_info_filename):
        """
        Used to get the best version of a model from disk.
        Args:
            model (Module): Model on which to update the weights.
            directory (string): Directory path where the model is located.
            device (string): Device on which the model should be loaded.
            model_info_filename (string): Model name saved on the disk to be loaded.
        """
        checkpoint = torch.load(os.path.join(directory, model_info_filename), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
