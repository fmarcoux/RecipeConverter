from torch import nn
import torch, os
from AudioFeatureExtraction import AudioFeatureExtraction


class AudioClassification(nn.Module):
    def __init__(self, num_class):
        super(AudioClassification, self).__init__()
        self.feature_extraction = AudioFeatureExtraction()

        self.fc = nn.Linear(256, num_class)
        self.fc_doa = nn.Linear(256, 3)

    def forward(self, x_in, h=None):
        """
        Method used to go through the CRNN model to compute a classification and doa estimate.
        Args:
            x_in (tensor): Frequency signal (N x C x T x F)
            h (tensor): If given, the context will be passed to the recurrent neural network layer, else the context is
                set to 0.

        Returns:
            out_classification (tensor): Prediction on the class for the given signal.
            out_doa (tensor): Prediction on the doa for the given signal.
            h (tensor): The updated context to be passed to the next iteration if needed.
        """
        out, h = self.feature_extraction.forward(x_in=x_in, h=h)

        out_classification = self.fc(out)[:, -1, :]

        out_doa = self.fc_doa(out)[:, -1, :]
        out_doa = out_doa / torch.sum(out_doa**2, dim=1).unsqueeze(1).sqrt()

        return out_classification, out_doa, h

    def load_best_model(self, directory, device, classification_file_name):
        """
        Used to get the best version of a model from disk.
        Args:
            directory (string): Directory path where the model is located.
            device (string): Device on which the model should be loaded.
            model_info_filename (string): Model name saved on the disk to be loaded.
        """
        checkpoint = torch.load(os.path.join(directory, classification_file_name), map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])

    def set_custom_eval_mode(self):
        """This is used when exporting a model to ONNX.
        It essentially put the network in evaluation mode except the batch normalisation layer, do not use this when using the pytorch inference.
        It is done this way so that the export onnx function exports correctly the layers settings  
        """
        self.train()
        self.feature_extraction.set_custom_eval()
        