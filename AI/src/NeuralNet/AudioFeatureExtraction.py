import os
import torch
from torch import nn


class AudioFeatureExtraction(nn.Module):
    """
    Model of the neural network that predicts the class and the doa on a Google dataset.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the audio feaure exatrction.
        """
        super(AudioFeatureExtraction, self).__init__()
        self.RNN = nn.GRU(input_size=64, num_layers=3, hidden_size=256, batch_first=True, dropout=0.5)

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d((1, 8))
        self.maxpool2 = nn.MaxPool2d((1, 8))
        self.maxpool3 = nn.MaxPool2d((1, 4))
        self.maxpool4 = nn.MaxPool2d((1, 2))

    def forward(self, x_in, h=None):
        """
        Method used to go through the CRNN model to compute a classification and doa estimate.
        Args:
            x_in (tensor): Frequency signal (N x C x T x F)
            h (tensor): If given, the context will be passed to the recurrent neural network layer, else the context is
                set to 0.

        Returns:
            out(tensor) : The neural network output after the rnn of size (1,256)
            h (tensor): The updated context to be passed to the next iteration if needed.
        """
        x1 = self.maxpool1(torch.relu(self.bn1(self.conv1(x_in))))
        x2 = self.maxpool2(torch.relu(self.bn2(self.conv2(x1))))
        x3 = self.maxpool3(torch.relu(self.bn3(self.conv3(x2))))
        x4 = self.maxpool4(torch.relu(self.bn4(self.conv4(x3))))

        x_rnn = torch.squeeze(x4, dim=-1).permute(0, 2, 1)

        if h is not None:
            out, h = self.RNN(x_rnn, h)
        else:
            out, h = self.RNN(x_rnn)

        return out, h

    def set_bn(self, is_train):
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train() if is_train else module.eval()
    
    def set_custom_eval(self):
        self.train()
        self.set_dropout()
        for module in self.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    
    def set_dropout(self):
        self.RNN.dropout=0

    def load_best_model(self, directory, device, feature_extraction_file_name):
        """
        Used to get the best version of a model from disk.
        Args:
            directory (string): Directory path where the model is located.
            device (string): Device on which the model should be loaded.
            model_info_filename (string): Model name saved on the disk to be loaded.
        """
        checkpoint = torch.load(os.path.join(directory, feature_extraction_file_name), map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
