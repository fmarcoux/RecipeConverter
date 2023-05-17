import torch
import torchaudio
import numpy as np


class DataTransformation:
    """Used to transform audio file or raw audio data into a input that is valid for the neural net"""

    def __init__(self, sample_rate: int, audio_length_in_seconds: float, n_fft: int  = 1024, hop_length: int = 256) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate
        self.num_samples = int(audio_length_in_seconds * sample_rate)
        self.transformation = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=None, normalized=True
        )
        self.amplitude_to_db =  torchaudio.transforms.AmplitudeToDB(stype="amplitude")

    def from_audio_file(self, audio_path):
        """Generate the Nerual net input from a audio file name
        Args:
            audio_path (string): The file name

        Returns:
            Tensor: The transformed audio data (spectrogram)
        """
        original_signal, _ = torchaudio.load(audio_path)
        return self.from_raw_data(original_signal)

    def from_spectrogram(self, spectrogram):
        """Generate the Nerual net input from a audio spectrogram
        Args:
            spectrogram (Tensor): The unmodified spectrogram

        Returns:
            Tensor: The transformed audio data (spectrogram)
        """
        freq_signal = spectrogram * self._generate_speech_gain_mask(seq_len=spectrogram.shape[2])
        return self.get_nn_inputs_from_frequency_signal(freq_signal)

    def from_raw_data(self, data):
        """Generate the Nerual net input from raw audio data
        Args:
            data (Tensor): The raw audio data

        Returns:
            Tensor: The transformed audio data (spectrogram)
        """
        signal = self.cut(data)
        signal = self.right_pad(signal)
        return self.from_spectrogram(self.transformation(signal))

    def class_certainty(self,class_prediction:torch.Tensor):
        """Used to compute the class predicted and the certainty of the prediction
        Args:
            class_prediction (torch.tensor): the class predictions 
        """
        class_prediction = torch.softmax(class_prediction, dim=1)
        pred = torch.argmax(class_prediction, dim=-1).item()
        certitude = class_prediction[:, pred].item()
        return pred,certitude
    
    def cut(self, signal):
        """
        Cut the signal if longer than the number of samples
        The number of samples is defined from the sample rate and audio length
        Args:
            signal (tensor): Audio signal.

        Returns:
            signal (tensor): Adjusted sequence length signal.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, 0 : self.num_samples]
        return signal

    def right_pad(self, signal):
        """
        Right pad the signal if shorter than the number of samples.
        The number of samples is defined from the sample rate and audio length
        Args:
            signal (tensor): Audio signal.

        Returns:
            signal (tensor): Adjusted sequence length signal.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal

    def _generate_speech_gain_mask(self, seq_len, min_val=0.8, max_val=1.2):
        """
        Generate a random gain mask to be applied on the signal frequencies. This helps to be more flexible on the
        device recording the signal.
        Args:
            seq_len (int): Sequence length of the frequency signal.
            min_val (float): Minimum gain value.
            max_val (float): Maximum gain value.

        Returns:
            speech_gain (ndarray): 3D array containing data with `float` type.
        """
        speech_gain = np.random.uniform(min_val, max_val, (20, 1))
        speech_gain = np.repeat(speech_gain, 26, axis=0)[:513, :]
        speech_gain = np.repeat(speech_gain, seq_len, axis=1)
        speech_gain = np.expand_dims(speech_gain, axis=0)
        speech_gain = np.repeat(speech_gain, 4, axis=0)
        return torch.tensor(speech_gain).float()

    def get_nn_inputs_from_frequency_signal(self,freq_signal):
        """
        Get the neural network inputs from the frequency signal given. This method takes the angle from the 4 channels,
        and the mean energy of the signal and concat them into a tensor.
        Args:
            freq_signal (tensor):  The frequency signal from the .wav file.

        Returns:
            input_rnn (tensor): The 5 channels input tensor to the neural network.
        """
        
        energy = torch.mean(self.amplitude_to_db(torch.abs(freq_signal)), dim=0, keepdim=True)
        signal_c1_im = torch.unsqueeze(torch.angle(freq_signal[0, :]), dim=0)
        signal_c2_im = torch.unsqueeze(torch.angle(freq_signal[1, :]), dim=0)
        signal_c3_im = torch.unsqueeze(torch.angle(freq_signal[2, :]), dim=0)
        signal_c4_im = torch.unsqueeze(torch.angle(freq_signal[3, :]), dim=0)
        input_rnn = torch.cat([energy, signal_c1_im, signal_c2_im, signal_c3_im, signal_c4_im], dim=0)
        return input_rnn.permute(0, 2, 1)
