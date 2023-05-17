import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from datetime import datetime

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(root, "NeuralNet"))

from Dataset import AcousticDataset
from AudioClassification import AudioClassification
from Trainer import Trainer
from DataTransformation import DataTransformation

load_dotenv(os.path.join(root, ".env"))  # loading local variables

GOOGLE_DATASET_PATH = os.getenv("GOOGLE_DATASET_PATH")
CUSTOM_DATASET_PATH = os.getenv("CUSTOM_DATASET_PATH")
GOOGLE_MODEL_INFO_FILE_NAME = os.getenv("GOOGLE_MODEL_INFO_FILE_NAME")
CUSTOM_MODEL_INFO_FILE_NAME = os.getenv("CUSTOM_MODEL_INFO_FILE_NAME")
FEATURE_EXTRACTION_FILE_NAME = os.getenv("FEATURE_EXTRACTION_MODEL_FILE_NAME")
GPU_NAME = os.getenv("GPU_NAME")
NUM_WORKERS = int(os.getenv("NUM_WORKERS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LR = float(os.getenv("LR"))  # Learning rate
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES"))
NOISE_FILE = os.getenv("NOISE_FILE")  # The noise added to audio clips when testing (if needed)
TEST_FILE = os.getenv("TEST_FILE")  # The audio file to test
ROOT_DIR = os.getenv("ROOT_DIR")  # The folder which contains the models
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))  # The weight devay for the trainig phase
N_FFT = int(os.getenv("N_FFT"))  # Parameter for spectrogram
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))  # Parameter for spectrogram
PATIENCE = int(os.getenv("PATIENCE"))  # Patience when training
RATIO_LOSS_DOA = float(os.getenv("RATIO_LOSS_DOA"))  # loss multiplier for doa error calculation
RATIO_NOISE = float(os.getenv("RATIO_NOISE"))  # loss multiplier for doa error calculation
TRAINING_SESSION_DIR_NAME = os.getenv("TRAINING_SESSION_DIR_NAME")  # the directory where figures will be saved
NB_TRAINING = int(os.getenv("NB_TRAINING")) # The number of training for each dataset
STEP = float(os.getenv("STEP")) #If multiple training for one dataset, this is the weight decay step
TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH") #The test validation dataset (see validation_ugv_dataset)
MULTIPLE_DATASET = os.getenv("MULTIPLE_DATASET") == "True" #To train over multiple dataset or not
MULTIPLE_DATASET_PATH = os.getenv("MULTIPLE_DATASET_PATH") #The path to the root of the multiple datasets


def main(
    train,
    display_validation,
    test,
    nb_epoch,
    use_google,
    session_name,
    debug,
    valid_onnx
):
    """
    Main function of the module. 
    Args:
        train (bool): Whether to train the model or not.
        display_validation (bool): Whether to display the predictions on the validation dataset or not.
        test (bool): Whether to display the predictions on the test dataset or not.
        nb_epoch (int): Number of epochs for which to train the network.
        use_google (bool): Whether to train with the Google dataset to do transfer learning afterwards, or to train on
            the custom dataset with transfer learning.
        training_name(str) : the name of the training , the figures will be saved under this directory 
    """

    try:
        os.mkdir(os.path.join(ROOT_DIR, TRAINING_SESSION_DIR_NAME))
    except FileExistsError:
        pass
    except Exception :
        raise

    device = GPU_NAME if torch.cuda.is_available() else "cpu"


    print("DEVICE:", device)
    
    if session_name == "":
        session_name = input(
            "Enter the name of the current execution, this name will be used to name the different figures : "
        )

    session_name = session_name.replace(".", "_").replace(" ", "")
    session_dir = os.path.join(TRAINING_SESSION_DIR_NAME, session_name)
    
    if train:
        validation_dataset = AcousticDataset(TEST_DATASET_PATH,False)
        
        validation_loader  = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        for training_nb in range(NB_TRAINING):

            if MULTIPLE_DATASET:
                dataset_paths = os.listdir(MULTIPLE_DATASET_PATH)
                for i in range(len(dataset_paths)):
                    dataset_paths[i] = os.path.join(MULTIPLE_DATASET_PATH, dataset_paths[i])
            else:
                dataset_paths = [GOOGLE_DATASET_PATH if use_google else CUSTOM_DATASET_PATH]

            print(f"datasets : {dataset_paths}")

            for dataset_path in dataset_paths:
                dataset_name = os.path.basename(dataset_path)
                dataset = AcousticDataset(dataset_path)
                num_classes = len(dataset.class_weight)
                
                training_type = "GOOGLE" if use_google else "CUSTOM"
                print(f"{NB_TRAINING} training(s) have been queued with the current parameters:\n"+
                    f"Learning rate : {LR}\n"
                    f"Weight decay : {WEIGHT_DECAY}\n"
                    f"Step : {STEP}\n"
                    f"Noise ratio  : {RATIO_NOISE}\n"
                    f"DOA ratio : {RATIO_LOSS_DOA}\n"
                    f"using the  {dataset_name} dataset\n"
                    f"number of class {num_classes}\n"
                    f"Total number of samples in the dataset : {len(dataset)}\n"
                    f"Weight classes computed for the dataset: {dataset.class_weight}"
                    )

                train_loader,_,_ = create_dataset_loaders(dataset=dataset)
                model = AudioClassification(num_class=num_classes)

                if not use_google:
                    model.feature_extraction.load_best_model(
                        directory=ROOT_DIR,
                        device=device,
                        feature_extraction_file_name=FEATURE_EXTRACTION_FILE_NAME,
                    )
                
                model.to(device)

                weigth_decay = WEIGHT_DECAY + training_nb * STEP

                current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
                training_name = f"{training_type}_LR_{round(LR,6)}_WD_{round(weigth_decay,6)}_NOISE_{round(RATIO_NOISE,6)}_DOA_{round(RATIO_LOSS_DOA,6)}_{session_name}_{dataset_name}_{current_time}"
                training_name = training_name.replace(".", "_")
                training_dir = os.path.join(TRAINING_SESSION_DIR_NAME, training_name)

                try:
                    os.mkdir(os.path.join(ROOT_DIR, TRAINING_SESSION_DIR_NAME, training_name))
                except FileExistsError:
                    pass
                except Exception :
                    raise

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=LR,
                    weight_decay=weigth_decay,
                )

                trainer = Trainer(
                    training_loader=train_loader,
                    validation_loader=validation_loader,
                    loss_function_classification=torch.nn.CrossEntropyLoss(weight=dataset.class_weight.to(device)),
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    MODEL_INFO_FILE_NAME=GOOGLE_MODEL_INFO_FILE_NAME if use_google else CUSTOM_MODEL_INFO_FILE_NAME,
                    FEATURE_EXTRACTION_FILE_NAME=FEATURE_EXTRACTION_FILE_NAME,
                    ROOT_DIR_PATH=ROOT_DIR,
                    patience=PATIENCE,
                    training_name=training_name,
                    training_session_dir_name=training_dir,
                    ratio_loss_doa=RATIO_LOSS_DOA,
                )
                trainer.train_loop(nb_epoch)

    if use_google:
        dataset = AcousticDataset(dataset_path=GOOGLE_DATASET_PATH)
        model = AudioClassification(num_class=len(dataset.class_weight))
        model.load_best_model(
            directory=ROOT_DIR,
            device=device,
            classification_file_name=GOOGLE_MODEL_INFO_FILE_NAME,
        )
    else:
        dataset = AcousticDataset(dataset_path=CUSTOM_DATASET_PATH)
        model = AudioClassification(num_class=len(dataset.class_weight))
        model.load_best_model(
            directory=ROOT_DIR,
            device=device,
            classification_file_name=CUSTOM_MODEL_INFO_FILE_NAME,
        )
    
    print(model)
    print("Model's number of parameters : {}".format(sum(p.numel() for p in model.parameters())))
    model.to(device=device)
    _, valid_loader, _ = create_dataset_loaders(dataset=dataset)

    if display_validation:
        visualize_predictions(
            model,
            valid_loader,
            device,
            dataset,
            dir=session_dir,
            title=f"Validation_{session_name}",
            debug=False,
        )

    if test:
        dataset = AcousticDataset(TEST_DATASET_PATH,False)
        
        test_loader  = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        visualize_predictions_onnx(onnx,test_loader,dataset,dir=session_dir,title=f"Test_{session_name}")

  
        model.eval()
        model.feature_extraction.set_bn(True)
        visualize_predictions(
            model,
            test_loader,
            device,
            dataset,
            dir=session_dir,
            title=f"Test_{session_name}",
            debug=debug,
        )


def create_dataset_loaders(dataset: AcousticDataset):
    """Split the entire dataset into multiple dataloader (training ,valid and test)
    The corresponding size are 70% , 20% and 10% of the initial size

    Args:
        dataset (AcousticDataset): The dataset to split

    Returns:
        DataLoader,DataLoader,DataLoader: The three dataloader
    """
    train_size = int(len(dataset) * 0.7)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size

    (
        training_sub_dataset,
        validation_sub_dataset,
        test_sub_dataset,
    ) = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(  # Batch size to one for real life simulation
        test_sub_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, valid_loader, test_loader


def visualize_predictions_onnx(onnx:NeuralNetworkOnnx,dataloader,dataset:AcousticDataset,dir,title):
    """
    Used to visualize the target and the predictions of the onnx model on some input .wav signals.
    Args:
        model (NeuralNetworkOnnx): The model used to perform the predictions (onnx).
        dataloader (Dataloader): The dataloader that provides the signals and the targets.
        device (string): The device on which to perform the computations.
        dataset (AcousticDataset): Dataset used to visualize predictions.
        dir(string) : The training directory to write the graph's in
        title (string): The title given to the plot.
    """
    confusion_mat = np.zeros((len(dataset.intToName), len(dataset.intToName)), dtype=int)  
    error =0
    total =0 
    for signals,labels,labels_doa in dataloader:
        pred,_,_ = onnx.from_spectrogram(signals)
        pred_index = dataset.nameToInt[pred]
        label = labels.item()
        try:
            confusion_mat[label][pred_index] += 1
        except:
            pass
        if label != pred_index:
            error += 1
        total += 1

    try:
        os.mkdir(os.path.join(ROOT_DIR, dir))
    except FileExistsError:
        pass
    except Exception :
        raise

    confusion_matrix(confusion_mat, dataset.intToName, total, error, "Test")
    plt.gcf()
    plt.savefig(os.path.join(ROOT_DIR, dir, title))

def visualize_predictions(model:AudioClassification, dataloader, device, dataset, dir, title, debug):
    """
    Used to visualize the target and the predictions of the model on some input .wav signals.
    Args:
        model (Module): The model used to perform the predictions.
        dataloader (Dataloader): The dataloader that provides the signals and the targets.
        device (string): The device on which to perform the computations.
        dataset (AcousticDataset): Dataset used to visualize predictions.
        dir(string) : The training directory to write the graph's in
        title (string): The title given to the plot.
        debug (bool) : to log the differences between the train and eval mode
    """
    try:
        os.mkdir(os.path.join(ROOT_DIR, dir))
    except FileExistsError:
        pass
    except Exception :
        raise
    errors = 0
    total = 0
    confusion_mat = np.zeros((len(dataset.intToName), len(dataset.intToName)), dtype=int)
    MSE_loss = torch.nn.MSELoss()
    doa_MSE = 0.0
    total_3D_angle_error = 0.0

    diff_pred = torch.zeros((1, len(dataset.intToName)))
    diff_doa = torch.zeros((1, 3))
    single_pred_array = []
    diff_cert = []


    with torch.no_grad():
        for signals, labels, labels_doa in dataloader:
            signals, labels, labels_doa = (
                signals.to(device),
                labels.to(device),
                labels_doa.to(device),
            )
            
            predictions_eval, doas_eval, _ = model(signals)
            predictions = torch.softmax(predictions_eval, dim=1)
            total_3D_angle_error += get_3d_angle_torch(doas_eval, labels_doa) * 180 / torch.pi

            for prediction, label, doa, label_doa in zip(predictions, labels, doas_eval, labels_doa):
                pred = torch.argmax(prediction, dim=-1).item()
                label = label.item()
                try:
                    confusion_mat[label][pred] += 1
                except:
                    pass
                if label != pred:
                    errors += 1
                total += 1

            doa_MSE += MSE_loss(doas_eval, labels_doa)

            if debug:
                diff_pred, diff_doa, cert, log = test_model_mode(
                    model, signals, dataset, diff_pred, diff_doa, labels.item(), True
                )
                diff_cert.append(cert)
                single_pred_array.append(log)

        total_3D_angle_error /= len(dataloader)

    if debug:
        log_test_to_file(
            os.path.join(ROOT_DIR, dir, "log.txt"),
            diff_pred,
            diff_doa,
            total,
            single_pred_array,
            diff_cert,
        )

    confusion_matrix(confusion_mat, dataset.intToName, total, errors, "Test")
    plt.gcf()
    plt.savefig(os.path.join(ROOT_DIR, dir, title))
    print(
        f"MSE DoA: {doa_MSE.item() / len(dataloader)} -- 3D DoA angle error:{round(total_3D_angle_error.item(), 2)} deg"
    )
    return


def test_model_mode(model, signal, dataset, diff_pred, diff_doa, label, isCleaned):

    predictions_eval, doas_eval, _ = model(signal)
    pred_eval = torch.softmax(predictions_eval, dim=1)[0]
    index_pred_eval = torch.argmax(pred_eval, dim=-1).item()
    label_eval = dataset.intToName[index_pred_eval]

    model.train()
    predictions_train, doas_train, _ = model(signal)
    pred_train = torch.softmax(predictions_train, dim=1)[0]
    index_pred_train = torch.argmax(pred_train, dim=-1).item()
    label_train = dataset.intToName[index_pred_train]

    diff_pred = torch.add(diff_pred, torch.abs((predictions_train - predictions_eval)))
    diff_doa = torch.add(diff_doa, torch.abs((doas_train - doas_eval)))
    cert_diff = abs(pred_eval[index_pred_eval] - pred_train[index_pred_train])

    if isCleaned and (label_eval == label_train and cert_diff < 0.05):
        return diff_pred, diff_doa, cert_diff, ("", "", "", "")

    return (
        diff_pred,
        diff_doa,
        cert_diff,
        (
            f"Correct label : {dataset.intToName[label]}\n",
            f"Eval  {label_eval}, {pred_eval[index_pred_eval]:2f} ,DATA: {predictions_eval}\n",
            f"Train {label_train}, {pred_train[index_pred_train]:2f} ,DATA: {predictions_train}\n",
            f"Certainty diff : {cert_diff:2f}\n\n",
        ),
    )


def log_test_to_file(path, diff_pred, diff_doa, num_samples, single_pred_array, cert):
    with open(path, "w") as f:
        f.write("All audio diff\n")
        for tuple in single_pred_array:
            f.write(f"{tuple[0]}{tuple[1]}{tuple[2]}{tuple[3]}")
        f.write(f"Mean difference on prediction: {torch.div(diff_pred,num_samples)[0]}\n")
        f.write(f"Mean difference on doa: {torch.div(diff_doa,num_samples)[0]}\n")
        f.write(f"Mean difference on certitude: {torch.div(torch.Tensor(cert),num_samples)[0]}\n")


def confusion_matrix(confusion_mat, intToName, total, errors, title):
    plt.clf()
    labels = [k for k in intToName.values()]
    c = plt.pcolormesh(confusion_mat, cmap="bone_r")
    plt.colorbar(c)
    plt.xticks(np.arange(0.5, len(labels)), labels, rotation=90, size=8)
    plt.yticks(np.arange(0.5, len(labels)), labels, size=8)
    plt.title(
        f"{title}: {total-errors}/{total}({100 - round(100 * errors / total, 2)}%)",
        fontweight="bold",
    )
    plt.tight_layout()
    for y in range(confusion_mat.shape[0]):
        for x in range(confusion_mat.shape[1]):
            if (confusion_mat[y, x] > 0):
                text = str(confusion_mat[y, x])
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    text % confusion_mat[y, x],
                    size=6,
                    horizontalalignment="center",
                    verticalalignment="center",
                )


def neural_network_inference(model, device, audio):
    """
    Used to load a .wav file on disk and predict the class and the doa of the signal.
    Args:
        model (Module): The model used to perform the predictions.
        device (string): The device on which to perform the computations.
        audio(string): The path to the audio file to test
    """
    data_transformation = DataTransformation(n_fft= N_FFT,hop_length= HOP_LENGTH, sample_rate =16000, audio_length_in_seconds= NUM_SAMPLES/16000)
    input_rnn = data_transformation.from_audio_file(torch.unsqueeze(audio,dim=0))
    prediction_class, prediction_doa, _ = model(input_rnn.to(device))
    index,certainty = data_transformation.class_certainty(prediction_class)
    print(f"prediction index : {index}, certitude : {certainty}, doa : {prediction_doa}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the neural network.")
    parser.add_argument(
        "-v",
        "--display_validation",
        action="store_true",
        help="Display the predictions of the neural network on the validation dataset.",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Display the predictions of the neural network on the test dataset.",
    )
    parser.add_argument(
        "-e",
        "--nb_epochs",
        action="store",
        type=int,
        default=100,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "-g",
        "--use_google",
        action="store_true",
        help="Use this option to set the dataset to Google. Training on the Google dataset for transfer learning on the"
        " custom dataset.",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug the training vs eval model",
    )

    parser.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        default="",
        help="The name of the training session, this will be used to name the training folder and different graphs",
    )
    
    parser.add_argument(
        "-o",
        "--onnx",
        action="store_true",
        help="use the onnx model for validation a test file",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(
        args.train,
        args.display_validation,
        args.predict,
        args.nb_epochs,
        args.use_google,
        args.name,
        args.debug,
        args.onnx
    )
