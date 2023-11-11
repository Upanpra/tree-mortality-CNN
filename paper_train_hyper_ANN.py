# import files for functions
import time
import os

from sklearn.metrics import confusion_matrix
import torch

from src.modelANN import MortalityANN
from src.trainer import trainCNN
from src.accuracy import get_Ytrue_YPredict
from src.functions import *
from src.data_loader import transforms_aug, data_loader

# setting up train and test folder


train_folder = "data/training_data/4classes/hyper/train_hyper"  # Assumes you have downloaded our data and placed it at the repo root
test_folder = "data/training_data/4classes/hyper/test_hyper"



mean, std = get_mean_std(train_folder, modality = 'hyper')

print(mean)
print(std)
print("Hyperspectral with 426 bands at 1 m resolution with ANN with STD")

# set all the below parameters change based on which data type cnn is being running for
INPUT_CHANNELS = 426
INPUT_SIZE = [128, 128]
NUM_CLASSES = 4            # The number of output classes. In this case, from 1 to 4
NUM_EPOCHS = 40            # change to 200 # The number of times we loop over the whole dataset during training
BATCH_SIZE = 16            # Change to 16 or 32 The size of input data took for one iteration of an epoch
LEARNING_RATE = 1e-3       # The speed of convergence

BASE_MODEL_OUTPUT_FOLDER = "provide path to save checkpoint"

model_name = "MortalityANN"

channel_indices = None  # Set to None to not select any channels
if channel_indices is not None:
    INPUT_CHANNELS = len(channel_indices)

timestamp = time.time()
checkpoint_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"model-HYPER-{model_name}-{timestamp}.pt")
bestmodel_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"best-test-model-HYPER{model_name}-{timestamp}.pt")
print(f"Saving model to: {bestmodel_path}")

# use transforms_aug function from data loader to do augmentation
transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

# use data_loader function from data loader
train_loader, test_loader = data_loader(train_folder, test_folder, INPUT_SIZE, BATCH_SIZE, mean, std, channel_indices=channel_indices)

# calling cnn functions: MortalityCNN
net = MortalityANN(input_size=[INPUT_CHANNELS] + INPUT_SIZE, first_layer_conv2d=True)
print(net)

start = time.time()
print(start/60)

net, train_history, test_history = trainCNN(net, train_loader, test_loader,
                                        num_epochs=NUM_EPOCHS, 
                                        learning_rate=LEARNING_RATE,
                                        modelsavepath=bestmodel_path,
                                        compute_accs=True)

end = time.time()
print(end)
print(f" Elapsed time: {(end-start)/60} minutes")
print(f" Elapsed time: {(end-start)/60/60} hours")
print(time.time())



# calling for accuracy
y_true, y_predict = get_Ytrue_YPredict(net, test_loader )

confusion_results = confusion_matrix(y_true, y_predict)
print(confusion_results)

confusion_results1 = confusion_matrix(y_true, y_predict, normalize="true")
print(confusion_results1)

confusion_results2 = confusion_matrix(y_true, y_predict, normalize="pred")
print(confusion_results2)

torch.save(net.state_dict(), checkpoint_path)